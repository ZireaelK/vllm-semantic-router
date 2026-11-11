package extproc

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

const gatewayReplayConfigTemplate = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
global:
  services:
    router_replay:
      enabled: true
      store_backend: memory
providers:
  defaults:
    model: primary-model
  models:
    - name: primary-model
      provider_model_id: primary-model
      api_format: openai
      reliability:
        retry_count: 0
        first_byte_timeout: {{TIMEOUT}}
      backend_refs:
        - name: primary
          endpoint: {{ENDPOINT}}
          protocol: http
          provider: vllm
routing:
  modelCards:
    - name: primary-model
  decisions:
    - name: default_route
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: primary-model
          use_reasoning: false
      algorithm:
        type: static
      plugins:
        - type: router_replay
          configuration:
            enabled: true
            capture_request_body: true
            capture_response_body: true
`

func setupGatewayReplayHarness(t *testing.T, configYAML string) (*OpenAIRouter, *httptest.Server) {
	t.Helper()
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(configYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(configPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = router.Close() })
	set, err := upstream.Build(router.Config, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = set.Close(ctx)
	})
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	handler, err := gateway.NewHandler(gateway.Options{
		Serving:  gateway.Static(gateway.Serving{Engine: routing.NewEngine(NewRouterService(router), opts), Upstream: set}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}
	frontend := httptest.NewServer(handler)
	t.Cleanup(frontend.Close)
	return router, frontend
}

func getLatestReplayRecord(router *OpenAIRouter) (routerreplay.RoutingRecord, bool) {
	if router.ReplayRecorder != nil {
		records := router.ReplayRecorder.ListAllRecords()
		if len(records) > 0 {
			return records[len(records)-1], true
		}
	}
	for _, rec := range router.ReplayRecorders {
		records := rec.ListAllRecords()
		if len(records) > 0 {
			return records[len(records)-1], true
		}
	}
	return routerreplay.RoutingRecord{}, false
}

func TestGatewayReplayConnectFailure(t *testing.T) {
	// Allocate and close a local TCP port to guarantee connection refused
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	deadAddr := listener.Addr().String()
	_ = listener.Close()

	cfgYAML := strings.Replace(gatewayReplayConfigTemplate, "{{ENDPOINT}}", deadAddr, 1)
	cfgYAML = strings.Replace(cfgYAML, "{{TIMEOUT}}", "5s", 1)

	router, frontend := setupGatewayReplayHarness(t, cfgYAML)

	reqBody := `{"model":"primary-model","messages":[{"role":"user","content":"hello"}]}`
	req, err := http.NewRequest(http.MethodPost, frontend.URL+"/v1/chat/completions", strings.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("unexpected request error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("client status = %d (%s), want 503", resp.StatusCode, string(body))
	}

	record, found := getLatestReplayRecord(router)
	if !found {
		t.Fatal("expected replay record to be recorded, but found none")
	}

	if record.ResponseStatus != http.StatusServiceUnavailable {
		t.Errorf("replay ResponseStatus = %d, want 503", record.ResponseStatus)
	}
	if record.LifecycleState != routerreplay.LifecycleFailed {
		t.Errorf("replay LifecycleState = %q, want %q", record.LifecycleState, routerreplay.LifecycleFailed)
	}
	if record.TerminalReason != "upstream_error_response" {
		t.Errorf("replay TerminalReason = %q, want %q", record.TerminalReason, "upstream_error_response")
	}
}

func TestGatewayReplayFirstByteTimeout(t *testing.T) {
	// Start an HTTP backend that sleeps longer than first_byte_timeout
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer backend.Close()

	backendAddr := strings.TrimPrefix(backend.URL, "http://")
	cfgYAML := strings.Replace(gatewayReplayConfigTemplate, "{{ENDPOINT}}", backendAddr, 1)
	cfgYAML = strings.Replace(cfgYAML, "{{TIMEOUT}}", "50ms", 1)

	router, frontend := setupGatewayReplayHarness(t, cfgYAML)

	reqBody := `{"model":"primary-model","messages":[{"role":"user","content":"hello"}]}`
	req, err := http.NewRequest(http.MethodPost, frontend.URL+"/v1/chat/completions", strings.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("unexpected request error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusGatewayTimeout {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("client status = %d (%s), want 504", resp.StatusCode, string(body))
	}

	record, found := getLatestReplayRecord(router)
	if !found {
		t.Fatal("expected replay record to be recorded, but found none")
	}

	if record.ResponseStatus != http.StatusGatewayTimeout {
		t.Errorf("replay ResponseStatus = %d, want 504", record.ResponseStatus)
	}
	if record.LifecycleState != routerreplay.LifecycleFailed {
		t.Errorf("replay LifecycleState = %q, want %q", record.LifecycleState, routerreplay.LifecycleFailed)
	}
	if record.TerminalReason != "upstream_error_response" {
		t.Errorf("replay TerminalReason = %q, want %q", record.TerminalReason, "upstream_error_response")
	}
}

func TestGatewayReplayClientCancel(t *testing.T) {
	// Start an HTTP backend that hangs
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()

	backendAddr := strings.TrimPrefix(backend.URL, "http://")
	cfgYAML := strings.Replace(gatewayReplayConfigTemplate, "{{ENDPOINT}}", backendAddr, 1)
	cfgYAML = strings.Replace(cfgYAML, "{{TIMEOUT}}", "10s", 1)

	router, frontend := setupGatewayReplayHarness(t, cfgYAML)

	ctx, cancel := context.WithCancel(context.Background())
	reqBody := `{"model":"primary-model","messages":[{"role":"user","content":"hello"}]}`
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, frontend.URL+"/v1/chat/completions", strings.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	resp, err := http.DefaultClient.Do(req)
	if resp != nil {
		_ = resp.Body.Close()
	}
	if err == nil || (!errors.Is(err, context.Canceled) && !strings.Contains(err.Error(), "canceled")) {
		t.Fatalf("expected client context cancellation error, got %v", err)
	}

	// Wait briefly for the server to finish session teardown
	var record routerreplay.RoutingRecord
	var found bool
	for range 20 {
		record, found = getLatestReplayRecord(router)
		if found && record.LifecycleState != "" && record.LifecycleState != routerreplay.LifecycleInProgress {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	if !found {
		t.Fatal("expected replay record to be recorded, but found none")
	}

	if record.LifecycleState != routerreplay.LifecycleAborted {
		t.Errorf("replay LifecycleState = %q, want %q", record.LifecycleState, routerreplay.LifecycleAborted)
	}
	if record.TerminalReason != "client_canceled" {
		t.Errorf("replay TerminalReason = %q, want %q", record.TerminalReason, "client_canceled")
	}
}

type mockImmediateUpstream struct {
	result *upstream.Result
}

func (m *mockImmediateUpstream) Execute(ctx context.Context, call *routing.Call, listener string) (*upstream.Result, error) {
	return m.result, nil
}

type sessionCapturingProcessor struct {
	inner       routing.Processor
	lastSession *routingSession
}

func (p *sessionCapturingProcessor) Open(ctx context.Context) (routing.Session, error) {
	sess, err := p.inner.Open(ctx)
	if err != nil {
		return nil, err
	}
	if rs, ok := sess.(*routingSession); ok {
		p.lastSession = rs
	}
	return sess, nil
}

func TestGatewayReplayFallbackImmediate(t *testing.T) {
	cfgYAML := strings.Replace(gatewayReplayConfigTemplate, "{{ENDPOINT}}", "127.0.0.1:8000", 1)
	cfgYAML = strings.Replace(cfgYAML, "{{TIMEOUT}}", "5s", 1)

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(cfgYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(configPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = router.Close() })

	mockUpstream := &mockImmediateUpstream{
		result: &upstream.Result{
			Immediate: &routing.Response{
				Status: http.StatusServiceUnavailable,
				Header: routing.Header{{Name: "content-type", Value: "application/json"}},
				Body:   []byte(`{"error":"all backends unavailable"}`),
			},
		},
	}

	capturing := &sessionCapturingProcessor{inner: NewRouterService(router)}
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	handler, err := gateway.NewHandler(gateway.Options{
		Serving: gateway.Static(gateway.Serving{
			Engine:   routing.NewEngine(capturing, opts),
			Upstream: mockUpstream,
		}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}

	reqBody := `{"model":"primary-model","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	// 1) Downstream HTTP client receives 503 status
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("client status = %d (%s), want 503", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "all backends unavailable") {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}

	// 2) ctx.UpstreamStatusCode is 503
	if capturing.lastSession == nil || capturing.lastSession.ctx == nil {
		t.Fatal("expected routingSession and RequestContext to be captured")
	}
	reqCtx := capturing.lastSession.ctx
	if reqCtx.UpstreamStatusCode != http.StatusServiceUnavailable {
		t.Errorf("ctx.UpstreamStatusCode = %d, want 503", reqCtx.UpstreamStatusCode)
	}

	// 3) Replay record in store has ResponseStatus == 503
	record, found := getLatestReplayRecord(router)
	if !found {
		t.Fatal("expected replay record to be recorded, but found none")
	}
	if record.ResponseStatus != http.StatusServiceUnavailable {
		t.Errorf("replay ResponseStatus = %d, want 503", record.ResponseStatus)
	}

	// 4) TerminalLifecycleState == routerreplay.LifecycleFailed
	if reqCtx.TerminalLifecycleState != routerreplay.LifecycleFailed {
		t.Errorf("ctx.TerminalLifecycleState = %q, want %q", reqCtx.TerminalLifecycleState, routerreplay.LifecycleFailed)
	}
	if record.LifecycleState != routerreplay.LifecycleFailed {
		t.Errorf("replay LifecycleState = %q, want %q", record.LifecycleState, routerreplay.LifecycleFailed)
	}

	// 5) TerminalLifecycleReason == "upstream_error_response"
	if reqCtx.TerminalLifecycleReason != "upstream_error_response" {
		t.Errorf("ctx.TerminalLifecycleReason = %q, want %q", reqCtx.TerminalLifecycleReason, "upstream_error_response")
	}
	if record.TerminalReason != "upstream_error_response" {
		t.Errorf("replay TerminalReason = %q, want %q", record.TerminalReason, "upstream_error_response")
	}
}
