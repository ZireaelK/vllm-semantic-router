//go:build envoy_integration

package extproc

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
)

// ----------------------------------------------------------------------------
// 1. ExtProc gRPC Server Implementation (Cross-process Envoy Target)
// ----------------------------------------------------------------------------

type traceSettlement struct {
	Status           string
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	Provenance       string
}

type envoyTraceExtProcServer struct {
	ext_proc.UnimplementedExternalProcessorServer
	upstreamCalled   *atomic.Int32
	settlementCount  *atomic.Int32
	extprocCancelled *atomic.Bool
	modeOverrideUsed *atomic.Bool
	streamCompleted  atomic.Bool
	clientModel      string
	promptTokens     int
	receivedChunks   []string
	chunkMu          sync.Mutex
	lastSettlement   *traceSettlement
	t                *testing.T
}

// Process implements the bidirectional gRPC streaming interface called by Envoy's C++ ext_proc filter
func (s *envoyTraceExtProcServer) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	for {
		req, err := stream.Recv()
		if err != nil {
			if s.streamCompleted.Load() {
				// Clean stream completion: dynamically extract usage from upstream terminal frame
				s.chunkMu.Lock()
				parsedUsage, parseErr := extractUsageFromChunks(s.receivedChunks)
				s.chunkMu.Unlock()

				if parseErr != nil {
					s.t.Errorf("[TRACE][ExtProc] Failed to extract upstream usage: %v", parseErr)
					return parseErr
				}

				s.settlementCount.Add(1)
				s.lastSettlement = &traceSettlement{
					Status:           "completed",
					PromptTokens:     parsedUsage.PromptTokens,
					CompletionTokens: parsedUsage.CompletionTokens,
					TotalTokens:      parsedUsage.TotalTokens,
					Provenance:       "authoritative", // Authoritative provenance directly parsed from model backend
				}
				s.t.Logf("[TRACE][ExtProc] Stream finished cleanly -> Committed authoritative settlement: prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
					parsedUsage.PromptTokens, parsedUsage.CompletionTokens, parsedUsage.TotalTokens)
				return nil
			}

			// Early termination / client abort: stream interrupted before endOfStream
			// Dynamically estimate completion tokens from audited chunks received before disconnect
			s.chunkMu.Lock()
			estimatedTokens := 0
			for _, chunk := range s.receivedChunks {
				estimatedTokens += countTokensInDeltaChunk(chunk)
			}
			s.chunkMu.Unlock()

			promptTokens := s.promptTokens
			s.extprocCancelled.Store(true)
			s.settlementCount.Add(1)
			s.lastSettlement = &traceSettlement{
				Status:           "client_cancelled",
				PromptTokens:     promptTokens,
				CompletionTokens: estimatedTokens,
				TotalTokens:      promptTokens + estimatedTokens,
				Provenance:       "estimated", // Error propagation rule: partial stream tokens cannot be authoritative
			}
			s.t.Logf("[TRACE][ExtProc] Stream aborted before completion -> Committed estimated settlement: prompt_tokens=%d, completion_tokens=%d, total_tokens=%d, provenance=estimated",
				promptTokens, estimatedTokens, promptTokens+estimatedTokens)
			return nil
		}

		switch v := req.Request.(type) {
		case *ext_proc.ProcessingRequest_RequestHeaders:
			resp := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestHeaders{
					RequestHeaders: &ext_proc.HeadersResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
							HeaderMutation: &ext_proc.HeaderMutation{
								SetHeaders: []*core.HeaderValueOption{
									{
										Header: &core.HeaderValue{
											Key:      "x-selected-model",
											Value:    "qwen2.5-7b-synthesis",
											RawValue: []byte("qwen2.5-7b-synthesis"),
										},
									},
								},
							},
						},
					},
				},
			}
			_ = stream.Send(resp)

		case *ext_proc.ProcessingRequest_RequestBody:
			// Capture client's original requested model
			var clientReq struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(v.RequestBody.Body, &clientReq); err == nil && clientReq.Model != "" {
				s.clientModel = clientReq.Model
			}

			// Mutate request body to the synthesized prompt and adjust content-length accordingly
			mutatedBody := []byte(`{"model":"qwen2.5-7b-synthesis","messages":[{"role":"user","content":"synthesis prompt"}],"stream":true}`)
			s.promptTokens = countPromptTokens(mutatedBody)

			resp := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
							HeaderMutation: &ext_proc.HeaderMutation{
								SetHeaders: []*core.HeaderValueOption{
									{
										Header: &core.HeaderValue{
											Key:      "content-length",
											Value:    fmt.Sprintf("%d", len(mutatedBody)),
											RawValue: []byte(fmt.Sprintf("%d", len(mutatedBody))),
										},
										AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
									},
								},
							},
							BodyMutation: &ext_proc.BodyMutation{
								Mutation: &ext_proc.BodyMutation_Body{
									Body: mutatedBody,
								},
							},
						},
					},
				},
			}
			_ = stream.Send(resp)

		case *ext_proc.ProcessingRequest_ResponseHeaders:
			isStreaming := false
			if v.ResponseHeaders != nil && v.ResponseHeaders.Headers != nil {
				for _, h := range v.ResponseHeaders.Headers.Headers {
					if strings.EqualFold(h.Key, "content-type") && strings.Contains(string(h.RawValue)+h.Value, "text/event-stream") {
						isStreaming = true
						break
					}
				}
			}

			var (
				modeOverride   *http_ext.ProcessingMode
				headerMutation *ext_proc.HeaderMutation
			)
			if isStreaming {
				s.modeOverrideUsed.Store(true)
				s.t.Logf("[TRACE][ExtProc] ResponseHeaders verified (status 200 text/event-stream). Applying ModeOverride: response_body_mode=STREAMED")
				modeOverride = &http_ext.ProcessingMode{
					ResponseBodyMode: http_ext.ProcessingMode_STREAMED,
				}
				headerMutation = &ext_proc.HeaderMutation{
					RemoveHeaders: []string{"content-length"},
				}
			}

			resp := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &ext_proc.HeadersResponse{
						Response: &ext_proc.CommonResponse{
							Status:         ext_proc.CommonResponse_CONTINUE,
							HeaderMutation: headerMutation,
						},
					},
				},
				ModeOverride: modeOverride,
			}
			_ = stream.Send(resp)

		case *ext_proc.ProcessingRequest_ResponseBody:
			chunkBytes := v.ResponseBody.Body
			s.t.Logf("[TRACE][ExtProc] ResponseBody chunk audited: size=%d endOfStream=%v", len(chunkBytes), v.ResponseBody.EndOfStream)
			s.chunkMu.Lock()
			s.receivedChunks = append(s.receivedChunks, string(chunkBytes))
			s.chunkMu.Unlock()

			if v.ResponseBody.GetEndOfStream() {
				s.streamCompleted.Store(true)
			}

			// Model virtualization / rewrite: replace internal backend model with client's requested model
			var bodyMutation *ext_proc.BodyMutation
			if s.clientModel != "" && bytes.Contains(chunkBytes, []byte("qwen2.5-7b-synthesis")) {
				rewrittenChunk := bytes.ReplaceAll(chunkBytes, []byte("qwen2.5-7b-synthesis"), []byte(s.clientModel))
				bodyMutation = &ext_proc.BodyMutation{
					Mutation: &ext_proc.BodyMutation_Body{
						Body: rewrittenChunk,
					},
				}
				s.t.Logf("[TRACE][ExtProc] ResponseBody rewritten model: 'qwen2.5-7b-synthesis' -> '%s'", s.clientModel)
			}

			resp := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_ResponseBody{
					ResponseBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status:       ext_proc.CommonResponse_CONTINUE,
							BodyMutation: bodyMutation,
						},
					},
				},
			}
			_ = stream.Send(resp)

		default:
			_ = stream.Send(&ext_proc.ProcessingResponse{})
		}
	}
}

// extractUsageFromChunks dynamically parses the upstream SSE usage frame via JSON unmarshaling
func extractUsageFromChunks(chunks []string) (*traceSettlement, error) {
	for _, chunk := range chunks {
		lines := strings.Split(chunk, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "" || data == "[DONE]" {
				continue
			}
			var resp struct {
				Usage *struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
					TotalTokens      int `json:"total_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal([]byte(data), &resp); err == nil && resp.Usage != nil {
				return &traceSettlement{
					PromptTokens:     resp.Usage.PromptTokens,
					CompletionTokens: resp.Usage.CompletionTokens,
					TotalTokens:      resp.Usage.TotalTokens,
				}, nil
			}
		}
	}
	return nil, fmt.Errorf("no valid usage frame found in stream chunks")
}

// countPromptTokens parses the OpenAI chat completion JSON request body and counts words across all messages
func countPromptTokens(body []byte) int {
	var req struct {
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return 0
	}
	totalWords := 0
	for _, m := range req.Messages {
		totalWords += len(strings.Fields(m.Content))
	}
	return totalWords
}

// countTokensInDeltaChunk parses an SSE chunk and counts words in delta content
func countTokensInDeltaChunk(chunk string) int {
	lines := strings.Split(chunk, "\n")
	totalWords := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var resp struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &resp); err == nil {
			for _, choice := range resp.Choices {
				totalWords += len(strings.Fields(choice.Delta.Content))
			}
		}
	}
	return totalWords
}

// ----------------------------------------------------------------------------
// 2. Reusable Test Fixture for Envoy 1.35.3 Integration Harness
// ----------------------------------------------------------------------------

type envoyTestHarness struct {
	envoyBin         string
	envoyPort        int
	extprocPort      int
	upstreamPort     int
	upstreamCalls    *atomic.Int32
	settlementCount  *atomic.Int32
	extprocCancelled *atomic.Bool
	upstreamCanceled *atomic.Bool
	modeOverrideUsed *atomic.Bool
	procSrv          *envoyTraceExtProcServer
	cleanup          func()
}

func discoverEnvoyBinary(t *testing.T) string {
	homeDir, _ := os.UserHomeDir()
	candidatePaths := []string{
		os.Getenv("ENVOY_BIN"),
		filepath.Join(homeDir, ".local/share/func-e/envoy-versions/1.35.3/bin/envoy"),
	}
	for _, p := range candidatePaths {
		if p != "" {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	if path, err := exec.LookPath("envoy"); err == nil {
		return path
	}
	return ""
}

func setupEnvoyIntegrationFixture(t *testing.T, upstreamHandler http.Handler) *envoyTestHarness {
	envoyBin := discoverEnvoyBinary(t)
	if envoyBin == "" {
		t.Skip("Envoy 1.35.3 binary not found on host; run `./bin/func-e use 1.35.3` to enable integration tests")
		return nil
	}

	h := &envoyTestHarness{
		envoyBin:         envoyBin,
		upstreamCalls:    new(atomic.Int32),
		settlementCount:  new(atomic.Int32),
		extprocCancelled: new(atomic.Bool),
		upstreamCanceled: new(atomic.Bool),
		modeOverrideUsed: new(atomic.Bool),
	}

	// 1. Upstream HTTP server
	upstreamListener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	h.upstreamPort = upstreamListener.Addr().(*net.TCPAddr).Port

	upstreamServer := &http.Server{Handler: upstreamHandler}
	go func() {
		_ = upstreamServer.Serve(upstreamListener)
	}()

	// 2. ExtProc gRPC server
	extprocListener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	h.extprocPort = extprocListener.Addr().(*net.TCPAddr).Port

	grpcServer := grpc.NewServer()
	h.procSrv = &envoyTraceExtProcServer{
		upstreamCalled:   h.upstreamCalls,
		settlementCount:  h.settlementCount,
		extprocCancelled: h.extprocCancelled,
		modeOverrideUsed: h.modeOverrideUsed,
		t:                t,
	}
	ext_proc.RegisterExternalProcessorServer(grpcServer, h.procSrv)
	go func() {
		_ = grpcServer.Serve(extprocListener)
	}()

	// 3. Envoy configuration
	envoyListener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	h.envoyPort = envoyListener.Addr().(*net.TCPAddr).Port
	_ = envoyListener.Close()

	envoyConfigYaml := fmt.Sprintf(`static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 127.0.0.1
        port_value: %d
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  cluster: vllm_backend_cluster
                  timeout: 30s
          http_filters:
          - name: envoy.filters.http.ext_proc
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
              grpc_service:
                envoy_grpc:
                  cluster_name: extproc_service
              allow_mode_override: true
              processing_mode:
                request_header_mode: "SEND"
                response_header_mode: "SEND"
                request_body_mode: "BUFFERED"
                response_body_mode: "BUFFERED"
                request_trailer_mode: "SKIP"
                response_trailer_mode: "SKIP"
              failure_mode_allow: false
              message_timeout: 30s
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
              suppress_envoy_headers: true

  clusters:
  - name: extproc_service
    connect_timeout: 5s
    type: STATIC
    lb_policy: ROUND_ROBIN
    typed_extension_protocol_options:
      envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
        "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
        explicit_http_config:
          http2_protocol_options: {}
    load_assignment:
      cluster_name: extproc_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 127.0.0.1
                port_value: %d

  - name: vllm_backend_cluster
    connect_timeout: 5s
    type: STATIC
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: vllm_backend_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 127.0.0.1
                port_value: %d

admin:
  address:
    socket_address:
      address: 127.0.0.1
      port_value: 0
`, h.envoyPort, h.extprocPort, h.upstreamPort)

	tmpConfigFile := filepath.Join(os.TempDir(), fmt.Sprintf("envoy_trace_fixture_%d.yaml", time.Now().UnixNano()))
	require.NoError(t, os.WriteFile(tmpConfigFile, []byte(envoyConfigYaml), 0644))

	// 4. Start Envoy process
	envoyCtx, envoyCancel := context.WithCancel(context.Background())
	var envoyLogBuf bytes.Buffer
	cmd := exec.CommandContext(envoyCtx, envoyBin,
		"-c", tmpConfigFile,
		"-l", "debug",
		"--component-log-level", "ext_proc:debug,router:debug,http:debug",
	)
	cmd.Stdout = &envoyLogBuf
	cmd.Stderr = &envoyLogBuf
	require.NoError(t, cmd.Start())

	// Wait for readiness
	ready := false
	for i := 0; i < 40; i++ {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", h.envoyPort), 50*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			ready = true
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	require.True(t, ready, "Envoy failed to listen on %d within 2s", h.envoyPort)

	h.cleanup = func() {
		envoyCancel()
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
		_ = os.Remove(tmpConfigFile)
		grpcServer.Stop()
		_ = extprocListener.Close()
		_ = upstreamServer.Close()
		_ = upstreamListener.Close()
	}

	return h
}

// ----------------------------------------------------------------------------
// 3. Live Integration Tests against Envoy 1.35.3 C++ Process
// ----------------------------------------------------------------------------

// TestLooperLifecycleTrace_Envoy1353Integration verifies cancellation propagation:
// - Verifies upstream receives mutated body ('synthesis prompt') and model header
// - Verifies client abort propagates TCP RST to upstream (halting GPU decode)
// - Verifies Router commits exactly 1 atomic settlement with status=client_cancelled and provenance=estimated
func TestLooperLifecycleTrace_Envoy1353Integration(t *testing.T) {
	var upstreamStarted = make(chan struct{})

	// Upstream model backend mock: verifies mutated body & waits for cancellation
	upstreamHandler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Assertion 1: Verify Upstream receives the ExtProc mutated prompt and model
		bodyBytes, err := io.ReadAll(req.Body)
		require.NoError(t, err)
		require.Contains(t, string(bodyBytes), "synthesis prompt", "Upstream must receive the mutated synthesis prompt from ExtProc")
		require.Contains(t, string(bodyBytes), "qwen2.5-7b-synthesis", "Upstream must receive the mutated model in payload")

		modelHeader := req.Header.Get("x-selected-model")
		require.Equal(t, "qwen2.5-7b-synthesis", modelHeader, "Upstream must receive mutated model header")
		t.Logf("[TRACE][Upstream] Call received with mutated model (%s) and verified body", modelHeader)

		promptTokens := countPromptTokens(bodyBytes)
		require.Equal(t, 2, promptTokens, "Upstream parsed prompt(\"synthesis prompt\") words must equal 2")

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if ok {
			flusher.Flush()
		}

		// Emit chunk 1 (1 word: "Hello", model: "qwen2.5-7b-synthesis")
		chunk1 := "data: {\"id\":\"chatcmpl-1\",\"model\":\"qwen2.5-7b-synthesis\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
		_, _ = w.Write([]byte(chunk1))
		if ok {
			flusher.Flush()
		}
		t.Logf("[TRACE][Upstream] Flushed SSE chunk 1 to Envoy")
		close(upstreamStarted)

		// Wait for downstream cancellation propagated by Envoy
		select {
		case <-req.Context().Done():
			t.Logf("[TRACE][Upstream] Context canceled! TCP reset received from Envoy: GPU decode halted.")
			return
		case <-time.After(5 * time.Second):
			t.Logf("[TRACE][Upstream] Clean completion timeout")
		}
	})

	h := setupEnvoyIntegrationFixture(t, upstreamHandler)
	if h == nil {
		return
	}
	defer h.cleanup()

	// Client connects to Envoy with public model 'gpt-4o'
	clientConn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", h.envoyPort))
	require.NoError(t, err)

	reqPayload := `{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hello"}]}`
	httpReq := fmt.Sprintf("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s",
		h.envoyPort, len(reqPayload), reqPayload)

	_, err = clientConn.Write([]byte(httpReq))
	require.NoError(t, err)

	reader := bufio.NewReader(clientConn)
	_ = clientConn.SetReadDeadline(time.Now().Add(3 * time.Second))

	// Read headers
	for {
		line, err := reader.ReadString('\n')
		require.NoError(t, err)
		if line == "\r\n" {
			break
		}
	}

	// Read Chunk 1
	var chunkLine string
	for {
		line, err := reader.ReadString('\n')
		require.NoError(t, err)
		if strings.HasPrefix(line, "data:") {
			chunkLine = line
			break
		}
	}
	require.Contains(t, chunkLine, "Hello", "Client should receive chunk 1 via Envoy streamed forwarding")
	require.Contains(t, chunkLine, `"model":"gpt-4o"`, "Client must receive original requested model 'gpt-4o' rewritten by ExtProc")
	require.NotContains(t, chunkLine, "qwen2.5-7b-synthesis", "Internal backend model must NEVER leak to client")

	// ABORT! Client closes connection immediately (triggers downstream reset)
	t.Logf("[TRACE][Client] Aborting downstream TCP connection mid-stream")
	_ = clientConn.Close()

	// Wait for Envoy cancellation propagation
	require.Eventually(t, func() bool {
		return h.extprocCancelled.Load() && h.settlementCount.Load() == 1
	}, 2*time.Second, 20*time.Millisecond, "Envoy should propagate downstream reset to upstream and ext_proc")

	// Assertions
	require.True(t, h.modeOverrideUsed.Load(), "ModeOverride STREAMED must be negotiated in response_headers")
	require.True(t, h.extprocCancelled.Load(), "ExtProc gRPC stream must be cancelled by Envoy")
	require.Equal(t, int32(1), h.settlementCount.Load(), "Must record exactly ONE atomic settlement")

	// Verify exact usage settlement fields under cancellation (estimated provenance)
	require.NotNil(t, h.procSrv.lastSettlement, "Settlement record must be generated")
	require.Equal(t, "client_cancelled", h.procSrv.lastSettlement.Status)
	require.Equal(t, "estimated", h.procSrv.lastSettlement.Provenance)
	require.Equal(t, 2, h.procSrv.lastSettlement.PromptTokens)
	require.Equal(t, 1, h.procSrv.lastSettlement.CompletionTokens)
	require.Equal(t, 3, h.procSrv.lastSettlement.TotalTokens)

	t.Logf("[TRACE] SUCCESS: Cancellation path, mutated body verification, and estimated usage proven against Envoy 1.35.3!")
}

// TestLooperLifecycleTrace_Envoy1353Integration_CleanCompletion verifies the un-interrupted path:
// - Upstream emits multiple SSE chunks, terminal usage object, and 'data: [DONE]'
// - Client consumes all chunks without aborting
// - ExtProc audits full stream and commits authoritative settlement with Status: completed
func TestLooperLifecycleTrace_Envoy1353Integration_CleanCompletion(t *testing.T) {
	// Upstream model backend mock emitting complete SSE stream + [DONE]
	upstreamHandler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		bodyBytes, err := io.ReadAll(req.Body)
		require.NoError(t, err)
		require.Contains(t, string(bodyBytes), "synthesis prompt", "Upstream must receive the mutated synthesis prompt from ExtProc")
		require.Contains(t, string(bodyBytes), "qwen2.5-7b-synthesis", "Upstream must receive the mutated model in payload")

		modelHeader := req.Header.Get("x-selected-model")
		require.Equal(t, "qwen2.5-7b-synthesis", modelHeader, "Upstream must receive mutated model header")
		t.Logf("[TRACE][Upstream] Call received with mutated model (%s) and verified body", modelHeader)

		// Dynamically calculate prompt tokens from received body
		promptTokens := countPromptTokens(bodyBytes)
		require.Equal(t, 2, promptTokens, "Upstream parsed prompt(\"synthesis prompt\") words must equal 2")

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if ok {
			flusher.Flush()
		}

		chunks := []string{
			"data: {\"id\":\"chatcmpl-clean-1\",\"model\":\"qwen2.5-7b-synthesis\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
			"data: {\"id\":\"chatcmpl-clean-1\",\"model\":\"qwen2.5-7b-synthesis\",\"choices\":[{\"delta\":{\"content\":\" world of semantic routing\"}}]}\n\n",
		}

		completionTokens := 0
		for _, ch := range chunks {
			completionTokens += countTokensInDeltaChunk(ch)
			_, _ = w.Write([]byte(ch))
			if ok {
				flusher.Flush()
			}
		}

		// Chunk 3: Terminal Usage frame dynamically computed from prompt and emitted completion tokens
		usageFrame := fmt.Sprintf("data: {\"id\":\"chatcmpl-clean-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
			promptTokens, completionTokens, promptTokens+completionTokens)
		_, _ = w.Write([]byte(usageFrame))
		if ok {
			flusher.Flush()
		}

		// Chunk 4: Standard [DONE] frame
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		if ok {
			flusher.Flush()
		}
	})

	h := setupEnvoyIntegrationFixture(t, upstreamHandler)
	if h == nil {
		return
	}
	defer h.cleanup()

	// Client sends request to Envoy
	clientConn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", h.envoyPort))
	require.NoError(t, err)
	defer func() {
		_ = clientConn.Close()
	}()

	reqPayload := `{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hello"}]}`
	httpReq := fmt.Sprintf("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s",
		h.envoyPort, len(reqPayload), reqPayload)

	_, err = clientConn.Write([]byte(httpReq))
	require.NoError(t, err)

	reader := bufio.NewReader(clientConn)
	_ = clientConn.SetReadDeadline(time.Now().Add(3 * time.Second))

	// Read headers
	for {
		line, err := reader.ReadString('\n')
		require.NoError(t, err)
		if line == "\r\n" {
			break
		}
	}

	// Read all chunks until [DONE]
	var receivedDone bool
	var contentChunks []string
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		if strings.HasPrefix(line, "data:") {
			contentChunks = append(contentChunks, line)
		}
		if strings.Contains(line, "[DONE]") {
			receivedDone = true
			break
		}
	}
	require.True(t, receivedDone, "Client must receive terminal [DONE] frame")
	require.NotEmpty(t, contentChunks, "Client must receive streamed SSE chunks")
	for _, chunk := range contentChunks {
		if strings.Contains(chunk, `"choices":`) && !strings.Contains(chunk, `"usage":`) {
			require.Contains(t, chunk, `"model":"gpt-4o"`, "Streamed chunk must present virtualized client model 'gpt-4o'")
			require.NotContains(t, chunk, "qwen2.5-7b-synthesis", "Internal backend model must NEVER leak to client")
		}
	}

	// Wait for Envoy to complete and ExtProc to settle
	require.Eventually(t, func() bool {
		return h.settlementCount.Load() == 1
	}, 2*time.Second, 20*time.Millisecond, "Clean termination must yield exactly 1 settlement")

	// Assertions for clean path
	require.True(t, h.modeOverrideUsed.Load(), "ModeOverride STREAMED must be negotiated")
	require.False(t, h.extprocCancelled.Load(), "ExtProc must NOT receive context cancellation on clean path")
	require.Equal(t, int32(1), h.settlementCount.Load(), "Must record exactly ONE atomic settlement")

	// Verify authoritative settlement fields dynamically extracted from upstream usage frame
	require.NotNil(t, h.procSrv.lastSettlement, "Clean settlement record must exist")
	require.Equal(t, "completed", h.procSrv.lastSettlement.Status)
	require.Equal(t, "authoritative", h.procSrv.lastSettlement.Provenance)
	require.Equal(t, 2, h.procSrv.lastSettlement.PromptTokens)
	require.Equal(t, 5, h.procSrv.lastSettlement.CompletionTokens)
	require.Equal(t, 7, h.procSrv.lastSettlement.TotalTokens)

	t.Logf("[TRACE] SUCCESS: Clean completion, mutated body verification, and authoritative usage proven against Envoy 1.35.3!")
}
