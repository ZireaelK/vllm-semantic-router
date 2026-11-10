package extproc

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// looperPlanTestSession adapts a routingSession for routing.Engine.Plan while
// preserving the exact Fallback(), Evidence(), Close(), and Reliability() behaviors
// of the underlying routingSession.
type looperPlanTestSession struct {
	*routingSession
	route string
	body  []byte
}

func (s *looperPlanTestSession) RequestHeaders(h routing.Header, eos bool) (*routing.Effect, error) {
	s.clientHeader = h.Clone()
	return &routing.Effect{}, nil
}

func (s *looperPlanTestSession) RequestBody(b []byte, eos bool) (*routing.Effect, error) {
	return &routing.Effect{
		Header: &routing.HeaderMutation{
			Set: []routing.HeaderOption{
				{Name: routing.RouteHeader, Value: s.route},
			},
		},
		Body: &routing.BodyMutation{
			Body: s.body,
		},
		ClearRouteCache: true,
	}, nil
}

type looperPlanTestProcessor struct {
	session routing.Session
}

func (p *looperPlanTestProcessor) Open(ctx context.Context) (routing.Session, error) {
	return p.session, nil
}

// parseHostPort parses host and port from a net.Addr.
func parseHostPort(t *testing.T, addr net.Addr) (string, int) {
	t.Helper()
	host, portStr, err := net.SplitHostPort(addr.String())
	require.NoError(t, err)
	port, err := strconv.Atoi(portStr)
	require.NoError(t, err)
	return host, port
}

func TestNativeFallbackLooperStreamingBudget(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)
	orch := router.FallbackOrchestrator

	// -------------------------------------------------------------------------
	// TestCase 1: Unit test verifying nativeFallbackApplies guards
	// When ctx.LooperStreamingActive = true, nativeFallbackApplies strictly returns false.
	// -------------------------------------------------------------------------
	t.Run("TestCase1_NativeFallbackApplies_LooperStreamingActiveGuards", func(t *testing.T) {
		// Baseline: normal request with fallback enabled and 2 eligible models -> true
		ctxNormal := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
		require.True(t, nativeFallbackApplies(orch, ctxNormal), "normal context with eligible models should be eligible for fallback")

		// Guard: ctx.LooperStreamingActive = true -> strictly false
		ctxStreaming := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
		ctxStreaming.LooperStreamingActive = true
		require.False(t, nativeFallbackApplies(orch, ctxStreaming), "LooperStreamingActive must strictly block fallback")

		// Guard: ctx.LooperRequest = true -> strictly false
		ctxLooperReq := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
		ctxLooperReq.LooperRequest = true
		require.False(t, nativeFallbackApplies(orch, ctxLooperReq), "LooperRequest must strictly block fallback")

		// Guard: ctx == nil -> strictly false
		require.False(t, nativeFallbackApplies(orch, nil), "nil RequestContext must safely return false")

		// Guard: orch == nil -> strictly false
		require.False(t, nativeFallbackApplies(nil, ctxNormal), "nil orchestrator must safely return false")
	})

	// -------------------------------------------------------------------------
	// TestCase 2: Verification that session.Fallback() returns nil and
	// sessionFallback.Next() immediately ends when LooperStreamingActive = true
	// -------------------------------------------------------------------------
	t.Run("TestCase2_SessionFallback_ReturnsNilAndNextEndsImmediately", func(t *testing.T) {
		t.Run("session_fallback_returns_nil_when_streaming_active", func(t *testing.T) {
			session := nativeFallbackSession(t, policy, "model-primary", "model-fallback-1")
			session.ctx.LooperStreamingActive = true
			session.ctx.LooperRemainingBudget = 1

			fb := session.Fallback()
			require.Nil(t, fb, "session.Fallback() must return nil when LooperStreamingActive is true")
		})

		t.Run("session_fallback_returns_nil_when_looper_request", func(t *testing.T) {
			session := nativeFallbackSession(t, policy, "model-primary", "model-fallback-1")
			session.ctx.LooperRequest = true

			fb := session.Fallback()
			require.Nil(t, fb, "session.Fallback() must return nil when LooperRequest is true")
		})

		t.Run("session_fallback_returns_instance_when_normal", func(t *testing.T) {
			session := nativeFallbackSession(t, policy, "model-primary", "model-fallback-1")
			session.ctx.LooperStreamingActive = false
			session.ctx.LooperRequest = false

			fb := session.Fallback()
			require.NotNil(t, fb, "session.Fallback() must return valid Fallback instance for normal requests")
		})

		t.Run("double_defense_sessionFallback_Next_ends_immediately_when_streaming_active", func(t *testing.T) {
			// Even if a sessionFallback is directly invoked, nativeFallbackApplies blocks candidate dispatch
			session := nativeFallbackSession(t, policy, "model-primary", "model-fallback-1")
			session.ctx.LooperStreamingActive = true
			session.ctx.LooperRemainingBudget = 1

			// Direct sessionFallback instance representing second line of defense
			chain := &sessionFallback{session: session}
			step, err := chain.Next(context.Background(), failedWith(http.StatusServiceUnavailable, "503 service unavailable"))
			require.NoError(t, err)
			require.Nil(t, step.Call, "chain.Next() must not prepare a fallback call when LooperStreamingActive is true")
			require.Nil(t, step.Immediate, "chain.Next() must not return an immediate response")
			require.False(t, session.ctx.FallbackAuditRecorded, "fallback audit must not be recorded")
		})
	})

	// -------------------------------------------------------------------------
	// TestCase 3: Realistic Native Gateway Engine.Plan / Upstream.Execute simulation
	// - Decision configured with cross-model fallback (2 candidates: primary synthesis & fallback)
	// - LooperStreamingActive = true, remaining budget = 1
	// - Synthesis target returns 503
	// - Verify Native Gateway absolutely never sends call to candidate, total calls <= budget (1),
	//   and never replays synthesis input or prompt to fallback candidate.
	// -------------------------------------------------------------------------
	t.Run("TestCase3_NativeGateway_EnginePlan_UpstreamExecute_BudgetEnforced", func(t *testing.T) {
		var (
			synthesisCalls atomic.Int64
			fallbackCalls  atomic.Int64
			mu             sync.Mutex
			synthesisBody  string
			fallbackBody   string
		)

		synthesisServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			bodyBytes, _ := io.ReadAll(r.Body)
			mu.Lock()
			synthesisBody = string(bodyBytes)
			mu.Unlock()
			synthesisCalls.Add(1)
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":{"message":"synthesis target unavailable","code":503}}`))
		}))
		defer synthesisServer.Close()

		fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			bodyBytes, _ := io.ReadAll(r.Body)
			mu.Lock()
			fallbackBody = string(bodyBytes)
			mu.Unlock()
			fallbackCalls.Add(1)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"id":"chatcmpl-fb","choices":[{"message":{"content":"fallback answer"}}]}`))
		}))
		defer fallbackServer.Close()

		synHost, synPort := parseHostPort(t, synthesisServer.Listener.Addr())
		fbHost, fbPort := parseHostPort(t, fallbackServer.Listener.Addr())

		topology := upstream.Topology{
			Clusters: []upstream.ClusterSpec{
				{
					Name:     "model-primary",
					LBPolicy: upstream.LBRoundRobin,
					Endpoints: []upstream.EndpointSpec{
						{Name: "syn-ep", Host: synHost, Port: synPort, Scheme: "http", Weight: 1},
					},
				},
				{
					Name:     "model-fallback-1",
					LBPolicy: upstream.LBRoundRobin,
					Endpoints: []upstream.EndpointSpec{
						{Name: "fb-ep", Host: fbHost, Port: fbPort, Scheme: "http", Weight: 1},
					},
				},
			},
			DefaultCluster: "model-primary",
		}

		set, err := upstream.New(topology, upstream.Options{})
		require.NoError(t, err)
		defer func() {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			_ = set.Close(ctx)
		}()

		rawSynthesisPayload := `{"model":"model-primary","messages":[{"role":"system","content":"Synthesis Instruction"},{"role":"user","content":"Intermediate drafts to synthesize"}]}`

		reqCtx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
		reqCtx.LooperStreamingActive = true
		reqCtx.LooperRemainingBudget = 1
		reqCtx.LooperTargetModel = "model-primary"
		fallbackEnabled := true
		reqCtx.VSRSelectedDecision = &config.Decision{
			Name: "looper_synthesis_decision",
			Fallback: &fallback.FallbackOverride{
				Enabled:     &fallbackEnabled,
				MaxAttempts: 2,
			},
		}

		session := &routingSession{
			router: router,
			ctx:    reqCtx,
			clientHeader: routing.Header{
				{Name: ":method", Value: "POST"},
				{Name: ":path", Value: "/v1/chat/completions"},
				{Name: "content-type", Value: "application/json"},
				{Name: routing.RouteHeader, Value: "model-primary"},
			},
		}

		planSession := &looperPlanTestSession{
			routingSession: session,
			route:          "model-primary",
			body:           []byte(rawSynthesisPayload),
		}

		opts := routing.DefaultOptions
		opts.ExecutesFallback = true
		engine := routing.NewEngine(&looperPlanTestProcessor{session: planSession}, opts)

		clientRequest := &routing.Request{
			Header: routing.Header{
				{Name: ":method", Value: "POST"},
				{Name: ":path", Value: "/v1/chat/completions"},
				{Name: "content-type", Value: "application/json"},
				{Name: routing.RouteHeader, Value: "model-primary"},
			},
			Body: []byte(rawSynthesisPayload),
		}

		// 1. Plan via Native Gateway Engine
		plan, err := engine.Plan(context.Background(), clientRequest)
		require.NoError(t, err)
		require.NotNil(t, plan.Call)
		require.Nil(t, plan.Call.Fallback, "Native Gateway Engine must NOT attach Fallback when LooperStreamingActive is true")

		// 2. Execute via Upstream Set
		result, err := set.Execute(context.Background(), plan.Call, "")
		require.NoError(t, err)
		require.NotNil(t, result)
		require.NotNil(t, result.Response)
		defer result.Response.Body.Close()

		// 3. Verifications
		require.Equal(t, http.StatusServiceUnavailable, result.Response.StatusCode, "Primary 503 response must be returned")
		require.Equal(t, 1, result.Hops, "Hops must be exactly 1, no fallback hops")
		require.Equal(t, int64(1), synthesisCalls.Load(), "Primary synthesis target must be called exactly once")
		require.Equal(t, int64(0), fallbackCalls.Load(), "Fallback candidate must NEVER be called")

		mu.Lock()
		require.Contains(t, synthesisBody, "Synthesis Instruction", "Synthesis target must receive synthesis prompt")
		require.Empty(t, fallbackBody, "Fallback candidate must receive nothing")
		mu.Unlock()

		// 4. Double defense verification under Upstream.Execute:
		// Even if a fallback object was manually attached to Call, sessionFallback.Next
		// must immediately stop and not call the candidate.
		callWithFallback := &routing.Call{
			Route: "model-primary",
			Request: routing.Request{
				Header: routing.Header{
					{Name: ":method", Value: "POST"},
					{Name: ":path", Value: "/v1/chat/completions"},
					{Name: "content-type", Value: "application/json"},
					{Name: routing.RouteHeader, Value: "model-primary"},
				},
				Body: []byte(rawSynthesisPayload),
			},
			Fallback: &sessionFallback{session: session},
		}

		result2, err2 := set.Execute(context.Background(), callWithFallback, "")
		require.NoError(t, err2)
		require.NotNil(t, result2)
		defer result2.Response.Body.Close()

		require.Equal(t, http.StatusServiceUnavailable, result2.Response.StatusCode)
		require.Equal(t, 1, result2.Hops, "Even with manual fallback, hops must remain 1")
		require.Equal(t, int64(2), synthesisCalls.Load())
		require.Equal(t, int64(0), fallbackCalls.Load(), "Fallback candidate must STILL never be called")
	})

	// -------------------------------------------------------------------------
	// TestCase 4: Control Comparison - Non-looper request DOES perform fallback
	// -------------------------------------------------------------------------
	t.Run("TestCase4_ControlComparison_NonLooperRequestPerformsFallback", func(t *testing.T) {
		var (
			primaryCalls  atomic.Int64
			fallbackCalls atomic.Int64
		)

		primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			primaryCalls.Add(1)
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":{"message":"primary unavailable","code":503}}`))
		}))
		defer primaryServer.Close()

		fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fallbackCalls.Add(1)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"id":"chatcmpl-fb","choices":[{"message":{"content":"candidate ok"}}]}`))
		}))
		defer fallbackServer.Close()

		priHost, priPort := parseHostPort(t, primaryServer.Listener.Addr())
		fbHost, fbPort := parseHostPort(t, fallbackServer.Listener.Addr())

		topology := upstream.Topology{
			Clusters: []upstream.ClusterSpec{
				{
					Name:     "model-primary",
					LBPolicy: upstream.LBRoundRobin,
					Endpoints: []upstream.EndpointSpec{
						{Name: "pri-ep", Host: priHost, Port: priPort, Scheme: "http", Weight: 1},
					},
				},
				{
					Name:     "model-fallback-1",
					LBPolicy: upstream.LBRoundRobin,
					Endpoints: []upstream.EndpointSpec{
						{Name: "fb-ep", Host: fbHost, Port: fbPort, Scheme: "http", Weight: 1},
					},
				},
			},
			DefaultCluster: "model-primary",
		}

		set, err := upstream.New(topology, upstream.Options{})
		require.NoError(t, err)
		defer func() {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			_ = set.Close(ctx)
		}()

		reqCtx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
		reqCtx.LooperStreamingActive = false
		reqCtx.LooperRequest = false

		session := &routingSession{
			router: router,
			ctx:    reqCtx,
			clientHeader: routing.Header{
				{Name: ":method", Value: "POST"},
				{Name: ":path", Value: "/v1/chat/completions"},
				{Name: "content-type", Value: "application/json"},
				{Name: routing.RouteHeader, Value: "model-primary"},
			},
		}

		planSession := &looperPlanTestSession{
			routingSession: session,
			route:          "model-primary",
			body:           []byte(`{"model":"model-primary","messages":[{"role":"user","content":"hello"}]}`),
		}

		opts := routing.DefaultOptions
		opts.ExecutesFallback = true
		engine := routing.NewEngine(&looperPlanTestProcessor{session: planSession}, opts)

		clientRequest := &routing.Request{
			Header: routing.Header{
				{Name: ":method", Value: "POST"},
				{Name: ":path", Value: "/v1/chat/completions"},
				{Name: "content-type", Value: "application/json"},
				{Name: routing.RouteHeader, Value: "model-primary"},
			},
			Body: []byte(`{"model":"model-primary","messages":[{"role":"user","content":"hello"}]}`),
		}

		plan, err := engine.Plan(context.Background(), clientRequest)
		require.NoError(t, err)
		require.NotNil(t, plan.Call.Fallback, "Standard request must attach Fallback")

		result, err := set.Execute(context.Background(), plan.Call, "")
		require.NoError(t, err)
		require.NotNil(t, result)
		defer result.Response.Body.Close()

		require.Equal(t, http.StatusOK, result.Response.StatusCode, "Fallback candidate response must be returned")
		require.Equal(t, 2, result.Hops, "Hops must be 2 (primary failed, fallback succeeded)")
		require.Equal(t, int64(1), primaryCalls.Load())
		require.Equal(t, int64(1), fallbackCalls.Load())
	})
}
