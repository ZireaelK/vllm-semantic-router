package extproc

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// TestLooperBudgetExhausted_EnvoyMode_ResponseDetectorBlocked verifies that for
// Envoy ExtProc looper streaming requests with a budget limit of 32 (31 panel
// calls + 1 synthesis call):
//  1. ExtProc response stage accounts for the final synthesis call dispatch on the shared ledger.
//  2. The ledger remaining after synthesis deduction is 0.
//  3. Subsequent response detector calls in observeResponseStageSignals fail with
//     budget.ErrExhausted and do not make extra physical calls (total physical calls <= 32).
func TestLooperBudgetExhausted_EnvoyMode_ResponseDetectorBlocked(t *testing.T) {
	detectorServer, detectorCalls := newHallucinationEndpointServer(t, []string{"unsupported claim"}, false)
	router, ctx := newHallucinationSignalRouter(t, detectorServer, "warn")

	// Limit of 32: 31 used by panel models during Looper streaming.
	execCtx, ledger := budget.WithLimit(context.Background(), 32)
	for i := 0; i < 31; i++ {
		require.NoError(t, budget.Consume(execCtx))
	}
	require.Equal(t, 31, ledger.Used())
	require.Equal(t, 1, ledger.Remaining())

	ctx.TraceContext = execCtx
	ctx.Context = execCtx
	ctx.LooperStreamingActive = true
	ctx.IsNativeGateway = false
	ctx.LooperRemainingBudget = 1

	// Envoy performs the final synthesis call outside of Go. ExtProc receives the
	// streaming response and triggers observeResponseStageSignals.
	router.observeResponseStageSignals(ctx, hallucinationAnswer)

	// The synthesis call must have consumed the 32nd budget call.
	require.Equal(t, 32, ledger.Used(), "synthesis call must consume 1 from the shared ledger")
	require.Equal(t, 0, ledger.Remaining(), "shared ledger must be completely exhausted (remaining 0)")
	require.Equal(t, 0, ctx.LooperRemainingBudget, "LooperRemainingBudget must be updated to 0")

	// The detector must NOT have executed a physical network call because budget was exhausted.
	require.Equal(t, int32(0), detectorCalls.Load(), "detector must not perform any physical call when budget is exhausted")

	// The signal must be recorded as failed.
	require.Equal(t, "hallucination_evaluation_failed", ctx.VSRSignalErrors[hallucinationSignalKey])

	// Directly consuming against the request context must return budget.ErrExhausted.
	require.True(t, errors.Is(budget.Consume(selectionRequestContext(ctx)), budget.ErrExhausted))
}

// TestLooperBudgetExhausted_NativeGatewayMode_SynthesisAndResponseStage verifies that for
// Native Gateway looper streaming requests with a budget limit of 32 (31 panel
// calls + 1 synthesis call):
//  1. Session budget ledger is propagated through Engine.Plan to Call.Ledger and Handler/Upstream.Execute.
//  2. The native synthesis execution consumes the 32nd budget call on the shared ledger.
//  3. observeResponseStageSignals does not double-consume on native gateway requests.
//  4. Subsequent response detector calls in observeResponseStageSignals fail with
//     budget.ErrExhausted and do not make extra physical calls (total physical calls <= 32).
func TestLooperBudgetExhausted_NativeGatewayMode_SynthesisAndResponseStage(t *testing.T) {
	detectorServer, detectorCalls := newHallucinationEndpointServer(t, []string{"unsupported claim"}, false)
	router, reqCtx := newHallucinationSignalRouter(t, detectorServer, "warn")

	var synthesisCalls atomic.Int32
	synthesisServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		synthesisCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"` + hallucinationAnswer + `"}}]}`))
	}))
	defer synthesisServer.Close()

	synHost, synPort := parseHostPort(t, synthesisServer.Listener.Addr())
	topology := upstream.Topology{
		Clusters: []upstream.ClusterSpec{
			{
				Name:     "model-primary",
				LBPolicy: upstream.LBRoundRobin,
				Endpoints: []upstream.EndpointSpec{
					{Name: "syn-ep", Host: synHost, Port: synPort, Scheme: "http", Weight: 1},
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

	// Limit of 32: 31 used by panel models during Looper streaming.
	execCtx, ledger := budget.WithLimit(context.Background(), 32)
	for i := 0; i < 31; i++ {
		require.NoError(t, budget.Consume(execCtx))
	}
	require.Equal(t, 31, ledger.Used())
	require.Equal(t, 1, ledger.Remaining())

	reqCtx.TraceContext = execCtx
	reqCtx.Context = execCtx
	reqCtx.LooperStreamingActive = true
	reqCtx.IsNativeGateway = true
	reqCtx.LooperRemainingBudget = 1

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
		body:           []byte(`{"model":"model-primary","messages":[{"role":"user","content":"Synthesis test"}]}`),
	}

	engine := routing.NewEngine(&looperPlanTestProcessor{session: planSession}, routing.DefaultOptions)

	clientRequest := &routing.Request{
		Header: routing.Header{
			{Name: ":method", Value: "POST"},
			{Name: ":path", Value: "/v1/chat/completions"},
			{Name: "content-type", Value: "application/json"},
			{Name: routing.RouteHeader, Value: "model-primary"},
		},
		Body: []byte(`{"model":"model-primary","messages":[{"role":"user","content":"Synthesis test"}]}`),
	}

	// 1. Plan synthesis via Native Gateway Engine
	plan, err := engine.Plan(context.Background(), clientRequest)
	require.NoError(t, err)
	require.NotNil(t, plan.Call)
	require.NotNil(t, plan.Call.Ledger, "plan.Call.Ledger must be populated from BudgetSession")
	require.Equal(t, ledger, plan.Call.Ledger)
	require.Equal(t, 1, plan.Call.Ledger.Remaining())

	// 2. Upstream Set executes synthesis call: consumes 32nd call on the ledger
	result, err := set.Execute(context.Background(), plan.Call, "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.Response)
	_ = result.Response.Body.Close()

	require.Equal(t, int32(1), synthesisCalls.Load(), "synthesis upstream must be called once")
	require.Equal(t, 32, ledger.Used(), "synthesis call must consume the 32nd budget call")
	require.Equal(t, 0, ledger.Remaining(), "budget must be completely exhausted after synthesis")

	// 3. Response observation runs
	router.observeResponseStageSignals(reqCtx, hallucinationAnswer)

	// Since reqCtx.IsNativeGateway is true, observeResponseStageSignals must not double-consume.
	require.Equal(t, 32, ledger.Used(), "observeResponseStageSignals must not double-consume for native gateway")
	require.Equal(t, 0, ledger.Remaining(), "budget remains 0")

	// The detector must NOT make any physical calls
	require.Equal(t, int32(0), detectorCalls.Load(), "detector must not perform any physical call when budget is exhausted")

	// Total physical calls across the whole request = 31 panels + 1 synthesis + 0 detector = 32 <= 32
	require.Equal(t, "hallucination_evaluation_failed", reqCtx.VSRSignalErrors[hallucinationSignalKey])
	require.True(t, errors.Is(budget.Consume(selectionRequestContext(reqCtx)), budget.ErrExhausted))
}

// TestLooperBudgetExhausted_EnvoyMode_200JSONDegradation_ResponseDetectorBlocked verifies that
// when an Envoy-dispatched Looper streaming synthesis request degrades to 200 application/json:
//  1. Upfront pre-reservation in addEnvoyReliabilityHeaders debits the 32nd call on the shared ledger.
//  2. The non-streaming response body processing (which bypasses observeResponseStageSignals and calls
//     prepareResponsePolicy -> scoreResponseStageSignals directly) encounters an exhausted ledger.
//  3. Response-stage detectors fail with budget.ErrExhausted and do NOT perform a 33rd physical call.
func TestLooperBudgetExhausted_EnvoyMode_200JSONDegradation_ResponseDetectorBlocked(t *testing.T) {
	detectorServer, detectorCalls := newHallucinationEndpointServer(t, []string{"unsupported claim"}, false)
	router, ctx := newHallucinationSignalRouter(t, detectorServer, "warn")

	// Limit of 32: 31 used by panel models during Looper streaming.
	execCtx, ledger := budget.WithLimit(context.Background(), 32)
	for i := 0; i < 31; i++ {
		require.NoError(t, budget.Consume(execCtx))
	}
	require.Equal(t, 31, ledger.Used())
	require.Equal(t, 1, ledger.Remaining())

	ctx.TraceContext = execCtx
	ctx.Context = execCtx
	ctx.LooperStreamingActive = true
	ctx.IsNativeGateway = false
	ctx.LooperRemainingBudget = 1

	// ExtProc pre-reserves the synthesis call before Envoy dispatch
	dispatchResp := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{
		RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{HeaderMutation: &ext_proc.HeaderMutation{}}},
	}}
	router.addEnvoyReliabilityHeaders(dispatchResp, ctx)

	require.Equal(t, 32, ledger.Used(), "synthesis call must consume 1 from the shared ledger upfront")
	require.Equal(t, 0, ledger.Remaining(), "shared ledger must be completely exhausted (remaining 0)")
	require.Equal(t, 0, ctx.LooperRemainingBudget, "LooperRemainingBudget must be updated to 0")
	require.True(t, ctx.EnvoyBudgetReserved)

	// Envoy dispatches and upstream responds with 200 application/json (JSON degradation)
	respHeadersReq := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "application/json"},
				},
			},
		},
	}
	_, err := router.handleResponseHeaders(respHeadersReq, ctx)
	require.NoError(t, err)

	jsonBody := []byte(`{"id":"chatcmpl-test","object":"chat.completion","model":"model-primary","choices":[{"index":0,"message":{"role":"assistant","content":"` + hallucinationAnswer + `"},"finish_reason":"stop"}]}`)
	resp := router.handleNonStreamingResponseBody(jsonBody, ctx, 10*time.Millisecond)
	require.NotNil(t, resp)

	// The detector must NOT have executed a physical network call because budget was exhausted upfront.
	require.Equal(t, int32(0), detectorCalls.Load(), "detector must not perform any physical call when budget is exhausted")

	// The signal must be recorded as failed.
	require.Equal(t, "hallucination_evaluation_failed", ctx.VSRSignalErrors[hallucinationSignalKey])

	// Directly consuming against the request context must return budget.ErrExhausted.
	require.True(t, errors.Is(budget.Consume(selectionRequestContext(ctx)), budget.ErrExhausted))
}

// TestLooperBudgetExhausted_EnvoyMode_Retries_ResponseDetectorBlocked verifies that
// when Envoy retries the final synthesis call:
// 1. Limit = 32, 30 panel calls used, remaining = 2, retries allowed = 1.
// 2. Both attempts (1 initial + 1 retry = 2 authorized attempts) are pre-reserved upfront.
// 3. Envoy performs 2 attempts (first fails, second succeeds) and returns x-envoy-attempt-count: 2.
// 4. Ledger remains at 0 remaining, blocking response detectors from exceeding budget (30 + 2 + 0 = 32).
func TestLooperBudgetExhausted_EnvoyMode_Retries_ResponseDetectorBlocked(t *testing.T) {
	detectorServer, detectorCalls := newHallucinationEndpointServer(t, []string{"unsupported claim"}, false)
	router, ctx := newHallucinationSignalRouter(t, detectorServer, "warn")

	one := 1
	ctx.VSRSelectedDecision.Reliability = &config.DecisionReliability{
		RetryCount: &one,
		RetryOn:    "5xx",
	}

	// Limit of 32: 30 used by panel models during Looper streaming.
	execCtx, ledger := budget.WithLimit(context.Background(), 32)
	for i := 0; i < 30; i++ {
		require.NoError(t, budget.Consume(execCtx))
	}
	require.Equal(t, 30, ledger.Used())
	require.Equal(t, 2, ledger.Remaining())

	ctx.TraceContext = execCtx
	ctx.Context = execCtx
	ctx.LooperStreamingActive = true
	ctx.IsNativeGateway = false
	ctx.LooperRemainingBudget = 2

	// Pre-reserve before Envoy dispatch: 1 initial attempt + 1 retry = 2 authorized attempts.
	dispatchResp := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{
		RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{HeaderMutation: &ext_proc.HeaderMutation{}}},
	}}
	router.addEnvoyReliabilityHeaders(dispatchResp, ctx)

	require.Equal(t, 32, ledger.Used(), "synthesis dispatch must pre-reserve both initial attempt and retry (2 calls)")
	require.Equal(t, 0, ledger.Remaining(), "shared ledger must be completely exhausted (remaining 0)")
	require.Equal(t, 0, ctx.LooperRemainingBudget, "LooperRemainingBudget must be updated to 0")
	require.Equal(t, 2, ctx.EnvoyAuthorizedAttempts)
	require.True(t, ctx.EnvoyBudgetReserved)

	// Envoy performs attempt 1 (fails with 503) and attempt 2 (succeeds with 200).
	// Envoy reports attempt count: 2 in response headers.
	respHeadersReq := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "text/event-stream"},
					{Key: "x-envoy-attempt-count", Value: "2"},
				},
			},
		},
	}
	_, err := router.handleResponseHeaders(respHeadersReq, ctx)
	require.NoError(t, err)

	// Since 2 attempts were authorized and 2 attempts were used, unused = 0.
	require.Equal(t, 32, ledger.Used())
	require.Equal(t, 0, ledger.Remaining())

	// Streaming response observation runs
	router.observeResponseStageSignals(ctx, hallucinationAnswer)

	// Detector must NOT execute any physical calls
	require.Equal(t, int32(0), detectorCalls.Load(), "detector must not perform any physical call when budget is exhausted")
	require.Equal(t, "hallucination_evaluation_failed", ctx.VSRSignalErrors[hallucinationSignalKey])
	require.True(t, errors.Is(budget.Consume(selectionRequestContext(ctx)), budget.ErrExhausted))
}

// TestLooperBudgetExhausted_EnvoyMode_AttemptCountRefund verifies that
// ExtProc does NOT trust unauthenticated x-envoy-attempt-count response headers
// to refund retry budget:
//  1. Limit = 32, 30 panel calls used, 2 attempts authorized and pre-reserved upfront (ledger remaining 0).
//  2. Even when upstream/proxy response carries x-envoy-attempt-count: 1, ExtProc conservatively retains
//     the pre-reserved retry budget (ledger remaining stays 0).
//  3. Subsequent response detector call is blocked with budget.ErrExhausted, strictly upholding the
//     budget limit (total physical calls <= 32).
func TestLooperBudgetExhausted_EnvoyMode_AttemptCountRefund(t *testing.T) {
	t.Run("no_refund_when_attempt_count_header_present", func(t *testing.T) {
		detectorServer, detectorCalls := newHallucinationEndpointServer(t, []string{"unsupported claim"}, false)
		router, ctx := newHallucinationSignalRouter(t, detectorServer, "warn")

		one := 1
		ctx.VSRSelectedDecision.Reliability = &config.DecisionReliability{
			RetryCount: &one,
			RetryOn:    "5xx",
		}

		// Limit of 32: 30 used by panel models during Looper streaming.
		execCtx, ledger := budget.WithLimit(context.Background(), 32)
		for i := 0; i < 30; i++ {
			require.NoError(t, budget.Consume(execCtx))
		}
		require.Equal(t, 30, ledger.Used())
		require.Equal(t, 2, ledger.Remaining())

		ctx.TraceContext = execCtx
		ctx.Context = execCtx
		ctx.LooperStreamingActive = true
		ctx.IsNativeGateway = false
		ctx.LooperRemainingBudget = 2

		// Pre-reserve before Envoy dispatch: 1 initial attempt + 1 retry = 2 authorized attempts.
		dispatchResp := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{HeaderMutation: &ext_proc.HeaderMutation{}}},
		}}
		router.addEnvoyReliabilityHeaders(dispatchResp, ctx)

		require.Equal(t, 32, ledger.Used(), "synthesis dispatch must pre-reserve 2 calls")
		require.Equal(t, 0, ledger.Remaining(), "shared ledger must be exhausted before dispatch")
		require.Equal(t, 2, ctx.EnvoyAuthorizedAttempts)

		// Upstream or intermediate proxy returns x-envoy-attempt-count: 1
		respHeadersReq := &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{
					Headers: []*core.HeaderValue{
						{Key: ":status", Value: "200"},
						{Key: "content-type", Value: "text/event-stream"},
						{Key: "x-envoy-attempt-count", Value: "1"},
					},
				},
			},
		}
		_, err := router.handleResponseHeaders(respHeadersReq, ctx)
		require.NoError(t, err)

		// ExtProc does NOT trust unauthenticated attempt count -> NO refund!
		require.Equal(t, 32, ledger.Used(), "reserved retry budget must remain debited (used stays 32)")
		require.Equal(t, 0, ledger.Remaining(), "shared ledger must remain at 0")
		require.Equal(t, 0, ctx.LooperRemainingBudget, "ctx.LooperRemainingBudget must remain 0")

		// Response detector runs but is BLOCKED with budget.ErrExhausted, 0 physical calls made!
		router.observeResponseStageSignals(ctx, hallucinationAnswer)

		require.Equal(t, int32(0), detectorCalls.Load(), "detector must NOT perform any physical call when budget is exhausted")
		require.Equal(t, 32, ledger.Used(), "total calls remain bounded at 32")
		require.Equal(t, 0, ledger.Remaining())
		require.Equal(t, "hallucination_evaluation_failed", ctx.VSRSignalErrors[hallucinationSignalKey])
		require.True(t, errors.Is(budget.Consume(selectionRequestContext(ctx)), budget.ErrExhausted))
	})

	t.Run("no_refund_when_attempt_count_header_is_absent", func(t *testing.T) {
		detectorServer, detectorCalls := newHallucinationEndpointServer(t, []string{"unsupported claim"}, false)
		router, ctx := newHallucinationSignalRouter(t, detectorServer, "warn")

		one := 1
		ctx.VSRSelectedDecision.Reliability = &config.DecisionReliability{
			RetryCount: &one,
			RetryOn:    "5xx",
		}

		execCtx, ledger := budget.WithLimit(context.Background(), 32)
		for i := 0; i < 30; i++ {
			require.NoError(t, budget.Consume(execCtx))
		}
		require.Equal(t, 30, ledger.Used())
		require.Equal(t, 2, ledger.Remaining())

		ctx.TraceContext = execCtx
		ctx.Context = execCtx
		ctx.LooperStreamingActive = true
		ctx.IsNativeGateway = false
		ctx.LooperRemainingBudget = 2

		dispatchResp := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{HeaderMutation: &ext_proc.HeaderMutation{}}},
		}}
		router.addEnvoyReliabilityHeaders(dispatchResp, ctx)

		require.Equal(t, 32, ledger.Used())
		require.Equal(t, 0, ledger.Remaining())

		// Response headers without x-envoy-attempt-count header
		respHeadersReq := &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{
					Headers: []*core.HeaderValue{
						{Key: ":status", Value: "200"},
						{Key: "content-type", Value: "text/event-stream"},
					},
				},
			},
		}
		_, err := router.handleResponseHeaders(respHeadersReq, ctx)
		require.NoError(t, err)

		// Without evidence, all authorized attempts remain reserved
		require.Equal(t, 32, ledger.Used(), "ledger must remain at 32 when attempt count header is absent")
		require.Equal(t, 0, ledger.Remaining())
		require.Equal(t, 0, ctx.LooperRemainingBudget)

		router.observeResponseStageSignals(ctx, hallucinationAnswer)
		require.Equal(t, int32(0), detectorCalls.Load(), "detector must not perform any physical call when budget remains reserved")
		require.Equal(t, "hallucination_evaluation_failed", ctx.VSRSignalErrors[hallucinationSignalKey])
		require.True(t, errors.Is(budget.Consume(selectionRequestContext(ctx)), budget.ErrExhausted))
	})
}
