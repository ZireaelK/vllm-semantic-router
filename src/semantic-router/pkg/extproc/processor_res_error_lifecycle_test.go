package extproc

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

type scriptedMockStream struct {
	MockStream
	errAfterRequests error
	cancelCtx        context.CancelFunc
	onSend           func(resp *ext_proc.ProcessingResponse) error
}

func (s *scriptedMockStream) Send(resp *ext_proc.ProcessingResponse) error {
	if s.onSend != nil {
		if err := s.onSend(resp); err != nil {
			return err
		}
	}
	return s.MockStream.Send(resp)
}

func (s *scriptedMockStream) Recv() (*ext_proc.ProcessingRequest, error) {
	if s.RecvIndex >= len(s.Requests) {
		if s.cancelCtx != nil {
			s.cancelCtx()
		}
		if s.errAfterRequests != nil {
			return nil, s.errAfterRequests
		}
		return nil, io.EOF
	}
	return s.MockStream.Recv()
}

func setupReceiveErrorLifecycleFixture(t *testing.T) (*httptest.Server, *OpenAIRouter, *store.MemoryStore) {
	// Mock looper backend returns 500 so fusion quorum fallback triggers.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))

	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)

	decision := &config.Decision{
		Name: "fusion-quorum-fallback-lifecycle",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:                "judge",
				AnalysisModels:       []string{"panel-a", "panel-b"},
				AnalysisMode:         config.FusionAnalysisModeOneCall,
				QuorumFailurePolicy:  config.FusionQuorumFailurePolicyFallback,
				QuorumFallbackTarget: "backup-model",
			},
		},
	}

	routerCfg := &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{
			Enabled: true,
		},
		Looper: config.LooperConfig{
			Endpoint: server.URL,
		},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"panel-a":      {Capabilities: []string{"chat"}},
				"panel-b":      {Capabilities: []string{"chat"}},
				"judge":        {Capabilities: []string{"chat"}},
				"backup-model": {Capabilities: []string{"chat"}},
			},
		},
		Recipes: []config.RoutingRecipe{
			{
				Name: config.DefaultRecipeName,
				Profile: config.RoutingProfile{
					Decisions: []config.Decision{*decision},
				},
			},
		},
	}

	classifier, err := classification.BuildClassifier(routerCfg, nil, nil, nil)
	require.NoError(t, err)

	router := &OpenAIRouter{
		Config:         routerCfg,
		Classifier:     classifier,
		ReplayRecorder: recorder,
	}

	return server, router, memStore
}

func buildStandardLifecycleRequests(t *testing.T, reqID string) (*ext_proc.ProcessingRequest, *ext_proc.ProcessingRequest, *ext_proc.ProcessingRequest) {
	headersReq := &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_RequestHeaders{
			RequestHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{
					Headers: []*core.HeaderValue{
						{Key: ":path", Value: "/v1/chat/completions"},
						{Key: ":method", Value: "POST"},
						{Key: "content-type", Value: "application/json"},
						{Key: "x-request-id", Value: reqID},
					},
				},
			},
		},
	}
	bodyReq := &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_RequestBody{
			RequestBody: &ext_proc.HttpBody{
				Body:        []byte(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello"}],"stream":true}`),
				EndOfStream: true,
			},
		},
	}
	respHeadersReq := &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{
					Headers: []*core.HeaderValue{
						{Key: ":status", Value: "200"},
						{Key: "content-type", Value: "text/event-stream"},
					},
				},
			},
		},
	}
	return headersReq, bodyReq, respHeadersReq
}

func TestLooperStreaming_ReceiveErrorLifecycle(t *testing.T) {
	t.Run("upstream error event chunk before client cancellation aligns to LifecycleFailed and response_encode_failed", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := buildStandardLifecycleRequests(t, "req-err-chunk-client-cancel")
		errorChunk := []byte("data: {\"error\":{\"message\":\"upstream model overloaded\",\"type\":\"server_error\",\"code\":\"server_error\"}}\n\n")
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        errorChunk,
					EndOfStream: false,
				},
			},
		}

		ctx, cancel := context.WithCancel(context.Background())
		stream := &scriptedMockStream{
			MockStream: MockStream{
				Requests:  []*ext_proc.ProcessingRequest{headersReq, bodyReq, respHeadersReq, respBodyReq},
				Responses: make([]*ext_proc.ProcessingResponse, 0),
				Ctx:       ctx,
			},
			errAfterRequests: context.Canceled,
			cancelCtx:        cancel,
		}

		err := router.Process(stream)
		require.NoError(t, err)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.StreamingAborted)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState, "replay lifecycle must be failed due to upstream error event")
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.NotEqual(t, "client_canceled", record.TerminalReason, "replay terminal reason must never be client_canceled when upstream sent an error event")
		require.Equal(t, "upstream_error_response", record.TerminalReason)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)

		// Assert terminal state idempotency: subsequent cleanup or cancellation cannot overwrite terminal state
		router.finalizeRouterReplay(capturedCtx, routerreplay.LifecycleAborted, "client_canceled")
		require.NoError(t, router.handleProcessReceiveError(capturedCtx, context.Canceled))

		recordAfter, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfter.LifecycleState, "subsequent cleanup must be idempotent")
		require.Equal(t, "upstream_error_response", recordAfter.TerminalReason)
		require.Equal(t, "response_encode_failed", recordAfter.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("upstream error event chunk before EOF aligns to LifecycleFailed and response_encode_failed", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := buildStandardLifecycleRequests(t, "req-err-chunk-eof")
		errorChunk := []byte("data: {\"error\":{\"message\":\"upstream model overloaded\",\"type\":\"server_error\",\"code\":\"server_error\"}}\n\n")
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        errorChunk,
					EndOfStream: false,
				},
			},
		}

		stream := &scriptedMockStream{
			MockStream: MockStream{
				Requests:  []*ext_proc.ProcessingRequest{headersReq, bodyReq, respHeadersReq, respBodyReq},
				Responses: make([]*ext_proc.ProcessingResponse, 0),
				Ctx:       context.Background(),
			},
			errAfterRequests: io.EOF,
		}

		err := router.Process(stream)
		require.NoError(t, err)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.StreamingAborted)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.NotEqual(t, "client_canceled", record.TerminalReason)
		require.NotEqual(t, "stream_ended_before_terminal_response", record.TerminalReason)
		require.Equal(t, "upstream_error_response", record.TerminalReason)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)

		// Assert idempotency
		router.finalizeRouterReplay(capturedCtx, routerreplay.LifecycleAborted, "client_canceled")
		recordAfter, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfter.LifecycleState)
		require.Equal(t, "upstream_error_response", recordAfter.TerminalReason)
	})

	t.Run("pure client cancellation without upstream error chunk aligns to LifecycleAborted and cancelled", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := buildStandardLifecycleRequests(t, "req-pure-client-cancel")

		ctx, cancel := context.WithCancel(context.Background())
		stream := &scriptedMockStream{
			MockStream: MockStream{
				Requests:  []*ext_proc.ProcessingRequest{headersReq, bodyReq, respHeadersReq},
				Responses: make([]*ext_proc.ProcessingResponse, 0),
				Ctx:       ctx,
			},
			errAfterRequests: context.Canceled,
			cancelCtx:        cancel,
		}

		err := router.Process(stream)
		require.NoError(t, err)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.StreamingAborted)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumCancelled), capturedCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState, "pure client cancel must be LifecycleAborted")
		require.Equal(t, "client_canceled", record.TerminalReason)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "cancelled", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("upstream error event chunk then Send cancellation aligns to LifecycleFailed and response_encode_failed", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := buildStandardLifecycleRequests(t, "req-err-chunk-send-cancel")
		errorChunk := []byte("data: {\"error\":{\"message\":\"upstream model overloaded\",\"type\":\"server_error\",\"code\":\"server_error\"}}\n\n")
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        errorChunk,
					EndOfStream: false,
				},
			},
		}

		ctx, cancel := context.WithCancel(context.Background())
		stream := &scriptedMockStream{
			MockStream: MockStream{
				Requests:  []*ext_proc.ProcessingRequest{headersReq, bodyReq, respHeadersReq, respBodyReq},
				Responses: make([]*ext_proc.ProcessingResponse, 0),
				Ctx:       ctx,
			},
			cancelCtx: cancel,
			onSend: func(resp *ext_proc.ProcessingResponse) error {
				if _, ok := resp.Response.(*ext_proc.ProcessingResponse_ResponseBody); ok {
					cancel()
					return context.Canceled
				}
				return nil
			},
		}

		err := router.Process(stream)
		require.ErrorIs(t, err, context.Canceled)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.StreamingAborted)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState, "replay lifecycle must be failed due to upstream error event")
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.NotEqual(t, "client_canceled", record.TerminalReason, "replay terminal reason must not be client_canceled")
		require.NotEqual(t, "request_canceled", record.TerminalReason, "replay terminal reason must not be request_canceled")
		require.Equal(t, "upstream_error_response", record.TerminalReason)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)

		// Assert terminal state idempotency: subsequent cleanup or cancellation cannot overwrite terminal state
		router.finalizeRouterReplay(capturedCtx, routerreplay.LifecycleAborted, "client_canceled")
		require.NoError(t, router.handleProcessReceiveError(capturedCtx, context.Canceled))

		recordAfter, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfter.LifecycleState, "subsequent cleanup must be idempotent")
		require.Equal(t, "upstream_error_response", recordAfter.TerminalReason)
	})

	t.Run("non-EOS malformed SSE chunk before client cancellation aligns to LifecycleFailed and response_encode_failed", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := buildStandardLifecycleRequests(t, "req-malformed-sse-client-cancel")
		malformedChunk := []byte("data: {invalid_json\n\n")
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        malformedChunk,
					EndOfStream: false,
				},
			},
		}

		ctx, cancel := context.WithCancel(context.Background())
		stream := &scriptedMockStream{
			MockStream: MockStream{
				Requests:  []*ext_proc.ProcessingRequest{headersReq, bodyReq, respHeadersReq, respBodyReq},
				Responses: make([]*ext_proc.ProcessingResponse, 0),
				Ctx:       ctx,
			},
			errAfterRequests: context.Canceled,
			cancelCtx:        cancel,
		}

		err := router.Process(stream)
		require.NoError(t, err)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.StreamingAborted)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState, "replay lifecycle must be failed due to malformed sse decode failure")
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.NotEqual(t, "client_canceled", record.TerminalReason)
		require.Equal(t, "upstream_error_response", record.TerminalReason)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("non-EOS malformed SSE chunk before EOF aligns to LifecycleFailed and response_encode_failed", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := buildStandardLifecycleRequests(t, "req-malformed-sse-eof")
		malformedChunk := []byte("data: {invalid_json\n\n")
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        malformedChunk,
					EndOfStream: false,
				},
			},
		}

		stream := &scriptedMockStream{
			MockStream: MockStream{
				Requests:  []*ext_proc.ProcessingRequest{headersReq, bodyReq, respHeadersReq, respBodyReq},
				Responses: make([]*ext_proc.ProcessingResponse, 0),
				Ctx:       context.Background(),
			},
			errAfterRequests: io.EOF,
		}

		err := router.Process(stream)
		require.NoError(t, err)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.StreamingAborted)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.NotEqual(t, "client_canceled", record.TerminalReason)
		require.NotEqual(t, "stream_ended_before_terminal_response", record.TerminalReason)
		require.Equal(t, "upstream_error_response", record.TerminalReason)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("pure client cancellation during Send aligns to LifecycleAborted and cancelled", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := buildStandardLifecycleRequests(t, "req-pure-client-send-cancel")
		validChunk := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":12345,\"model\":\"panel-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n")
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        validChunk,
					EndOfStream: false,
				},
			},
		}

		ctx, cancel := context.WithCancel(context.Background())
		stream := &scriptedMockStream{
			MockStream: MockStream{
				Requests:  []*ext_proc.ProcessingRequest{headersReq, bodyReq, respHeadersReq, respBodyReq},
				Responses: make([]*ext_proc.ProcessingResponse, 0),
				Ctx:       ctx,
			},
			cancelCtx: cancel,
			onSend: func(resp *ext_proc.ProcessingResponse) error {
				if _, ok := resp.Response.(*ext_proc.ProcessingResponse_ResponseBody); ok {
					cancel()
					return context.Canceled
				}
				return nil
			},
		}

		err := router.Process(stream)
		require.ErrorIs(t, err, context.Canceled)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.StreamingAborted)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumCancelled), capturedCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState, "pure client cancel must be LifecycleAborted")
		require.Equal(t, "client_canceled", record.TerminalReason)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "cancelled", record.RouteDiagnostics.FusionQuorum.Disposition)
	})
}
