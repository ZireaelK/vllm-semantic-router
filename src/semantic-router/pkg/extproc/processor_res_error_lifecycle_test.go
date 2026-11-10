package extproc

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
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

	t.Run("native session upstream reset with alive parent context aligns to LifecycleFailed and response_encode_failed", func(t *testing.T) {
		inflight.Reset()
		defer inflight.Reset()

		coreObs, observedLogs := observer.New(zapcore.InfoLevel)
		restoreLogger := zap.ReplaceGlobals(zap.New(coreObs))
		defer restoreLogger()

		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		ctx := context.Background()
		rawSession, err := router.Open(ctx)
		require.NoError(t, err)

		session, ok := rawSession.(*routingSession)
		require.True(t, ok)

		// 1. RequestHeaders
		headers := routing.Header{
			{Name: ":path", Value: "/v1/chat/completions"},
			{Name: ":method", Value: "POST"},
			{Name: "content-type", Value: "application/json"},
			{Name: "x-request-id", Value: "req-native-reset-alive"},
		}
		effect, err := session.RequestHeaders(headers, false)
		require.NoError(t, err)
		require.NotNil(t, effect)

		// 2. RequestBody (triggers quorum fallback to backup-model)
		body := []byte(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello"}],"stream":true}`)
		effect, err = session.RequestBody(body, true)
		require.NoError(t, err)
		require.NotNil(t, effect)

		capturedCtx := session.ctx
		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.LooperStreamingActive)
		require.NotNil(t, capturedCtx.PendingLooperQuorumOutcome)
		require.Equal(t, looper.FusionQuorumFallbackReady, capturedCtx.PendingLooperQuorumOutcome.Disposition)

		// Simulate open inflight admission for fallback model
		fallbackModel := capturedCtx.VSRSelectedModel
		require.Equal(t, "backup-model", fallbackModel)
		capturedCtx.InflightModel = fallbackModel
		capturedCtx.InflightToken = inflight.Begin(fallbackModel)
		require.NotZero(t, capturedCtx.InflightToken, "inflight token must be acquired")
		require.Equal(t, 1, inflight.Get(fallbackModel), "inflight count must be 1")

		// 3. ResponseHeaders
		respHeaders := routing.Header{
			{Name: ":status", Value: "200"},
			{Name: "content-type", Value: "text/event-stream"},
		}
		effect, err = session.ResponseHeaders(respHeaders, false)
		require.NoError(t, err)
		require.NotNil(t, effect)
		require.True(t, capturedCtx.IsStreamingResponse)
		require.Equal(t, 200, capturedCtx.UpstreamStatusCode)

		// 4. ResponseBody (receive valid SSE partial chunk)
		validChunk := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":12345,\"model\":\"backup-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n")
		effect, err = session.ResponseBody(validChunk, false)
		require.NoError(t, err)
		require.NotNil(t, effect)
		require.NotNil(t, capturedCtx.SemanticStreamState)
		require.Nil(t, capturedCtx.SemanticStreamState.failed, "valid chunk must not register failure")

		// 5. Upstream connection resets / drops while parent context is still alive
		require.NoError(t, session.ctx.Context.Err(), "parent context must be alive and unexpired")
		upstreamResetErr := errors.New("connection reset by peer")
		session.Close(upstreamResetErr)

		// Assertions:
		// 1) Quorum outcome is response_encode_failed (NOT cancelled)
		require.True(t, capturedCtx.StreamingAborted)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)
		require.NotEqual(t, string(looper.FusionQuorumCancelled), capturedCtx.VSRFusionQuorum.Disposition)

		// 2) usage log abort_reason is upstream_response_failed (NOT client_canceled)
		var usageLogs []observer.LoggedEntry
		for _, entry := range observedLogs.All() {
			if entry.ContextMap()["event"] == "llm_usage" {
				usageLogs = append(usageLogs, entry)
			}
		}
		require.NotEmpty(t, usageLogs, "llm_usage audit log must be emitted")
		usageFields := usageLogs[len(usageLogs)-1].ContextMap()
		require.Equal(t, "upstream_response_failed", usageFields["abort_reason"])
		require.NotEqual(t, "client_canceled", usageFields["abort_reason"])

		// 3) Replay record is LifecycleFailed (NOT LifecycleAborted)
		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "upstream_response_failed", record.TerminalReason)
		require.NotEqual(t, "client_canceled", record.TerminalReason)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
		require.NotEqual(t, "cancelled", record.RouteDiagnostics.FusionQuorum.Disposition)

		// 4) Inflight count cleared to 0 and resources correctly released
		require.Zero(t, capturedCtx.InflightToken, "inflight token must be cleared to 0")
		require.Zero(t, inflight.Get(capturedCtx.InflightModel), "inflight count must be 0")
	})

	t.Run("native session upstream reset with canceled parent context aligns to LifecycleAborted and cancelled", func(t *testing.T) {
		inflight.Reset()
		defer inflight.Reset()

		coreObs, observedLogs := observer.New(zapcore.InfoLevel)
		restoreLogger := zap.ReplaceGlobals(zap.New(coreObs))
		defer restoreLogger()

		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		ctx, cancel := context.WithCancel(context.Background())
		rawSession, err := router.Open(ctx)
		require.NoError(t, err)

		session, ok := rawSession.(*routingSession)
		require.True(t, ok)

		// 1. RequestHeaders
		headers := routing.Header{
			{Name: ":path", Value: "/v1/chat/completions"},
			{Name: ":method", Value: "POST"},
			{Name: "content-type", Value: "application/json"},
			{Name: "x-request-id", Value: "req-native-reset-canceled"},
		}
		effect, err := session.RequestHeaders(headers, false)
		require.NoError(t, err)
		require.NotNil(t, effect)

		// 2. RequestBody
		body := []byte(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello"}],"stream":true}`)
		effect, err = session.RequestBody(body, true)
		require.NoError(t, err)
		require.NotNil(t, effect)

		capturedCtx := session.ctx
		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.LooperStreamingActive)
		require.NotNil(t, capturedCtx.PendingLooperQuorumOutcome)

		// Simulate open inflight admission for fallback model
		fallbackModel := capturedCtx.VSRSelectedModel
		require.Equal(t, "backup-model", fallbackModel)
		capturedCtx.InflightModel = fallbackModel
		capturedCtx.InflightToken = inflight.Begin(fallbackModel)
		require.NotZero(t, capturedCtx.InflightToken, "inflight token must be acquired")
		require.Equal(t, 1, inflight.Get(fallbackModel), "inflight count must be 1")

		// 3. ResponseHeaders
		respHeaders := routing.Header{
			{Name: ":status", Value: "200"},
			{Name: "content-type", Value: "text/event-stream"},
		}
		effect, err = session.ResponseHeaders(respHeaders, false)
		require.NoError(t, err)
		require.NotNil(t, effect)

		// 4. ResponseBody (receive valid SSE partial chunk)
		validChunk := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":12345,\"model\":\"backup-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n")
		effect, err = session.ResponseBody(validChunk, false)
		require.NoError(t, err)
		require.NotNil(t, effect)

		// 5. Explicitly cancel parent context, simulating client disconnect
		cancel()

		// session.Close with upstream error
		session.Close(errors.New("connection reset by peer"))

		// Assertions:
		// 1) Quorum outcome is cancelled
		require.True(t, capturedCtx.StreamingAborted)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumCancelled), capturedCtx.VSRFusionQuorum.Disposition)

		// 2) usage log abort_reason is client_canceled
		var usageLogs []observer.LoggedEntry
		for _, entry := range observedLogs.All() {
			if entry.ContextMap()["event"] == "llm_usage" {
				usageLogs = append(usageLogs, entry)
			}
		}
		require.NotEmpty(t, usageLogs, "llm_usage audit log must be emitted")
		usageFields := usageLogs[len(usageLogs)-1].ContextMap()
		require.Equal(t, "client_canceled", usageFields["abort_reason"])

		// 3) Replay record is LifecycleAborted and client_canceled
		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "client_canceled", record.TerminalReason)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "cancelled", record.RouteDiagnostics.FusionQuorum.Disposition)

		// 4) Inflight count cleared to 0
		require.Zero(t, capturedCtx.InflightToken)
		require.Zero(t, inflight.Get(capturedCtx.InflightModel))
	})

	t.Run("non-streaming 200 application/json malformed body in Process returns 502 and aligns replay to LifecycleFailed and response_decode_failed", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, _ := buildStandardLifecycleRequests(t, "req-malformed-json-process")
		respHeadersReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: ":status", Value: "200"},
							{Key: "content-type", Value: "application/json"},
						},
					},
				},
			},
		}
		malformedJSONBody := []byte("{invalid_json_body_syntax")
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        malformedJSONBody,
					EndOfStream: true,
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
			errAfterRequests: io.EOF,
			cancelCtx:        cancel,
		}

		err := router.Process(stream)
		require.NoError(t, err)

		// 1. Assert client received 502 ImmediateResponse
		require.NotEmpty(t, stream.Responses)
		var immediateResp *ext_proc.ImmediateResponse
		for _, resp := range stream.Responses {
			if imm := resp.GetImmediateResponse(); imm != nil {
				immediateResp = imm
				break
			}
		}
		require.NotNil(t, immediateResp, "client must receive immediate error response")
		require.Equal(t, uint32(502), uint32(immediateResp.GetStatus().GetCode()))
		require.Contains(t, string(immediateResp.GetBody()), "The selected model returned an invalid response")

		// 2. Assert Quorum terminal disposition is response_encode_failed
		require.NotNil(t, capturedCtx)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)
		require.NotEqual(t, string(looper.FusionQuorumCancelled), capturedCtx.VSRFusionQuorum.Disposition)

		// 3. Assert Replay record is LifecycleFailed with reason "response_decode_failed"
		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState, "replay lifecycle must be failed due to decode error")
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "response_decode_failed", record.TerminalReason, "replay reason must be response_decode_failed")
		require.NotEqual(t, "client_canceled", record.TerminalReason)
		require.NotEqual(t, "stream_ended_before_terminal_response", record.TerminalReason)
		require.Equal(t, 502, record.ResponseStatus)
		require.Contains(t, string(record.ResponseBody), "The selected model returned an invalid response")

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)

		// 4. Simulate subsequent Recv EOF and client cancellation: assert idempotency
		router.finalizeRouterReplay(capturedCtx, routerreplay.LifecycleAborted, "client_canceled")
		require.NoError(t, router.handleProcessReceiveError(capturedCtx, context.Canceled))
		require.NoError(t, router.handleProcessReceiveError(capturedCtx, io.EOF))

		recordAfter, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfter.LifecycleState, "subsequent cleanup must be idempotent")
		require.Equal(t, "response_decode_failed", recordAfter.TerminalReason)
		require.Equal(t, "response_encode_failed", recordAfter.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("non-streaming 200 application/json malformed body in Process before client cancellation aligns to LifecycleFailed and response_decode_failed", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, _ := buildStandardLifecycleRequests(t, "req-malformed-json-cancel")
		respHeadersReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: ":status", Value: "200"},
							{Key: "content-type", Value: "application/json"},
						},
					},
				},
			},
		}
		malformedJSONBody := []byte("{invalid_json_body_syntax")
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        malformedJSONBody,
					EndOfStream: true,
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

		require.NotEmpty(t, stream.Responses)
		var immediateResp *ext_proc.ImmediateResponse
		for _, resp := range stream.Responses {
			if imm := resp.GetImmediateResponse(); imm != nil {
				immediateResp = imm
				break
			}
		}
		require.NotNil(t, immediateResp)
		require.Equal(t, uint32(502), uint32(immediateResp.GetStatus().GetCode()))

		require.NotNil(t, capturedCtx)
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome)
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)

		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "response_decode_failed", record.TerminalReason)
		require.NotEqual(t, "client_canceled", record.TerminalReason)
		require.Equal(t, 502, record.ResponseStatus)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("native session non-streaming malformed JSON response aligns to LifecycleFailed and response_decode_failed after Close", func(t *testing.T) {
		server, router, _ := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		ctx := context.Background()
		rawSession, err := router.Open(ctx)
		require.NoError(t, err)

		session, ok := rawSession.(*routingSession)
		require.True(t, ok)
		capturedCtx := session.ctx

		// 1. RequestHeaders
		headers := routing.Header{
			{Name: ":path", Value: "/v1/chat/completions"},
			{Name: ":method", Value: "POST"},
			{Name: "content-type", Value: "application/json"},
			{Name: "x-request-id", Value: "req-native-malformed-json"},
		}
		effect, err := session.RequestHeaders(headers, false)
		require.NoError(t, err)
		require.NotNil(t, effect)

		// 2. RequestBody (triggers quorum fallback)
		body := []byte(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello"}],"stream":true}`)
		effect, err = session.RequestBody(body, true)
		require.NoError(t, err)
		require.NotNil(t, effect)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.LooperStreamingActive)
		require.NotNil(t, capturedCtx.PendingLooperQuorumOutcome)

		// 3. ResponseHeaders (200 application/json)
		respHeaders := routing.Header{
			{Name: ":status", Value: "200"},
			{Name: "content-type", Value: "application/json"},
		}
		effect, err = session.ResponseHeaders(respHeaders, false)
		require.NoError(t, err)
		require.NotNil(t, effect)
		require.False(t, capturedCtx.IsStreamingResponse)
		require.Equal(t, 200, capturedCtx.UpstreamStatusCode)

		// 4. ResponseBody (malformed JSON)
		malformedJSON := []byte("{malformed_json_syntax")
		effect, err = session.ResponseBody(malformedJSON, true)
		require.NoError(t, err)
		require.NotNil(t, effect)
		require.NotNil(t, effect.Immediate, "native session ResponseBody must return immediate response on decode error")
		require.Equal(t, 502, effect.Immediate.Status)
		require.Contains(t, string(effect.Immediate.Body), "The selected model returned an invalid response")

		// 5. Close session (graceful EOF or transport finish)
		session.Close(nil)

		// 6. Assertions
		require.Nil(t, capturedCtx.PendingLooperQuorumOutcome, "pending outcome must be finalized")
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)
		require.NotEqual(t, string(looper.FusionQuorumCancelled), capturedCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, capturedCtx.RouterReplayID)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState, "replay lifecycle must be failed")
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "response_decode_failed", record.TerminalReason)
		require.NotEqual(t, "stream_ended_before_terminal_response", record.TerminalReason)
		require.NotEqual(t, "client_canceled", record.TerminalReason)
		require.Equal(t, 502, record.ResponseStatus)

		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)

		// Assert idempotency on second close with cancellation or receive error
		router.finalizeRouterReplay(capturedCtx, routerreplay.LifecycleAborted, "client_canceled")
		require.NoError(t, router.handleProcessReceiveError(capturedCtx, context.Canceled))
		recordAfter, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfter.LifecycleState, "subsequent receive error must be idempotent")
		require.Equal(t, "response_decode_failed", recordAfter.TerminalReason)
	})

	t.Run("non-streaming 200 application/json malformed body with initial store failure retries on receive error cleanup", func(t *testing.T) {
		server, router, memStore := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		faultStore := &faultInjectingStore{
			Storage:                  memStore,
			failUpdateLifecycleCount: 1,
			failUpdateLifecycleErr:   errors.New("db connection timeout"),
		}
		router.ReplayRecorder = routerreplay.NewRecorder(faultStore)

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, _ := buildStandardLifecycleRequests(t, "req-malformed-json-retry-store-fail")
		respHeadersReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: ":status", Value: "200"},
							{Key: "content-type", Value: "application/json"},
						},
					},
				},
			},
		}
		malformedJSONBody := []byte("{invalid_json_body_syntax")
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        malformedJSONBody,
					EndOfStream: true,
				},
			},
		}

		var sendCalled bool
		ctx, cancel := context.WithCancel(context.Background())
		stream := &scriptedMockStream{
			MockStream: MockStream{
				Requests:  []*ext_proc.ProcessingRequest{headersReq, bodyReq, respHeadersReq, respBodyReq},
				Responses: make([]*ext_proc.ProcessingResponse, 0),
				Ctx:       ctx,
			},
			errAfterRequests: context.Canceled,
			cancelCtx:        cancel,
			onSend: func(resp *ext_proc.ProcessingResponse) error {
				if resp.GetImmediateResponse() != nil {
					sendCalled = true
					// First UpdateLifecycle failed:
					// 1. RouterReplayCommitted must NOT be true
					require.False(t, capturedCtx.RouterReplayCommitted, "RouterReplayCommitted must be false after initial storage failure")
					// 2. Initial terminal decision must be recorded on ctx
					require.Equal(t, routerreplay.LifecycleFailed, capturedCtx.TerminalLifecycleState)
					require.Equal(t, "response_decode_failed", capturedCtx.TerminalLifecycleReason)
					// 3. Underlying record in storage must still be in_progress
					rec, found, err := memStore.Get(context.Background(), capturedCtx.RouterReplayID)
					require.NoError(t, err)
					require.True(t, found)
					require.Equal(t, store.LifecycleInProgress, rec.LifecycleState, "underlying record must still be in_progress")
				}
				return nil
			},
		}

		err := router.Process(stream)
		require.NoError(t, err)
		require.True(t, sendCalled)

		// After Process finished, Recv returned context.Canceled which triggered handleProcessReceiveError
		// 1. Second UpdateLifecycle call succeeded, RouterReplayCommitted is true
		require.True(t, capturedCtx.RouterReplayCommitted, "RouterReplayCommitted must be true after retry during cleanup")

		// 2. Record must have transitioned to LifecycleFailed with response_decode_failed (NOT overwritten by client_canceled)
		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "response_decode_failed", record.TerminalReason)
		require.NotEqual(t, "client_canceled", record.TerminalReason)
		require.NotEqual(t, "stream_ended_before_terminal_response", record.TerminalReason)
		require.Equal(t, 502, record.ResponseStatus)

		// 3. Assert idempotency on subsequent retry / cleanup calls
		callsBefore := faultStore.updateLifecycleCalls
		router.finalizeRouterReplay(capturedCtx, routerreplay.LifecycleAborted, "client_canceled")
		require.NoError(t, router.handleProcessReceiveError(capturedCtx, context.Canceled))
		require.NoError(t, router.handleProcessReceiveError(capturedCtx, io.EOF))

		require.Equal(t, callsBefore, faultStore.updateLifecycleCalls, "subsequent finalization must be strictly idempotent")
		recordAfter, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfter.LifecycleState)
		require.Equal(t, "response_decode_failed", recordAfter.TerminalReason)
	})

	t.Run("native session non-streaming malformed JSON with initial store failure retries on Close", func(t *testing.T) {
		server, router, memStore := setupReceiveErrorLifecycleFixture(t)
		defer server.Close()

		faultStore := &faultInjectingStore{
			Storage:                  memStore,
			failUpdateLifecycleCount: 1,
			failUpdateLifecycleErr:   errors.New("db connection timeout"),
		}
		router.ReplayRecorder = routerreplay.NewRecorder(faultStore)

		ctx := context.Background()
		rawSession, err := router.Open(ctx)
		require.NoError(t, err)

		session, ok := rawSession.(*routingSession)
		require.True(t, ok)
		capturedCtx := session.ctx

		// 1. RequestHeaders
		headers := routing.Header{
			{Name: ":path", Value: "/v1/chat/completions"},
			{Name: ":method", Value: "POST"},
			{Name: "content-type", Value: "application/json"},
			{Name: "x-request-id", Value: "req-native-store-retry"},
		}
		effect, err := session.RequestHeaders(headers, false)
		require.NoError(t, err)
		require.NotNil(t, effect)

		// 2. RequestBody
		body := []byte(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello"}],"stream":true}`)
		effect, err = session.RequestBody(body, true)
		require.NoError(t, err)
		require.NotNil(t, effect)

		// 3. ResponseHeaders
		respHeaders := routing.Header{
			{Name: ":status", Value: "200"},
			{Name: "content-type", Value: "application/json"},
		}
		effect, err = session.ResponseHeaders(respHeaders, false)
		require.NoError(t, err)
		require.NotNil(t, effect)

		// 4. ResponseBody with malformed JSON
		malformedJSON := []byte("{malformed_json_syntax")
		effect, err = session.ResponseBody(malformedJSON, true)
		require.NoError(t, err)
		require.NotNil(t, effect)
		require.NotNil(t, effect.Immediate)
		require.Equal(t, 502, effect.Immediate.Status)

		// First UpdateLifecycle failed:
		require.False(t, capturedCtx.RouterReplayCommitted, "RouterReplayCommitted must be false after store failure")
		require.Equal(t, routerreplay.LifecycleFailed, capturedCtx.TerminalLifecycleState)
		require.Equal(t, "response_decode_failed", capturedCtx.TerminalLifecycleReason)

		recBefore, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, store.LifecycleInProgress, recBefore.LifecycleState)

		// 5. Subsequent Close with client cancellation
		session.Close(context.Canceled)

		// Second persistence succeeded:
		require.True(t, capturedCtx.RouterReplayCommitted, "RouterReplayCommitted must be true after retry on Close")

		record, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "response_decode_failed", record.TerminalReason, "must keep initial terminal reason, not overwritten by client_canceled")
		require.Equal(t, 502, record.ResponseStatus)

		// 6. Assert idempotency
		callsBefore := faultStore.updateLifecycleCalls
		session.Close(nil)
		router.finalizeRouterReplay(capturedCtx, routerreplay.LifecycleAborted, "client_canceled")
		require.NoError(t, router.handleProcessReceiveError(capturedCtx, io.EOF))
		require.Equal(t, callsBefore, faultStore.updateLifecycleCalls, "subsequent finalization must be idempotent")

		recordAfter, found := router.ReplayRecorder.GetRecord(capturedCtx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfter.LifecycleState)
		require.Equal(t, "response_decode_failed", recordAfter.TerminalReason)
	})
}

type faultInjectingStore struct {
	store.Storage
	mu                       sync.Mutex
	failUpdateLifecycleCount int
	failUpdateLifecycleErr   error
	updateLifecycleCalls     int
}

func (f *faultInjectingStore) UpdateLifecycle(ctx context.Context, id string, state string, endedAt time.Time, durationMS int64, reason string) error {
	f.mu.Lock()
	f.updateLifecycleCalls++
	if f.failUpdateLifecycleCount > 0 {
		f.failUpdateLifecycleCount--
		err := f.failUpdateLifecycleErr
		if err == nil {
			err = errors.New("db connection timeout")
		}
		f.mu.Unlock()
		return err
	}
	f.mu.Unlock()
	return f.Storage.UpdateLifecycle(ctx, id, state, endedAt, durationMS, reason)
}
