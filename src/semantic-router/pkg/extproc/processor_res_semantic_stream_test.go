package extproc

import (
	"context"
	"errors"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// TestNonLooperStreaming_MissingUsageRetainsLifecycleCompleted verifies that for
// ordinary non-Looper requests, a complete and valid SSE stream (with valid text
// chunks, finish_reason, and [DONE]) whose upstream does not return usage statistics
// is successfully committed as LifecycleCompleted with "response_complete", and is
// NOT erroneously aborted or committed as LifecycleFailed with "upstream_stream_incomplete".
func TestNonLooperStreaming_MissingUsageRetainsLifecycleCompleted(t *testing.T) {
	t.Run("without_prior_usage_completes_with_nil_tokens", func(t *testing.T) {
		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()

		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{Enabled: true},
			},
			ReplayRecorder: recorder,
		}

		reqCtx := &RequestContext{
			RequestID:                "req-non-looper-no-usage",
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			IsStreamingResponse:      true,
			LooperStreamingActive:    false,
			SourceFormat:             llmprotocol.OpenAIChatV1,
			TargetFormat:             llmprotocol.OpenAIChatV1,
			TraceContext:             context.Background(),
			UpstreamStatusCode:       200,
			RequestModel:             "gpt-4",
		}
		router.startRouterReplay(reqCtx, "gpt-4", "gpt-4", "default")

		// Upstream chunks without any usage chunk
		chunk1 := "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello world\"}}]}\n\n"
		chunk2 := "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
		chunk3 := "data: [DONE]\n\n"

		resp1 := router.handleSemanticStreamingResponseBody([]byte(chunk1), false, reqCtx)
		require.NotNil(t, resp1)
		require.False(t, reqCtx.StreamingAborted)

		resp2 := router.handleSemanticStreamingResponseBody([]byte(chunk2), false, reqCtx)
		require.NotNil(t, resp2)
		require.False(t, reqCtx.StreamingAborted)

		resp3 := router.handleSemanticStreamingResponseBody([]byte(chunk3), true, reqCtx)
		require.NotNil(t, resp3)

		// Verify streaming is NOT aborted
		require.False(t, reqCtx.StreamingAborted, "streaming must not be marked aborted when usage is missing on complete stream")

		// Verify replay record is committed as LifecycleCompleted (not LifecycleFailed)
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState, "complete stream without usage must finalize as LifecycleCompleted")
		require.Equal(t, "response_complete", record.TerminalReason, "complete stream without usage must have TerminalReason 'response_complete'")
		require.NotEqual(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.NotEqual(t, "upstream_stream_incomplete", record.TerminalReason)

		// Verify usage is empty/nil since upstream did not provide authoritative usage
		require.Nil(t, record.PromptTokens, "prompt tokens must be nil when upstream provided no usage")
		require.Nil(t, record.CompletionTokens, "completion tokens must be nil when upstream provided no usage")
		require.Nil(t, record.TotalTokens, "total tokens must be nil when upstream provided no usage")
	})

	t.Run("with_estimated_usage_preserves_estimation_and_completes", func(t *testing.T) {
		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()

		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{Enabled: true},
			},
			ReplayRecorder: recorder,
		}

		reqCtx := &RequestContext{
			RequestID:                "req-non-looper-estimated-usage",
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			IsStreamingResponse:      true,
			LooperStreamingActive:    false,
			SourceFormat:             llmprotocol.OpenAIChatV1,
			TargetFormat:             llmprotocol.OpenAIChatV1,
			TraceContext:             context.Background(),
			UpstreamStatusCode:       200,
			RequestModel:             "gpt-4",
		}
		router.startRouterReplay(reqCtx, "gpt-4", "gpt-4", "default")
		router.updateRouterReplayUsageCost(reqCtx, routerreplay.UsageCost{
			PromptTokens: replayIntPtr(25),
			TotalTokens:  replayIntPtr(25),
		})

		chunk1 := "data: {\"id\":\"chatcmpl-est\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello estimated\"}}]}\n\n"
		chunk2 := "data: {\"id\":\"chatcmpl-est\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
		chunk3 := "data: [DONE]\n\n"

		router.handleSemanticStreamingResponseBody([]byte(chunk1), false, reqCtx)
		router.handleSemanticStreamingResponseBody([]byte(chunk2), false, reqCtx)
		router.handleSemanticStreamingResponseBody([]byte(chunk3), true, reqCtx)

		require.False(t, reqCtx.StreamingAborted, "streaming must not be marked aborted when usage is missing on complete stream")
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
		require.Equal(t, "response_complete", record.TerminalReason)
		require.NotNil(t, record.PromptTokens)
		require.Equal(t, 25, *record.PromptTokens, "estimated prompt tokens must be preserved, not overwritten or cleared as aborted")
	})
}

// TestNonLooperStreaming_TruncatedStreamMarksLifecycleFailed acts as negative control:
// an SSE stream that closes prematurely without finish_reason and missing [DONE]
// must still be correctly marked as StreamingAborted and LifecycleFailed with "upstream_stream_incomplete".
func TestNonLooperStreaming_TruncatedStreamMarksLifecycleFailed(t *testing.T) {
	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()

	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{Enabled: true},
		},
		ReplayRecorder: recorder,
	}

	reqCtx := &RequestContext{
		RequestID:                "req-non-looper-truncated",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		IsStreamingResponse:      true,
		LooperStreamingActive:    false,
		SourceFormat:             llmprotocol.OpenAIChatV1,
		TargetFormat:             llmprotocol.OpenAIChatV1,
		TraceContext:             context.Background(),
		UpstreamStatusCode:       200,
		RequestModel:             "gpt-4",
	}
	router.startRouterReplay(reqCtx, "gpt-4", "gpt-4", "default")

	// Upstream truncated chunk: partial content, no finish_reason, no [DONE]
	truncatedChunk := "data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Truncated response...\"}}]}\n\n"

	router.handleSemanticStreamingResponseBody([]byte(truncatedChunk), false, reqCtx)

	// Early EOF / endOfStream without [DONE]
	router.handleSemanticStreamingResponseBody(nil, true, reqCtx)

	// Verify streaming IS marked aborted
	require.True(t, reqCtx.StreamingAborted, "truncated stream must be marked aborted")

	// Verify replay record is committed as LifecycleFailed with upstream_stream_incomplete
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState, "truncated stream must finalize as LifecycleFailed")
	require.Equal(t, "upstream_stream_incomplete", record.TerminalReason, "truncated stream must have TerminalReason 'upstream_stream_incomplete'")
}

// TestRoutingSession_PhaseErrorLifecycleSymmetry verifies that routingSession.phase
// delegates to resolveReplayLifecycleState on errors, ensuring 100% contract symmetry
// with Envoy Process (processor_core.go).
func TestRoutingSession_PhaseErrorLifecycleSymmetry(t *testing.T) {
	t.Run("client_cancellation_during_phase_resolves_to_LifecycleAborted", func(t *testing.T) {
		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()

		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{Enabled: true},
			},
			ReplayRecorder: recorder,
		}

		session := router.newRoutingSession(context.Background(), nil)
		session.ctx.RequestID = "req-session-cancel"
		session.ctx.RouterReplayRecorder = recorder
		session.ctx.RouterReplayPluginConfig = &replayConfig
		router.startRouterReplay(session.ctx, "gpt-4", "gpt-4", "default")

		// Simulate client cancellation error returned by a reply func in phase
		_, err := session.phase(func() (*ext_proc.ProcessingResponse, error) {
			return nil, context.Canceled
		}, func(*ext_proc.ProcessingResponse) {})

		require.Error(t, err)
		require.True(t, errors.Is(err, context.Canceled))

		record, ok := recorder.GetRecord(session.ctx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState,
			"client cancellation in session phase must resolve to LifecycleAborted via resolveReplayLifecycleState")
		require.Equal(t, "client_canceled", record.TerminalReason)
	})

	t.Run("upstream_error_status_during_phase_resolves_to_upstream_error_response", func(t *testing.T) {
		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()

		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{Enabled: true},
			},
			ReplayRecorder: recorder,
		}

		session := router.newRoutingSession(context.Background(), nil)
		session.ctx.RequestID = "req-session-upstream-err"
		session.ctx.RouterReplayRecorder = recorder
		session.ctx.RouterReplayPluginConfig = &replayConfig
		session.ctx.UpstreamStatusCode = 502
		router.startRouterReplay(session.ctx, "gpt-4", "gpt-4", "default")

		_, err := session.phase(func() (*ext_proc.ProcessingResponse, error) {
			return nil, errors.New("upstream connection reset")
		}, func(*ext_proc.ProcessingResponse) {})

		require.Error(t, err)

		record, ok := recorder.GetRecord(session.ctx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "upstream_error_response", record.TerminalReason,
			"non-2xx upstream status code in session phase must resolve to upstream_error_response via resolveReplayLifecycleState")
	})

	t.Run("grpc_canceled_during_phase_resolves_to_LifecycleAborted", func(t *testing.T) {
		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()

		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{Enabled: true},
			},
			ReplayRecorder: recorder,
		}

		session := router.newRoutingSession(context.Background(), nil)
		session.ctx.RequestID = "req-session-grpc-cancel"
		session.ctx.RouterReplayRecorder = recorder
		session.ctx.RouterReplayPluginConfig = &replayConfig
		router.startRouterReplay(session.ctx, "gpt-4", "gpt-4", "default")

		_, err := session.phase(func() (*ext_proc.ProcessingResponse, error) {
			return nil, status.Error(codes.Canceled, "request canceled by client")
		}, func(*ext_proc.ProcessingResponse) {})

		require.Error(t, err)

		record, ok := recorder.GetRecord(session.ctx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "client_canceled", record.TerminalReason)
	})
}
