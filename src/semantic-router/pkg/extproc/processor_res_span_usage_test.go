package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestLooperStreaming_UpstreamSpanUsage(t *testing.T) {
	spanAttrsMap := func(attrs []attribute.KeyValue) map[string]attribute.Value {
		res := make(map[string]attribute.Value, len(attrs))
		for _, a := range attrs {
			res[string(a.Key)] = a.Value
		}
		return res
	}

	t.Run("SSE_Streaming", func(t *testing.T) {
		coreObs, observedLogs := observer.New(zapcore.InfoLevel)
		restoreLogger := zap.ReplaceGlobals(zap.New(coreObs))
		t.Cleanup(restoreLogger)

		exporter := tracetest.NewInMemoryExporter()
		tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
		previousTP := otel.GetTracerProvider()
		otel.SetTracerProvider(tp)
		t.Cleanup(func() {
			_ = tp.Shutdown(context.Background())
			otel.SetTracerProvider(previousTP)
		})

		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()
		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{Enabled: true},
			},
			ReplayRecorder: recorder,
		}

		tracer := tp.Tracer("test-tracer")
		traceCtx, upstreamSpan := tracer.Start(context.Background(), tracing.SpanUpstreamRequest)

		// Final upstream model reports 40 prompt / 10 completion
		streamState := &semanticResponseStreamState{
			responseID: "chatcmpl-stream-test",
			model:      "judge",
			stop:       llmprotocol.StopEndTurn,
			terminal:   true,
			items: map[int]*semanticStreamItem{
				0: {role: llmprotocol.RoleAssistant, text: "synthesized", completed: true},
			},
			order: []int{0},
			usage: llmprotocol.Usage{
				State: llmprotocol.UsageAvailable,
				InputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(40),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				OutputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(10),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				Total: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(50),
					Provenance: llmprotocol.UsageAuthoritative,
				},
			},
		}

		reqCtx := &RequestContext{
			RequestID:               "req-sse-span-usage",
			RequestModel:            "virtual-entrypoint",
			VSRSelectedModel:        "judge",
			LooperTargetModel:       "judge",
			VSRSelectedDecisionName: "fusion-stream",
			VSRSelectionMethod:      "cascade",
			SourceFormat:            llmprotocol.OpenAIChatV1,
			TargetFormat:            llmprotocol.OpenAIChatV1,
			IsStreamingResponse:     true,
			LooperStreamingActive:   true,
			// Panel intermediate stage consumed 100 prompt / 20 completion
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     100,
				CompletionTokens: 20,
				TotalTokens:      120,
			},
			SemanticStreamState:      streamState,
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			TraceContext:             traceCtx,
			UpstreamSpan:             upstreamSpan,
			StartTime:                time.Now().Add(-100 * time.Millisecond),
		}
		router.startRouterReplay(reqCtx, "virtual-entrypoint", "judge", "fusion-stream")

		router.finalizeSemanticStreamingResponse(reqCtx, nil)
		require.False(t, reqCtx.StreamingAborted)
		require.True(t, reqCtx.LooperUsageSettled)

		reqCtx.UpstreamSpan.End()

		// 1. Verify upstream span has strictly single-stage physical model tokens (40 input / 10 output)
		spans := exporter.GetSpans()
		require.Len(t, spans, 1)
		attrs := spanAttrsMap(spans[0].Attributes)
		require.Equal(t, int64(40), attrs["gen_ai.usage.input_tokens"].AsInt64())
		require.Equal(t, int64(10), attrs["gen_ai.usage.output_tokens"].AsInt64())

		// 2. Verify Replay record has full cascaded tokens (140 prompt / 30 completion)
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.Equal(t, 140, *record.PromptTokens)
		require.Equal(t, 30, *record.CompletionTokens)
		require.Equal(t, 170, *record.TotalTokens)

		// 3. Verify settlement logs (looper_streaming_completed and llm_usage) have cascaded tokens (140 / 30 / 170)
		var completedFound, usageFound bool
		for _, entry := range observedLogs.All() {
			cmap := entry.ContextMap()
			if cmap["event"] == "looper_streaming_completed" {
				completedFound = true
				require.EqualValues(t, 140, cmap["prompt_tokens"])
				require.EqualValues(t, 30, cmap["completion_tokens"])
				require.EqualValues(t, 170, cmap["total_tokens"])
			}
			if cmap["event"] == "llm_usage" {
				usageFound = true
				require.EqualValues(t, 140, cmap["prompt_tokens"])
				require.EqualValues(t, 30, cmap["completion_tokens"])
				require.EqualValues(t, 170, cmap["total_tokens"])
			}
		}
		require.True(t, completedFound, "expected looper_streaming_completed event in logs")
		require.True(t, usageFound, "expected llm_usage event in logs")
	})

	t.Run("JSON_Downgrade_200", func(t *testing.T) {
		coreObs, observedLogs := observer.New(zapcore.InfoLevel)
		restoreLogger := zap.ReplaceGlobals(zap.New(coreObs))
		t.Cleanup(restoreLogger)

		exporter := tracetest.NewInMemoryExporter()
		tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
		previousTP := otel.GetTracerProvider()
		otel.SetTracerProvider(tp)
		t.Cleanup(func() {
			_ = tp.Shutdown(context.Background())
			otel.SetTracerProvider(previousTP)
		})

		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()
		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{Enabled: true},
			},
			ReplayRecorder: recorder,
		}

		tracer := tp.Tracer("test-tracer")
		traceCtx, upstreamSpan := tracer.Start(context.Background(), tracing.SpanUpstreamRequest)

		reqCtx := &RequestContext{
			RequestID:               "req-json-downgrade-usage",
			RequestModel:            "virtual-entrypoint",
			VSRSelectedModel:        "judge",
			LooperTargetModel:       "judge",
			VSRSelectedDecisionName: "fusion-json",
			VSRSelectionMethod:      "cascade",
			SourceFormat:            llmprotocol.OpenAIChatV1,
			TargetFormat:            llmprotocol.OpenAIChatV1,
			LooperStreamingActive:   true,
			// Panel intermediate stage consumed 100 prompt / 20 completion
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     100,
				CompletionTokens: 20,
				TotalTokens:      120,
			},
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			TraceContext:             traceCtx,
			UpstreamSpan:             upstreamSpan,
			StartTime:                time.Now().Add(-100 * time.Millisecond),
		}
		router.startRouterReplay(reqCtx, "virtual-entrypoint", "judge", "fusion-json")

		// Final upstream model response body reporting 40 prompt / 10 completion
		responseBody := []byte(`{
			"id": "chatcmpl-json-test",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "judge",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}
		}`)

		resp := router.handleNonStreamingResponseBody(responseBody, reqCtx, 100*time.Millisecond)
		require.NotNil(t, resp)
		require.True(t, reqCtx.LooperUsageSettled)

		reqCtx.UpstreamSpan.End()

		// 1. Verify upstream span has strictly single-stage physical model tokens (40 input / 10 output)
		spans := exporter.GetSpans()
		require.Len(t, spans, 1)
		attrs := spanAttrsMap(spans[0].Attributes)
		require.Equal(t, int64(40), attrs["gen_ai.usage.input_tokens"].AsInt64())
		require.Equal(t, int64(10), attrs["gen_ai.usage.output_tokens"].AsInt64())

		// 2. Verify Replay record has full cascaded tokens (140 prompt / 30 completion)
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.Equal(t, 140, *record.PromptTokens)
		require.Equal(t, 30, *record.CompletionTokens)
		require.Equal(t, 170, *record.TotalTokens)

		// 3. Verify settlement log (llm_usage) has cascaded tokens (140 / 30 / 170)
		var usageFound bool
		for _, entry := range observedLogs.All() {
			cmap := entry.ContextMap()
			if cmap["event"] == "llm_usage" {
				usageFound = true
				require.EqualValues(t, 140, cmap["prompt_tokens"])
				require.EqualValues(t, 30, cmap["completion_tokens"])
				require.EqualValues(t, 170, cmap["total_tokens"])
			}
		}
		require.True(t, usageFound, "expected llm_usage event in logs")
	})
}
