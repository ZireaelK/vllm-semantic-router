package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// semanticResponseStreamState is request-scoped reconstruction state for
// telemetry, replay, memory, and cache policy. It consumes neutral events from
// the same codec contract as buffered responses and never inspects client SSE JSON.
type semanticResponseStreamState struct {
	responseID string
	model      string
	stop       llmprotocol.StopReason
	stopSeq    string
	usage      llmprotocol.Usage
	items      map[int]*semanticStreamItem
	order      []int
	terminal   bool
	failed     *llmprotocol.ProtocolError
}

type semanticStreamItem struct {
	id           string
	role         llmprotocol.Role
	text         string
	refusal      string
	reasoning    string
	reasoningSig string
	toolCall     *llmprotocol.ToolCall
	completed    bool
}

func (r *OpenAIRouter) handleSemanticStreamingResponseBody(
	responseBody []byte,
	endOfStream bool,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	recordStreamingTTFT(ctx)
	r.initializeSemanticResponseStream(ctx)
	buffers := semanticStreamBuffers{}
	buffers.push(responseBody, ctx)
	if endOfStream {
		buffers.finalize(ctx)
		r.finalizeSemanticStreamingResponse(ctx, buffers.streamErr)
	}
	return buffers.processingResponse(ctx)
}

func (r *OpenAIRouter) initializeSemanticResponseStream(ctx *RequestContext) {
	if err := r.ensureSemanticResponseStream(ctx); err != nil {
		ctx.StreamingAborted = true
		logging.ComponentErrorEvent("extproc", "neutral_stream_init_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
	}
}

type semanticStreamBuffers struct {
	translated []byte
	public     []byte
	errorFrame []byte
	streamErr  error
}

func (buffers *semanticStreamBuffers) push(responseBody []byte, ctx *RequestContext) {
	if len(responseBody) == 0 {
		return
	}
	if buffers.streamErr == nil && ctx != nil {
		if ctx.Context != nil && errors.Is(ctx.Context.Err(), context.Canceled) {
			buffers.recordError(ctx, ctx.Context.Err(), true)
		} else if ctx.TraceContext != nil && errors.Is(ctx.TraceContext.Err(), context.Canceled) {
			buffers.recordError(ctx, ctx.TraceContext.Err(), true)
		}
	}
	if ctx.PublicChatUsageFilter != nil {
		filtered, err := ctx.PublicChatUsageFilter.Push(responseBody)
		buffers.public = append(buffers.public, filtered...)
		buffers.recordError(ctx, err, true)
	}
	if ctx.ProtocolResponseStream != nil {
		frames, events, diagnostics, err := ctx.ProtocolResponseStream.Push(responseBody)
		buffers.translated = appendProtocolFrames(buffers.translated, frames)
		if frame := failureFrame(frames, events); frame != nil {
			buffers.errorFrame = frame
		}
		observeProtocolStream(ctx, events, diagnostics)
		buffers.recordError(ctx, err, true)
	}
}

func (buffers *semanticStreamBuffers) finalize(ctx *RequestContext) {
	if buffers.streamErr == nil && ctx != nil {
		if ctx.Context != nil && errors.Is(ctx.Context.Err(), context.Canceled) {
			buffers.recordError(ctx, ctx.Context.Err(), true)
		} else if ctx.TraceContext != nil && errors.Is(ctx.TraceContext.Err(), context.Canceled) {
			buffers.recordError(ctx, ctx.TraceContext.Err(), true)
		}
	}
	if ctx.ProtocolResponseStream != nil {
		// A failure the Router raised, in this read or an earlier one, is not the
		// provider's error event, so its frame stays on the abort path.
		routerFailed := buffers.streamErr != nil || ctx.StreamingAborted
		frames, events, diagnostics, err := ctx.ProtocolResponseStream.Finalize(buffers.streamErr)
		buffers.translated = appendProtocolFrames(buffers.translated, frames)
		if frame := failureFrame(frames, events); frame != nil && !routerFailed {
			buffers.errorFrame = frame
		}
		observeProtocolStream(ctx, events, diagnostics)
		buffers.recordError(ctx, err, true)
	}
	if ctx.PublicChatUsageFilter != nil {
		filtered, err := ctx.PublicChatUsageFilter.Finalize()
		buffers.public = append(buffers.public, filtered...)
		buffers.recordError(ctx, err, false)
	}
}

func (buffers *semanticStreamBuffers) recordError(ctx *RequestContext, err error, overwrite bool) {
	if err == nil || !overwrite && buffers.streamErr != nil {
		return
	}
	buffers.streamErr = err
	if ctx != nil {
		ctx.StreamingAborted = true
		if ctx.SemanticStreamState != nil && (overwrite || ctx.SemanticStreamState.failed == nil) {
			var protoErr *llmprotocol.ProtocolError
			if errors.As(err, &protoErr) {
				ctx.SemanticStreamState.failed = protoErr
			} else if errors.Is(err, context.Canceled) || status.Code(err) == codes.Canceled {
				ctx.SemanticStreamState.failed = llmprotocol.NewError(
					llmprotocol.ErrorUpstreamUnavailable,
					"stream_canceled",
					err.Error(),
					err,
				)
			} else if errors.Is(err, context.DeadlineExceeded) || status.Code(err) == codes.DeadlineExceeded {
				ctx.SemanticStreamState.failed = llmprotocol.NewError(
					llmprotocol.ErrorUpstreamUnavailable,
					"stream_timeout",
					err.Error(),
					err,
				)
			} else {
				ctx.SemanticStreamState.failed = llmprotocol.NewError(
					llmprotocol.ErrorInternal,
					"stream_decode_failed",
					err.Error(),
					nil,
				)
			}
		}
	}
}

func appendProtocolFrames(body []byte, frames [][]byte) []byte {
	for _, frame := range frames {
		body = append(body, frame...)
	}
	return body
}

func observeProtocolStream(
	ctx *RequestContext,
	events []llmprotocol.Event,
	diagnostics []llmprotocol.Diagnostic,
) {
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	// Streaming headers have already been emitted. Keep late compatibility
	// warnings observable through the same counter and structured diagnostics
	// as buffered responses instead of retaining them only in request state.
	for _, diagnostic := range diagnostics {
		recordProtocolDiagnostic(ctx, normalizeProtocol(string(ctx.SourceFormat)), normalizeProtocol(string(ctx.TargetFormat)), diagnostic)
	}
	ctx.SemanticStreamState.observe(events)
}

func extractAuthoritativeStreamUsage(state *semanticResponseStreamState) (int, int, bool) {
	if state == nil || state.usage.State != llmprotocol.UsageAvailable {
		return 0, 0, false
	}
	metrics := responseUsageFromSemanticUsage(state.usage)
	if !metrics.invalid {
		return metrics.promptTokens, metrics.completionTokens, true
	}
	return 0, 0, false
}

func estimateAbortedLooperOutputTokens(state *semanticResponseStreamState) int {
	if state == nil || len(state.items) == 0 {
		return 0
	}
	keys := make([]int, 0, len(state.items))
	for k := range state.items {
		keys = append(keys, k)
	}
	sort.Ints(keys)

	var b strings.Builder
	for _, k := range keys {
		item := state.items[k]
		if item == nil {
			continue
		}
		if item.text != "" {
			b.WriteString(item.text)
		}
		if item.reasoning != "" {
			b.WriteString(item.reasoning)
		}
		if item.refusal != "" {
			b.WriteString(item.refusal)
		}
		if item.toolCall != nil {
			if item.toolCall.Name != "" {
				b.WriteString(item.toolCall.Name)
			}
			if item.toolCall.Arguments != "" {
				b.WriteString(item.toolCall.Arguments)
			}
		}
	}
	combined := b.String()
	if combined == "" {
		return 0
	}
	return contextcompression.EstimateTokens(combined)
}

// failureFrame returns the frame the encoder wrote for an upstream failure in
// this batch. The Chat encoder writes a failure as the batch's last frame
// (TestChatStreamFailureEndsItsBatch).
func failureFrame(frames [][]byte, events []llmprotocol.Event) []byte {
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseFailed && len(frames) > 0 {
			return frames[len(frames)-1]
		}
	}
	return nil
}

func (buffers *semanticStreamBuffers) processingResponse(ctx *RequestContext) *ext_proc.ProcessingResponse {
	// A Chat client of a Chat upstream gets the public filter's output up to an
	// upstream error event, then the Router's error frame for it, however the
	// upstream body is split into reads. A client that needs a rewrite keeps the
	// codec's frames.
	if ctx.PublicChatUsageFilter.UpstreamFailed() && len(buffers.errorFrame) > 0 && !requiresClientResponseRewrite(ctx) {
		return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{Body: append(buffers.public, buffers.errorFrame...)},
		}, nil)
	}
	if ctx.ProtocolResponseStream != nil &&
		(requiresClientResponseRewrite(ctx) || ctx.StreamingAborted) {
		return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{Body: buffers.translated},
		}, nil)
	}
	if ctx.PublicChatUsageFilter != nil {
		return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{Body: buffers.public},
		}, nil)
	}
	return buildResponseBodyContinueResponse(nil, nil)
}

func recordStreamingTTFT(ctx *RequestContext) {
	if ctx == nil || ctx.TTFTRecorded || ctx.ProcessingStartTime.IsZero() || ctx.RequestModel == "" {
		return
	}

	ttft := time.Since(ctx.ProcessingStartTime).Seconds()
	if ttft <= 0 {
		return
	}

	ctx.TTFTSeconds = ttft
	ctx.TTFTRecorded = true
	if !ctx.LooperStreamingActive {
		metrics.RecordModelFirstResponseObservation(ctx.RequestModel, ttft)
		latency.UpdateTTFT(ctx.RequestModel, ttft)
		ctx.CacheWarmthEstimate = latency.EstimateCacheProbability(latency.CacheEstimationInput{
			Model:       ctx.RequestModel,
			TTFTSeconds: ttft,
		})
	}
	maybeEmitTransitionEvent(ctx)
	logging.Debugf("Recorded TTFT on first streamed body chunk: model=%q, TTFT=%.4fs", ctx.RequestModel, ttft)
}

func (r *OpenAIRouter) ensureSemanticResponseStream(ctx *RequestContext) error {
	if ctx == nil {
		return fmt.Errorf("request context is unavailable")
	}
	if ctx.ProtocolResponseStream != nil {
		return nil
	}
	engine, err := r.protocolEngineForBackend(ctx)
	if err != nil {
		return err
	}
	source, target := responseWireFormats(ctx)
	streamContext := llmprotocol.StreamContext{
		Context: ctx.TraceContext, Source: source, Target: target,
		Options:     clientStreamOptions(ctx),
		PublicModel: ctx.RequestModel, PreviousResponseID: responseObjectPreviousID(ctx),
	}
	mutation := clientStreamMutation(ctx, source)
	if responseID := responseObjectPublicID(ctx); responseID != "" {
		streamContext.ResponseID = responseID
	}
	stream, err := engine.NewStreamWithMutation(source, target, streamContext, mutation)
	if err != nil {
		return err
	}
	ctx.ProtocolResponseStream = stream
	if source == llmprotocol.OpenAIChatV1 && target == llmprotocol.OpenAIChatV1 {
		ctx.PublicChatUsageFilter = protocolcodec.NewChatPublicStreamFilter(
			llmprotocol.DefaultPolicy().Limits.SSEFrameBytes, streamUsageRequestedByClient(ctx),
		)
		if ctx.LooperStreamingActive && ctx.RequestModel != "" {
			ctx.PublicChatUsageFilter.SetPublicModel(ctx.RequestModel)
		}
	}
	ctx.SemanticStreamState = &semanticResponseStreamState{
		usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		items: make(map[int]*semanticStreamItem),
	}
	return nil
}

func streamUsageRequestedByClient(ctx *RequestContext) bool {
	options := clientStreamOptions(ctx)
	return options.IncludeUsage != nil && *options.IncludeUsage
}

//nolint:gocognit,cyclop,funlen // Observation is an exhaustive reducer over the closed neutral event contract.
func (state *semanticResponseStreamState) observe(events []llmprotocol.Event) {
	if state == nil {
		return
	}
	for _, event := range events {
		if event.ResponseID != "" {
			state.responseID = event.ResponseID
		}
		if event.Model != "" {
			state.model = event.Model
		}
		if event.Usage != nil {
			state.usage = *event.Usage
		}
		if event.StopReason != "" {
			state.stop = event.StopReason
		}
		switch event.Type {
		case llmprotocol.EventOutputItemStarted:
			item := state.item(event.ItemIndex)
			item.id = event.ItemID
			item.role = event.Role
			if event.ToolCall != nil {
				call := *event.ToolCall
				item.toolCall = &call
			}
			if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
				item.reasoningSig = event.Content.Signature
			}
		case llmprotocol.EventOutputTextDelta:
			item := state.item(event.ItemIndex)
			if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
				item.refusal += event.Delta
			} else {
				item.text += event.Delta
			}
		case llmprotocol.EventReasoningDelta:
			item := state.item(event.ItemIndex)
			item.reasoning += event.Delta
			if event.Content != nil && event.Content.Signature != "" {
				item.reasoningSig = event.Content.Signature
			}
		case llmprotocol.EventToolCallDelta:
			item := state.item(event.ItemIndex)
			if item.toolCall == nil {
				item.toolCall = &llmprotocol.ToolCall{}
			}
			if event.ToolCall != nil {
				if event.ToolCall.Kind != "" {
					item.toolCall.Kind = event.ToolCall.Kind
				}
				if event.ToolCall.ID != "" {
					item.toolCall.ID = event.ToolCall.ID
				}
				if event.ToolCall.Name != "" {
					item.toolCall.Name = event.ToolCall.Name
				}
				item.toolCall.Arguments += event.ToolCall.Arguments
			}
		case llmprotocol.EventOutputItemCompleted:
			item := state.item(event.ItemIndex)
			item.completed = true
			if event.ToolCall != nil {
				call := *event.ToolCall
				item.toolCall = &call
			}
		case llmprotocol.EventResponseCompleted:
			state.terminal = true
			state.stopSeq = event.MatchedStopSequence
		case llmprotocol.EventResponseFailed:
			state.terminal = true
			state.failed = event.Error
		}
	}
}

func (state *semanticResponseStreamState) item(index int) *semanticStreamItem {
	item := state.items[index]
	if item != nil {
		return item
	}
	item = &semanticStreamItem{role: llmprotocol.RoleAssistant}
	state.items[index] = item
	state.order = append(state.order, index)
	return item
}

//nolint:cyclop // Response assembly validates every terminal stream invariant in one place.
func (state *semanticResponseStreamState) response() (*llmprotocol.Response, error) {
	if state == nil {
		return nil, fmt.Errorf("semantic stream incomplete: nil stream state")
	}
	if state.failed != nil {
		if state.failed.Code != "" && state.failed.Message != "" {
			return nil, fmt.Errorf("semantic stream failed: %s: %s", state.failed.Code, state.failed.Message)
		}
		if state.failed.Code != "" {
			return nil, fmt.Errorf("semantic stream failed: %s", state.failed.Code)
		}
		return nil, fmt.Errorf("semantic stream failed: %s", state.failed.Message)
	}
	if !state.terminal {
		return nil, fmt.Errorf("semantic stream incomplete: stream ended without terminal event")
	}
	indices := append([]int(nil), state.order...)
	sort.SliceStable(indices, func(left, right int) bool { return indices[left] < indices[right] })
	output := make([]llmprotocol.OutputItem, 0, len(indices))
	for _, index := range indices {
		item := state.items[index]
		if item == nil || !item.completed {
			return nil, fmt.Errorf("semantic stream output item is incomplete")
		}
		contents := make([]llmprotocol.Content, 0, 4)
		if item.reasoning != "" || item.reasoningSig != "" {
			contents = append(contents, llmprotocol.Content{
				Kind: llmprotocol.ContentReasoning, Text: item.reasoning,
				Signature: item.reasoningSig,
			})
		}
		if item.refusal != "" {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: item.refusal})
		}
		if item.text != "" {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: item.text})
		}
		if item.toolCall != nil {
			call := *item.toolCall
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &call})
		}
		if len(contents) == 0 {
			return nil, fmt.Errorf("semantic stream output item is empty")
		}
		itemID := item.id
		if itemID == "" {
			itemID = fmt.Sprintf("item_%d", index)
		}
		output = append(output, llmprotocol.OutputItem{ID: itemID, Role: item.role, Content: contents})
	}
	if state.stop == "" {
		state.stop = llmprotocol.StopUnknown
	}
	return &llmprotocol.Response{
		Generation: 1, ID: state.responseID, CreatedAt: time.Now().UTC(),
		Model: state.model, Output: output, StopReason: state.stop,
		MatchedStopSequence: state.stopSeq, Usage: state.usage,
	}, nil
}

func (r *OpenAIRouter) finalizeSemanticStreamingResponse(ctx *RequestContext, streamErr error) {
	if ctx == nil || ctx.StreamingComplete {
		return
	}
	ctx.StreamingComplete = true
	if streamErr != nil || ctx.SemanticStreamState == nil || !ctx.SemanticStreamState.terminal {
		ctx.StreamingAborted = true
	}
	semanticResponse, responseErr := ctx.SemanticStreamState.response()
	if responseErr == nil {
		ctx.SemanticResponse = semanticResponse
	} else {
		ctx.StreamingAborted = true
	}
	completionLatency := time.Duration(0)
	if !ctx.StartTime.IsZero() {
		completionLatency = time.Since(ctx.StartTime)
		if ctx.RequestModel != "" && !ctx.LooperStreamingActive {
			metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency.Seconds())
		}
	}
	inflight.End(ctx.InflightModel, ctx.InflightToken)
	ctx.InflightToken = 0

	usage := r.takeNeutralResponseUsage(ctx)
	upstreamUsage := usage
	if ctx != nil && ctx.LooperStreamingActive {
		upstreamUsage = r.takeNeutralResponseUpstreamUsage(ctx)
	}
	// Decoded stream events carry the client-facing model, not the provider's.
	observeUpstreamResponse(ctx, "", upstreamUsage)
	r.reportSemanticStreamingUsage(ctx, completionLatency, usage)
	if ctx.LooperStreamingActive {
		if !usage.invalid && !ctx.StreamingAborted && responseErr == nil {
			ctx.LooperUsageSettled = true
			logging.ComponentEvent("extproc", "looper_streaming_completed", map[string]interface{}{
				"request_id":        ctx.RequestID,
				"decision":          ctx.VSRSelectedDecisionName,
				"algorithm":         ctx.VSRSelectionMethod,
				"models_used":       ctx.LooperModelsUsed,
				"iterations":        ctx.LooperIterations,
				"selected_model":    firstNonEmpty(ctx.LooperTargetModel, ctx.VSRSelectedModel),
				"prompt_tokens":     usage.promptTokens,
				"completion_tokens": usage.completionTokens,
				"total_tokens":      responseUsageTotal(usage),
				"trace_recorded":    ctx.LooperTrace != nil,
			})
		} else {
			abortErr := streamErr
			if abortErr == nil {
				if ctx.SemanticStreamState != nil && ctx.SemanticStreamState.failed != nil {
					abortErr = ctx.SemanticStreamState.failed
				} else {
					abortErr = responseErr
				}
			}
			r.settleAbortedLooperStreamingUsage(ctx, abortErr)
		}
	}

	calibrationPromptTokens := usage.promptTokens
	if ctx.LooperStreamingActive {
		calibrationPromptTokens = 0
		if !upstreamUsage.invalid {
			calibrationPromptTokens = upstreamUsage.promptTokens
		}
	}
	r.calibrateTokenEstimatorForModel(
		ctx,
		calibrationTargetModel(ctx),
		streamingCalibrationByteLen(ctx),
		calibrationPromptTokens,
	)

	if responseErr != nil || usage.invalid {
		ctx.StreamingAborted = true
		abortErr := streamErr
		if abortErr == nil {
			abortErr = responseErr
		}
		replayState := routerreplay.LifecycleFailed
		replayReason := "upstream_stream_incomplete"
		if isLooperCancellation(ctx, abortErr) {
			replayState = routerreplay.LifecycleAborted
			replayReason = "client_canceled"
		}
		r.finalizeRouterReplay(ctx, replayState, replayReason)
	}

	if responseErr != nil {
		logging.ComponentWarnEvent("extproc", "neutral_stream_reconstruction_skipped", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      responseErr.Error(),
		})
		r.recordUnscheduledResponseMemoryStore(ctx, "skipped", "stream_incomplete", true)
		if ctx.PendingLooperQuorumOutcome != nil {
			finalizeLooperQuorumOutcome(ctx, ctx.PendingLooperQuorumOutcome, ctx.VSRSelectedDecision, false)
			ctx.PendingLooperQuorumOutcome = nil
		}
		return
	}
	recordPrimaryOutputDigest(ctx, semanticResponse)
	r.observeResponseStageSignals(ctx, semanticAssistantContent(semanticResponse))
	encoded, err := r.encodeClientResponse(*semanticResponse, ctx)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "neutral_stream_replay_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		r.recordUnscheduledResponseMemoryStore(ctx, "skipped", "stream_encode_failed", true)
		if ctx.PendingLooperQuorumOutcome != nil {
			finalizeLooperQuorumOutcome(ctx, ctx.PendingLooperQuorumOutcome, ctx.VSRSelectedDecision, false)
			ctx.PendingLooperQuorumOutcome = nil
		}
		return
	}
	if !ctx.LooperStreamingActive {
		r.updateResponseCache(ctx, encoded)
	}
	r.updateKVAddressRegistry(ctx)

	if ctx.PendingLooperQuorumOutcome != nil {
		if ctx.StreamingAborted {
			ctx.PendingLooperQuorumOutcome.Disposition = looper.FusionQuorumResponseEncodeFailed
			finalizeLooperQuorumOutcome(ctx, ctx.PendingLooperQuorumOutcome, ctx.VSRSelectedDecision, false)
		} else {
			finalizeLooperQuorumOutcome(ctx, ctx.PendingLooperQuorumOutcome, ctx.VSRSelectedDecision, true)
		}
		ctx.PendingLooperQuorumOutcome = nil
	}

	r.scheduleSemanticResponseMemoryStore(ctx, semanticResponse)
	r.persistResponseObject(ctx)
	r.attachRouterReplayResponse(ctx, attachLooperTraceToReplaySnapshot(ctx, encoded), true)
}

func streamingCalibrationByteLen(ctx *RequestContext) int {
	if ctx == nil {
		return 0
	}
	if ctx.LooperStreamingActive {
		if ctx.LooperSynthesisTextBytes > 0 {
			return ctx.LooperSynthesisTextBytes
		}
		return 0
	}
	return tokenCalibrationByteLen(ctx)
}

func attachLooperTraceToReplaySnapshot(ctx *RequestContext, encoded []byte) []byte {
	if ctx == nil || !ctx.LooperStreamingActive || ctx.LooperTrace == nil || len(encoded) == 0 {
		return encoded
	}
	tracePayload := ctx.LooperTrace
	if fusionTrace, ok := ctx.LooperTrace.(*looper.FusionTrace); ok && fusionTrace != nil {
		tracePayload = looper.ProjectFusionPublicTrace(fusionTrace)
	}
	traceJSON, err := json.Marshal(tracePayload)
	if err != nil || len(traceJSON) == 0 || string(traceJSON) == "null" {
		return encoded
	}
	field := "looper_trace"
	switch ctx.VSRSelectionMethod {
	case config.DecisionAlgorithmFusion:
		field = "fusion"
	case config.DecisionAlgorithmWorkflows:
		field = "flow"
	}
	return restoreLooperFieldJSON(encoded, traceJSON, field)
}

func (r *OpenAIRouter) reportSemanticStreamingUsage(
	ctx *RequestContext,
	completionLatency time.Duration,
	usage responseUsageMetrics,
) {
	if ctx == nil {
		return
	}
	pricing := r.sessionTurnPricing(ctx.RequestModel)
	if ctx.LooperStreamingActive {
		pricing = sessiontelemetry.TurnPricing{}
	}
	recordSessionTurnOutcome(ctx, usage, pricing)
	if usage.invalid {
		return
	}
	totalTokens := responseUsageTotal(usage)
	if r.RateLimiter != nil && ctx.RateLimitCtx != nil && totalTokens > 0 {
		r.RateLimiter.Report(*ctx.RateLimitCtx, ratelimit.TokenUsage{
			InputTokens:  usage.promptTokens,
			OutputTokens: usage.completionTokens,
			TotalTokens:  totalTokens,
		})
	}
	if totalTokens > 0 {
		recordSessionTurn(ctx, usage, pricing)
	}
	if ctx.RequestModel == "" {
		return
	}
	if !ctx.LooperStreamingActive {
		recordModelUsageTokens(ctx.RequestModel, usage)
		metrics.RecordModelWindowedRequest(
			ctx.RequestModel,
			completionLatency.Seconds(),
			int64(usage.promptTokens),
			int64(usage.completionTokens),
		)
	}
	if usage.completionTokens > 0 && completionLatency > 0 {
		timePerToken := completionLatency.Seconds() / float64(usage.completionTokens)
		if !ctx.LooperStreamingActive {
			metrics.RecordModelResponseDurationPerOutputToken(ctx.RequestModel, timePerToken)
			latency.UpdateTPOT(ctx.RequestModel, timePerToken)
		}
	}
	replayUsage := r.recordResponseCost(ctx, completionLatency, usage)
	r.updateRouterReplayUsageCost(ctx, replayUsage)
	if !ctx.LooperStreamingActive {
		r.observeRouterLearningUsageTelemetry(ctx, completionLatency, usage, replayUsage)
	}
}
