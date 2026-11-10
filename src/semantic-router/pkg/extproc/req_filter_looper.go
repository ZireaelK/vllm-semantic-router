/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// isLooperRequest checks if the incoming request is from looper (internal request)
// If so, extproc should skip plugin processing to avoid recursion
func (r *OpenAIRouter) isLooperRequest(ctx *RequestContext) bool {
	return ctx.LooperRequest
}

// shouldUseLooper checks if the decision requires looper execution
// Returns true if:
// - Decision has an Algorithm configured AND
// - Decision has at least one ModelRef (ReMoM supports single model) AND
// - Looper endpoint is configured in router config
func (r *OpenAIRouter) shouldUseLooper(decision *config.Decision) bool {
	if decision == nil || r.Config == nil {
		return false
	}
	if decision.Algorithm == nil {
		return false
	}
	if !config.IsLooperAlgorithmType(decision.Algorithm.Type) {
		return false
	}

	return hasLooperModelInputs(decision)
}

func hasLooperModelInputs(decision *config.Decision) bool {
	switch decision.Algorithm.Type {
	case config.DecisionAlgorithmReMoM:
		return len(decision.ModelRefs) >= 1
	case config.DecisionAlgorithmFusion:
		return len(decision.ModelRefs) >= 1 || hasFusionAnalysisModels(decision)
	case config.DecisionAlgorithmWorkflows:
		return len(decision.ModelRefs) >= 1
	default:
		return len(decision.ModelRefs) > 1
	}
}

func hasFusionAnalysisModels(decision *config.Decision) bool {
	return decision.Algorithm.Fusion != nil &&
		len(decision.Algorithm.Fusion.AnalysisModels) > 0
}

// looperConstructionFailed answers a decision whose Looper algorithm cannot
// be built.
func (r *OpenAIRouter) looperConstructionFailed(
	decision *config.Decision,
	reqCtx *RequestContext,
	err error,
) *ext_proc.ProcessingResponse {
	r.logLooperConstructionFailure(decision, reqCtx, err)
	return r.createErrorResponse(500, "Looper construction failed.")
}

func (r *OpenAIRouter) logLooperConstructionFailure(decision *config.Decision, reqCtx *RequestContext, err error) {
	logging.ComponentErrorEvent("extproc", "looper_construction_failed", map[string]interface{}{
		"request_id": reqCtx.RequestID,
		"decision":   decision.Name,
		"algorithm":  decision.Algorithm.Type,
		"error":      err.Error(),
	})
}

// looperModelClient is the client for the Router's own model calls outside a
// request graph, such as a prompt selector's or a context recovery's: hops
// served in process. It is nil when the router has nothing to send hops with.
func (r *OpenAIRouter) looperModelClient() *looper.Client {
	hops := r.looperHops()
	if hops == nil {
		return nil
	}
	return looper.NewHopClient(&r.Config.Looper, hops)
}

// handleLooperExecution executes the looper for multi-model decisions
func (r *OpenAIRouter) handleLooperExecution(
	ctx context.Context,
	request *llmprotocol.Request,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if r.WorkflowStateService != nil && decision != nil && decision.Algorithm != nil &&
		decision.Algorithm.Type == config.DecisionAlgorithmWorkflows {
		if !r.WorkflowStateService.Acquire() {
			return r.createErrorResponse(503, "Router is shutting down"), nil
		}
		defer r.WorkflowStateService.Release()
	}

	looperReq, errorResponse := r.buildLooperRequest(request, decision, reqCtx)
	if errorResponse != nil {
		return errorResponse, nil
	}

	ownership := looper.ResolveFinalStageOwnership(looperReq)
	if r.shouldStreamLooperFinalStage(ownership, reqCtx, request, decision) {
		return r.handleLooperStreamingDispatch(ctx, looperReq, decision, request.Model, reqCtx)
	}

	resp, errorResponse := r.executeLooper(ctx, looperReq, decision, request.Model, reqCtx)
	if errorResponse != nil {
		return errorResponse, nil
	}

	return r.finalizeLooperBufferedResponse(resp, decision, request.Model, reqCtx)
}

func (r *OpenAIRouter) shouldStreamLooperFinalStage(
	ownership looper.StageOwnership,
	reqCtx *RequestContext,
	request *llmprotocol.Request,
	decision *config.Decision,
) bool {
	if r == nil || r.Config == nil || !r.Config.Looper.IsStreamingEnabled() {
		return false
	}
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.Type != config.DecisionAlgorithmFusion {
		return false
	}
	if reqCtx == nil || !reqCtx.ExpectStreamingResponse {
		return false
	}
	if request == nil || !request.Stream {
		return false
	}
	if request.CandidateCount != nil && *request.CandidateCount > 1 {
		return false
	}
	if reqCtx.SemanticRequest != nil && reqCtx.SemanticRequest.CandidateCount != nil && *reqCtx.SemanticRequest.CandidateCount > 1 {
		return false
	}
	if reqCtx.SourceFormat != llmprotocol.OpenAIChatV1 {
		return false
	}
	if isResponseAPIRequest(reqCtx) {
		return false
	}

	targetModel := strings.TrimSpace(ownership.TargetModel)
	if targetModel == "" && decision.Algorithm.Fusion != nil {
		targetModel = strings.TrimSpace(decision.Algorithm.Fusion.Model)
	}
	if targetModel == "" && len(decision.ModelRefs) > 0 {
		targetModel = strings.TrimSpace(decision.ModelRefs[0].Model)
	}
	if targetModel == "" {
		return false
	}

	format, err := wireFormatForModel(r.Config.GetModelAPIFormat(targetModel))
	if err != nil || format != llmprotocol.OpenAIChatV1 {
		return false
	}
	if reqCtx != nil {
		backendModel := reqCtx.backendModelForCandidate(targetModel)
		if backendModel != "" && backendModel != targetModel {
			format, err := wireFormatForModel(r.Config.GetModelAPIFormat(backendModel))
			if err != nil || format != llmprotocol.OpenAIChatV1 {
				return false
			}
		}
	}

	var fallbackTarget string
	if decision.Algorithm.Fusion != nil {
		fallbackTarget = strings.TrimSpace(decision.Algorithm.Fusion.QuorumFallbackTarget)
	}
	if fallbackTarget != "" {
		format, err := wireFormatForModel(r.Config.GetModelAPIFormat(fallbackTarget))
		if err != nil || format != llmprotocol.OpenAIChatV1 {
			return false
		}
		if reqCtx != nil {
			backendFallback := reqCtx.backendModelForCandidate(fallbackTarget)
			if backendFallback != "" && backendFallback != fallbackTarget {
				format, err := wireFormatForModel(r.Config.GetModelAPIFormat(backendFallback))
				if err != nil || format != llmprotocol.OpenAIChatV1 {
					return false
				}
			}
		}
	}

	return ownership.Eligibility == looper.StreamingEligible
}

func (r *OpenAIRouter) handleLooperStreamingDispatch(
	ctx context.Context,
	looperReq *looper.Request,
	decision *config.Decision,
	originalModel string,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	limit := config.MaxUpstreamCallsPerRequest
	if decision != nil && decision.Algorithm != nil && decision.Algorithm.Budget != nil && decision.Algorithm.Budget.MaxCalls > 0 {
		limit = decision.Algorithm.Budget.MaxCalls
	}
	var ledger *budget.Ledger
	ctx, ledger = budget.WithLimit(ctx, limit)

	runtimeConfig := r.Config.Looper
	for _, entrypoint := range r.Config.EffectiveEntrypoints(config.ChatAPI) {
		runtimeConfig.EntrypointModels = append(runtimeConfig.EntrypointModels, entrypoint.ModelNames...)
	}
	hops := r.looperHops()
	if hops == nil {
		return r.looperConstructionFailed(decision, reqCtx, errNoLooperHops), nil
	}
	client := looper.NewHopClient(&runtimeConfig, hops)
	looperInstance, err := looper.FactoryWithClientAndWorkflowState(&runtimeConfig, decision.Algorithm.Type, client, r.WorkflowStateService)
	if err != nil {
		return r.looperConstructionFailed(decision, reqCtx, err), nil
	}

	streamingLooper, ok := looperInstance.(looper.StreamingLooper)
	if !ok {
		resp, errorResponse := r.executeLooper(ctx, looperReq, decision, originalModel, reqCtx)
		if errorResponse != nil {
			return errorResponse, nil
		}
		return r.finalizeLooperBufferedResponse(resp, decision, originalModel, reqCtx)
	}

	ledger, _ = budget.Current(ctx)
	plan, err := streamingLooper.PrepareStreamingSynthesis(ctx, looperReq)
	if err == nil && ledger != nil && ledger.Remaining() < 1 {
		err = budget.ErrExhausted
	} else if err != nil && ledger != nil && ledger.Remaining() < 1 && !errors.Is(err, budget.ErrExhausted) && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		err = errors.Join(budget.ErrExhausted, err)
	}
	if err != nil {
		logging.ComponentErrorEvent("extproc", "looper_streaming_prepare_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"decision":   decision.Name,
			"algorithm":  decision.Algorithm.Type,
			"error":      err.Error(),
		})
		evidence, _ := looperFailureEvidenceFromError(err)
		if plan != nil {
			if evidence.algorithm == "" && decision != nil && decision.Algorithm != nil {
				evidence.algorithm = decision.Algorithm.Type
			}
			if len(evidence.modelsUsed) == 0 {
				evidence.modelsUsed = plan.ModelsUsed
			}
			if evidence.iterations == 0 {
				evidence.iterations = plan.Iterations
			}
			if evidence.usage.IsZero() {
				evidence.usage = plan.IntermediateUsage
			}
			if evidence.fusionQuorum == nil {
				evidence.fusionQuorum = plan.QuorumOutcome
			}
		}
		if evidence.algorithm == "" && decision != nil && decision.Algorithm != nil {
			evidence.algorithm = decision.Algorithm.Type
		}
		var quorumOutcome *looper.FusionQuorumOutcome
		if outcome, ok := looper.FusionQuorumOutcomeFromError(err); ok {
			quorumOutcome = outcome
			evidence.fusionQuorum = quorumOutcome
		} else if plan != nil && plan.QuorumOutcome != nil {
			quorumOutcome = plan.QuorumOutcome
		}
		finalizeLooperQuorumOutcome(reqCtx, quorumOutcome, decision, false)

		statusCode := 502
		lifecycleState := routerreplay.LifecycleFailed
		reason := "looper_streaming_prepare_failed"
		msg := "Looper intermediate execution failed"

		var windowErr *looper.StageContextWindowError
		if errors.Is(err, context.DeadlineExceeded) {
			statusCode = 504
			reason = "looper_streaming_deadline_exceeded"
			msg = "Looper streaming deadline exceeded"
			if reqCtx != nil {
				reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamTimeout, "request_timeout", "request timed out", err)
			}
		} else if errors.Is(err, context.Canceled) || (quorumOutcome != nil && quorumOutcome.Disposition == looper.FusionQuorumCancelled) {
			statusCode = 500
			lifecycleState = routerreplay.LifecycleAborted
			reason = "client_cancelled_during_looper_prepare"
			msg = "Client cancelled during looper prepare"
			if reqCtx != nil {
				reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInternal, "request_canceled", "request canceled", err)
			}
		} else if errors.As(err, &windowErr) || (quorumOutcome != nil && quorumOutcome.Disposition == looper.FusionQuorumBudgetExhausted) {
			statusCode = 400
			reason = "looper_stage_context_exceeded"
			msg = "Looper stage context window exceeded"
			if reqCtx != nil {
				reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "context_length_exceeded", "Looper stage context window exceeded", err)
			}
		} else if errors.Is(err, budget.ErrExhausted) {
			statusCode = 400
			reason = "looper_budget_exhausted"
			msg = "Looper budget exhausted"
			if reqCtx != nil {
				reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "budget_exhausted", "Looper budget exhausted", err)
			}
			observed := 0
			if ledger != nil {
				observed = ledger.Used()
			}
			algorithm := ""
			if decision != nil && decision.Algorithm != nil {
				algorithm = decision.Algorithm.Type
			}
			logging.ComponentErrorEvent("extproc", "looper_call_budget_exhausted", map[string]interface{}{
				"request_id": reqCtx.RequestID,
				"decision":   decision.Name,
				"algorithm":  algorithm,
				"limit":      limit,
				"hops":       observed,
				"calls":      observed,
			})
		}

		return r.recordLooperFailureWithLifecycle(
			reqCtx, originalModel, decision, statusCode,
			msg, lifecycleState, reason,
			evidence,
		), nil
	}

	targetModel := strings.TrimSpace(plan.TargetModel)
	if targetModel == "" && plan.SynthesisParams != nil {
		targetModel = strings.TrimSpace(plan.SynthesisParams.Model)
	}

	if reqCtx != nil {
		if ledger != nil {
			reqCtx.LooperRemainingBudget = ledger.Remaining()
		}
		reqCtx.LooperIntermediateUsage = plan.IntermediateUsage
		reqCtx.LooperModelsUsed = plan.ModelsUsed
		reqCtx.LooperIterations = plan.Iterations
		reqCtx.LooperTrace = plan.Trace
		reqCtx.LooperTargetModel = targetModel
		reqCtx.RequestModel = originalModel
		reqCtx.VSRSelectedModel = targetModel
		if decision != nil && decision.Algorithm != nil {
			reqCtx.VSRSelectionMethod = decision.Algorithm.Type
		}
		if decision != nil {
			if reqCtx.VSRSelectedDecision == nil {
				reqCtx.VSRSelectedDecision = decision
			}
			reqCtx.VSRSelectedDecisionName = decision.Name
		}
		reqCtx.LooperStreamingActive = true
	}

	success := false
	defer func() {
		if !success {
			if rec := recover(); rec != nil {
				panic(rec)
			}
			reqCtx.LooperStreamingActive = false
		}
	}()

	rawSynthesisBody, err := json.Marshal(plan.SynthesisParams)
	if err != nil {
		finalizeLooperQuorumOutcome(reqCtx, plan.QuorumOutcome, decision, false)
		return r.recordLooperFailure(
			reqCtx, originalModel, decision, 502,
			"Looper streaming synthesis params marshal failed", "looper_stream_marshal_failed",
			looperFailureEvidence{
				algorithm:    decision.Algorithm.Type,
				modelsUsed:   plan.ModelsUsed,
				iterations:   plan.Iterations,
				usage:        plan.IntermediateUsage,
				fusionQuorum: plan.QuorumOutcome,
			},
		), nil
	}

	dispatchResponse, synthesisTextBytes, err := r.buildLooperStreamingDispatchResponse(
		rawSynthesisBody, targetModel, decision, reqCtx,
	)
	if err != nil {
		finalizeLooperQuorumOutcome(reqCtx, plan.QuorumOutcome, decision, false)
		statusCode := 502
		lifecycleState := routerreplay.LifecycleFailed
		reason := "looper_stream_dispatch_encode_failed"
		msg := "Looper streaming dispatch encoding failed"

		var maskingErr *maskingDispatchError
		var budgetErr *selection.RequestBudgetError
		var windowErr *looper.StageContextWindowError
		if errors.As(err, &maskingErr) {
			statusCode = http.StatusServiceUnavailable // 503
			reason = "looper_stream_dispatch_masking_unavailable"
			msg = maskingUnavailableMessage
		} else if errors.Is(err, context.DeadlineExceeded) {
			statusCode = 504
			reason = "looper_stream_dispatch_deadline_exceeded"
			msg = "Looper streaming dispatch deadline exceeded"
			if reqCtx != nil {
				reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamTimeout, "request_timeout", "request timed out", err)
			}
		} else if errors.Is(err, context.Canceled) {
			statusCode = 500
			lifecycleState = routerreplay.LifecycleAborted
			reason = "client_cancelled_during_looper_dispatch"
			msg = "Client cancelled during looper dispatch"
			if reqCtx != nil {
				reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInternal, "request_canceled", "request canceled", err)
			}
		} else if errors.As(err, &windowErr) {
			statusCode = 400
			reason = "looper_stage_context_exceeded"
			msg = "Looper stage context window exceeded"
			if reqCtx != nil {
				reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "context_length_exceeded", "Looper stage context window exceeded", err)
			}
		} else if errors.Is(err, budget.ErrExhausted) {
			statusCode = 400
			reason = "looper_budget_exhausted"
			msg = "Looper budget exhausted"
			if reqCtx != nil {
				reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "budget_exhausted", "Looper budget exhausted", err)
			}
		} else if errors.As(err, &budgetErr) {
			statusCode = 400
			reason = "looper_stream_dispatch_bad_request"
			msg = "Looper streaming dispatch rejected: " + err.Error()
			if reqCtx != nil {
				reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, budgetErr.Code, budgetErr.Message, err)
			}
		} else if reqCtx != nil && reqCtx.ImmediateProtocolError != nil {
			statusCode = protocolErrorToHTTPStatus(reqCtx.ImmediateProtocolError)
			reason = "looper_stream_dispatch_protocol_error"
			msg = reqCtx.ImmediateProtocolError.Message
		}

		return r.recordLooperFailureWithLifecycle(
			reqCtx, originalModel, decision, statusCode,
			msg, lifecycleState, reason,
			looperFailureEvidence{
				algorithm:    decision.Algorithm.Type,
				modelsUsed:   plan.ModelsUsed,
				iterations:   plan.Iterations,
				usage:        plan.IntermediateUsage,
				fusionQuorum: plan.QuorumOutcome,
			},
		), nil
	}
	if dispatchResponse.GetImmediateResponse() != nil {
		finalizeLooperQuorumOutcome(reqCtx, plan.QuorumOutcome, decision, false)
		r.recordLooperImmediateDispatchFailure(
			reqCtx, originalModel, decision, dispatchResponse,
			looperFailureEvidence{
				algorithm:    decision.Algorithm.Type,
				modelsUsed:   plan.ModelsUsed,
				iterations:   plan.Iterations,
				usage:        plan.IntermediateUsage,
				fusionQuorum: plan.QuorumOutcome,
			},
		)
		return dispatchResponse, nil
	}

	success = true
	// Update RequestContext state for subsequent response stream handling and audit
	reqCtx.LooperTargetModel = targetModel
	reqCtx.LooperSynthesisTextBytes = synthesisTextBytes
	if reqCtx.LooperSynthesisPromptTokens <= 0 && synthesisTextBytes > 0 {
		reqCtx.LooperSynthesisPromptTokens = max(1, (synthesisTextBytes+3)/4)
	}
	reqCtx.LooperIntermediateUsage = plan.IntermediateUsage
	if ledger != nil {
		reqCtx.LooperRemainingBudget = ledger.Remaining()
	}
	reqCtx.LooperTrace = plan.Trace
	reqCtx.LooperModelsUsed = plan.ModelsUsed
	reqCtx.LooperIterations = plan.Iterations
	reqCtx.ResponsePath = headers.ResponsePathLooper
	reqCtx.VSRSelectedModel = targetModel
	reqCtx.RequestModel = originalModel // Preserve client's original requested model for rewrite & accounting
	reqCtx.VSRSelectionMethod = decision.Algorithm.Type
	if reqCtx.VSRSelectedDecision == nil && decision != nil {
		reqCtx.VSRSelectedDecision = decision
	}
	if reqCtx.VSRSelectedDecisionName == "" && decision != nil {
		reqCtx.VSRSelectedDecisionName = decision.Name
	}

	if plan.QuorumOutcome != nil {
		reqCtx.PendingLooperQuorumOutcome = plan.QuorumOutcome
	}

	// Start replay so downstream stream completion or cancellation can record usage and lifecycle
	r.startRouterReplay(reqCtx, originalModel, targetModel, decision.Name)

	mutatedBody := dispatchResponse.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	logging.ComponentDebugEvent("extproc", "looper_streaming_dispatch_prepared", map[string]interface{}{
		"request_id":     reqCtx.RequestID,
		"original_model": originalModel,
		"target_model":   targetModel,
		"decision":       decision.Name,
		"body_bytes":     len(mutatedBody),
	})

	return dispatchResponse, nil
}

func updateLooperSynthesisPromptMetrics(reqCtx *RequestContext, fallback *llmprotocol.Request) int {
	target := fallback
	if reqCtx != nil && reqCtx.SemanticRequest != nil {
		target = reqCtx.SemanticRequest
	}
	if target == nil {
		return 0
	}
	signals := extractSemanticRequestSignals(target)
	if reqCtx != nil {
		reqCtx.LooperSynthesisPromptTokens = signals.ContextTokenFloor
		if reqCtx.LooperSynthesisPromptTokens <= 0 && signals.ContextTextBytes > 0 {
			reqCtx.LooperSynthesisPromptTokens = max(1, (signals.ContextTextBytes+3)/4)
		}
	}
	return signals.ContextTextBytes
}

func looperSynthesisSamplingTextBytes(reqCtx *RequestContext, fallback *llmprotocol.Request) int {
	return updateLooperSynthesisPromptMetrics(reqCtx, fallback)
}

func (r *OpenAIRouter) buildLooperStreamingDispatchResponse(
	rawSynthesisBody []byte,
	targetModel string,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, int, error) {
	if reqCtx != nil && reqCtx.TraceContext != nil && reqCtx.TraceContext.Err() != nil {
		return nil, 0, reqCtx.TraceContext.Err()
	}

	if reqCtx != nil {
		if reqCtx.VSRSelectedDecision == nil && decision != nil {
			reqCtx.VSRSelectedDecision = decision
		}
		if reqCtx.VSRSelectedDecisionName == "" && decision != nil {
			reqCtx.VSRSelectedDecisionName = decision.Name
		}
		clientSemanticReq := reqCtx.SemanticRequest
		clientEnvelope := reqCtx.ProtocolEnvelope
		defer func() {
			reqCtx.SemanticRequest = clientSemanticReq
			reqCtx.ProtocolEnvelope = clientEnvelope
		}()
	}

	engine, err := r.protocolEngine()
	if err != nil {
		return nil, 0, err
	}
	clientOptions := clientStreamOptions(reqCtx)
	synthesisReq, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, rawSynthesisBody)
	if err != nil {
		return nil, 0, err
	}
	synthesisReq.Stream = true
	synthesisReq.StreamOptions = clientOptions
	synthesisReq.Generation++

	// Retire the client's raw byte-replay envelope because the outgoing payload
	// is now the synthesized/fallback stage request.
	reqCtx.ProtocolEnvelope = llmprotocol.Envelope{}

	useReasoning, reasoningEffort := r.getReasoningInfoFromDecision(decision, targetModel)
	applyLooperReasoningContext(reqCtx, targetModel, useReasoning, reasoningEffort)

	decisionName := ""
	if decision != nil {
		decisionName = decision.Name
	}

	backendModel := reqCtx.backendModelForCandidate(targetModel)
	_, _, hasConfiguredBackend, _ := r.Config.ResolvePrimaryBackendForModel(backendModel)
	if hasConfiguredBackend || r.Config.Looper.Endpoint == "" {
		dispatch, prepErr := r.prepareProviderDispatch(&synthesisReq, targetModel, decisionName, useReasoning, reqCtx)
		if prepErr != nil {
			return nil, 0, prepErr
		}
		// Keep decision stage controls for final provider adaptation.
		dispatch.decisionName = decisionName
		reqCtx.SemanticRequest.Stream = true
		response := r.buildProviderDispatchResponse(dispatch, reqCtx)
		if response.GetImmediateResponse() != nil {
			return response, updateLooperSynthesisPromptMetrics(reqCtx, &synthesisReq), nil
		}
		finalized, finErr := r.finalizeProviderDispatchResponse(dispatch, response, reqCtx)
		if finErr != nil {
			return nil, 0, finErr
		}
		return finalized, updateLooperSynthesisPromptMetrics(reqCtx, &synthesisReq), nil
	}

	// Legacy Looper.Endpoint-only test fixtures have no VLLMEndpoints; still
	// route through encodeDispatchRequest and routeHeaderState so include_usage,
	// external model mapping, credentials, and headers remain uniform.
	targetFormat, fmtErr := wireFormatForModel(r.Config.GetModelAPIFormat(backendModel))
	if fmtErr != nil {
		targetFormat = llmprotocol.OpenAIChatV1
	}
	dispatch := &providerDispatch{
		logicalModel:  targetModel,
		backendModel:  backendModel,
		upstreamModel: r.Config.ResolveExternalModelID(targetModel, ""),
		targetFormat:  targetFormat,
		decisionName:  decisionName,
		useReasoning:  useReasoning,
	}
	synthesisReq.Model = dispatch.upstreamModel
	synthesisReq.Stream = true
	synthesisReq.Generation++
	if reqCtx != nil {
		reqCtx.TargetFormat = dispatch.targetFormat
		reqCtx.SemanticRequest = &synthesisReq
		reqCtx.RequestModel = dispatch.logicalModel
		reqCtx.VSRSelectedModel = dispatch.logicalModel
		if reqCtx.SemanticRequest != nil {
			if _, decisionErr := r.applyDispatchDecision(reqCtx.SemanticRequest, dispatch, reqCtx); decisionErr != nil {
				return nil, 0, decisionErr
			}
			if _, paramsErr := r.applyDispatchRequestParams(reqCtx.SemanticRequest, reqCtx); paramsErr != nil {
				return nil, 0, paramsErr
			}
		}
	}

	state := &routeHeaderState{
		setHeaders:    r.startUpstreamSpanAndInjectHeaders(dispatch, reqCtx),
		removeHeaders: []string{"content-length"},
	}
	if r.CredentialResolver != nil {
		if credErr := r.appendProviderCredential(state, dispatch.effectiveBackendModel(), "", reqCtx); credErr != nil {
			return credErr, updateLooperSynthesisPromptMetrics(reqCtx, &synthesisReq), nil
		}
	}
	appendRoutingHeaders(&state.setHeaders, dispatch.effectiveBackendModel())
	setProviderRequestPath(&state.setHeaders, nil, dispatch.targetFormat)
	r.applyDecisionHeaderMutations(state, reqCtx)

	if err := r.applyMaskingBeforeDispatch(reqCtx); err != nil {
		return nil, 0, err
	}
	mutatedBody, err := r.encodeDispatchRequest(reqCtx)
	if err != nil {
		return nil, 0, err
	}
	mutatedBody, err = r.adaptProviderRequest(mutatedBody, dispatch, reqCtx)
	if err != nil {
		return nil, 0, err
	}
	appendContentLengthHeader(&state.setHeaders, len(mutatedBody))
	response := buildRequestBodyContinueResponse(state, nil, r.shouldClearRouteCache())
	bindPreparedDispatchArtifact(response.GetRequestBody().GetResponse(), reqCtx, mutatedBody, dispatch.targetFormat)
	return response, updateLooperSynthesisPromptMetrics(reqCtx, &synthesisReq), nil
}

func (r *OpenAIRouter) finalizeLooperBufferedResponse(
	resp *looper.Response,
	decision *config.Decision,
	originalModel string,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	// The looper's final model must land on the request context before the
	// response headers are built. Runtime model selection has already put a
	// preliminary panel candidate in VSRSelectedModel, and appendLooperRoutingFacts
	// prefers that value, so building headers first would advertise a panel model
	// as the responder even when a fallback target produced the answer.
	reqCtx.RequestModel = resp.Model
	reqCtx.VSRSelectedModel = resp.Model
	reqCtx.VSRSelectionMethod = resp.AlgorithmType

	response, semanticResponse, clientBody, err := r.prepareLooperResponse(resp, reqCtx)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "looper_response_encode_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"format":     reqCtx.SourceFormat,
			"error":      err.Error(),
		})
		// Encoding failed, so an error response is returned instead of the
		// recovered fallback.
		// Promote the terminal disposition first, then route through the shared
		// failure recorder. That helper is what actually creates the Replay
		// record: the usage, status, body, and quorum writers all no-op while
		// RouterReplayID is empty, so recording evidence without starting Replay
		// would persist nothing and return no Replay ID header.
		finalizeLooperQuorumOutcome(reqCtx, resp.QuorumOutcome, decision, false)
		return r.recordLooperFailure(
			reqCtx, originalModel, decision, 502,
			"Looper returned an invalid response", "looper_response_encode_failed",
			looperFailureEvidence{
				algorithm:    resp.AlgorithmType,
				modelsUsed:   resp.ModelsUsed,
				iterations:   resp.Iterations,
				usage:        resp.Usage,
				fusionQuorum: resp.QuorumOutcome,
			},
		), nil
	}
	r.recordSuccessfulLooperExecution(
		resp, originalModel, decision, reqCtx, semanticResponse, clientBody,
	)
	// Replay starts inside the recorder, so its ID only exists now. Attach it to
	// the already-built immediate response, as the failure path does, otherwise a
	// successful looper response omits the header for the record just created.
	addRouterReplayHeaderToImmediateResponse(response, reqCtx.RouterReplayID)
	return response, nil
}

func (r *OpenAIRouter) executeLooper(
	ctx context.Context,
	looperReq *looper.Request,
	decision *config.Decision,
	originalModel string,
	reqCtx *RequestContext,
) (*looper.Response, *ext_proc.ProcessingResponse) {
	runtimeConfig := r.Config.Looper
	for _, entrypoint := range r.Config.EffectiveEntrypoints(config.ChatAPI) {
		runtimeConfig.EntrypointModels = append(runtimeConfig.EntrypointModels, entrypoint.ModelNames...)
	}
	program, err := looper.Template(&runtimeConfig, decision.Algorithm.Type, r.WorkflowStateService)
	if err != nil {
		return nil, r.looperConstructionFailed(decision, reqCtx, err)
	}
	hops := r.looperHops()
	if hops == nil {
		return nil, r.looperConstructionFailed(decision, reqCtx, errNoLooperHops)
	}
	return r.executeLooperGraph(ctx, program, hops, looperReq, originalModel, decision, reqCtx)
}

func (r *OpenAIRouter) buildLooperRequest(
	request *llmprotocol.Request,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*looper.Request, *ext_proc.ProcessingResponse) {
	modelRefs := decision.ModelRefs
	if reqCtx.VSREligibleModelRefs != nil {
		modelRefs = reqCtx.VSREligibleModelRefs
	}
	// Build looper request.
	// Responses uses buffered aggregation so its codec can enforce the existing
	// target capability and projection gates on the complete neutral result.
	// This controls internal execution only: the final transport gate renders
	// the client's requested JSON or SSE representation independently.
	streaming := reqCtx.ExpectStreamingResponse
	if isResponseAPIRequest(reqCtx) {
		streaming = false
	}
	logging.ComponentEvent("extproc", "looper_execution_started", map[string]interface{}{
		"request_id":       reqCtx.RequestID,
		"decision":         decision.Name,
		"algorithm":        decision.Algorithm.Type,
		"candidate_models": len(modelRefs),
		"streaming":        streaming,
		"response_api":     isResponseAPIRequest(reqCtx),
	})
	if _, err := r.applyDispatchRequestParams(request, reqCtx); err != nil {
		return nil, r.createErrorResponse(400, "Invalid request parameter policy")
	}
	engine, err := r.protocolEngine()
	if err == nil {
		var encoded protocolcodec.RequestResult
		encoded, err = engine.EncodeRequest(llmprotocol.OpenAIChatV1, *request, llmprotocol.Envelope{})
		if err == nil {
			var openAIRequest *openai.ChatCompletionNewParams
			openAIRequest, err = parseOpenAIRequest(encoded.Body)
			if err == nil {
				looperReq := &looper.Request{
					OriginalRequest:       openAIRequest,
					CandidateRequirements: r.candidateRequirements(reqCtx).Clone(),
					PermittedModels:       looperPermittedModels(modelRefs, decision.Algorithm),
					Grounding:             r.groundingForRecipe(reqCtx.Routing.RecipeName()),
					BaseContextTokens:     reqCtx.VSRContextTokenCount,
					ModelRefs:             modelRefs,
					ModelParams:           r.getModelParams(),
					Algorithm:             decision.Algorithm,
					IsStreaming:           streaming,
					DecisionName:          decision.Name,
					RecipeName:            reqCtx.Routing.RecipeName(),
					OutputContract:        decision.OutputContract,
					OutputContractSpec:    decision.OutputContractSpec,
					ToolStageGate:         r.looperToolStageGate(request, decision, reqCtx),
				}
				if params := decision.GetRequestParamsConfig(); params != nil && params.MaxTokensLimit != nil {
					limit := *params.MaxTokensLimit
					looperReq.MaxTokensLimit = &limit
				}
				return looperReq, nil
			}
		}
	}
	if err != nil {
		return nil, r.recordLooperFailure(
			reqCtx, request.Model, decision, 400,
			"Looper cannot represent the request semantics", "looper_request_unsupported",
		)
	}
	return nil, r.recordLooperFailure(
		reqCtx, request.Model, decision, 400,
		"Looper cannot represent the request semantics", "looper_request_unsupported",
	)
}

func (r *OpenAIRouter) recordSuccessfulLooperExecution(
	resp *looper.Response,
	originalModel string,
	decision *config.Decision,
	reqCtx *RequestContext,
	semanticResponse *llmprotocol.Response,
	clientBody []byte,
) {
	// Protocol encoding succeeded and the immediate response is about to be
	// returned to Envoy, so promote a ready fallback to its terminal disposition
	// and emit the single quorum metric.
	finalizeLooperQuorumOutcome(reqCtx, resp.QuorumOutcome, decision, true)

	// Capture router replay information if enabled. Detailed attempts remain in
	// Replay; the public response surface keeps only aggregate Looper headers.
	reqCtx.VSRLooperDiagnostics = looperReplayDiagnostics(resp.ExecutionTrace)
	r.startRouterReplay(reqCtx, originalModel, resp.Model, decision.Name)
	r.updateLooperReplayUsage(reqCtx, resp.Usage)
	if resp.Usage.PromptTokens > 0 || resp.Usage.CompletionTokens > 0 {
		// Looper usage is an aggregate across potentially differently priced
		// calls. Preserve continuity and token totals without charging the whole
		// cascade at the final response model's price.
		recordSessionTurn(reqCtx, responseUsageMetrics{
			promptTokens:     int(resp.Usage.PromptTokens),
			completionTokens: int(resp.Usage.CompletionTokens),
		}, sessiontelemetry.TurnPricing{})
	}

	// Update router replay with success status (looper returns immediate response with 200)
	r.updateRouterReplayStatus(reqCtx, 200, false)

	// Attach response body to router replay record
	r.attachRouterReplayResponse(reqCtx, clientBody, true)

	// Memory auto_store: the normal response body pipeline (where
	// scheduleResponseMemoryStore runs) is bypassed by ImmediateResponse.
	// A buffered looper result is decoded once into the same neutral response
	// contract used by direct inference. Streaming memory storage is deferred
	// until a terminal semantic accumulator is available.
	if semanticResponse != nil {
		r.scheduleSemanticResponseMemoryStore(reqCtx, semanticResponse)
	}
}

func (r *OpenAIRouter) updateLooperReplayUsage(ctx *RequestContext, usage looper.TokenUsage) {
	promptTokens := int(usage.PromptTokens)
	completionTokens := int(usage.CompletionTokens)
	totalTokens := int(usage.TotalTokens)
	if promptTokens == 0 && completionTokens == 0 && totalTokens == 0 {
		return
	}

	// A looper result can aggregate differently priced model calls. Persist its
	// exact reported token counts, but leave pricing fields unset rather than
	// attributing the whole cascade to the selected model's price.
	r.updateRouterReplayUsageCost(ctx, routerreplay.UsageCost{
		PromptTokens:     replayIntPtr(promptTokens),
		CompletionTokens: replayIntPtr(completionTokens),
		TotalTokens:      replayIntPtr(totalTokens),
	})
}

func (r *OpenAIRouter) groundingForRecipe(recipe config.RecipeName) *looper.GroundingBackends {
	if r.RecipeClassifiers != nil {
		classifier, ok := r.RecipeClassifiers.ForRecipe(recipe)
		if !ok {
			return nil
		}
		return classifier.GroundingBackends()
	}
	if recipe == "" || recipe == config.DefaultRecipeName {
		return r.Classifier.GroundingBackends()
	}
	return nil
}

func looperPermittedModels(refs []config.ModelRef, algorithm *config.AlgorithmConfig) []string {
	var models []string
	for _, ref := range refs {
		models = append(models, ref.Model)
		if ref.LoRAName != "" {
			models = append(models, ref.LoRAName)
		}
	}
	return append(models, explicitAlgorithmModels(algorithm)...)
}
