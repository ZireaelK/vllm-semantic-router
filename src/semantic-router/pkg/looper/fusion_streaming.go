package looper

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var _ StreamingLooper = (*FusionLooper)(nil)

// PrepareStreamingSynthesis executes intermediate candidate analysis panels
// (buffered) and constructs the final synthesis request payload without
// executing the final upstream HTTP call directly.
func (l *FusionLooper) PrepareStreamingSynthesis(
	ctx context.Context,
	req *Request,
) (*StreamingSynthesisPlan, error) {
	if req == nil || req.OriginalRequest == nil {
		return nil, fmt.Errorf("fusion streaming synthesis requires a valid request and original params")
	}

	cfg := l.resolveFusionExecutionConfig(req)
	if len(cfg.AnalysisModels) == 0 {
		return nil, fmt.Errorf("fusion analysis_models cannot be empty")
	}
	if cfg.Model == "" {
		cfg.Model = cfg.AnalysisModels[0]
	}
	if err := validateFusionExecutionConfig(cfg); err != nil {
		return nil, err
	}
	if err := l.validateFusionModels(cfg); err != nil {
		return nil, err
	}

	logging.ComponentEvent("looper", "fusion_streaming_synthesis_prepare_started", map[string]interface{}{
		"decision":        req.DecisionName,
		"judge_model":     cfg.Model,
		"analysis_models": len(cfg.AnalysisModels),
		"analysis_mode":   cfg.AnalysisMode,
	})

	panel, err := l.executeFusionPanel(ctx, req, cfg)
	if err != nil {
		return l.applyStreamingQuorumFailurePolicy(ctx, req, cfg, panel, err)
	}
	if len(panel.failedModels) > 0 {
		logging.ComponentWarnEvent("looper", "fusion_panel_partial", map[string]interface{}{
			"decision":  req.DecisionName,
			"responses": len(panel.responses),
			"failures":  len(panel.failedModels),
		})
	}

	groundedPanel, groundingScores, groundingMode, err := l.applyGrounding(ctx, req, cfg, panel.responses)
	if err != nil {
		return nil, newFusionPanelStageError(err, panel, panel.usage, len(panel.dispatchedModels))
	}

	panelAttempts := len(panel.dispatchedModels)
	var (
		prompt            string
		intermediateUsage = panel.usage
		analysis          *FusionAnalysis
		iterations        = panelAttempts
	)

	switch cfg.AnalysisMode {
	case config.FusionAnalysisModeSeparate:
		var (
			analysisResp *ModelResponse
			analysisErr  error
		)
		analysis, analysisResp, analysisErr = l.runFusionAnalysis(
			ctx, req, cfg, groundedPanel, groundingScores, panelAttempts+1)
		if analysisResp != nil {
			intermediateUsage = intermediateUsage.Add(analysisResp)
		}
		if analysisErr != nil {
			return nil, newFusionPanelStageError(analysisErr, panel, intermediateUsage, panelAttempts+1, cfg.Model)
		}
		iterations++

		original := extractOriginalContent(req.OriginalRequest)
		outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract)
		prompt = buildFusionFinalPrompt(cfg, original, outputContract, groundedPanel, analysis)
		if notes := groundingSynthesisNotes(groundingScores, cfg.GroundingPolicy); notes != "" {
			prompt = prompt + "\n\n" + notes
		}

	case config.FusionAnalysisModeOneCall, config.FusionAnalysisModeNone:
		original := extractOriginalContent(req.OriginalRequest)
		outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract)
		prompt = buildFusionModeFinalPrompt(cfg, original, outputContract, groundedPanel)
		if notes := groundingSynthesisNotes(groundingScores, cfg.GroundingPolicy); notes != "" {
			prompt = prompt + "\n\n" + notes
		}

	default:
		unsupportedErr := fmt.Errorf("unsupported fusion analysis_mode %q", cfg.AnalysisMode)
		return nil, newFusionPanelStageError(unsupportedErr, panel, intermediateUsage, panelAttempts)
	}

	finalReq := appendFusionStageMessage(req.OriginalRequest, prompt)
	finalReq.Model = cfg.Model
	if cfg.Temperature != nil {
		finalReq.Temperature = openai.Float(*cfg.Temperature)
	}
	if cfg.MaxCompletionTokens > 0 {
		finalReq.MaxTokens = (openai.ChatCompletionNewParams{}).MaxTokens
		finalReq.MaxCompletionTokens = openai.Int(int64(cfg.MaxCompletionTokens))
	}
	var stageAdditionalModels []string
	if cfg.AnalysisMode == config.FusionAnalysisModeSeparate {
		stageAdditionalModels = []string{cfg.Model}
	}
	if err := capLooperStageOutput(req, finalReq); err != nil {
		return nil, newFusionPanelStageError(err, panel, intermediateUsage, iterations, stageAdditionalModels...)
	}
	if req != nil && req.ToolStageGate != nil && (len(finalReq.Tools) > 0 || len(finalReq.Functions) > 0) {
		if gated := req.ToolStageGate(llmprotocol.TrustedStageFinal, finalReq); gated != nil {
			finalReq = gated
		}
	}
	if err := validateLooperStageContext(req, finalReq, cfg.Model); err != nil {
		return nil, newFusionPanelStageError(err, panel, intermediateUsage, iterations, stageAdditionalModels...)
	}

	trace := buildFusionTrace(cfg, groundedPanel, panel.failedModels, analysis, groundingMode, groundingScores)
	modelsUsed := orderedFusionModelsUsed(panel.dispatchedModels, cfg.Model)

	return &StreamingSynthesisPlan{
		SynthesisParams:   finalReq,
		TargetModel:       cfg.Model,
		IntermediateUsage: intermediateUsage,
		Trace:             trace,
		ModelsUsed:        modelsUsed,
		Iterations:        iterations + 1,
		QuorumOutcome:     nil,
	}, nil
}

func (l *FusionLooper) applyStreamingQuorumFailurePolicy(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panel fusionPanelOutcome,
	panelErr error,
) (*StreamingSynthesisPlan, error) {
	var quorumErr *FusionQuorumError
	if !errors.As(panelErr, &quorumErr) {
		return nil, panelErr
	}
	evidence := quorumErr.Evidence()
	policy := string(resolvedQuorumFailurePolicy(cfg))

	if ctxErr := fusionCallerBudgetErr(ctx); ctxErr != nil {
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumCancelled, "")
		return nil, fmt.Errorf("%w: %w", ctxErr, quorumErr.WithPolicyOutcome(policy, "", FusionQuorumCancelled, evidence.Usage))
	}

	if cfg.QuorumFailurePolicy != config.FusionQuorumFailurePolicyFallback {
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumFailed, "")
		return nil, quorumErr.WithPolicyOutcome(policy, "", FusionQuorumFailed, evidence.Usage)
	}

	target := cfg.QuorumFallbackTarget
	if target == "" {
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumFailed, "")
		return nil, fmt.Errorf("fusion quorum fallback requested without a target: %w",
			quorumErr.WithPolicyOutcome(policy, "", FusionQuorumFailed, evidence.Usage))
	}

	fallbackReq := req.OriginalRequest
	if outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract); outputContract != "" {
		fallbackReq = appendFusionStageMessage(req.OriginalRequest, outputContract)
	} else if fallbackReq != nil {
		data, err := json.Marshal(fallbackReq)
		if err != nil {
			return nil, fmt.Errorf("marshal fallback request: %w", err)
		}
		var cloned openai.ChatCompletionNewParams
		if err := json.Unmarshal(data, &cloned); err != nil {
			return nil, fmt.Errorf("unmarshal fallback request: %w", err)
		}
		fallbackReq = &cloned
	}

	if fallbackReq != nil {
		fallbackReq.Model = target
		if cfg.Temperature != nil {
			fallbackReq.Temperature = openai.Float(*cfg.Temperature)
		}
		if cfg.MaxCompletionTokens > 0 {
			fallbackReq.MaxTokens = (openai.ChatCompletionNewParams{}).MaxTokens
			fallbackReq.MaxCompletionTokens = openai.Int(int64(cfg.MaxCompletionTokens))
		}
	}
	if err := capLooperStageOutput(req, fallbackReq); err != nil {
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumFailed, "")
		return nil, fmt.Errorf("cap fallback stage output: %w",
			quorumErr.WithPolicyOutcome(policy, "", FusionQuorumFailed, evidence.Usage))
	}
	if req != nil && req.ToolStageGate != nil && fallbackReq != nil && (len(fallbackReq.Tools) > 0 || len(fallbackReq.Functions) > 0) {
		if gated := req.ToolStageGate(llmprotocol.TrustedStageFinal, fallbackReq); gated != nil {
			fallbackReq = gated
		}
	}
	if err := validateLooperStageContext(req, fallbackReq, target); err != nil {
		disposition := fusionQuorumFallbackDisposition(ctx, err)
		l.logQuorumDecision(req, cfg, evidence, disposition, target)
		return nil, fmt.Errorf("validate fallback stage context: %w: %w",
			err, quorumErr.WithPolicyOutcome(policy, target, disposition, evidence.Usage))
	}

	modelsUsed := append(append([]string{}, panel.dispatchedModels...), target)
	iterations := len(panel.dispatchedModels) + 1
	evidence.Usage = panel.usage

	l.logQuorumDecision(req, cfg, evidence, FusionQuorumFallbackReady, target)
	quorumOutcome := newFusionQuorumOutcome(evidence, policy, target, FusionQuorumFallbackReady)

	return &StreamingSynthesisPlan{
		SynthesisParams:   fallbackReq,
		TargetModel:       target,
		IntermediateUsage: panel.usage,
		Trace:             nil,
		ModelsUsed:        modelsUsed,
		Iterations:        iterations,
		QuorumOutcome:     quorumOutcome,
	}, nil
}

func newFusionPanelStageError(cause error, panel fusionPanelOutcome, usage TokenUsage, iterations int, additionalModels ...string) error {
	modelsUsed := append([]string(nil), panel.dispatchedModels...)
	modelsUsed = append(modelsUsed, additionalModels...)
	return NewFusionStageFailureError(cause, modelsUsed, iterations, usage)
}
