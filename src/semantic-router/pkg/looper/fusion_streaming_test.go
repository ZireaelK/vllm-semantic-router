package looper

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestPrepareStreamingSynthesis_OneCallMode(t *testing.T) {
	var (
		seenMu     sync.Mutex
		seenModels []string
	)
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		seenMu.Lock()
		seenModels = append(seenModels, model)
		seenMu.Unlock()
		switch model {
		case "panel-a":
			return "panel A perspective", http.StatusOK
		case "panel-b":
			return "panel B perspective", http.StatusOK
		default:
			t.Errorf("unexpected model called during prepare: %s", model)
			return "unexpected", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
			AnalysisMode:   config.FusionAnalysisModeOneCall,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	// Verify target model
	assert.Equal(t, "judge", plan.TargetModel)
	assert.Equal(t, "judge", string(plan.SynthesisParams.Model))

	// Verify only panel models were called, judge was NOT called yet (to be dispatched by Envoy)
	seenMu.Lock()
	assert.ElementsMatch(t, []string{"panel-a", "panel-b"}, seenModels)
	seenMu.Unlock()

	// Verify prompt contains panel responses
	messages := plan.SynthesisParams.Messages
	require.NotEmpty(t, messages)
	lastMsg := messages[len(messages)-1]
	require.NotEmpty(t, lastMsg)

	// Verify models used & accounting
	assert.Contains(t, plan.ModelsUsed, "panel-a")
	assert.Contains(t, plan.ModelsUsed, "panel-b")
	assert.Contains(t, plan.ModelsUsed, "judge")
	assert.Greater(t, plan.IntermediateUsage.TotalTokens, int64(0))
	assert.NotNil(t, plan.Trace)
	assert.Nil(t, plan.QuorumOutcome)
}

func TestPrepareStreamingSynthesis_SeparateAnalysisMode(t *testing.T) {
	var (
		seenMu     sync.Mutex
		seenModels []string
	)
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		seenMu.Lock()
		seenModels = append(seenModels, model)
		seenMu.Unlock()
		switch model {
		case "panel-a":
			return "panel A perspective", http.StatusOK
		case "panel-b":
			return "panel B perspective", http.StatusOK
		case "judge":
			// Intermediate judge analysis call
			require.Contains(t, prompt, "return only valid JSON")
			return `{"consensus":["common fact"],"contradictions":[],"partial_coverage":[],"unique_insights":["insight 1"],"blind_spots":[]}`, http.StatusOK
		default:
			t.Errorf("unexpected model called during prepare: %s", model)
			return "unexpected", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
			AnalysisMode:   config.FusionAnalysisModeSeparate,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	// Verify judge was called once for structured analysis, but not for final synthesis
	seenMu.Lock()
	require.Len(t, seenModels, 3)
	assert.ElementsMatch(t, []string{"panel-a", "panel-b"}, seenModels[:2])
	assert.Equal(t, "judge", seenModels[2])
	seenMu.Unlock()

	// Verify trace contains structured analysis
	require.NotNil(t, plan.Trace)
	fusionTrace, ok := plan.Trace.(*FusionTrace)
	require.True(t, ok, "trace should be *FusionTrace")
	require.NotNil(t, fusionTrace.Analysis)
	assert.Equal(t, []string{"common fact"}, fusionTrace.Analysis.Consensus)

	// Verify synthesis params
	assert.Equal(t, "judge", plan.TargetModel)
	assert.Equal(t, "judge", string(plan.SynthesisParams.Model))
	assert.Equal(t, 4, plan.Iterations)
}

func TestPrepareStreamingSynthesis_QuorumFallback(t *testing.T) {
	// All panel models fail -> trigger quorum fallback
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel error", http.StatusInternalServerError
	})
	defer server.Close()

	temp := 0.5
	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                "judge",
			AnalysisModels:       []string{"panel-a", "panel-b"},
			QuorumFailurePolicy:  config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget: "backup-model",
			Temperature:          &temp,
			MaxCompletionTokens:  512,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	// Fallback target is selected for direct upstream streaming
	assert.Equal(t, "backup-model", plan.TargetModel)
	assert.Equal(t, "backup-model", string(plan.SynthesisParams.Model))
	assert.Equal(t, temp, plan.SynthesisParams.Temperature.Value)
	assert.Equal(t, int64(512), plan.SynthesisParams.MaxCompletionTokens.Value)

	// Quorum outcome evidence is preserved
	require.NotNil(t, plan.QuorumOutcome)
	assert.Equal(t, FusionQuorumFallbackReady, plan.QuorumOutcome.Disposition)
	assert.Equal(t, "backup-model", plan.QuorumOutcome.FallbackTarget)
	assert.Contains(t, plan.ModelsUsed, "backup-model")
}

func TestPrepareStreamingSynthesis_QuorumFailureAborts(t *testing.T) {
	// Panel fails and policy is fail
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel error", http.StatusInternalServerError
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:               "judge",
			AnalysisModels:      []string{"panel-a", "panel-b"},
			QuorumFailurePolicy: config.FusionQuorumFailurePolicyFail,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.Error(t, err)
	assert.Nil(t, plan)
	assert.True(t, strings.Contains(err.Error(), "quorum") || strings.Contains(err.Error(), "panel"))
}

func TestPrepareStreamingSynthesis_PreservesSamplingParams(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel ok", http.StatusOK
	})
	defer server.Close()

	temp := 0.7
	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:               "judge",
			AnalysisModels:      []string{"panel-a"},
			AnalysisMode:        config.FusionAnalysisModeOneCall,
			Temperature:         &temp,
			MaxCompletionTokens: 1024,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	assert.Equal(t, temp, plan.SynthesisParams.Temperature.Value)
	assert.Equal(t, int64(1024), plan.SynthesisParams.MaxCompletionTokens.Value)
}

func TestPrepareStreamingSynthesis_ToolStageGate_SkipsWhenNoTools(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel ok", http.StatusOK
	})
	defer server.Close()

	gateCalled := false
	req := newFusionTestRequest()
	req.IsStreaming = true
	req.ToolStageGate = func(stage llmprotocol.TrustedStage, r *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
		gateCalled = true
		return r
	}
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a"},
			AnalysisMode:   config.FusionAnalysisModeOneCall,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)
	assert.False(t, gateCalled, "gate should not be called when request has no tools")
}

func TestPrepareStreamingSynthesis_ToolStageGate_InvokedWithTools(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel ok", http.StatusOK
	})
	defer server.Close()

	var recordedStage llmprotocol.TrustedStage
	gateCalled := false
	req := newFusionTestRequest()
	req.OriginalRequest = twoToolRequest(t)
	req.IsStreaming = true
	req.ToolStageGate = func(stage llmprotocol.TrustedStage, r *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
		gateCalled = true
		recordedStage = stage
		return r
	}
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a"},
			AnalysisMode:   config.FusionAnalysisModeOneCall,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)
	assert.True(t, gateCalled, "gate should be called when request has tools")
	assert.Equal(t, llmprotocol.TrustedStageFinal, recordedStage)
}

func TestPrepareStreamingSynthesis_ToolStageGate_DefensiveAgainstNilReturn(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel ok", http.StatusOK
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.OriginalRequest = twoToolRequest(t)
	req.IsStreaming = true
	req.ToolStageGate = func(stage llmprotocol.TrustedStage, r *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
		return nil
	}
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a"},
			AnalysisMode:   config.FusionAnalysisModeOneCall,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)
	assert.NotNil(t, plan.SynthesisParams, "SynthesisParams must not be overwritten with nil")
}

func TestPrepareStreamingSynthesis_CapLooperStageOutput(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel ok", http.StatusOK
	})
	defer server.Close()

	limit := 128
	req := newFusionTestRequest()
	req.IsStreaming = true
	req.MaxTokensLimit = &limit
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:               "judge",
			AnalysisModels:      []string{"panel-a"},
			AnalysisMode:        config.FusionAnalysisModeOneCall,
			MaxCompletionTokens: 1024,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)
	assert.Equal(t, int64(128), plan.SynthesisParams.MaxCompletionTokens.Value)
}

func TestPrepareStreamingSynthesis_QuorumFallback_ToolGateAndCapOutput(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "fail", http.StatusInternalServerError
	})
	defer server.Close()

	limit := 64
	var recordedStage llmprotocol.TrustedStage
	gateCalled := false
	req := newFusionTestRequest()
	req.OriginalRequest = twoToolRequest(t)
	req.IsStreaming = true
	req.MaxTokensLimit = &limit
	req.ToolStageGate = func(stage llmprotocol.TrustedStage, r *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
		gateCalled = true
		recordedStage = stage
		return r
	}
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                "judge",
			AnalysisModels:       []string{"panel-a"},
			AnalysisMode:         config.FusionAnalysisModeOneCall,
			QuorumFailurePolicy:  config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget: "fallback-target",
			MaxCompletionTokens:  512,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)
	assert.Equal(t, "fallback-target", plan.TargetModel)
	assert.True(t, gateCalled, "fallback should call ToolStageGate when tools present")
	assert.Equal(t, llmprotocol.TrustedStageFinal, recordedStage)
	assert.Equal(t, int64(64), plan.SynthesisParams.MaxCompletionTokens.Value)
}

func TestPrepareStreamingSynthesis_QuorumFallback_ToolGateSkipsWhenNoTools(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "fail", http.StatusInternalServerError
	})
	defer server.Close()

	gateCalled := false
	req := newFusionTestRequest()
	req.IsStreaming = true
	req.ToolStageGate = func(stage llmprotocol.TrustedStage, r *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
		gateCalled = true
		return r
	}
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                "judge",
			AnalysisModels:       []string{"panel-a"},
			AnalysisMode:         config.FusionAnalysisModeOneCall,
			QuorumFailurePolicy:  config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget: "fallback-target",
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)
	assert.Equal(t, "fallback-target", plan.TargetModel)
	assert.False(t, gateCalled, "fallback should not call ToolStageGate when request has no tools")
}

func TestNewFusionPanelStageError(t *testing.T) {
	panel := fusionPanelOutcome{
		dispatchedModels: []string{"m1", "m2"},
	}
	totalUsage := TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	cause := assert.AnError

	err := newFusionPanelStageError(cause, panel, totalUsage, 3, "judge")
	require.Error(t, err)

	var stageErr *FusionStageFailureError
	require.ErrorAs(t, err, &stageErr)
	assert.Equal(t, cause, stageErr.Cause)
	assert.Equal(t, []string{"m1", "m2", "judge"}, stageErr.ModelsUsed)
	assert.Equal(t, 3, stageErr.Iterations)
	assert.Equal(t, totalUsage, stageErr.Usage)
	assert.Equal(t, "fusion stage failure: assert.AnError general error for testing", stageErr.Error())
	assert.ErrorIs(t, stageErr, cause)

	var qErr *FusionQuorumError
	assert.False(t, errors.As(err, &qErr), "must not return FusionQuorumError for stage failures")
}

func TestFusionExecute_IntermediateStageFailurePreservesUsage(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		switch model {
		case "panel-a":
			return "panel A answer", http.StatusOK
		case "panel-b":
			return "panel B answer", http.StatusOK
		case "judge":
			return "judge unavailable", http.StatusBadGateway
		default:
			return "err", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
			AnalysisMode:   config.FusionAnalysisModeOneCall,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	resp, err := looper.Execute(context.Background(), req)
	require.Error(t, err)
	assert.Nil(t, resp)

	var qErr *FusionQuorumError
	assert.False(t, errors.As(err, &qErr), "intermediate stage failure must not be classified as FusionQuorumError")

	var stageErr *FusionStageFailureError
	require.ErrorAs(t, err, &stageErr)
	assert.Greater(t, stageErr.Usage.TotalTokens, int64(0), "panel usage must be preserved when judge stage fails")
	assert.Equal(t, []string{"panel-a", "panel-b", "judge"}, stageErr.ModelsUsed)
	assert.Equal(t, 3, stageErr.Iterations)
}

func TestPrepareStreamingSynthesis_IntermediateStageContextExceededReturnsFusionStageFailureError(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		switch model {
		case "panel-a":
			return "panel A answer", http.StatusOK
		case "panel-b":
			return "panel B answer", http.StatusOK
		default:
			return "err", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.BaseContextTokens = 100
	req.ModelRefs = []config.ModelRef{{Model: "judge"}}
	req.ModelParams = map[string]config.ModelParams{
		"judge": {ContextWindowSize: 50}, // exceeds window
	}
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
			AnalysisMode:   config.FusionAnalysisModeOneCall,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.Error(t, err)
	assert.Nil(t, plan)

	var qErr *FusionQuorumError
	assert.False(t, errors.As(err, &qErr), "intermediate stage failure must not be classified as FusionQuorumError")

	var stageErr *FusionStageFailureError
	require.ErrorAs(t, err, &stageErr)
	assert.Greater(t, stageErr.Usage.TotalTokens, int64(0), "panel usage must be preserved")
	assert.Equal(t, []string{"panel-a", "panel-b"}, stageErr.ModelsUsed)
	assert.Equal(t, 2, stageErr.Iterations)
}

func TestPrepareStreamingSynthesis_QuorumFallback_StageContextExceeded(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "fail", http.StatusInternalServerError
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.BaseContextTokens = 100
	req.ModelRefs = []config.ModelRef{{Model: "fallback-target"}}
	req.ModelParams = map[string]config.ModelParams{
		"fallback-target": {ContextWindowSize: 50}, // smaller than BaseContextTokens -> exceeds
	}
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                "judge",
			AnalysisModels:       []string{"panel-a"},
			AnalysisMode:         config.FusionAnalysisModeOneCall,
			QuorumFailurePolicy:  config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget: "fallback-target",
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.Error(t, err)
	assert.Nil(t, plan)

	var qErr *FusionQuorumError
	require.ErrorAs(t, err, &qErr)
	outcome, ok := FusionQuorumOutcomeFromError(err)
	require.True(t, ok)
	require.NotNil(t, outcome)
	assert.Equal(t, "fallback", outcome.SelectedPolicy)
	assert.Equal(t, "fallback-target", outcome.FallbackTarget)
	assert.Equal(t, FusionQuorumBudgetExhausted, outcome.Disposition)
}

func TestPrepareStreamingSynthesis_AnalysisErrorReturnsFusionStageFailureError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		switch model {
		case "panel-a":
			return "panel A answer", http.StatusOK
		case "panel-b":
			return "panel B answer", http.StatusOK
		case "judge":
			cancel()
			return "judge error", http.StatusInternalServerError
		default:
			return "err", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
			AnalysisMode:   config.FusionAnalysisModeSeparate,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(ctx, req)
	require.Error(t, err)
	assert.Nil(t, plan)

	var qErr *FusionQuorumError
	assert.False(t, errors.As(err, &qErr), "analysis error must not be classified as FusionQuorumError")

	var stageErr *FusionStageFailureError
	require.ErrorAs(t, err, &stageErr)
	assert.Greater(t, stageErr.Usage.TotalTokens, int64(0), "panel usage must be preserved")
	assert.Equal(t, []string{"panel-a", "panel-b", "judge"}, stageErr.ModelsUsed)
	assert.Equal(t, 3, stageErr.Iterations)
}


