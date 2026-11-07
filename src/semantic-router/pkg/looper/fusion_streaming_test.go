package looper

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	assert.Equal(t, 3, plan.Iterations)
}

func TestPrepareStreamingSynthesis_QuorumFallback(t *testing.T) {
	// All panel models fail -> trigger quorum fallback
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel error", http.StatusInternalServerError
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                "judge",
			AnalysisModels:       []string{"panel-a", "panel-b"},
			QuorumFailurePolicy:  config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget: "backup-model",
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	// Fallback target is selected for direct upstream streaming
	assert.Equal(t, "backup-model", plan.TargetModel)
	assert.Equal(t, "backup-model", string(plan.SynthesisParams.Model))

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
