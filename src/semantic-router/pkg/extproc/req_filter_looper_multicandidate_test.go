package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func int64Ptr(v int64) *int64 {
	return &v
}

func TestLooperStreaming_MultiCandidate(t *testing.T) {
	// Test case 1: Verify request.CandidateCount = ptr(2) or reqCtx.SemanticRequest.CandidateCount = ptr(2)
	// causes shouldStreamLooperFinalStage to return false, safely falling back to buffered mode.
	t.Run("TestCase1_CandidateCountGreaterThanOne_FallsBackToBuffered", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				Looper: config.LooperConfig{
					Streaming: config.LooperStreamingConfig{Enabled: nil}, // default enabled
				},
				BackendModels: config.BackendModels{
					ModelConfig: map[string]config.ModelParams{
						"judge":   {APIFormat: config.APIFormatOpenAI},
						"panel-a": {APIFormat: config.APIFormatOpenAI},
					},
				},
			},
		}

		ownership := looper.StageOwnership{
			Eligibility: looper.StreamingEligible,
			TargetModel: "judge",
		}
		decision := &config.Decision{
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model: "judge",
				},
			},
		}

		// When request.CandidateCount > 1, shouldStreamLooperFinalStage returns false
		reqCtx := &RequestContext{
			ExpectStreamingResponse: true,
			SourceFormat:            llmprotocol.OpenAIChatV1,
		}
		request := &llmprotocol.Request{
			Stream:         true,
			CandidateCount: int64Ptr(2),
		}
		require.False(t, router.shouldStreamLooperFinalStage(ownership, reqCtx, request, decision),
			"request with CandidateCount=2 must not enter native streaming")

		// When request.CandidateCount is nil, but reqCtx.SemanticRequest.CandidateCount > 1, returns false
		reqCtxWithSemanticReq := &RequestContext{
			ExpectStreamingResponse: true,
			SourceFormat:            llmprotocol.OpenAIChatV1,
			SemanticRequest: &llmprotocol.Request{
				Stream:         true,
				CandidateCount: int64Ptr(3),
			},
		}
		requestWithoutCount := &llmprotocol.Request{
			Stream: true,
		}
		require.False(t, router.shouldStreamLooperFinalStage(ownership, reqCtxWithSemanticReq, requestWithoutCount, decision),
			"reqCtx.SemanticRequest with CandidateCount=3 must not enter native streaming")
	})

	// Test case 2: Verify request.CandidateCount = nil or request.CandidateCount = ptr(1)
	// normally allows native streaming (returns true).
	t.Run("TestCase2_CandidateCountNilOrOne_AllowsNativeStreaming", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				Looper: config.LooperConfig{
					Streaming: config.LooperStreamingConfig{Enabled: nil},
				},
				BackendModels: config.BackendModels{
					ModelConfig: map[string]config.ModelParams{
						"judge":   {APIFormat: config.APIFormatOpenAI},
						"panel-a": {APIFormat: config.APIFormatOpenAI},
					},
				},
			},
		}

		ownership := looper.StageOwnership{
			Eligibility: looper.StreamingEligible,
			TargetModel: "judge",
		}
		decision := &config.Decision{
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model: "judge",
				},
			},
		}

		// CandidateCount = nil -> allowed (returns true)
		reqCtx := &RequestContext{
			ExpectStreamingResponse: true,
			SourceFormat:            llmprotocol.OpenAIChatV1,
		}
		requestNil := &llmprotocol.Request{
			Stream:         true,
			CandidateCount: nil,
		}
		require.True(t, router.shouldStreamLooperFinalStage(ownership, reqCtx, requestNil, decision),
			"request with CandidateCount=nil must be allowed into native streaming")

		// CandidateCount = 1 -> allowed (returns true)
		requestOne := &llmprotocol.Request{
			Stream:         true,
			CandidateCount: int64Ptr(1),
		}
		reqCtx.SemanticRequest = requestOne
		require.True(t, router.shouldStreamLooperFinalStage(ownership, reqCtx, requestOne, decision),
			"request with CandidateCount=1 must be allowed into native streaming")
	})

	// Test case 3: Verify end-to-end that an n: 2 Fusion request under default config
	// completes normally via buffered mode and does not trigger stream codec exception.
	t.Run("TestCase3_EndToEnd_FusionMultiCandidateBufferedExecution", func(t *testing.T) {
		// First verify why native streaming would fail with multi-choice chunks:
		// Chat stream decoder rejects multiple choices in a frame or index != 0.
		decoder := (protocolcodec.OpenAIChatCodec{}).NewDecoder(llmprotocol.StreamContext{
			Source:      llmprotocol.OpenAIChatV1,
			Target:      llmprotocol.OpenAIChatV1,
			PublicModel: "judge",
		}, llmprotocol.DefaultPolicy())
		multiChoiceChunk := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"judge\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}},{\"index\":1,\"delta\":{\"content\":\"b\"}}]}\n\n")
		_, _, errCodec := decoder.Push(multiChoiceChunk)
		require.Error(t, errCodec, "native chat stream decoder must reject multi-choice chunk")
		require.Contains(t, errCodec.Error(), "stream_multiple_choices")

		// Now test end-to-end that an n: 2 Fusion request falls back to buffered mode,
		// completes cleanly, and returns 200 OK without triggering the stream codec exception.
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Model string `json:"model"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
			// Upstream returns two choices (n: 2)
			writeFusionReplayCompletion(w, req.Model, []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Synthesized choice 1 from " + req.Model,
					},
					"finish_reason": "stop",
				},
				{
					"index": 1,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Synthesized choice 2 from " + req.Model,
					},
					"finish_reason": "stop",
				},
			}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
		}))
		defer server.Close()

		router, reqCtx, request := quorumFallbackRouter(server.URL)
		router.Config.ClearRouteCache = true
		reqCtx.ExpectStreamingResponse = true
		request.Stream = true
		request.CandidateCount = int64Ptr(2)
		reqCtx.SemanticRequest.CandidateCount = int64Ptr(2)

		decision := &config.Decision{
			Name: "fusion-stream-multicandidate",
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a", "panel-b"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}

		response, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
		require.NoError(t, err, "buffered execution must complete without error")
		require.NotNil(t, response)

		// Verify fallback to buffered ImmediateResponse (not native streaming CONTINUE)
		require.Nil(t, response.GetRequestBody(), "expected buffered execution, not native streaming CONTINUE")
		immediate := response.GetImmediateResponse()
		require.NotNil(t, immediate, "expected ImmediateResponse for buffered execution fallback")
		require.EqualValues(t, 200, immediate.GetStatus().GetCode())

		// Verify response body is valid and terminates properly with [DONE]
		bodyStr := string(immediate.GetBody())
		require.NotEmpty(t, bodyStr)
		require.Contains(t, bodyStr, "data: [DONE]\n\n")
		require.Contains(t, bodyStr, "chatcmpl-fusion-")
		require.True(t, strings.HasPrefix(bodyStr, "data: "))
	})
}
