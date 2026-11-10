package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// TestLooperStreaming_SystemPrompt verifies system_prompt injection behavior for Looper streaming:
// Case 1: Normal synthesis final stage dispatches with configured system_prompt without modifying client request.
// Case 2: Quorum fallback final stage dispatches to fallback target with configured system_prompt.
// Case 3: Comparison with buffered mode (Looper.Streaming.Enabled = false), verifying consistent injection.
func TestLooperStreaming_SystemPrompt(t *testing.T) {
	t.Run("NormalSynthesis", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var request struct {
				Model string `json:"model"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
			writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Panel response from " + request.Model,
				},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
		}))
		defer server.Close()

		router, reqCtx, request := quorumFallbackRouter(server.URL)
		router.Config.ClearRouteCache = true
		reqCtx.ExpectStreamingResponse = true
		request.Stream = true

		const configuredPrompt = "You are an expert AI mathematician."
		decision := &config.Decision{
			Name: "fusion-synthesis-system-prompt",
			Plugins: []config.DecisionPlugin{
				{
					Type: config.DecisionPluginSystemPrompt,
					Configuration: config.MustStructuredPayload(map[string]interface{}{
						"enabled":       true,
						"system_prompt": configuredPrompt,
						"mode":          "insert",
					}),
				},
			},
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a", "panel-b"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}

		origUserContent := request.Messages[0].Content[0].Text
		origModel := request.Model

		response, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, response)

		// Must be a CONTINUE response (Native Streaming final dispatch)
		require.Nil(t, response.GetImmediateResponse())
		reqBodyResp := response.GetRequestBody()
		require.NotNil(t, reqBodyResp)
		common := reqBodyResp.GetResponse()
		require.NotNil(t, common)
		require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())

		// Dispatched body must carry the configured system prompt
		require.NotNil(t, common.GetBodyMutation())
		mutatedBody := common.GetBodyMutation().GetBody()
		require.NotEmpty(t, mutatedBody)

		var parsedReq struct {
			Model    string `json:"model"`
			Stream   bool   `json:"stream"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(mutatedBody, &parsedReq))
		require.Equal(t, "judge", parsedReq.Model)
		require.True(t, parsedReq.Stream)

		// Assert system instruction is injected exactly once
		var systemPromptCount int
		var foundConfiguredPrompt bool
		for _, msg := range parsedReq.Messages {
			if msg.Role == "system" {
				systemPromptCount++
				if strings.Contains(msg.Content, configuredPrompt) {
					foundConfiguredPrompt = true
				}
			}
		}
		require.True(t, foundConfiguredPrompt, "dispatched upstream body must contain the configured system prompt")
		require.Equal(t, 1, systemPromptCount, "system prompt must be injected exactly once")

		// Verify client's original request was NOT modified or corrupted
		require.Equal(t, origModel, request.Model)
		require.Len(t, request.Messages, 1)
		require.Equal(t, llmprotocol.RoleUser, request.Messages[0].Role)
		require.Equal(t, origUserContent, request.Messages[0].Content[0].Text)

		// Verify context state
		require.True(t, reqCtx.VSRInjectedSystemPrompt, "VSRInjectedSystemPrompt flag must be true")
		require.True(t, reqCtx.LooperStreamingActive)
		require.Equal(t, "judge", reqCtx.VSRSelectedModel)
	})

	t.Run("QuorumFallback", func(t *testing.T) {
		// Server returns 500 for panel models
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer server.Close()

		router, reqCtx, request := quorumFallbackRouter(server.URL)
		router.Config.ClearRouteCache = true
		reqCtx.ExpectStreamingResponse = true
		request.Stream = true

		const configuredPrompt = "You are a reliable fallback assistant."
		decision := &config.Decision{
			Name: "fusion-fallback-system-prompt",
			Plugins: []config.DecisionPlugin{
				{
					Type: config.DecisionPluginSystemPrompt,
					Configuration: config.MustStructuredPayload(map[string]interface{}{
						"enabled":       true,
						"system_prompt": configuredPrompt,
						"mode":          "insert",
					}),
				},
			},
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

		origUserContent := request.Messages[0].Content[0].Text
		origModel := request.Model

		response, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, response)

		// Must fall back to streaming dispatch for backup-model
		require.Nil(t, response.GetImmediateResponse())
		reqBodyResp := response.GetRequestBody()
		require.NotNil(t, reqBodyResp)
		common := reqBodyResp.GetResponse()
		require.NotNil(t, common)
		require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())

		// Dispatched body to backup-model must have the configured system prompt
		require.NotNil(t, common.GetBodyMutation())
		mutatedBody := common.GetBodyMutation().GetBody()
		require.NotEmpty(t, mutatedBody)

		var parsedReq struct {
			Model    string `json:"model"`
			Stream   bool   `json:"stream"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(mutatedBody, &parsedReq))
		require.Equal(t, "backup-model", parsedReq.Model)
		require.True(t, parsedReq.Stream)

		var systemPromptCount int
		var foundConfiguredPrompt bool
		for _, msg := range parsedReq.Messages {
			if msg.Role == "system" {
				systemPromptCount++
				if strings.Contains(msg.Content, configuredPrompt) {
					foundConfiguredPrompt = true
				}
			}
		}
		require.True(t, foundConfiguredPrompt, "dispatched fallback body must contain the configured system prompt")
		require.Equal(t, 1, systemPromptCount, "fallback system prompt must be injected exactly once")

		// Verify client's original request was NOT modified
		require.Equal(t, origModel, request.Model)
		require.Len(t, request.Messages, 1)
		require.Equal(t, llmprotocol.RoleUser, request.Messages[0].Role)
		require.Equal(t, origUserContent, request.Messages[0].Content[0].Text)

		// Context checks
		require.True(t, reqCtx.VSRInjectedSystemPrompt)
		require.True(t, reqCtx.LooperStreamingActive)
		require.Equal(t, "backup-model", reqCtx.VSRSelectedModel)
	})

	t.Run("BufferedModeComparison", func(t *testing.T) {
		const prompt = "Act as a helpful security analyst."
		decision := &config.Decision{
			Name: "fusion-buffered-comparison",
			Plugins: []config.DecisionPlugin{
				{
					Type: config.DecisionPluginSystemPrompt,
					Configuration: config.MustStructuredPayload(map[string]interface{}{
						"enabled":       true,
						"system_prompt": prompt,
						"mode":          "insert",
					}),
				},
			},
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a", "panel-b"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var request struct {
				Model string `json:"model"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
			writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Answer from " + request.Model,
				},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
		}))
		defer server.Close()

		// 1. Streaming mode execution
		routerStream, reqCtxStream, reqStream := quorumFallbackRouter(server.URL)
		routerStream.Config.ClearRouteCache = true
		reqCtxStream.ExpectStreamingResponse = true
		reqStream.Stream = true

		respStream, err := routerStream.handleLooperExecution(context.Background(), reqStream, decision, reqCtxStream)
		require.NoError(t, err)
		require.NotNil(t, respStream)
		require.Nil(t, respStream.GetImmediateResponse())
		streamBodyBytes := respStream.GetRequestBody().GetResponse().GetBodyMutation().GetBody()

		var streamReqParsed struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(streamBodyBytes, &streamReqParsed))

		var streamInjectedPrompt string
		for _, msg := range streamReqParsed.Messages {
			if msg.Role == "system" {
				streamInjectedPrompt = msg.Content
				break
			}
		}
		require.Equal(t, prompt, streamInjectedPrompt, "streaming mode must have injected the configured system prompt")
		require.True(t, reqCtxStream.VSRInjectedSystemPrompt)

		// 2. Buffered mode execution (modifyRequestBodyForLooper in req_filter_looper_internal.go:82)
		routerBuf, reqCtxBuf, _ := quorumFallbackRouter(server.URL)
		reqCtxBuf.VSRSelectedDecision = decision
		reqCtxBuf.VSRSelectedDecisionName = decision.Name

		bufferedReq := testNeutralRequest("judge", "hello")
		modifyErr := routerBuf.modifyRequestBodyForLooper(bufferedReq, "judge", decision.Name, false, reqCtxBuf)
		require.NoError(t, modifyErr)

		var bufferedInjectedPrompt string
		for _, inst := range bufferedReq.Instructions {
			if inst.Role == llmprotocol.RoleSystem && len(inst.Content) > 0 {
				bufferedInjectedPrompt = inst.Content[0].Text
				break
			}
		}
		require.Equal(t, prompt, bufferedInjectedPrompt, "buffered mode must have injected the configured system prompt")
		require.True(t, reqCtxBuf.VSRInjectedSystemPrompt)

		// 3. Verify injection behavior consistency between streaming and buffered mode
		assert.Equal(t, streamInjectedPrompt, bufferedInjectedPrompt,
			"injected system_prompt must be strictly identical between streaming and buffered modes")

		// 4. Test handleLooperExecution with Streaming.Enabled = false (buffered fallback)
		routerBufferedExec, reqCtxBufferedExec, reqBufferedExec := quorumFallbackRouter(server.URL)
		disabled := false
		routerBufferedExec.Config.Looper.Streaming.Enabled = &disabled
		assert.False(t, routerBufferedExec.Config.Looper.IsStreamingEnabled())

		respBufferedExec, err := routerBufferedExec.handleLooperExecution(context.Background(), reqBufferedExec, decision, reqCtxBufferedExec)
		require.NoError(t, err)
		require.NotNil(t, respBufferedExec)
		// In buffered mode, handleLooperExecution returns an ImmediateResponse (200 OK)
		require.NotNil(t, respBufferedExec.GetImmediateResponse())
		assert.Equal(t, typev3.StatusCode_OK, respBufferedExec.GetImmediateResponse().GetStatus().GetCode())
	})
}
