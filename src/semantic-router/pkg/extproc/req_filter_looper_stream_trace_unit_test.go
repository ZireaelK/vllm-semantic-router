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
	"net/http"
	"net/http/httptest"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestLooperStreamingDispatch_NativeContinue(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Panel answer from " + request.Model,
			},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
	}))
	defer server.Close()

	router, reqCtx, request := quorumFallbackRouter(server.URL)
	router.Config.ClearRouteCache = true
	reqCtx.ExpectStreamingResponse = true
	request.Stream = true

	decision := &config.Decision{
		Name: "fusion-stream",
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
	require.NoError(t, err)
	require.NotNil(t, response)

	// In Native Streaming, RequestBody returns a CONTINUE response (not ImmediateResponse)
	require.Nil(t, response.GetImmediateResponse())
	reqBodyResp := response.GetRequestBody()
	require.NotNil(t, reqBodyResp)
	common := reqBodyResp.GetResponse()
	require.NotNil(t, common)
	require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())
	require.True(t, common.GetClearRouteCache(), "ClearRouteCache must be true to re-match cluster in Envoy")

	// Verify header mutations: x-selected-model must be the synthesis target model
	var selectedModelHeader string
	for _, h := range common.GetHeaderMutation().GetSetHeaders() {
		if h.GetHeader().GetKey() == "x-selected-model" {
			selectedModelHeader = string(h.GetHeader().GetRawValue())
		}
	}
	require.Equal(t, "judge", selectedModelHeader)

	// Verify request body mutation carries the synthesis prompt and stream=true
	require.NotNil(t, common.GetBodyMutation())
	mutatedBody := common.GetBodyMutation().GetBody()
	require.NotEmpty(t, mutatedBody)

	var parsedReq map[string]interface{}
	require.NoError(t, json.Unmarshal(mutatedBody, &parsedReq))
	require.Equal(t, true, parsedReq["stream"])
	require.Equal(t, "judge", parsedReq["model"])

	// Verify context state
	require.True(t, reqCtx.LooperStreamingActive)
	require.Equal(t, "judge", reqCtx.VSRSelectedModel)
	require.Equal(t, "router-entrypoint", reqCtx.RequestModel)
	require.Equal(t, int64(20), reqCtx.LooperIntermediateUsage.PromptTokens) // 2 panel models * 10
	require.Equal(t, int64(10), reqCtx.LooperIntermediateUsage.CompletionTokens)
	require.Equal(t, int64(30), reqCtx.LooperIntermediateUsage.TotalTokens)
	require.NotEmpty(t, reqCtx.RouterReplayID)
}

func TestLooperStreamingDispatch_QuorumFallbackStreams(t *testing.T) {
	// Panel server where panel-a and panel-b fail (500)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	router, reqCtx, request := quorumFallbackRouter(server.URL)
	router.Config.ClearRouteCache = true
	reqCtx.ExpectStreamingResponse = true
	request.Stream = true

	decision := &config.Decision{
		Name: "fusion-quorum-fallback",
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

	response, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, response)

	// Falls back to direct streaming to backup-model
	require.Nil(t, response.GetImmediateResponse())
	reqBodyResp := response.GetRequestBody()
	require.NotNil(t, reqBodyResp)
	common := reqBodyResp.GetResponse()
	require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())

	var selectedModelHeader string
	for _, h := range common.GetHeaderMutation().GetSetHeaders() {
		if h.GetHeader().GetKey() == "x-selected-model" {
			selectedModelHeader = string(h.GetHeader().GetRawValue())
		}
	}
	require.Equal(t, "backup-model", selectedModelHeader)
	require.Equal(t, "backup-model", reqCtx.VSRSelectedModel)
	require.True(t, reqCtx.LooperStreamingActive)
	require.NotNil(t, reqCtx.LooperQuorumOutcome)
}

func TestLooperStreamingDispatch_FailFastOnQuorumFailure(t *testing.T) {
	// Server where panel fails with 500 and policy is abort
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	router, reqCtx, request := quorumFallbackRouter(server.URL)
	reqCtx.ExpectStreamingResponse = true
	request.Stream = true

	decision := &config.Decision{
		Name: "fusion-quorum-abort",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:               "judge",
				AnalysisModels:      []string{"panel-a", "panel-b"},
				QuorumFailurePolicy: config.FusionQuorumFailurePolicyFail,
			},
		},
	}

	response, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, response)

	// Must return ImmediateResponse (502) and NOT forward to Envoy
	require.NotNil(t, response.GetImmediateResponse())
	require.EqualValues(t, 502, response.GetImmediateResponse().GetStatus().GetCode())
	require.Nil(t, response.GetRequestBody())
	require.False(t, reqCtx.LooperStreamingActive)

	// Verify replay failure recorded
	record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
}

func TestLooperStreamingResponse_AuthoritativeUsageSettlement(t *testing.T) {
	router := &OpenAIRouter{}
	reqCtx := &RequestContext{
		LooperStreamingActive: true,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     25,
			CompletionTokens: 15,
			TotalTokens:      40,
		},
		SemanticResponse: &llmprotocol.Response{
			Usage: llmprotocol.Usage{
				State: llmprotocol.UsageAvailable,
				InputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(10),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				OutputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(5),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				Total: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(15),
					Provenance: llmprotocol.UsageAuthoritative,
				},
			},
		},
	}

	usage := router.takeNeutralResponseUsage(reqCtx)
	require.False(t, usage.invalid)
	// Authoritative settlement combines intermediate panel tokens + synthesis upstream tokens
	require.Equal(t, 35, usage.promptTokens)     // 25 + 10
	require.Equal(t, 20, usage.completionTokens) // 15 + 5
	require.Equal(t, 55, usage.totalTokens)      // 40 + 15
}

func TestLooperStreamingResponse_ClientCancelledEstimatedSettlement(t *testing.T) {
	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{
				Enabled: true,
			},
		},
		ReplayRecorder: recorder,
	}

	reqCtx := &RequestContext{
		RequestID:                 "req-stream-cancel",
		RouterReplayRecorder:      recorder,
		RouterReplayPluginConfig:  &replayConfig,
		IsStreamingResponse:       true,
		StreamingComplete:         false,
		LooperStreamingActive:     true,
		LooperAuditedOutputTokens: 8,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     20,
			CompletionTokens: 10,
			TotalTokens:      30,
		},
	}

	router.startRouterReplay(reqCtx, "client-model", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	// Trigger client cancel error in handleProcessReceiveError
	receiveErr := context.Canceled
	retErr := router.handleProcessReceiveError(reqCtx, receiveErr)
	require.NoError(t, retErr)
	require.True(t, reqCtx.StreamingAborted)

	// Verify replay record has estimated settlement
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_canceled", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 20, *record.PromptTokens)
	require.Equal(t, 18, *record.CompletionTokens) // 10 (intermediate) + 8 (audited deltas)
	require.Equal(t, 38, *record.TotalTokens)
}

func replayInt64Ptr(v int64) *int64 {
	return &v
}
