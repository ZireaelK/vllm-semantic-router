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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
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
	streamOpts, ok := parsedReq["stream_options"].(map[string]interface{})
	require.True(t, ok, "encodeDispatchRequest must inject stream_options onto synthesis dispatch")
	require.Equal(t, true, streamOpts["include_usage"])

	// Verify context state
	require.True(t, reqCtx.LooperStreamingActive)
	require.Equal(t, "judge", reqCtx.VSRSelectedModel)
	require.Equal(t, "judge", reqCtx.LooperTargetModel)
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
	require.NotNil(t, reqCtx.VSRFusionQuorum)
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

func TestLooperStreamingPipeline_EndToEndIncludeUsageAuthAndModelMapping(t *testing.T) {
	panelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Panel insight from " + request.Model,
			},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18})
	}))
	defer panelServer.Close()

	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: panel-a
      provider_model_id: provider/panel-a
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: ` + panelServer.URL + `
          api_key: sk-panel-a
    - name: panel-b
      provider_model_id: provider/panel-b
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: ` + panelServer.URL + `
          api_key: sk-panel-b
    - name: judge
      provider_model_id: provider/judge-v2
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: ` + panelServer.URL + `
          api_key: sk-judge-secret
routing: {}
`))
	require.NoError(t, err)
	cfg.ClearRouteCache = true
	cfg.Looper.Endpoint = panelServer.URL
	cfg.RouterReplay.Enabled = true

	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
		ReplayRecorder:     recorder,
	}

	// Client does NOT pass stream_options.include_usage; router must inject it upstream.
	request := testNeutralRequest("router-entrypoint", "Compare distributed consensus protocols")
	request.Stream = true
	reqCtx := &RequestContext{
		RequestID:                "req-e2e-looper-stream",
		Headers:                  map[string]string{headers.VSRDebug: "true"},
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		ExpectStreamingResponse:  true,
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
	}

	decision := &config.Decision{
		Name: "fusion-e2e-stream",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a", "panel-b"},
				AnalysisMode:   config.FusionAnalysisModeNone,
			},
		},
	}
	reqCtx.VSRSelectedDecision = decision
	reqCtx.VSRSelectedDecisionName = decision.Name

	// Phase 1: RequestBody executes intermediate Panel and returns CONTINUE for Synthesis
	reqResp, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
	require.NoError(t, err)
	require.Nil(t, reqResp.GetImmediateResponse())
	common := reqResp.GetRequestBody().GetResponse()
	require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())
	require.True(t, common.GetClearRouteCache())

	setHeaders := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	require.Equal(t, "judge", setHeaders[headers.SelectedModel])
	require.Equal(t, "Bearer sk-judge-secret", setHeaders["Authorization"])
	require.Equal(t, "/v1/chat/completions", setHeaders[":path"])

	var dispatchedPayload map[string]interface{}
	require.NoError(t, json.Unmarshal(common.GetBodyMutation().GetBody(), &dispatchedPayload))
	require.Equal(t, "provider/judge-v2", dispatchedPayload["model"], "external provider_model_id mapping must be applied")
	require.Equal(t, true, dispatchedPayload["stream"])
	streamOpts, ok := dispatchedPayload["stream_options"].(map[string]interface{})
	require.True(t, ok, "stream_options must be injected by encodeDispatchRequest")
	require.Equal(t, true, streamOpts["include_usage"])
	// Original client preference must stay false/unset so public SSE filter strips usage frame
	require.False(t, streamUsageRequestedByClient(reqCtx))

	// Phase 2: Upstream ResponseHeaders arrives with 200 text/event-stream
	resHeadersReq := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "text/event-stream"},
					{Key: "content-length", Value: "999"},
				},
			},
		},
	}
	resHeadersResp, err := router.handleResponseHeaders(resHeadersReq, reqCtx)
	require.NoError(t, err)
	require.Equal(t, http_ext.ProcessingMode_STREAMED, resHeadersResp.GetModeOverride().GetResponseBodyMode())
	resHeaderMap := headerValuesByName(resHeadersResp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders())
	require.Equal(t, headers.ResponsePathLooper, resHeaderMap[headers.VSRResponsePath])
	require.Equal(t, "panel-a,panel-b,judge", resHeaderMap[headers.VSRLooperModelsUsed])

	// Phase 3: Upstream streams SSE chunks ending with include_usage terminal chunk + [DONE]
	sseChunks := []string{
		"data: {\"id\":\"chatcmpl-synth-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"provider/judge-v2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Raft and \"},\"finish_reason\":null}]}\n\n",
		"data: {\"id\":\"chatcmpl-synth-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"provider/judge-v2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Paxos converge.\"},\"finish_reason\":\"stop\"}]}\n\n",
		"data: {\"id\":\"chatcmpl-synth-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"provider/judge-v2\",\"choices\":[],\"usage\":{\"prompt_tokens\":18,\"completion_tokens\":7,\"total_tokens\":25}}\n\n" +
			"data: [DONE]\n\n",
	}

	var clientVisibleSSE strings.Builder
	for idx, chunk := range sseChunks {
		bodyResp, bodyErr := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{
				Body:        []byte(chunk),
				EndOfStream: idx == len(sseChunks)-1,
			},
		}, reqCtx)
		require.NoError(t, bodyErr)
		if mut := bodyResp.GetResponseBody().GetResponse().GetBodyMutation(); mut != nil {
			clientVisibleSSE.Write(mut.GetBody())
		}
	}

	// Client stream gets virtualized public model and stripped forced usage frame
	renderedSSE := clientVisibleSSE.String()
	require.Contains(t, renderedSSE, `"model":"router-entrypoint"`)
	require.NotContains(t, renderedSSE, `"provider/judge-v2"`)
	require.NotContains(t, renderedSSE, `"prompt_tokens":18`)

	// Verify takeNeutralResponseUsage combined Panel (24 prompt + 12 completion = 36) + Synthesis (18 prompt + 7 completion = 25)
	settledUsage := router.takeNeutralResponseUsage(reqCtx)
	require.False(t, settledUsage.invalid)
	require.Equal(t, 42, settledUsage.promptTokens)     // 24 + 18
	require.Equal(t, 19, settledUsage.completionTokens) // 12 + 7
	require.Equal(t, 61, settledUsage.totalTokens)      // 36 + 25

	// Verify Replay record persisted authoritative total and Fusion trace
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 42, *record.PromptTokens)
	require.Equal(t, 19, *record.CompletionTokens)
	require.Equal(t, 61, *record.TotalTokens)
	require.Contains(t, record.ResponseBody, `"fusion"`)
}

func TestLooperStreamingDisconnect_PrefillBeforeResponseHeadersPersistsAuditAndReplay(t *testing.T) {
	coreObs, observedLogs := observer.New(zapcore.InfoLevel)
	restoreLogger := zap.ReplaceGlobals(zap.New(coreObs))
	t.Cleanup(restoreLogger)

	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{Enabled: true},
		},
		ReplayRecorder: recorder,
	}

	// Simulate client disconnecting during upstream prefill/queueing:
	// RequestBody returned CONTINUE (LooperStreamingActive == true), but
	// upstream ResponseHeaders has NOT arrived yet (IsStreamingResponse == false).
	reqCtx := &RequestContext{
		RequestID:                "req-prefill-disconnect",
		RequestModel:             "router-entrypoint",
		VSRSelectedModel:         "judge",
		LooperTargetModel:        "judge",
		VSRSelectedDecisionName:  "fusion-stream",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		IsStreamingResponse:      false,
		StreamingComplete:        false,
		LooperStreamingActive:    true,
		LooperModelsUsed:         []string{"panel-a", "panel-b"},
		LooperIterations:         2,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     34,
			CompletionTokens: 16,
			TotalTokens:      50,
		},
	}

	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	retErr := router.handleProcessReceiveError(reqCtx, context.Canceled)
	require.NoError(t, retErr)
	require.True(t, reqCtx.StreamingAborted)
	require.True(t, reqCtx.LooperUsageSettled)

	// Calling handleProcessReceiveError a second time must be idempotent
	require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))

	// 1. Verify Replay record preserved intermediate tokens
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_canceled", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 34, *record.PromptTokens)
	require.Equal(t, 16, *record.CompletionTokens)
	require.Equal(t, 50, *record.TotalTokens)

	// 2. Verify structured llm_usage audit log emitted exactly once with prefill stage
	var usageLogs []observer.LoggedEntry
	for _, entry := range observedLogs.All() {
		if entry.ContextMap()["event"] == "llm_usage" {
			usageLogs = append(usageLogs, entry)
		}
	}
	require.Len(t, usageLogs, 1)
	fields := usageLogs[0].ContextMap()
	require.Equal(t, "req-prefill-disconnect", fields["request_id"])
	require.Equal(t, "judge", fields["target_model"])
	require.Equal(t, "upstream_prefill", fields["abort_stage"])
	require.Equal(t, int64(34), fields["prompt_tokens"])
	require.Equal(t, int64(16), fields["completion_tokens"])
	require.Equal(t, int64(50), fields["total_tokens"])
}

func TestLooperStreamingTokenCalibration_UsesPhysicalTargetModelNeverAlias(t *testing.T) {
	classifier, err := classification.BuildClassifier(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ContextRules: []config.ContextRule{{
					Name:      "window",
					MinTokens: config.TokenCount("100K"),
				}},
			},
		},
	}, nil, nil, nil)
	require.NoError(t, err)

	router := &OpenAIRouter{
		Config:     &config.RouterConfig{},
		Classifier: classifier,
	}

	const synthesisBytes = 8000
	const synthesisPromptTokens = 2000 // exact 4.0 bytes/token for physical model "judge"

	for i := 0; i < 12; i++ {
		streamState := &semanticResponseStreamState{
			responseID: "chatcmpl-cal",
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
					Value:      replayInt64Ptr(synthesisPromptTokens),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				OutputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(50),
					Provenance: llmprotocol.UsageAuthoritative,
				},
			},
		}
		ctx := &RequestContext{
			RequestID:                "req-cal",
			SourceFormat:             llmprotocol.OpenAIChatV1,
			RequestModel:             "auto", // abstract client alias
			VSRSelectedModel:         "judge",
			LooperStreamingActive:    true,
			LooperTargetModel:        "judge",
			LooperSynthesisTextBytes: synthesisBytes,
			// Large intermediate panel tokens must NOT poison judge's ratio
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     20000,
				CompletionTokens: 5000,
				TotalTokens:      25000,
			},
			SemanticStreamState: streamState,
		}
		ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
		router.finalizeSemanticStreamingResponse(ctx, nil)
	}

	judgeMean, _, judgeSamples, judgeCalibrated := classifier.TokenCalibrationRatio("model:judge")
	require.True(t, judgeCalibrated)
	require.Equal(t, int64(12), judgeSamples)
	require.InDelta(t, 4.0, judgeMean, 0.05, "calibration must use synthesis-only prompt tokens and bytes")

	_, _, autoSamples, _ := classifier.TokenCalibrationRatio("model:auto")
	require.Zero(t, autoSamples, "abstract client alias 'auto' must never be calibrated")
}

func TestObserveProtocolStream_CJKAbortAccumulatesEstimatedTokens(t *testing.T) {
	cjkSentence := "这是一段用于测试流式中断计费保护的中文输出内容，包含多个分句。"
	expectedTokens := contextcompression.EstimateTokens(cjkSentence)
	require.Greater(t, expectedTokens, 1, "space-free Chinese sentence must not collapse to 1 token")

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
		RequestID:                "req-cjk-stream-abort",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		IsStreamingResponse:      true,
		LooperStreamingActive:    true,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     20,
			CompletionTokens: 10,
			TotalTokens:      30,
		},
	}
	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

	observeProtocolStream(reqCtx, []llmprotocol.Event{
		{Type: llmprotocol.EventOutputTextDelta, Delta: cjkSentence},
	}, nil)
	require.Equal(t, expectedTokens, reqCtx.LooperAuditedOutputTokens)

	require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 10+expectedTokens, *record.CompletionTokens)
}

func replayInt64Ptr(v int64) *int64 {
	return &v
}

func TestBuildLooperStreamingDispatchResponse_PreservesClientSemanticRequestContext(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
				"judge": {Capabilities: []string{"chat"}},
			}},
		},
	}
	originalSemanticReq := &llmprotocol.Request{
		Model: "original-user-model",
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "client prompt"}}},
		},
	}
	originalEnvelope := llmprotocol.Envelope{Request: []byte(`{"client":"raw"}`)}

	reqCtx := &RequestContext{
		RequestID:        "req-anti-pollution",
		SourceFormat:     llmprotocol.OpenAIChatV1,
		SemanticRequest:  originalSemanticReq,
		ProtocolEnvelope: originalEnvelope,
	}

	decision := &config.Decision{
		Name: "test-decision",
	}

	synthesisBody := []byte(`{"model":"judge","messages":[{"role":"system","content":"synthesis prompt"}]}`)
	_, _, _ = router.buildLooperStreamingDispatchResponse(synthesisBody, "judge", decision, reqCtx)

	require.Same(t, originalSemanticReq, reqCtx.SemanticRequest, "reqCtx.SemanticRequest must be restored to client's original request")
	require.Equal(t, originalEnvelope, reqCtx.ProtocolEnvelope, "reqCtx.ProtocolEnvelope must be restored to client's original envelope")
}

func TestHandleLooperStreamingDispatch_ImmediateResponseRecordsIntermediateUsage(t *testing.T) {
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
		RequestID:                "req-immediate-fail",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		ExpectStreamingResponse:  true,
		SourceFormat:             llmprotocol.OpenAIChatV1,
		RequestModel:             "entrypoint-model",
	}

	decision := &config.Decision{
		Name: "fusion-stream",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
		},
	}

	immediateResp := router.createErrorResponse(503, "immediate dispatch rejected")

	evidence := looperFailureEvidence{
		algorithm:  decision.Algorithm.Type,
		modelsUsed: []string{"panel-a", "panel-b"},
		iterations: 2,
		usage: looper.TokenUsage{
			PromptTokens:     150,
			CompletionTokens: 80,
			TotalTokens:      230,
		},
	}

	router.recordLooperImmediateDispatchFailure(reqCtx, "entrypoint-model", decision, immediateResp, evidence)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "looper_stream_dispatch_rejected", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 150, *record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 80, *record.CompletionTokens)
}

func TestFinalizeSemanticStreamingResponse_AbnormalStreamSetsLifecycleFailed(t *testing.T) {
	t.Run("responseErr sets LifecycleFailed and resists EOF", func(t *testing.T) {
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
			RequestID:                "req-abnormal-stream",
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			IsStreamingResponse:      true,
			LooperStreamingActive:    true,
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     30,
				CompletionTokens: 15,
				TotalTokens:      45,
			},
			SemanticStreamState: &semanticResponseStreamState{
				terminal: false,
			},
		}
		router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

		router.finalizeSemanticStreamingResponse(reqCtx, nil)

		require.True(t, reqCtx.StreamingAborted)
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", record.TerminalReason)

		require.NoError(t, router.handleProcessReceiveError(reqCtx, io.EOF))
		recordAfterEOF, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfterEOF.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", recordAfterEOF.TerminalReason)
	})

	t.Run("usage invalid sets LifecycleFailed and resists client canceled", func(t *testing.T) {
		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()

		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{Enabled: true},
			},
			ReplayRecorder: recorder,
		}

		streamState := &semanticResponseStreamState{
			responseID: "resp-1",
			model:      "judge",
			stop:       llmprotocol.StopEndTurn,
			terminal:   true,
			items: map[int]*semanticStreamItem{
				0: {completed: true, text: "completed text"},
			},
			order: []int{0},
			usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		}

		reqCtx := &RequestContext{
			RequestID:                "req-invalid-usage-stream",
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			IsStreamingResponse:      true,
			LooperStreamingActive:    true,
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
			SemanticStreamState: streamState,
		}
		router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

		router.finalizeSemanticStreamingResponse(reqCtx, nil)

		require.True(t, reqCtx.StreamingAborted)
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", record.TerminalReason)

		require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))
		recordAfterCancel, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfterCancel.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", recordAfterCancel.TerminalReason)
	})
}
