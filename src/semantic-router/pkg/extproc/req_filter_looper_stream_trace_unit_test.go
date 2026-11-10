package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
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
	require.NotNil(t, reqCtx.PendingLooperQuorumOutcome)
	require.Nil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must remain unfinalized during dispatch phase")
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
	require.NotContains(t, record.ResponseBody, `"analysis_mode"`)
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
				Total: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(synthesisPromptTokens + 50),
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
		SemanticStreamState: &semanticResponseStreamState{
			items: make(map[int]*semanticStreamItem),
		},
	}
	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

	observeProtocolStream(reqCtx, []llmprotocol.Event{
		{Type: llmprotocol.EventOutputTextDelta, Delta: cjkSentence},
	}, nil)

	require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 10+expectedTokens, *record.CompletionTokens)
}

func TestSettleAbortedLooperStreamingUsage_IncludesToolCallsAndAvoidsChunkInflation(t *testing.T) {
	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{Enabled: true},
		},
		ReplayRecorder: recorder,
	}

	toolName := "get_weather"
	toolArgs := `{"location": "Hangzhou, Zhejiang", "unit": "celsius"}`
	combinedText := toolName + toolArgs
	expectedTokens := contextcompression.EstimateTokens(combinedText)

	reqCtx := &RequestContext{
		RequestID:                "req-tool-stream-abort",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		IsStreamingResponse:      true,
		LooperStreamingActive:    true,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     20,
			CompletionTokens: 5,
			TotalTokens:      25,
		},
		SemanticStreamState: &semanticResponseStreamState{
			items: make(map[int]*semanticStreamItem),
		},
	}
	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

	// Emit tool call across multiple small delta chunks
	observeProtocolStream(reqCtx, []llmprotocol.Event{
		{
			Type:      llmprotocol.EventToolCallDelta,
			ItemIndex: 0,
			ToolCall:  &llmprotocol.ToolCall{Name: toolName},
		},
		{
			Type:      llmprotocol.EventToolCallDelta,
			ItemIndex: 0,
			ToolCall:  &llmprotocol.ToolCall{Arguments: `{"location":`},
		},
		{
			Type:      llmprotocol.EventToolCallDelta,
			ItemIndex: 0,
			ToolCall:  &llmprotocol.ToolCall{Arguments: ` "Hangzhou, Zhejiang",`},
		},
		{
			Type:      llmprotocol.EventToolCallDelta,
			ItemIndex: 0,
			ToolCall:  &llmprotocol.ToolCall{Arguments: ` "unit": "celsius"}`},
		},
	}, nil)

	require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 5+expectedTokens, *record.CompletionTokens)
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

func TestBuildLooperStreamingDispatchResponse_ContextCancelledReturnsError(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	reqCtx := &RequestContext{
		RequestID:    "req-cancelled",
		TraceContext: ctx,
	}

	decision := &config.Decision{Name: "test-decision"}
	synthesisBody := []byte(`{"model":"judge","messages":[{"role":"system","content":"synthesis prompt"}]}`)

	resp, n, err := router.buildLooperStreamingDispatchResponse(synthesisBody, "judge", decision, reqCtx)
	require.ErrorIs(t, err, context.Canceled)
	require.Nil(t, resp)
	require.Equal(t, 0, n)
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

func TestLooperStreaming_ShouldStreamLooperFinalStageGate(t *testing.T) {
	enabledFalse := false
	routerDisabled := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{
				Streaming: config.LooperStreamingConfig{Enabled: &enabledFalse},
			},
		},
	}
	routerEnabled := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{
				Streaming: config.LooperStreamingConfig{Enabled: nil}, // default true
			},
		},
	}

	reqCtx := &RequestContext{
		ExpectStreamingResponse: true,
		SourceFormat:            llmprotocol.OpenAIChatV1,
	}
	request := &llmprotocol.Request{Stream: true}
	ownership := looper.StageOwnership{
		Eligibility: looper.StreamingEligible,
		TargetModel: "judge",
	}

	fusionDecision := &config.Decision{
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion},
	}
	confidenceDecision := &config.Decision{
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmConfidence},
	}

	// Gate disabled -> false
	require.False(t, routerDisabled.shouldStreamLooperFinalStage(ownership, reqCtx, request, fusionDecision))

	// Non-fusion algorithm -> false
	require.False(t, routerEnabled.shouldStreamLooperFinalStage(ownership, reqCtx, request, confidenceDecision))

	// Target model cannot be determined -> false (safety fallback to buffered path)
	emptyOwnership := looper.StageOwnership{Eligibility: looper.StreamingEligible}
	require.False(t, routerEnabled.shouldStreamLooperFinalStage(emptyOwnership, reqCtx, request, fusionDecision))

	// Fusion algorithm and enabled with valid target model -> true
	require.True(t, routerEnabled.shouldStreamLooperFinalStage(ownership, reqCtx, request, fusionDecision))

	t.Run("AnthropicAnalysisModelFallsBackToBuffering", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				Looper: config.LooperConfig{
					Streaming: config.LooperStreamingConfig{Enabled: nil},
				},
				BackendModels: config.BackendModels{
					ModelConfig: map[string]config.ModelParams{
						"chat-model":      {APIFormat: config.APIFormatOpenAI},
						"anthropic-model": {APIFormat: config.APIFormatAnthropic},
					},
				},
			},
		}

		decision := &config.Decision{
			Name: "fusion-anthropic-analysis",
			ModelRefs: []config.ModelRef{
				{Model: "chat-model"},
			},
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					AnalysisModels: []string{"anthropic-model"},
				},
			},
		}

		// When looper request resolves stage ownership without explicit fusion.model,
		// TargetModel is resolved to AnalysisModels[0].
		looperReq := &looper.Request{
			Algorithm:   decision.Algorithm,
			ModelRefs:   decision.ModelRefs,
			IsStreaming: true,
		}
		stageOwnership := looper.ResolveFinalStageOwnership(looperReq)
		require.Equal(t, "anthropic-model", stageOwnership.TargetModel)
		require.Equal(t, looper.StreamingEligible, stageOwnership.Eligibility)

		// Even though decision.ModelRefs[0] is an OpenAI Chat model, the actual synthesis
		// target is anthropic-model (non-OpenAI format), so the gate must return false.
		require.False(t, router.shouldStreamLooperFinalStage(stageOwnership, reqCtx, request, decision))
	})

	t.Run("HandleLooperExecutionFallsBackToBufferedWhenAnalysisModelIsAnthropic", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Model string `json:"model"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
			writeFusionReplayCompletion(w, req.Model, []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Synthesized response from " + req.Model,
				},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
		}))
		defer server.Close()

		router, execReqCtx, execRequest := quorumFallbackRouter(server.URL)
		router.Config.BackendModels.ModelConfig["chat-model"] = config.ModelParams{
			APIFormat:    config.APIFormatOpenAI,
			Capabilities: []string{"chat"},
		}
		router.Config.BackendModels.ModelConfig["anthropic-model"] = config.ModelParams{
			APIFormat:    config.APIFormatAnthropic,
			Capabilities: []string{"chat"},
		}
		execReqCtx.ExpectStreamingResponse = true
		execRequest.Stream = true

		decision := &config.Decision{
			Name: "fusion-fallback-to-buffered",
			ModelRefs: []config.ModelRef{
				{Model: "chat-model"},
			},
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					AnalysisModels: []string{"anthropic-model"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}

		response, err := router.handleLooperExecution(context.Background(), execRequest, decision, execReqCtx)
		require.NoError(t, err)
		require.NotNil(t, response)

		// In buffered mode, response returns an ImmediateResponse instead of native streaming CONTINUE
		require.NotNil(t, response.GetImmediateResponse(), "expected immediate buffered response when streaming gate falls back")
		require.Nil(t, response.GetRequestBody(), "did not expect native streaming continue response")
	})
}

func TestLooperStreaming_BlocksOuterFallbackAndSettlesUpstreamTransportError(t *testing.T) {
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
		RequestID:                "req-transport-err",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		LooperStreamingActive:    true,
		UpstreamStatusCode:       502,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}

	// 1. Blocks outer fallback
	require.False(t, router.shouldAttemptFallback(reqCtx))

	// 2. Upstream transport error settles intermediate usage before attaching replay response
	router.startRouterReplay(reqCtx, "entrypoint", "judge", "fusion-stream")
	_ = router.handleUpstreamTransportError([]byte(`{"error":"bad gateway"}`), reqCtx)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "upstream_error_response", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 100, *record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 50, *record.CompletionTokens)
}

func TestLooperStreaming_PrepareCancelledRecordsLifecycleAborted(t *testing.T) {
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
		RequestID:                "req-prepare-cancel",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
	}

	decision := &config.Decision{
		Name:      "fusion-stream",
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion},
	}

	evidence := looperFailureEvidence{
		algorithm:  decision.Algorithm.Type,
		modelsUsed: []string{"panel-a"},
		iterations: 1,
		usage: looper.TokenUsage{
			PromptTokens:     40,
			CompletionTokens: 20,
			TotalTokens:      60,
		},
	}

	resp := router.recordLooperFailureWithLifecycle(
		reqCtx, "entrypoint", decision, 500,
		"Client cancelled during looper prepare",
		routerreplay.LifecycleAborted,
		"client_cancelled_during_looper_prepare",
		evidence,
	)
	require.NotNil(t, resp)
	require.NotNil(t, resp.GetImmediateResponse())
	require.Equal(t, int32(500), int32(resp.GetImmediateResponse().GetStatus().GetCode()))

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 500, record.ResponseStatus)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_cancelled_during_looper_prepare", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 40, *record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 20, *record.CompletionTokens)
}

func TestLooperStreaming_ProtocolErrorStatusMappingAndCancellation(t *testing.T) {
	// 1. Verify protocolErrorToHTTPStatus
	require.Equal(t, 404, protocolErrorToHTTPStatus(llmprotocol.NewError(llmprotocol.ErrorNotFound, "not_found", "not found", nil)))
	require.Equal(t, 401, protocolErrorToHTTPStatus(llmprotocol.NewError(llmprotocol.ErrorAuthentication, "auth", "unauthorized", nil)))
	require.Equal(t, 403, protocolErrorToHTTPStatus(llmprotocol.NewError(llmprotocol.ErrorPermission, "perm", "forbidden", nil)))
	require.Equal(t, 429, protocolErrorToHTTPStatus(llmprotocol.NewError(llmprotocol.ErrorRateLimited, "rate", "rate limited", nil)))
	require.Equal(t, 504, protocolErrorToHTTPStatus(llmprotocol.NewError(llmprotocol.ErrorUpstreamTimeout, "timeout", "timeout", nil)))
	require.Equal(t, 502, protocolErrorToHTTPStatus(llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "unavailable", "unavailable", nil)))
	require.Equal(t, 500, protocolErrorToHTTPStatus(llmprotocol.NewError(llmprotocol.ErrorInternal, "internal", "internal", nil)))
	require.Equal(t, 400, protocolErrorToHTTPStatus(llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "bad_request", "invalid", nil)))
	require.Equal(t, 400, protocolErrorToHTTPStatus(llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "feat", "unsupported", nil)))

	// 2. Cancellation sets ImmediateProtocolError and status 500
	reqCtx := &RequestContext{}
	cancelErr := context.Canceled
	if errors.Is(cancelErr, context.Canceled) {
		reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInternal, "request_canceled", "request canceled", cancelErr)
	}
	require.NotNil(t, reqCtx.ImmediateProtocolError)
	require.Equal(t, llmprotocol.ErrorInternal, reqCtx.ImmediateProtocolError.Category)
	require.Equal(t, 500, protocolErrorToHTTPStatus(reqCtx.ImmediateProtocolError))

	// 3. Budget error sets ImmediateProtocolError with ErrorInvalidRequest
	budgetErr := &selection.RequestBudgetError{Code: "context_length_exceeded", Message: "exceeds context window"}
	var asBudget *selection.RequestBudgetError
	if errors.As(budgetErr, &asBudget) {
		reqCtx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, asBudget.Code, asBudget.Message, budgetErr)
	}
	require.NotNil(t, reqCtx.ImmediateProtocolError)
	require.Equal(t, llmprotocol.ErrorInvalidRequest, reqCtx.ImmediateProtocolError.Category)
	require.Equal(t, "context_length_exceeded", reqCtx.ImmediateProtocolError.Code)
	require.Equal(t, "exceeds context window", reqCtx.ImmediateProtocolError.Message)
	require.Equal(t, 400, protocolErrorToHTTPStatus(reqCtx.ImmediateProtocolError))

	// 4. encodeImmediateResponseForClient syncs top-level Envoy immediate response status
	router := &OpenAIRouter{}
	ctx404 := &RequestContext{
		SourceFormat:           llmprotocol.OpenAIChatV1,
		ImmediateProtocolError: llmprotocol.NewError(llmprotocol.ErrorNotFound, "model_not_found", "model not found", nil),
	}
	resp := router.createErrorResponse(http.StatusBadRequest, "initial bad request")
	require.Equal(t, uint32(http.StatusBadRequest), uint32(resp.GetImmediateResponse().GetStatus().GetCode()))
	resp = router.encodeImmediateResponseForClient(resp, ctx404)
	require.Equal(t, uint32(http.StatusNotFound), uint32(resp.GetImmediateResponse().GetStatus().GetCode()))
}

func TestLooperStreaming_ReplayRecordsEncodedErrorAndStatus(t *testing.T) {
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
		RequestID:                "req-replay-sync-failure",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		SourceFormat:             llmprotocol.AnthropicMessagesV1,
		ImmediateProtocolError:   llmprotocol.NewError(llmprotocol.ErrorNotFound, "not_found", "resource not found", nil),
	}

	decision := &config.Decision{
		Name: "fusion-stream",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
		},
	}

	resp := router.recordLooperFailureWithLifecycle(
		reqCtx, "entrypoint-model", decision, http.StatusBadRequest,
		"resource not found", routerreplay.LifecycleFailed, "looper_stream_dispatch_protocol_error",
	)

	// The returned response should have status 404 from ImmediateProtocolError
	require.Equal(t, uint32(http.StatusNotFound), uint32(resp.GetImmediateResponse().GetStatus().GetCode()))

	// Replay record should have synced ResponseStatus = 404 and Anthropic wire format body
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, http.StatusNotFound, record.ResponseStatus)
	require.Contains(t, record.ResponseBody, `"type":"error"`)
	require.Contains(t, record.ResponseBody, `"not_found_error"`)

	// Also verify recordLooperImmediateDispatchFailure synchronizes encoded body and status
	reqCtx2 := &RequestContext{
		RequestID:                "req-replay-sync-immediate-dispatch",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		SourceFormat:             llmprotocol.AnthropicMessagesV1,
		ImmediateProtocolError:   llmprotocol.NewError(llmprotocol.ErrorRateLimited, "rate_limit_exceeded", "too many requests", nil),
	}
	dispatchResp := router.createErrorResponse(http.StatusBadRequest, "rate limited")
	router.recordLooperImmediateDispatchFailure(reqCtx2, "entrypoint-model", decision, dispatchResp, looperFailureEvidence{
		algorithm:  decision.Algorithm.Type,
		modelsUsed: []string{"model-a"},
		iterations: 1,
	})

	require.Equal(t, uint32(http.StatusTooManyRequests), uint32(dispatchResp.GetImmediateResponse().GetStatus().GetCode()))
	record2, ok2 := recorder.GetRecord(reqCtx2.RouterReplayID)
	require.True(t, ok2)
	require.Equal(t, http.StatusTooManyRequests, record2.ResponseStatus)
	require.Contains(t, record2.ResponseBody, `"type":"error"`)
	require.Contains(t, record2.ResponseBody, `"rate_limit_error"`)
}

func TestLooperStreaming_CandidateRequirementsPermitsFusionAndFallback(t *testing.T) {
	router := strictCandidateRouter()

	decision := &config.Decision{
		Name:      "fusion-stream-test",
		ModelRefs: []config.ModelRef{{Model: "text"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:                "vision",
				AnalysisModels:       []string{"text"},
				QuorumFallbackTarget: "small",
			},
		},
	}

	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "describe"}}}},
		Sampling: llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(512)},
	}

	// Case 1: When LooperStreamingActive is false and not looper hop -> denied (outside decision permission scope)
	ctxInactive := &RequestContext{
		VSRSelectedDecision:   decision,
		LooperStreamingActive: false,
		LooperRequest:         false,
	}
	errInactive := router.validateDispatchRequirements(request, &providerDispatch{
		logicalModel: "vision",
	}, ctxInactive)
	require.Error(t, errInactive)
	require.ErrorIs(t, errInactive, selection.ErrNoEligibleCandidates)

	// Case 2: When LooperStreamingActive is true -> synthesis model allowed
	ctxActive := &RequestContext{
		VSRSelectedDecision:   decision,
		LooperStreamingActive: true,
	}
	errActiveSynthesis := router.validateDispatchRequirements(request, &providerDispatch{
		logicalModel: "vision",
	}, ctxActive)
	require.NoError(t, errActiveSynthesis)

	// Case 3: When LooperStreamingActive is true -> fallback model allowed
	errActiveFallback := router.validateDispatchRequirements(request, &providerDispatch{
		logicalModel: "small",
	}, ctxActive)
	require.NoError(t, errActiveFallback)
}

func TestLooperStreaming_EmptyBody5xxMaintainsLifecycleFailed(t *testing.T) {
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
		RequestID:                "req-empty-5xx",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		LooperStreamingActive:    true,
		UpstreamStatusCode:       502,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
	router.startRouterReplay(reqCtx, "entrypoint", "judge", "fusion-stream")

	// Simulate EOF (empty body) on 502 response
	err := router.handleProcessReceiveError(reqCtx, io.EOF)
	require.NoError(t, err)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "upstream_error_response", record.TerminalReason)
}

func TestLooperStreaming_EmptyBody3xxMaintainsLifecycleFailed(t *testing.T) {
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
		RequestID:                "req-empty-3xx",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		LooperStreamingActive:    true,
		UpstreamStatusCode:       304,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
	router.startRouterReplay(reqCtx, "entrypoint", "judge", "fusion-stream")

	// Simulate EOF (empty body) on 304 response
	err := router.handleProcessReceiveError(reqCtx, io.EOF)
	require.NoError(t, err)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "upstream_error_response", record.TerminalReason)
}

func TestLooperStreaming_TelemetryAndEstimatorIsolation(t *testing.T) {
	reqCtx := &RequestContext{
		RequestID:             "req-telemetry-isolation",
		RequestModel:          "virtual-entrypoint",
		LooperStreamingActive: true,
		LooperTargetModel:     "physical-judge",
	}

	// recordStreamingTTFT skips metrics and cache probability on RequestModel
	recordStreamingTTFT(reqCtx)
	require.Zero(t, reqCtx.CacheWarmthEstimate)

	// calibrateTokenEstimatorForModel uses byteLen for context_compression when LooperStreamingActive
	classifier, err := classification.BuildClassifier(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ContextRules: []config.ContextRule{{
					Name:      "beyond_window",
					MinTokens: config.TokenCount("200K"),
				}},
			},
		},
	}, nil, nil, nil)
	require.NoError(t, err)

	router := &OpenAIRouter{
		Classifier: classifier,
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"physical-judge": {Capabilities: []string{"chat"}},
				},
			},
		},
	}

	reqCtx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	reqCtx.LooperSynthesisTextBytes = 1234
	router.calibrateTokenEstimatorForModel(reqCtx, "physical-judge", 1234, 100)

	// reportSemanticStreamingUsage skips single-model metrics and TPOT updates when LooperStreamingActive
	router.reportSemanticStreamingUsage(reqCtx, 100*time.Millisecond, responseUsageMetrics{
		promptTokens:     10,
		completionTokens: 20,
	})
}

func TestLooperStreaming_MaskingErrorMapsTo503(t *testing.T) {
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
		RequestID:                "req-masking-503",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
	}

	decision := &config.Decision{
		Name:      "fusion-masking",
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion},
	}

	maskingErr := &maskingDispatchError{cause: errors.New("pii service unavailable")}
	var matchedMaskingErr *maskingDispatchError
	require.True(t, errors.As(maskingErr, &matchedMaskingErr))

	resp := router.recordLooperFailureWithLifecycle(
		reqCtx, "entrypoint", decision, http.StatusServiceUnavailable,
		maskingUnavailableMessage, routerreplay.LifecycleFailed,
		"looper_stream_dispatch_masking_unavailable",
		looperFailureEvidence{
			algorithm: decision.Algorithm.Type,
			usage: looper.TokenUsage{
				PromptTokens:     120,
				CompletionTokens: 45,
				TotalTokens:      165,
			},
		},
	)

	require.NotNil(t, resp)
	require.NotNil(t, resp.GetImmediateResponse())
	require.Equal(t, int32(503), int32(resp.GetImmediateResponse().GetStatus().GetCode()))

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 503, record.ResponseStatus)
	require.Equal(t, "looper_stream_dispatch_masking_unavailable", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 120, *record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 45, *record.CompletionTokens)

	// Verify that when streaming dispatch encounters a maskingDispatchError,
	// handleLooperStreamingDispatch maps it to 503 and "looper_stream_dispatch_masking_unavailable",
	// and persists intermediate token usage into replay.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqPayload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&reqPayload))
		writeFusionReplayCompletion(w, reqPayload.Model, []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Panel answer from " + reqPayload.Model,
			},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165})
	}))
	defer server.Close()

	streamRouter, streamReqCtx, protoReq := quorumFallbackRouter(server.URL)
	streamRouter.Config.ClearRouteCache = true
	streamReqCtx.ExpectStreamingResponse = true
	protoReq.Stream = true

	streamDecision := &config.Decision{
		Name: "fusion-stream-masking",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a"},
				AnalysisMode:   config.FusionAnalysisModeOneCall,
			},
		},
		Plugins: []config.DecisionPlugin{{
			Type:          config.DecisionPluginMasking,
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
		}},
	}
	streamReqCtx.VSRSelectedDecision = streamDecision

	looperReq, errResp := streamRouter.buildLooperRequest(protoReq, streamDecision, streamReqCtx)
	require.Nil(t, errResp)
	require.NotNil(t, looperReq)

	streamResp, err := streamRouter.handleLooperStreamingDispatch(context.Background(), looperReq, streamDecision, protoReq.Model, streamReqCtx)
	require.NoError(t, err)
	require.NotNil(t, streamResp)
	require.NotNil(t, streamResp.GetImmediateResponse())
	require.Equal(t, int32(503), int32(streamResp.GetImmediateResponse().GetStatus().GetCode()))

	streamRecord, ok := streamRouter.ReplayRecorder.GetRecord(streamReqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 503, streamRecord.ResponseStatus)
	require.Equal(t, "looper_stream_dispatch_masking_unavailable", streamRecord.TerminalReason)
	require.NotNil(t, streamRecord.PromptTokens)
	require.Equal(t, 120, *streamRecord.PromptTokens)
	require.NotNil(t, streamRecord.CompletionTokens)
	require.Equal(t, 45, *streamRecord.CompletionTokens)
}

func TestLooperStreaming_NonStreamingResponseBodyProtection(t *testing.T) {
	router := &OpenAIRouter{}

	reqCtx := &RequestContext{
		RequestID:             "req-non-streaming-guard",
		RequestModel:          "virtual-entrypoint",
		LooperStreamingActive: true,
		InflightToken:         100,
	}

	// When LooperStreamingActive is true, reportNonStreamingUsage does not panic and uses empty pricing
	router.reportNonStreamingUsage(reqCtx, 100*time.Millisecond, responseUsageMetrics{
		promptTokens:     50,
		completionTokens: 25,
	})
	// Inflight token should be cleared
	require.Equal(t, uint64(0), reqCtx.InflightToken)

	// Verify non-streaming response body handling under LooperStreamingActive:
	// Settles usage, records both synthesized and intermediate tokens into Replay,
	// and prevents subsequent EOF from overwriting usage or marking aborted.
	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router.Config = &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{Enabled: true},
	}
	router.ReplayRecorder = recorder

	reqCtx.RouterReplayRecorder = recorder
	reqCtx.RouterReplayPluginConfig = &replayConfig
	reqCtx.LooperIntermediateUsage = looper.TokenUsage{
		PromptTokens:     30,
		CompletionTokens: 20,
		TotalTokens:      50,
	}
	router.startRouterReplay(reqCtx, "virtual-entrypoint", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	responseBody := []byte(`{
		"id": "chatcmpl-nonstream",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "judge",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	resp := router.handleNonStreamingResponseBody(responseBody, reqCtx, 100*time.Millisecond)
	require.NotNil(t, resp)
	require.True(t, reqCtx.LooperUsageSettled)

	// Verify model rewrite in response body mutation
	bodyMutation := resp.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	require.NotEmpty(t, bodyMutation)
	var clientResp map[string]interface{}
	require.NoError(t, json.Unmarshal(bodyMutation, &clientResp))
	require.Equal(t, "virtual-entrypoint", clientResp["model"])

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 40, *record.PromptTokens)     // 10 + 30
	require.Equal(t, 25, *record.CompletionTokens) // 5 + 20
	require.Equal(t, 65, *record.TotalTokens)      // 15 + 50

	// Verify EOF does not overwrite usage or mark aborted
	err := router.handleProcessReceiveError(reqCtx, io.EOF)
	require.NoError(t, err)
	require.False(t, reqCtx.StreamingAborted)

	recordAfterEOF, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 40, *recordAfterEOF.PromptTokens)
	require.Equal(t, 25, *recordAfterEOF.CompletionTokens)
	require.Equal(t, 65, *recordAfterEOF.TotalTokens)
}

func TestLooperStreaming_NonStreamingResponseModelRewriteAndTraceMounting(t *testing.T) {
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
		RequestID:                "req-nonstream-model-trace",
		RequestModel:             "virtual-entrypoint",
		LooperStreamingActive:    true,
		VSRSelectionMethod:       config.DecisionAlgorithmFusion,
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		LooperTrace: &looper.FusionTrace{
			AnalysisMode: config.FusionAnalysisModeOneCall,
			Analysis: &looper.FusionAnalysis{
				Consensus: []string{"consensus item"},
			},
			Responses: []looper.FusionPanelResponse{
				{Model: "panel-a", Content: "panel response"},
			},
			FailedModels: []looper.FusionFailedModel{
				{Model: "panel-b", Error: "500 error"},
			},
			JudgeModel: "physical-judge",
		},
	}

	router.startRouterReplay(reqCtx, "virtual-entrypoint", "physical-judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	responseBody := []byte(`{
		"id": "chatcmpl-test",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "physical-judge",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}
	}`)

	resp := router.handleNonStreamingResponseBody(responseBody, reqCtx, 100*time.Millisecond)
	require.NotNil(t, resp)

	// 1. Client wire body: model must be rewritten to RequestModel ("virtual-entrypoint")
	clientBody := resp.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	require.NotEmpty(t, clientBody)
	var clientJSON map[string]interface{}
	require.NoError(t, json.Unmarshal(clientBody, &clientJSON))
	require.Equal(t, "virtual-entrypoint", clientJSON["model"], "public model name must be rewritten to requested model")
	_, clientHasFusion := clientJSON["fusion"]
	require.False(t, clientHasFusion, "client wire body must not leak fusion trace")

	// 2. Replay snapshot: must contain projected fusion trace, with analysis_mode stripped
	rec, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotEmpty(t, rec.ResponseBody)
	var replayJSON map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(rec.ResponseBody), &replayJSON))
	fusionVal, hasFusion := replayJSON["fusion"]
	require.True(t, hasFusion, "replay snapshot must have fusion trace attached")
	fusionMap, ok := fusionVal.(map[string]interface{})
	require.True(t, ok)
	_, hasAnalysisMode := fusionMap["analysis_mode"]
	require.False(t, hasAnalysisMode, "internal analysis_mode must not leak into replay snapshot")
	require.Equal(t, "physical-judge", fusionMap["judge_model"])
}

func TestLooperStreaming_NonStreamingResponseCalibratorIsolation(t *testing.T) {
	classifier, err := classification.BuildClassifier(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ContextRules: []config.ContextRule{{
					Name:      "beyond_window",
					MinTokens: config.TokenCount("200K"),
				}},
			},
		},
	}, nil, nil, nil)
	require.NoError(t, err)

	router := &OpenAIRouter{
		Classifier: classifier,
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"physical-judge": {Capabilities: []string{"chat"}},
				},
			},
		},
	}

	reqCtx := &RequestContext{
		RequestID:                "req-calibrator-test",
		RequestModel:             "virtual-entrypoint",
		LooperStreamingActive:    true,
		LooperTargetModel:        "physical-judge",
		LooperSynthesisTextBytes: 8000,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     20000,
			CompletionTokens: 5000,
			TotalTokens:      25000,
		},
	}
	reqCtx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})

	// Upstream 200 response with synthesis prompt_tokens = 2000
	responseBody := []byte(`{
		"id": "chatcmpl-calib",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "physical-judge",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 2000, "completion_tokens": 500, "total_tokens": 2500}
	}`)

	resp := router.handleNonStreamingResponseBody(responseBody, reqCtx, 100*time.Millisecond)
	require.NotNil(t, resp)

	// Calibrator must observe synthesis bytes (8000) / synthesis prompt tokens (2000) = 4.0
	// If it mistakenly used intermediate + synthesis = 22000 tokens, ratio would be 8000 / 22000 = 0.3636
	mean, _, samples, _ := classifier.TokenCalibrationRatio("model:physical-judge")
	require.Equal(t, int64(1), samples)
	require.InDelta(t, 4.0, mean, 0.001)
}

func TestLooperStreaming_NonStreamingResponseInvalidUsageSettlement(t *testing.T) {
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
		RequestID:                "req-invalid-usage",
		RequestModel:             "virtual-entrypoint",
		LooperStreamingActive:    true,
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     35,
			CompletionTokens: 25,
			TotalTokens:      60,
		},
	}
	router.startRouterReplay(reqCtx, "virtual-entrypoint", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	// Upstream 200 response missing "usage" field entirely
	responseBody := []byte(`{
		"id": "chatcmpl-nousage",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "judge",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}]
	}`)

	resp := router.handleNonStreamingResponseBody(responseBody, reqCtx, 100*time.Millisecond)
	require.NotNil(t, resp)

	// Must settle aborted looper streaming usage:
	// LooperUsageSettled and StreamingAborted should be true
	require.True(t, reqCtx.LooperUsageSettled, "usage must be marked settled")
	require.True(t, reqCtx.StreamingAborted, "streaming must be marked aborted on invalid usage")

	// Intermediate usage must be preserved and settled into Replay, NOT zeroed out
	rec, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, rec.PromptTokens)
	require.NotNil(t, rec.CompletionTokens)
	require.NotNil(t, rec.TotalTokens)
	require.Equal(t, 35, *rec.PromptTokens)
	require.Equal(t, 25, *rec.CompletionTokens)
	require.Equal(t, 60, *rec.TotalTokens)
}

func TestLooperStreaming_ReplaySnapshotFusionProjection(t *testing.T) {
	ctx := &RequestContext{
		LooperStreamingActive: true,
		VSRSelectionMethod:    config.DecisionAlgorithmFusion,
		LooperTrace: &looper.FusionTrace{
			AnalysisMode: config.FusionAnalysisModeOneCall,
			Analysis: &looper.FusionAnalysis{
				Consensus: []string{"consensus item"},
			},
			Responses: []looper.FusionPanelResponse{
				{Model: "panel-a", Content: "panel response"},
			},
			FailedModels: []looper.FusionFailedModel{
				{Model: "panel-b", Error: "500 error"},
			},
			JudgeModel: "judge",
		},
	}

	rawResponse := []byte(`{"id":"chatcmpl-test","choices":[{"index":0,"message":{"role":"assistant","content":"hello"}}]}`)
	result := attachLooperTraceToReplaySnapshot(ctx, rawResponse)
	require.NotEmpty(t, result)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(result, &parsed))

	fusionVal, hasFusion := parsed["fusion"]
	require.True(t, hasFusion, "fusion trace should be attached to replay snapshot")

	fusionMap, ok := fusionVal.(map[string]interface{})
	require.True(t, ok, "fusion should be a JSON object")

	// analysis_mode must NOT be present (sanitized/projected via ProjectFusionPublicTrace)
	_, hasAnalysisMode := fusionMap["analysis_mode"]
	require.False(t, hasAnalysisMode, "internal analysis_mode must not leak into replay snapshot")

	// Public fields must be present
	require.Equal(t, "judge", fusionMap["judge_model"])
	require.NotNil(t, fusionMap["analysis"])
	require.NotNil(t, fusionMap["responses"])
	require.NotNil(t, fusionMap["failed_models"])
}

func TestLooperStreamingDispatch_LegacyEndpointAppliesRequestParams(t *testing.T) {
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
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(2048)

	limit := 64
	decision := &config.Decision{
		Name: "fusion-stream-params",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a", "panel-b"},
				AnalysisMode:   config.FusionAnalysisModeOneCall,
			},
		},
		Plugins: []config.DecisionPlugin{
			{
				Type: "request_params",
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"max_tokens_limit": limit,
				}),
			},
		},
	}
	reqCtx.VSRSelectedDecision = decision
	reqCtx.VSRSelectedDecisionName = decision.Name

	response, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, response)
	require.Nil(t, response.GetImmediateResponse())

	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var dispatchedPayload map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &dispatchedPayload))

	// Max tokens limit of 64 should be applied
	if maxCompletionTokens, ok := dispatchedPayload["max_completion_tokens"].(float64); ok {
		require.Equal(t, float64(64), maxCompletionTokens)
	} else if maxTokens, ok := dispatchedPayload["max_tokens"].(float64); ok {
		require.Equal(t, float64(64), maxTokens)
	} else {
		t.Fatalf("neither max_completion_tokens nor max_tokens found in dispatched payload: %v", dispatchedPayload)
	}
}

func TestLooperStreamingDispatch_BuildResponseDoesNotMutateClientRequest(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{
				Endpoint: "http://looper.legacy.local",
			},
		},
	}
	clientReq := &llmprotocol.Request{
		Model: "client-model",
		Sampling: llmprotocol.Sampling{
			MaxOutputTokens: llmprotocol.Int64(2048),
		},
	}
	decision := &config.Decision{
		Name: "test-decision",
		Plugins: []config.DecisionPlugin{
			{
				Type: "request_params",
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"max_tokens_limit": 64,
				}),
			},
		},
	}
	reqCtx := &RequestContext{
		RequestID:               "req-legacy-test",
		SourceFormat:            llmprotocol.OpenAIChatV1,
		SemanticRequest:         clientReq,
		VSRSelectedDecision:     decision,
		VSRSelectedDecisionName: decision.Name,
	}

	rawSynthesis := []byte(`{"model":"synth-model","messages":[{"role":"user","content":"hi"}],"max_tokens":500}`)
	resp, _, err := router.buildLooperStreamingDispatchResponse(rawSynthesis, "synth-model", decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, resp)

	// Dispatched synthesis body has request_params applied (500 capped to 64)
	body := resp.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var dispatched map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &dispatched))
	if maxTokens, ok := dispatched["max_tokens"].(float64); ok {
		require.Equal(t, float64(64), maxTokens)
	} else if maxCompletionTokens, ok := dispatched["max_completion_tokens"].(float64); ok {
		require.Equal(t, float64(64), maxCompletionTokens)
	} else {
		t.Fatalf("neither max_tokens nor max_completion_tokens found: %v", dispatched)
	}

	// Client request restored after deferred exit must NOT have had request_params applied
	require.Equal(t, clientReq, reqCtx.SemanticRequest)
	require.Equal(t, int64(2048), *clientReq.Sampling.MaxOutputTokens)
}

func TestLooperExecutionErrorResponse_SanitizesClientVisibleMessage(t *testing.T) {
	router := &OpenAIRouter{}
	reqCtx := &RequestContext{
		RequestID:    "req-123",
		SourceFormat: llmprotocol.OpenAIChatV1,
	}
	decision := &config.Decision{
		Name: "test-decision",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmConfidence,
		},
	}
	internalErr := errors.New("upstream connection refused at 10.0.0.123:8080: secret internal failure")
	resp := router.looperExecutionErrorResponse(internalErr, "original-model", decision, reqCtx)
	require.NotNil(t, resp)
	require.NotNil(t, reqCtx.ImmediateProtocolError)
	// Must not leak internal details to client-visible protocol error
	require.Equal(t, "Looper execution failed", reqCtx.ImmediateProtocolError.Message)
	require.Equal(t, "looper_execution_failed", reqCtx.ImmediateProtocolError.Code)
	require.Equal(t, llmprotocol.ErrorInternal, reqCtx.ImmediateProtocolError.Category)
	// Raw cause is preserved internally
	require.ErrorIs(t, reqCtx.ImmediateProtocolError, internalErr)

	// Encoded immediate response payload should contain sanitized message and NOT leak internal error text
	immediate := resp.GetImmediateResponse()
	require.NotNil(t, immediate)
	bodyStr := string(immediate.GetBody())
	require.Contains(t, bodyStr, "Looper execution failed")
	require.NotContains(t, bodyStr, "10.0.0.123:8080")
	require.NotContains(t, bodyStr, "secret internal failure")
}

func TestLooperFailureEvidenceFromError_FusionStageFailureError(t *testing.T) {
	stageErr := looper.NewFusionStageFailureError(
		errors.New("judge model overloaded"),
		[]string{"panel-a", "panel-b", "judge"},
		3,
		looper.TokenUsage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150},
	)
	evidence, ok := looperFailureEvidenceFromError(stageErr)
	require.True(t, ok)
	require.Equal(t, config.DecisionAlgorithmFusion, evidence.algorithm)
	require.Equal(t, []string{"panel-a", "panel-b", "judge"}, evidence.modelsUsed)
	require.Equal(t, 3, evidence.iterations)
	require.Equal(t, int64(150), evidence.usage.TotalTokens)
	require.Nil(t, evidence.fusionQuorum, "fusionQuorum must remain nil to avoid false quorum alarm")

	router := &OpenAIRouter{}
	reqCtx := &RequestContext{
		RequestID:    "req-stage-fail",
		SourceFormat: llmprotocol.OpenAIChatV1,
	}
	decision := &config.Decision{
		Name: "test-fusion-stage",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
		},
	}
	resp := router.looperExecutionErrorResponse(stageErr, "original-model", decision, reqCtx)
	require.NotNil(t, resp)
	require.Nil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must not be populated when quorum was met")
}

func TestEstimateAbortedLooperOutputTokens_MapKeyOrderDeterminism(t *testing.T) {
	// Construct state with non-sequential out-of-order map keys
	state := &semanticResponseStreamState{
		items: map[int]*semanticStreamItem{
			5: {text: " world"},
			2: {text: "hello"},
			0: {text: "say: "},
		},
	}
	tokens := estimateAbortedLooperOutputTokens(state)
	expectedTokens := contextcompression.EstimateTokens("say: hello world")
	require.Equal(t, expectedTokens, tokens)
}

func TestLooperStreamingAbort_AuthoritativeUsageBeforeDoneSettlement(t *testing.T) {
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
		RequestID:                "req-authoritative-before-done",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		IsStreamingResponse:      true,
		StreamingComplete:        false,
		LooperStreamingActive:    true,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     30,
			CompletionTokens: 15,
			TotalTokens:      45,
		},
		SemanticStreamState: &semanticResponseStreamState{
			items: make(map[int]*semanticStreamItem),
			usage: llmprotocol.Usage{
				State: llmprotocol.UsageAvailable,
				InputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(18),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				OutputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(7),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				Total: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(25),
					Provenance: llmprotocol.UsageAuthoritative,
				},
			},
		},
	}

	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	// Simulate client abort/disconnect before [DONE] arrival
	err := router.handleProcessReceiveError(reqCtx, context.Canceled)
	require.NoError(t, err)
	require.True(t, reqCtx.StreamingAborted)
	require.True(t, reqCtx.LooperUsageSettled)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_canceled", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)

	// Authoritative usage:
	// PromptTokens: 30 (intermediate) + 18 (synthesis upstream authoritative) = 48
	// CompletionTokens: 15 (intermediate) + 7 (synthesis upstream authoritative) = 22
	// TotalTokens: 48 + 22 = 70
	require.Equal(t, 48, *record.PromptTokens)
	require.Equal(t, 22, *record.CompletionTokens)
	require.Equal(t, 70, *record.TotalTokens)
}

func TestLooperStreamingAbort_OutputChunksWithoutUsageFrameEstimatedWithPromptFloor(t *testing.T) {
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

	chunkText := "Synthesis response output chunk generated by judge model before client disconnected."
	expectedEstimatedOutput := contextcompression.EstimateTokens(chunkText)
	require.Greater(t, expectedEstimatedOutput, 0)

	synthesisPromptTokens := 42
	reqCtx := &RequestContext{
		RequestID:                   "req-chunk-without-usage",
		RouterReplayRecorder:        recorder,
		RouterReplayPluginConfig:    &replayConfig,
		IsStreamingResponse:         true,
		StreamingComplete:           false,
		LooperStreamingActive:       true,
		LooperSynthesisPromptTokens: synthesisPromptTokens,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     20,
			CompletionTokens: 10,
			TotalTokens:      30,
		},
		SemanticStreamState: &semanticResponseStreamState{
			items: make(map[int]*semanticStreamItem),
		},
	}

	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	// Observe delta chunks without usage tail frame
	observeProtocolStream(reqCtx, []llmprotocol.Event{
		{
			Type:      llmprotocol.EventOutputTextDelta,
			ItemIndex: 0,
			Delta:     chunkText,
		},
	}, nil)

	// Simulate client cancellation
	err := router.handleProcessReceiveError(reqCtx, context.Canceled)
	require.NoError(t, err)
	require.True(t, reqCtx.StreamingAborted)
	require.True(t, reqCtx.LooperUsageSettled)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_canceled", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)

	// Input Tokens must not be zeroed out: 20 (intermediate) + 42 (synthesis prompt) = 62
	require.Equal(t, 20+synthesisPromptTokens, *record.PromptTokens)
	// Output Tokens: 10 (intermediate) + estimated output
	require.Equal(t, 10+expectedEstimatedOutput, *record.CompletionTokens)
	require.Equal(t, 20+synthesisPromptTokens+10+expectedEstimatedOutput, *record.TotalTokens)
}

func TestLooperStreamingAbort_OutputChunksWithoutUsageFrameFallbackToSynthesisBytes(t *testing.T) {
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

	chunkText := "Partial output before cancel."
	expectedEstimatedOutput := contextcompression.EstimateTokens(chunkText)
	require.Greater(t, expectedEstimatedOutput, 0)

	synthesisBytes := 120
	expectedEstimatedPromptFloor := (synthesisBytes + 3) / 4 // 30 tokens

	reqCtx := &RequestContext{
		RequestID:                   "req-chunk-bytes-fallback",
		RouterReplayRecorder:        recorder,
		RouterReplayPluginConfig:    &replayConfig,
		IsStreamingResponse:         true,
		StreamingComplete:           false,
		LooperStreamingActive:       true,
		LooperSynthesisPromptTokens: 0, // Not explicitly set, fallback to bytes
		LooperSynthesisTextBytes:    synthesisBytes,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     15,
			CompletionTokens: 5,
			TotalTokens:      20,
		},
		SemanticStreamState: &semanticResponseStreamState{
			items: make(map[int]*semanticStreamItem),
		},
	}

	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	observeProtocolStream(reqCtx, []llmprotocol.Event{
		{
			Type:      llmprotocol.EventOutputTextDelta,
			ItemIndex: 0,
			Delta:     chunkText,
		},
	}, nil)

	err := router.handleProcessReceiveError(reqCtx, context.Canceled)
	require.NoError(t, err)
	require.True(t, reqCtx.StreamingAborted)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 15+expectedEstimatedPromptFloor, *record.PromptTokens)
	require.Equal(t, 5+expectedEstimatedOutput, *record.CompletionTokens)
	require.Equal(t, 15+expectedEstimatedPromptFloor+5+expectedEstimatedOutput, *record.TotalTokens)
}

func TestLooperStreamingAbort_CompressedLargeRequestSettlesPostCompressionPromptTokens(t *testing.T) {
	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()

	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.ReplayRecorder = recorder
	router.Config.RouterReplay = config.RouterReplayConfig{
		Enabled: true,
	}
	router.Config.DefaultModel = "virtual-entrypoint"
	router.Config.CandidateRequirements = &config.CandidateRequirements{
		Capabilities: config.CandidateCapabilitiesDeclared,
		Context:      config.CandidateContextKnownLimits,
	}
	params := router.Config.ModelConfig[model]
	params.Capabilities = []string{"chat"}
	params.ContextWindowSize = 32768
	params.MaxOutputTokens = 8192
	router.Config.ModelConfig[model] = params

	payload, err := config.NewStructuredPayload(map[string]any{
		"enabled": true,
		"targets": map[string]any{
			"current_user": map[string]any{"mode": "truncate"},
			"history":      map[string]any{"mode": "extractive", "min_tokens": 2000, "target_tokens": 1000},
			"tool_outputs": map[string]any{"mode": "preserve"},
		},
	})
	require.NoError(t, err)

	decision := &config.Decision{
		Name:      "fusion-compression-decision",
		ModelRefs: []config.ModelRef{{Model: model}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model: model,
			},
		},
		Plugins: []config.DecisionPlugin{
			{Type: "context_compression", Configuration: payload},
		},
	}
	router.Config.Decisions = []config.Decision{*decision}
	router.Config.Signals.ContextRules = []config.ContextRule{{
		Name:      "window",
		MinTokens: config.TokenCount("100K"),
	}}

	classifier, err := classification.BuildClassifier(router.Config, nil, nil, nil)
	require.NoError(t, err)
	router.Classifier = classifier

	clientReq := &llmprotocol.Request{
		Model: "client-entrypoint",
		Messages: []llmprotocol.Message{
			{Role: "user", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "original client query"}}},
		},
	}
	reqCtx := &RequestContext{
		RequestID:                "req-compress-abort-trace",
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          clientReq,
		VSRSelectedDecision:      decision,
		VSRSelectedDecisionName:  decision.Name,
		TraceContext:             context.Background(),
		ExpectStreamingResponse:  true,
		IsStreamingResponse:      true,
		StreamingComplete:        false,
		LooperStreamingActive:    true,
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     20,
			CompletionTokens: 10,
			TotalTokens:      30,
		},
		SemanticStreamState: &semanticResponseStreamState{
			items: make(map[int]*semanticStreamItem),
		},
	}
	reqCtx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})

	// Large uncompressed synthesis prompt (80000+ bytes, ~20000+ tokens)
	largeText := "HEAD instruction\n" + strings.Repeat(" a", 40000) + "\nTAIL instruction"
	rawSynthesisBody := []byte(fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":%q}]}`, model, largeText))

	initialSignals := extractSemanticRequestSignals(&llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: "user", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: largeText}}},
		},
	})
	uncompressedPromptTokens := initialSignals.ContextTokenFloor
	require.GreaterOrEqual(t, uncompressedPromptTokens, 10000)

	// Call buildLooperStreamingDispatchResponse
	resp, synthesisTextBytes, err := router.buildLooperStreamingDispatchResponse(
		rawSynthesisBody, model, decision, reqCtx,
	)
	require.NoError(t, err)
	require.NotNil(t, resp)

	// Compression should have run
	require.True(t, reqCtx.ContextCompressionApplied)
	require.Greater(t, reqCtx.ContextCompressionBefore, reqCtx.ContextCompressionAfter)

	// LooperSynthesisPromptTokens must reflect the post-compression prompt tokens floor
	compressedPromptTokens := reqCtx.LooperSynthesisPromptTokens
	require.Greater(t, compressedPromptTokens, 0)
	require.Less(t, compressedPromptTokens, uncompressedPromptTokens,
		"LooperSynthesisPromptTokens must reflect compressed tokens, not uncompressed initial tokens")

	reqCtx.LooperTargetModel = model
	reqCtx.LooperSynthesisTextBytes = synthesisTextBytes
	router.startRouterReplay(reqCtx, "virtual-entrypoint", model, decision.Name)
	require.NotEmpty(t, reqCtx.RouterReplayID)

	// Downstream output chunks arrive without usage tail frame
	chunkText := "Streaming output tokens from judge model before abort."
	expectedEstimatedOutput := contextcompression.EstimateTokens(chunkText)
	require.Greater(t, expectedEstimatedOutput, 0)

	observeProtocolStream(reqCtx, []llmprotocol.Event{
		{
			Type:      llmprotocol.EventOutputTextDelta,
			ItemIndex: 0,
			Delta:     chunkText,
		},
	}, nil)

	// Client cancels stream before done/usage frame
	err = router.handleProcessReceiveError(reqCtx, context.Canceled)
	require.NoError(t, err)
	require.True(t, reqCtx.StreamingAborted)
	require.True(t, reqCtx.LooperUsageSettled)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_canceled", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)

	// Prompt tokens: 20 intermediate + compressedPromptTokens.
	// Must NOT equal 20 + uncompressedPromptTokens.
	require.Equal(t, 20+compressedPromptTokens, *record.PromptTokens,
		"settled prompt tokens must reflect post-compression tokens floor")
	require.NotEqual(t, 20+uncompressedPromptTokens, *record.PromptTokens,
		"settled prompt tokens must not reflect pre-compression tokens floor")
	require.Equal(t, 10+expectedEstimatedOutput, *record.CompletionTokens)
	require.Equal(t, 20+compressedPromptTokens+10+expectedEstimatedOutput, *record.TotalTokens)
}

func TestLooperStreaming_SynthesisTextBytesReflectsPostCompression(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Config.DefaultModel = "virtual-entrypoint"
	router.Config.CandidateRequirements = &config.CandidateRequirements{
		Capabilities: config.CandidateCapabilitiesDeclared,
		Context:      config.CandidateContextKnownLimits,
	}
	params := router.Config.ModelConfig[model]
	params.Capabilities = []string{"chat"}
	params.ContextWindowSize = 32768
	params.MaxOutputTokens = 8192
	router.Config.ModelConfig[model] = params

	payload, err := config.NewStructuredPayload(map[string]any{
		"enabled": true,
		"targets": map[string]any{
			"current_user": map[string]any{"mode": "truncate"},
			"history":      map[string]any{"mode": "extractive", "min_tokens": 2000, "target_tokens": 1000},
			"tool_outputs": map[string]any{"mode": "preserve"},
		},
	})
	require.NoError(t, err)

	decision := &config.Decision{
		Name:      "fusion-compression-decision",
		ModelRefs: []config.ModelRef{{Model: model}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model: model,
			},
		},
		Plugins: []config.DecisionPlugin{
			{Type: "context_compression", Configuration: payload},
		},
	}
	router.Config.Decisions = []config.Decision{*decision}
	router.Config.Signals.ContextRules = []config.ContextRule{{
		Name:      "window",
		MinTokens: config.TokenCount("100K"),
	}}

	classifier, err := classification.BuildClassifier(router.Config, nil, nil, nil)
	require.NoError(t, err)
	router.Classifier = classifier

	clientReq := &llmprotocol.Request{
		Model: "client-entrypoint",
		Messages: []llmprotocol.Message{
			{Role: "user", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "original client query"}}},
		},
	}
	reqCtx := &RequestContext{
		RequestID:               "req-stream-compress-test",
		SourceFormat:            llmprotocol.OpenAIChatV1,
		SemanticRequest:         clientReq,
		VSRSelectedDecision:     decision,
		VSRSelectedDecisionName: decision.Name,
		TraceContext:            context.Background(),
		ExpectStreamingResponse: true,
	}
	reqCtx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})

	// 80000+ bytes uncompressed synthesis prompt
	largeText := "HEAD instruction\n" + strings.Repeat(" a", 40000) + "\nTAIL instruction"
	rawSynthesisBody := []byte(fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":%q}]}`, model, largeText))

	initialSignals := extractSemanticRequestSignals(&llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: "user", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: largeText}}},
		},
	})
	initialBytes := initialSignals.ContextTextBytes
	require.GreaterOrEqual(t, initialBytes, 8000, "initial uncompressed bytes should be at least 8000")

	// Call buildLooperStreamingDispatchResponse
	resp, synthesisTextBytes, err := router.buildLooperStreamingDispatchResponse(
		rawSynthesisBody, model, decision, reqCtx,
	)
	require.NoError(t, err)
	require.NotNil(t, resp)

	// Context compression must have been applied during prepareProviderDispatch
	require.True(t, reqCtx.ContextCompressionApplied)
	require.Greater(t, reqCtx.ContextCompressionBefore, reqCtx.ContextCompressionAfter)

	// synthesisTextBytes must reflect the post-compression bytes, NOT the uncompressed 8000+ bytes!
	require.Less(t, synthesisTextBytes, initialBytes, "synthesisTextBytes must be strictly less than uncompressed bytes")

	// Verify that the dispatched outbound body text length matches synthesisTextBytes exactly
	engine, err := router.protocolEngine()
	require.NoError(t, err)
	outboundBody := resp.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	decodedReq, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, outboundBody)
	require.NoError(t, err)
	outboundTextBytes := extractSemanticRequestSignals(&decodedReq).ContextTextBytes
	require.Equal(t, outboundTextBytes, synthesisTextBytes, "synthesisTextBytes must exactly match the outbound dispatched payload")

	// Verify reqCtx.LooperSynthesisTextBytes reflects the post-compression bytes
	reqCtx.LooperTargetModel = model
	reqCtx.LooperStreamingActive = true
	reqCtx.LooperSynthesisTextBytes = synthesisTextBytes
	require.Equal(t, synthesisTextBytes, reqCtx.LooperSynthesisTextBytes)
	require.Less(t, reqCtx.LooperSynthesisTextBytes, initialBytes)

	// Verify reqCtx.LooperSynthesisPromptTokens reflects the post-compression ContextTokenFloor
	outboundSignals := extractSemanticRequestSignals(&decodedReq)
	require.Equal(t, outboundSignals.ContextTokenFloor, reqCtx.LooperSynthesisPromptTokens, "LooperSynthesisPromptTokens must reflect post-compression prompt tokens")
	require.Less(t, reqCtx.LooperSynthesisPromptTokens, initialSignals.ContextTokenFloor, "LooperSynthesisPromptTokens must be strictly less than initial uncompressed tokens")

	// Verify client SemanticRequest was preserved and restored
	require.Same(t, clientReq, reqCtx.SemanticRequest, "client SemanticRequest must be restored after dispatch response building")

	// Verify calibration: upstream returns prompt_tokens corresponding to the compressed size (~4 bytes/token)
	actualTokens := synthesisTextBytes / 4
	for i := 0; i < 10; i++ {
		router.calibrateTokenEstimatorForModel(reqCtx, model, streamingCalibrationByteLen(reqCtx), actualTokens)
	}

	calibratedClassifier := router.classifierForRequest(reqCtx)
	require.NotNil(t, calibratedClassifier)
	mean, _, samples, calibrated := calibratedClassifier.TokenCalibrationRatio("model:" + model)
	require.True(t, calibrated)
	require.Equal(t, int64(10), samples)
	expectedRatio := float64(synthesisTextBytes) / float64(actualTokens)
	require.InDelta(t, expectedRatio, mean, 0.01)
	require.Less(t, mean, 10.0, "calibrator ratio must not be poisoned by uncompressed bytes")
}

func TestLooperStreaming_PrepareDeadlineExceededYields504(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqPayload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&reqPayload))
		writeFusionReplayCompletion(w, reqPayload.Model, []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Panel answer from " + reqPayload.Model,
			},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140})
	}))
	defer server.Close()

	streamRouter, streamReqCtx, protoReq := quorumFallbackRouter(server.URL)
	streamRouter.Config.ClearRouteCache = true
	streamReqCtx.ExpectStreamingResponse = true
	protoReq.Stream = true

	streamDecision := &config.Decision{
		Name: "fusion-stream-prepare-timeout",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a"},
				AnalysisMode:   config.FusionAnalysisModeOneCall,
			},
		},
	}
	streamReqCtx.VSRSelectedDecision = streamDecision

	looperReq, errResp := streamRouter.buildLooperRequest(protoReq, streamDecision, streamReqCtx)
	require.Nil(t, errResp)
	require.NotNil(t, looperReq)

	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-100*time.Millisecond))
	defer cancel()

	streamResp, err := streamRouter.handleLooperStreamingDispatch(ctx, looperReq, streamDecision, protoReq.Model, streamReqCtx)
	require.NoError(t, err)
	require.NotNil(t, streamResp)
	require.NotNil(t, streamResp.GetImmediateResponse())
	require.Equal(t, int32(504), int32(streamResp.GetImmediateResponse().GetStatus().GetCode()))

	require.NotNil(t, streamReqCtx.ImmediateProtocolError)
	require.Equal(t, llmprotocol.ErrorUpstreamTimeout, streamReqCtx.ImmediateProtocolError.Category)
	require.Equal(t, "request_timeout", streamReqCtx.ImmediateProtocolError.Code)

	record, ok := streamRouter.ReplayRecorder.GetRecord(streamReqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 504, record.ResponseStatus)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "looper_streaming_deadline_exceeded", record.TerminalReason)
}

func TestLooperStreaming_DispatchDeadlineExceededYields504(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqPayload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&reqPayload))
		writeFusionReplayCompletion(w, reqPayload.Model, []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Panel answer from " + reqPayload.Model,
			},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165})
	}))
	defer server.Close()

	streamRouter, streamReqCtx, protoReq := quorumFallbackRouter(server.URL)
	streamRouter.Config.ClearRouteCache = true
	streamReqCtx.ExpectStreamingResponse = true
	protoReq.Stream = true

	streamDecision := &config.Decision{
		Name: "fusion-stream-dispatch-timeout",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a"},
				AnalysisMode:   config.FusionAnalysisModeOneCall,
			},
		},
	}
	streamReqCtx.VSRSelectedDecision = streamDecision

	looperReq, errResp := streamRouter.buildLooperRequest(protoReq, streamDecision, streamReqCtx)
	require.Nil(t, errResp)
	require.NotNil(t, looperReq)

	// Set expired TraceContext so prepare succeeds but dispatch fails with DeadlineExceeded
	traceCtx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-100*time.Millisecond))
	defer cancel()
	streamReqCtx.TraceContext = traceCtx

	streamResp, err := streamRouter.handleLooperStreamingDispatch(context.Background(), looperReq, streamDecision, protoReq.Model, streamReqCtx)
	require.NoError(t, err)
	require.NotNil(t, streamResp)
	require.NotNil(t, streamResp.GetImmediateResponse())
	require.Equal(t, int32(504), int32(streamResp.GetImmediateResponse().GetStatus().GetCode()))

	require.NotNil(t, streamReqCtx.ImmediateProtocolError)
	require.Equal(t, llmprotocol.ErrorUpstreamTimeout, streamReqCtx.ImmediateProtocolError.Category)
	require.Equal(t, "request_timeout", streamReqCtx.ImmediateProtocolError.Code)

	record, ok := streamRouter.ReplayRecorder.GetRecord(streamReqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 504, record.ResponseStatus)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "looper_stream_dispatch_deadline_exceeded", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 120, *record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 45, *record.CompletionTokens)
}

func TestLooperStreaming_StageContextExceededYields400(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqPayload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&reqPayload))
		writeFusionReplayCompletion(w, reqPayload.Model, []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Panel answer from " + reqPayload.Model,
			},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165})
	}))
	defer server.Close()

	streamRouter, streamReqCtx, protoReq := quorumFallbackRouter(server.URL)
	streamRouter.Config.ClearRouteCache = true
	streamReqCtx.ExpectStreamingResponse = true
	protoReq.Stream = true

	// Set tiny ContextWindowSize on judge so validateLooperStageContext fails during prepare
	streamRouter.Config.BackendModels.ModelConfig["judge"] = config.ModelParams{
		Capabilities:      []string{"chat"},
		ContextWindowSize: 5,
	}

	streamDecision := &config.Decision{
		Name: "fusion-stream-stage-context-exceeded",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a"},
				AnalysisMode:   config.FusionAnalysisModeOneCall,
			},
		},
	}
	streamReqCtx.VSRSelectedDecision = streamDecision
	streamReqCtx.VSRContextTokenCount = 100

	looperReq, errResp := streamRouter.buildLooperRequest(protoReq, streamDecision, streamReqCtx)
	require.Nil(t, errResp)
	require.NotNil(t, looperReq)

	streamResp, err := streamRouter.handleLooperStreamingDispatch(context.Background(), looperReq, streamDecision, protoReq.Model, streamReqCtx)
	require.NoError(t, err)
	require.NotNil(t, streamResp)
	require.NotNil(t, streamResp.GetImmediateResponse())
	require.Equal(t, int32(400), int32(streamResp.GetImmediateResponse().GetStatus().GetCode()))

	require.NotNil(t, streamReqCtx.ImmediateProtocolError)
	require.Equal(t, llmprotocol.ErrorInvalidRequest, streamReqCtx.ImmediateProtocolError.Category)
	require.Equal(t, "context_length_exceeded", streamReqCtx.ImmediateProtocolError.Code)

	record, ok := streamRouter.ReplayRecorder.GetRecord(streamReqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 400, record.ResponseStatus)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "looper_stage_context_exceeded", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 120, *record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 45, *record.CompletionTokens)
}

func TestLooperStreaming_BudgetExhaustedYields400(t *testing.T) {
	t.Run("zero_budget_fails_closed_immediately", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var reqPayload struct {
				Model string `json:"model"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&reqPayload))
			writeFusionReplayCompletion(w, reqPayload.Model, []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Panel answer from " + reqPayload.Model,
				},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165})
		}))
		defer server.Close()

		streamRouter, streamReqCtx, protoReq := quorumFallbackRouter(server.URL)
		streamRouter.Config.ClearRouteCache = true
		streamReqCtx.ExpectStreamingResponse = true
		protoReq.Stream = true

		streamDecision := &config.Decision{
			Name: "fusion-stream-budget-exhausted-zero",
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}
		streamReqCtx.VSRSelectedDecision = streamDecision

		looperReq, errResp := streamRouter.buildLooperRequest(protoReq, streamDecision, streamReqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Install a budget ledger with 0 limit so first call triggers budget.ErrExhausted
		ctx, _ := budget.WithLimit(context.Background(), 0)

		streamResp, err := streamRouter.handleLooperStreamingDispatch(ctx, looperReq, streamDecision, protoReq.Model, streamReqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.NotNil(t, streamResp.GetImmediateResponse())
		require.Equal(t, int32(400), int32(streamResp.GetImmediateResponse().GetStatus().GetCode()))

		require.NotNil(t, streamReqCtx.ImmediateProtocolError)
		require.Equal(t, llmprotocol.ErrorInvalidRequest, streamReqCtx.ImmediateProtocolError.Category)
		require.Equal(t, "budget_exhausted", streamReqCtx.ImmediateProtocolError.Code)

		record, ok := streamRouter.ReplayRecorder.GetRecord(streamReqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, 400, record.ResponseStatus)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "looper_budget_exhausted", record.TerminalReason)
	})

	t.Run("exhausted_budget_after_prepare_fails_closed_reserving_final_leg", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var reqPayload struct {
				Model string `json:"model"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&reqPayload))
			writeFusionReplayCompletion(w, reqPayload.Model, []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Panel answer from " + reqPayload.Model,
				},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165})
		}))
		defer server.Close()

		streamRouter, streamReqCtx, protoReq := quorumFallbackRouter(server.URL)
		streamRouter.Config.ClearRouteCache = true
		streamReqCtx.ExpectStreamingResponse = true
		protoReq.Stream = true

		// Set budget to 1 call. The intermediate panel analysis spends that 1 call.
		// No quota remains for the final Envoy synthesis call (Remaining < 1),
		// so streaming dispatch must fail-closed with 400 budget exhausted.
		streamDecision := &config.Decision{
			Name: "fusion-stream-budget-exhausted-after-prepare",
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Budget: &config.AlgorithmBudget{
					MaxCalls: 1,
				},
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}
		streamReqCtx.VSRSelectedDecision = streamDecision

		looperReq, errResp := streamRouter.buildLooperRequest(protoReq, streamDecision, streamReqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		streamResp, err := streamRouter.handleLooperStreamingDispatch(context.Background(), looperReq, streamDecision, protoReq.Model, streamReqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.NotNil(t, streamResp.GetImmediateResponse())
		require.Equal(t, int32(400), int32(streamResp.GetImmediateResponse().GetStatus().GetCode()))

		require.NotNil(t, streamReqCtx.ImmediateProtocolError)
		require.Equal(t, llmprotocol.ErrorInvalidRequest, streamReqCtx.ImmediateProtocolError.Category)
		require.Equal(t, "budget_exhausted", streamReqCtx.ImmediateProtocolError.Code)

		record, ok := streamRouter.ReplayRecorder.GetRecord(streamReqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, 400, record.ResponseStatus)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "looper_budget_exhausted", record.TerminalReason)
	})
}

func TestLooperStreaming_PanicDuringDispatchConstructionSettlesIntermediateTokens(t *testing.T) {
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
		}}, map[string]int64{"prompt_tokens": 24, "completion_tokens": 12, "total_tokens": 36})
	}))
	defer server.Close()

	router, reqCtx, request := quorumFallbackRouter(server.URL)
	router.Config.ClearRouteCache = true
	reqCtx.ExpectStreamingResponse = true
	request.Stream = true

	decision := &config.Decision{
		Name: "fusion-stream-dispatch-panic",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a", "panel-b"},
				AnalysisMode:   config.FusionAnalysisModeOneCall,
			},
		},
	}
	reqCtx.VSRSelectedDecision = decision
	reqCtx.VSRSelectedDecisionName = decision.Name

	// Wrap execution and simulate a panic during dispatch response construction:
	// Once PrepareStreamingSynthesis succeeds, intermediate tokens are mounted immediately on reqCtx.
	// Even if a subsequent panic occurs during dispatch, the outer recover guard must settle the tokens to Replay.
	var recoveredPanic any
	func() {
		defer func() {
			if rec := recover(); rec != nil {
				recoveredPanic = rec
				if reqCtx != nil {
					router.settleAbortedLooperStreamingUsage(reqCtx, fmt.Errorf("processor_panic: %v", rec))
				}
				router.finalizeRouterReplay(reqCtx, routerreplay.LifecycleFailed, "processor_panic")
			}
		}()

		looperReq, errResp := router.buildLooperRequest(request, decision, reqCtx)
		require.Nil(t, errResp)

		runtimeConfig := router.Config.Looper
		hops := router.looperHops()
		require.NotNil(t, hops)
		client := looper.NewHopClient(&runtimeConfig, hops)
		looperInstance, err := looper.FactoryWithClientAndWorkflowState(&runtimeConfig, decision.Algorithm.Type, client, router.WorkflowStateService)
		require.NoError(t, err)

		streamingLooper, ok := looperInstance.(looper.StreamingLooper)
		require.True(t, ok)

		plan, err := streamingLooper.PrepareStreamingSynthesis(context.Background(), looperReq)
		require.NoError(t, err)

		// Verify immediate mounting before dispatch building
		reqCtx.LooperIntermediateUsage = plan.IntermediateUsage
		reqCtx.LooperModelsUsed = plan.ModelsUsed
		reqCtx.LooperIterations = plan.Iterations
		reqCtx.LooperTrace = plan.Trace
		reqCtx.LooperStreamingActive = true

		// Simulate panic during dispatch building (e.g. OOM or serialization panic)
		panic("simulated panic during streaming dispatch serialization")
	}()

	require.NotNil(t, recoveredPanic)
	require.Equal(t, "simulated panic during streaming dispatch serialization", recoveredPanic)
	require.True(t, reqCtx.LooperUsageSettled)
	require.True(t, reqCtx.StreamingAborted)

	// Verify replay record exists and holds intermediate token usage with LifecycleFailed and processor_panic
	record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "processor_panic", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 48, *record.PromptTokens)
	require.Equal(t, 24, *record.CompletionTokens)
	require.Equal(t, 72, *record.TotalTokens)
}

func TestLooperStreaming_PanicDuringResponseStreamProcessingSettlesTokens(t *testing.T) {
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
		RequestID:                 "req-stream-panic-response",
		RequestModel:              "client-model",
		VSRSelectedModel:          "judge",
		LooperTargetModel:         "judge",
		VSRSelectedDecisionName:   "fusion-stream",
		RouterReplayRecorder:      recorder,
		RouterReplayPluginConfig:  &replayConfig,
		IsStreamingResponse:       true,
		StreamingComplete:         false,
		LooperStreamingActive:     true,
		LooperAuditedOutputTokens: 10,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     45,
			CompletionTokens: 15,
			TotalTokens:      60,
		},
	}

	router.startRouterReplay(reqCtx, "client-model", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	// Simulate panic during downstream SSE response stream processing
	var recoveredPanic any
	func() {
		defer func() {
			if rec := recover(); rec != nil {
				recoveredPanic = rec
				if reqCtx != nil {
					router.settleAbortedLooperStreamingUsage(reqCtx, fmt.Errorf("processor_panic: %v", rec))
				}
				router.finalizeRouterReplay(reqCtx, routerreplay.LifecycleFailed, "processor_panic")
			}
		}()
		panic("downstream SSE chunk processing panic")
	}()

	require.NotNil(t, recoveredPanic)
	require.True(t, reqCtx.StreamingAborted)
	require.True(t, reqCtx.LooperUsageSettled)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "processor_panic", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 45, *record.PromptTokens)
	require.Equal(t, 25, *record.CompletionTokens) // 15 intermediate + 10 audited output tokens
	require.Equal(t, 70, *record.TotalTokens)
}

type looperStreamingPanicMockStream struct {
	MockStream
	panicOnResponseBody bool
	panicOnRequestBody  bool
	panicMsg            string
}

func (s *looperStreamingPanicMockStream) Send(resp *ext_proc.ProcessingResponse) error {
	if s.panicOnRequestBody && resp.GetRequestBody() != nil {
		panic(s.panicMsg)
	}
	if s.panicOnResponseBody && resp.GetResponseBody() != nil {
		panic(s.panicMsg)
	}
	return s.MockStream.Send(resp)
}

func TestLooperStreaming_ProcessRealEntrypointPanicSettlesIntermediateTokens(t *testing.T) {
	newFixture := func(t *testing.T) (*httptest.Server, *OpenAIRouter, *store.MemoryStore) {
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
			}}, map[string]int64{"prompt_tokens": 24, "completion_tokens": 12, "total_tokens": 36})
		}))

		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)

		decision := &config.Decision{
			Name: "fusion-stream-dispatch-panic",
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a", "panel-b"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}

		routerCfg := &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{
				Enabled: true,
			},
			Looper: config.LooperConfig{
				Endpoint: server.URL,
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"panel-a": {Capabilities: []string{"chat"}},
					"panel-b": {Capabilities: []string{"chat"}},
					"judge":   {Capabilities: []string{"chat"}},
				},
			},
			Recipes: []config.RoutingRecipe{
				{
					Name: config.DefaultRecipeName,
					Profile: config.RoutingProfile{
						Decisions: []config.Decision{*decision},
					},
				},
			},
		}

		classifier, err := classification.BuildClassifier(routerCfg, nil, nil, nil)
		require.NoError(t, err)

		router := &OpenAIRouter{
			Config:         routerCfg,
			Classifier:     classifier,
			ReplayRecorder: recorder,
		}

		return server, router, memStore
	}

	t.Run("panic during request body dispatch handle settles tokens", func(t *testing.T) {
		server, router, memStore := newFixture(t)
		defer server.Close()

		headersReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: "content-type", Value: "application/json"},
							{Key: "x-request-id", Value: "req-e2e-panic-dispatch"},
						},
					},
				},
			},
		}
		bodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body:        []byte(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello"}],"stream":true}`),
					EndOfStream: true,
				},
			},
		}

		stream := &looperStreamingPanicMockStream{
			MockStream: *NewMockStream([]*ext_proc.ProcessingRequest{
				headersReq,
				bodyReq,
			}),
			panicOnRequestBody: true,
			panicMsg:           "simulated dispatch phase panic",
		}

		// Call production Process entrypoint directly without manual recover
		err := router.Process(stream)
		require.Error(t, err)

		st, ok := status.FromError(err)
		require.True(t, ok)
		require.Equal(t, codes.Internal, st.Code())
		require.Contains(t, st.Message(), "simulated dispatch phase panic")

		// Assert Replay record captured intermediate tokens and marked LifecycleFailed with processor_panic
		records, err := memStore.List(context.Background())
		require.NoError(t, err)
		require.Len(t, records, 1)

		record := records[0]
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "processor_panic", record.TerminalReason)
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.Equal(t, 48, *record.PromptTokens)
		require.Equal(t, 24, *record.CompletionTokens)
		require.Equal(t, 72, *record.TotalTokens)
	})

	t.Run("panic during response body streaming handle settles tokens", func(t *testing.T) {
		server, router, memStore := newFixture(t)
		defer server.Close()

		headersReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: "content-type", Value: "application/json"},
							{Key: "x-request-id", Value: "req-e2e-panic-stream"},
						},
					},
				},
			},
		}
		bodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body:        []byte(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello"}],"stream":true}`),
					EndOfStream: true,
				},
			},
		}
		respHeadersReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: "content-type", Value: "text/event-stream"},
							{Key: ":status", Value: "200"},
						},
					},
				},
			},
		}
		respBodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        []byte("data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"model\":\"judge\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial answer\"}}]}\n\n"),
					EndOfStream: false,
				},
			},
		}

		stream := &looperStreamingPanicMockStream{
			MockStream: *NewMockStream([]*ext_proc.ProcessingRequest{
				headersReq,
				bodyReq,
				respHeadersReq,
				respBodyReq,
			}),
			panicOnResponseBody: true,
			panicMsg:            "simulated response chunk panic",
		}

		// Call production Process entrypoint directly without manual recover
		err := router.Process(stream)
		require.Error(t, err)

		st, ok := status.FromError(err)
		require.True(t, ok)
		require.Equal(t, codes.Internal, st.Code())
		require.Contains(t, st.Message(), "simulated response chunk panic")

		// Assert Replay record captured intermediate tokens plus estimated chunk tokens
		records, err := memStore.List(context.Background())
		require.NoError(t, err)
		require.Len(t, records, 1)

		record := records[0]
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "processor_panic", record.TerminalReason)
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.GreaterOrEqual(t, *record.PromptTokens, 48)
		require.GreaterOrEqual(t, *record.CompletionTokens, 24)
		require.Equal(t, *record.PromptTokens+*record.CompletionTokens, *record.TotalTokens)
	})
}

func TestLooperStreaming_QuorumFallbackResponsePhaseFinalization(t *testing.T) {
	newQuorumFallbackFixture := func(t *testing.T) (*httptest.Server, *OpenAIRouter, *RequestContext, *config.Decision) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))

		router, reqCtx, request := quorumFallbackRouter(server.URL)
		router.Config.ClearRouteCache = true
		reqCtx.ExpectStreamingResponse = true
		request.Stream = true

		decision := &config.Decision{
			Name: "fusion-quorum-fallback-delayed",
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
		require.Nil(t, response.GetImmediateResponse(), "fallback dispatch must continue to Envoy, not return immediate error")
		require.True(t, reqCtx.LooperStreamingActive)
		require.Equal(t, "backup-model", reqCtx.VSRSelectedModel)

		// At dispatch, Quorum outcome MUST be pending and NOT prematurely finalized as fallback_served
		require.NotNil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be staged during dispatch")
		require.Equal(t, looper.FusionQuorumFallbackReady, reqCtx.PendingLooperQuorumOutcome.Disposition)
		require.Nil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must remain unfinalized before response phase")

		return server, router, reqCtx, decision
	}

	t.Run("streaming response success finalizes fallback_served", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.IsStreamingResponse = true
		reqCtx.SemanticStreamState = &semanticResponseStreamState{
			responseID: "resp-stream-ok",
			model:      "backup-model",
			stop:       llmprotocol.StopEndTurn,
			terminal:   true,
			items: map[int]*semanticStreamItem{
				0: {completed: true, text: "successful fallback stream answer"},
			},
			order: []int{0},
			usage: llmprotocol.Usage{
				State: llmprotocol.UsageAvailable,
				InputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(20),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				OutputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(10),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				Total: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(30),
					Provenance: llmprotocol.UsageAuthoritative,
				},
			},
		}

		router.finalizeSemanticStreamingResponse(reqCtx, nil)

		require.False(t, reqCtx.StreamingAborted)
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after finalization")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated upon successful response")
		require.Equal(t, string(looper.FusionQuorumFallbackServed), reqCtx.VSRFusionQuorum.Disposition,
			"successful delivery must finalize to fallback_served")
		require.Equal(t, "backup-model", reqCtx.VSRFusionQuorum.FallbackTarget)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
		require.Equal(t, "response_complete", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "fallback_served", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("streaming response abort finalizes to cancelled instead of fallback_served", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.IsStreamingResponse = true
		router.settleAbortedLooperStreamingUsage(reqCtx, errors.New("client connection dropped"))
		router.finalizeRouterReplay(reqCtx, routerreplay.LifecycleAborted, "client_canceled")

		require.True(t, reqCtx.StreamingAborted)
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after abort")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on abort")
		require.NotEqual(t, string(looper.FusionQuorumFallbackServed), reqCtx.VSRFusionQuorum.Disposition,
			"aborted response must NOT be recorded as fallback_served")
		require.Equal(t, string(looper.FusionQuorumCancelled), reqCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "client_canceled", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "cancelled", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("client cancellation aligns replay aborted client_canceled and quorum cancelled", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.IsStreamingResponse = true
		require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))

		require.True(t, reqCtx.StreamingAborted)
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after client cancel")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on client cancel")
		require.Equal(t, string(looper.FusionQuorumCancelled), reqCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "client_canceled", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "cancelled", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("upstream 502 empty body eof aligns replay failed upstream_error_response and quorum response_encode_failed", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.IsStreamingResponse = true
		reqCtx.UpstreamStatusCode = http.StatusBadGateway
		require.NoError(t, router.handleProcessReceiveError(reqCtx, io.EOF))

		require.True(t, reqCtx.StreamingAborted)
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after upstream 502 eof")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on upstream 502 error")
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), reqCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "upstream_error_response", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("upstream stream missing done aligns replay failed upstream_stream_incomplete and quorum response_encode_failed", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.IsStreamingResponse = true
		reqCtx.SemanticStreamState = &semanticResponseStreamState{
			responseID: "resp-stream-incomplete",
			model:      "backup-model",
			terminal:   false,
			items: map[int]*semanticStreamItem{
				0: {completed: false, text: "partial text"},
			},
		}

		router.finalizeSemanticStreamingResponse(reqCtx, errors.New("stream_incomplete: missing [DONE]"))

		require.True(t, reqCtx.StreamingAborted)
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after stream incomplete")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on stream incomplete")
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), reqCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("internal panic aligns replay failed processor_panic and quorum response_encode_failed", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.IsStreamingResponse = true
		panicErr := fmt.Errorf("processor_panic: unexpected nil pointer")
		router.settleAbortedLooperStreamingUsage(reqCtx, panicErr)
		router.finalizeRouterReplay(reqCtx, routerreplay.LifecycleFailed, "processor_panic")

		require.True(t, reqCtx.StreamingAborted)
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after panic")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on panic")
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), reqCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "processor_panic", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("replay fusion quorum update failure logs structured error", func(t *testing.T) {
		coreObs, observedLogs := observer.New(zapcore.WarnLevel)
		restoreLogger := zap.ReplaceGlobals(zap.New(coreObs))
		defer restoreLogger()

		server, _, reqCtx, decision := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.RouterReplayID = "non-existent-replay-id"
		finalizeLooperQuorumOutcome(reqCtx, reqCtx.PendingLooperQuorumOutcome, decision, true)

		var found bool
		for _, entry := range observedLogs.All() {
			if entry.Message == "router_replay_fusion_quorum_update_failed" {
				found = true
				fields := entry.ContextMap()
				require.Equal(t, reqCtx.RequestID, fields["request_id"])
				require.Equal(t, "non-existent-replay-id", fields["replay_id"])
				require.NotEmpty(t, fields["error"])
				break
			}
		}
		require.True(t, found, "router_replay_fusion_quorum_update_failed error log must be emitted")
	})

	t.Run("upstream 502 transport error finalizes to response_encode_failed instead of fallback_served", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.UpstreamStatusCode = 502
		resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"bad gateway from upstream"}}`), reqCtx)
		require.NotNil(t, resp)

		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after transport error")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on upstream error")
		require.NotEqual(t, string(looper.FusionQuorumFallbackServed), reqCtx.VSRFusionQuorum.Disposition,
			"upstream 502 error must NOT be recorded as fallback_served")
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), reqCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("non-streaming 200 response finalizes fallback_served", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		body := []byte(`{
			"id": "chatcmpl-fallback-200",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "backup-model",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "non-streaming fallback answer"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
		}`)

		resp := router.handleNonStreamingResponseBody(body, reqCtx, 100*time.Millisecond)
		require.NotNil(t, resp)
		require.Nil(t, resp.GetImmediateResponse(), "valid 200 body must continue without immediate error")

		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after non-streaming completion")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on 200 completion")
		require.Equal(t, string(looper.FusionQuorumFallbackServed), reqCtx.VSRFusionQuorum.Disposition,
			"200 non-streaming completion must finalize to fallback_served")
	})

	t.Run("non-streaming decode error finalizes to response_encode_failed instead of fallback_served", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		resp := router.handleNonStreamingResponseBody([]byte("not valid json"), reqCtx, 100*time.Millisecond)
		require.NotNil(t, resp)
		require.NotNil(t, resp.GetImmediateResponse())

		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after decode error")
		require.NotNil(t, reqCtx.VSRFusionQuorum)
		require.NotEqual(t, string(looper.FusionQuorumFallbackServed), reqCtx.VSRFusionQuorum.Disposition,
			"decode error must NOT be recorded as fallback_served")
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), reqCtx.VSRFusionQuorum.Disposition)
	})

	t.Run("real sse missing done aligns replay failed upstream_stream_incomplete and quorum response_encode_failed", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.IsStreamingResponse = true
		sseChunk := []byte("data: {\"id\":\"chatcmpl-stream-nodone\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"backup-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial streaming fallback content\"},\"finish_reason\":null}]}\n\n")

		resp := router.handleSemanticStreamingResponseBody(sseChunk, true, reqCtx)
		require.NotNil(t, resp)

		require.True(t, reqCtx.StreamingAborted)
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after stream incomplete")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on stream incomplete")
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), reqCtx.VSRFusionQuorum.Disposition, "missing [DONE] must NEVER be classified as cancelled")
		require.NotEqual(t, string(looper.FusionQuorumCancelled), reqCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)

		require.True(t, reqCtx.LooperUsageSettled, "usage must be settled")
		require.Greater(t, reqCtx.LooperAuditedOutputTokens, 0, "partial streamed tokens must be audited/settled")
	})

	t.Run("real sse malformed chunk aligns replay failed and quorum response_encode_failed", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.IsStreamingResponse = true
		malformedChunk := []byte("data: {\"id\":\"chatcmpl-stream-broken\",invalid_json\n\n")

		resp := router.handleSemanticStreamingResponseBody(malformedChunk, true, reqCtx)
		require.NotNil(t, resp)

		require.True(t, reqCtx.StreamingAborted)
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after decode error")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on decode error")
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), reqCtx.VSRFusionQuorum.Disposition)
		require.NotEqual(t, string(looper.FusionQuorumCancelled), reqCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("real sse error event chunk aligns quorum response_encode_failed", func(t *testing.T) {
		server, router, reqCtx, _ := newQuorumFallbackFixture(t)
		defer server.Close()

		reqCtx.IsStreamingResponse = true
		errorChunk := []byte("data: {\"error\":{\"message\":\"upstream model overloaded\",\"type\":\"server_error\",\"code\":\"server_error\"}}\n\n")

		resp := router.handleSemanticStreamingResponseBody(errorChunk, true, reqCtx)
		require.NotNil(t, resp)

		require.True(t, reqCtx.StreamingAborted)
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after error event")
		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on error event")
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), reqCtx.VSRFusionQuorum.Disposition)
		require.NotEqual(t, string(looper.FusionQuorumCancelled), reqCtx.VSRFusionQuorum.Disposition)

		require.NotEmpty(t, reqCtx.RouterReplayID)
		record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("real sse downstream real cancellation aligns replay aborted client_canceled and quorum cancelled", func(t *testing.T) {
		t.Run("context cancelled", func(t *testing.T) {
			server, router, reqCtx, _ := newQuorumFallbackFixture(t)
			defer server.Close()

			reqCtx.IsStreamingResponse = true
			cancelCtx, cancel := context.WithCancel(context.Background())
			reqCtx.Context = cancelCtx
			cancel() // downstream client cancels

			chunk := []byte("data: {\"id\":\"chatcmpl-stream-cancel\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"backup-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"streamed before client disconnect\"},\"finish_reason\":null}]}\n\n")

			resp := router.handleSemanticStreamingResponseBody(chunk, true, reqCtx)
			require.NotNil(t, resp)

			require.True(t, reqCtx.StreamingAborted)
			require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after cancellation")
			require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on cancellation")
			require.Equal(t, string(looper.FusionQuorumCancelled), reqCtx.VSRFusionQuorum.Disposition)

			require.NotEmpty(t, reqCtx.RouterReplayID)
			record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
			require.True(t, ok)
			require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
			require.Equal(t, "client_canceled", record.TerminalReason)
			require.NotNil(t, record.RouteDiagnostics)
			require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
			require.Equal(t, "cancelled", record.RouteDiagnostics.FusionQuorum.Disposition)
		})

		t.Run("passed context.Canceled", func(t *testing.T) {
			server, router, reqCtx, _ := newQuorumFallbackFixture(t)
			defer server.Close()

			reqCtx.IsStreamingResponse = true
			router.settleAbortedLooperStreamingUsage(reqCtx, context.Canceled)
			router.finalizeRouterReplay(reqCtx, routerreplay.LifecycleAborted, "client_canceled")

			require.True(t, reqCtx.StreamingAborted)
			require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after cancellation")
			require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated on cancellation")
			require.Equal(t, string(looper.FusionQuorumCancelled), reqCtx.VSRFusionQuorum.Disposition)

			require.NotEmpty(t, reqCtx.RouterReplayID)
			record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
			require.True(t, ok)
			require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
			require.Equal(t, "client_canceled", record.TerminalReason)
			require.NotNil(t, record.RouteDiagnostics)
			require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
			require.Equal(t, "cancelled", record.RouteDiagnostics.FusionQuorum.Disposition)
		})
	})
}

func TestLooperStreaming_EnvoyMaxRetriesBudgetClamping(t *testing.T) {
	newServer := func() *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var reqPayload struct {
				Model string `json:"model"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&reqPayload))
			writeFusionReplayCompletion(w, reqPayload.Model, []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Panel answer from " + reqPayload.Model,
				},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
		}))
	}

	t.Run("ample budget (remaining 30) with model configured 1 retry does NOT amplify retries", func(t *testing.T) {
		server := newServer()
		defer server.Close()

		router, reqCtx, protoReq := quorumFallbackRouter(server.URL)
		router.Config.ClearRouteCache = true
		reqCtx.ExpectStreamingResponse = true
		protoReq.Stream = true

		modelCfg := router.Config.BackendModels.ModelConfig["judge"]
		modelCfg.Reliability = config.ProviderReliability{RetryCount: 1}
		router.Config.BackendModels.ModelConfig["judge"] = modelCfg

		decision := &config.Decision{
			Name: "fusion-stream-ample-budget",
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Parent ledger with limit 31: 1 call used by panel-a, 30 remaining
		execCtx, _ := budget.WithLimit(context.Background(), 31)
		streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.Nil(t, streamResp.GetImmediateResponse())
		require.True(t, reqCtx.LooperStreamingActive)
		require.Equal(t, 30, reqCtx.LooperRemainingBudget, "remaining budget must be recorded as 30")

		router.addEnvoyReliabilityHeaders(streamResp, reqCtx)

		headers := headerValuesByName(streamResp.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders())
		require.Equal(t, "1", headers["x-envoy-max-retries"],
			"x-envoy-max-retries must be strictly 1 and never amplified by remaining budget")
		require.NotEqual(t, "29", headers["x-envoy-max-retries"],
			"x-envoy-max-retries must NOT be amplified to budget - 1")
	})

	t.Run("ample budget (remaining 30) with unconfigured retries defaults to 0", func(t *testing.T) {
		server := newServer()
		defer server.Close()

		router, reqCtx, protoReq := quorumFallbackRouter(server.URL)
		router.Config.ClearRouteCache = true
		reqCtx.ExpectStreamingResponse = true
		protoReq.Stream = true

		decision := &config.Decision{
			Name: "fusion-stream-ample-budget-unconfigured-retry",
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Parent ledger with limit 31: 1 call used by panel-a, 30 remaining
		execCtx, _ := budget.WithLimit(context.Background(), 31)
		streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.Nil(t, streamResp.GetImmediateResponse())
		require.True(t, reqCtx.LooperStreamingActive)
		require.Equal(t, 30, reqCtx.LooperRemainingBudget, "remaining budget must be recorded as 30")

		router.addEnvoyReliabilityHeaders(streamResp, reqCtx)

		headers := headerValuesByName(streamResp.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders())
		require.Equal(t, "0", headers["x-envoy-max-retries"],
			"x-envoy-max-retries must be strictly 0 when no retries configured, never amplified to budget - 1")
		require.NotEqual(t, "29", headers["x-envoy-max-retries"])
	})

	t.Run("LooperRemainingBudget == 3 with 5 retries clamps retries to 2", func(t *testing.T) {
		five := 5
		cases := []struct {
			name          string
			modelRetries  int
			decisionRetry *int
		}{
			{
				name:          "decision_configured_5_retries",
				modelRetries:  0,
				decisionRetry: &five,
			},
			{
				name:          "model_configured_5_retries",
				modelRetries:  5,
				decisionRetry: nil,
			},
			{
				name:          "both_configured_5_retries",
				modelRetries:  5,
				decisionRetry: &five,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				server := newServer()
				defer server.Close()

				router, reqCtx, protoReq := quorumFallbackRouter(server.URL)
				router.Config.ClearRouteCache = true
				reqCtx.ExpectStreamingResponse = true
				protoReq.Stream = true

				if tc.modelRetries > 0 {
					modelCfg := router.Config.BackendModels.ModelConfig["judge"]
					modelCfg.Reliability = config.ProviderReliability{RetryCount: tc.modelRetries}
					router.Config.BackendModels.ModelConfig["judge"] = modelCfg
				}

				decision := &config.Decision{
					Name: "fusion-stream-budget-3",
					Algorithm: &config.AlgorithmConfig{
						Type: config.DecisionAlgorithmFusion,
						Fusion: &config.FusionAlgorithmConfig{
							Model:          "judge",
							AnalysisModels: []string{"panel-a"},
							AnalysisMode:   config.FusionAnalysisModeOneCall,
						},
					},
				}
				if tc.decisionRetry != nil {
					decision.Reliability = &config.DecisionReliability{
						RetryCount: tc.decisionRetry,
					}
				}
				reqCtx.VSRSelectedDecision = decision

				looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
				require.Nil(t, errResp)
				require.NotNil(t, looperReq)

				// Parent ledger with limit 4: 1 used by panel-a, 3 remaining
				execCtx, _ := budget.WithLimit(context.Background(), 4)
				streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
				require.NoError(t, err)
				require.NotNil(t, streamResp)
				require.Nil(t, streamResp.GetImmediateResponse())
				require.True(t, reqCtx.LooperStreamingActive)
				require.Equal(t, 3, reqCtx.LooperRemainingBudget, "remaining budget must be recorded as 3")

				router.addEnvoyReliabilityHeaders(streamResp, reqCtx)

				headers := headerValuesByName(streamResp.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders())
				require.Equal(t, "2", headers["x-envoy-max-retries"],
					"x-envoy-max-retries must be clamped to 2 when LooperRemainingBudget == 3 and 5 retries configured")
			})
		}
	})

	t.Run("LooperRemainingBudget == 1 clamps retries to 0 regardless of model or decision retry config", func(t *testing.T) {
		five := 5
		cases := []struct {
			name          string
			modelRetries  int
			decisionRetry *int
		}{
			{
				name:          "target_model_retries_only",
				modelRetries:  3,
				decisionRetry: nil,
			},
			{
				name:          "decision_retries_only",
				modelRetries:  0,
				decisionRetry: &five,
			},
			{
				name:          "both_model_and_decision_retries",
				modelRetries:  3,
				decisionRetry: &five,
			},
			{
				name:          "neither_configured",
				modelRetries:  0,
				decisionRetry: nil,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				server := newServer()
				defer server.Close()

				router, reqCtx, protoReq := quorumFallbackRouter(server.URL)
				router.Config.ClearRouteCache = true
				reqCtx.ExpectStreamingResponse = true
				protoReq.Stream = true

				if tc.modelRetries > 0 {
					modelCfg := router.Config.BackendModels.ModelConfig["judge"]
					modelCfg.Reliability = config.ProviderReliability{RetryCount: tc.modelRetries}
					router.Config.BackendModels.ModelConfig["judge"] = modelCfg
				}

				decision := &config.Decision{
					Name: "fusion-stream-budget-1",
					Algorithm: &config.AlgorithmConfig{
						Type: config.DecisionAlgorithmFusion,
						Fusion: &config.FusionAlgorithmConfig{
							Model:          "judge",
							AnalysisModels: []string{"panel-a"},
							AnalysisMode:   config.FusionAnalysisModeOneCall,
						},
					},
				}
				if tc.decisionRetry != nil {
					decision.Reliability = &config.DecisionReliability{
						RetryCount: tc.decisionRetry,
					}
				}
				reqCtx.VSRSelectedDecision = decision

				looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
				require.Nil(t, errResp)
				require.NotNil(t, looperReq)

				// Parent ledger with limit 2: 1 used by panel-a, 1 remaining
				execCtx, _ := budget.WithLimit(context.Background(), 2)
				streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
				require.NoError(t, err)
				require.NotNil(t, streamResp)
				require.Nil(t, streamResp.GetImmediateResponse())
				require.True(t, reqCtx.LooperStreamingActive)
				require.Equal(t, 1, reqCtx.LooperRemainingBudget, "remaining budget must be recorded as 1")

				router.addEnvoyReliabilityHeaders(streamResp, reqCtx)

				headers := headerValuesByName(streamResp.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders())
				require.Equal(t, "0", headers["x-envoy-max-retries"],
					"x-envoy-max-retries must be clamped to 0 when LooperRemainingBudget == 1")
			})
		}
	})
}

type looperRealPipelineMockStream struct {
	MockStream
	cancelAfter int
	cancelFn    context.CancelFunc
}

func (s *looperRealPipelineMockStream) Recv() (*ext_proc.ProcessingRequest, error) {
	if s.cancelAfter > 0 && s.RecvIndex >= s.cancelAfter {
		if s.cancelFn != nil {
			s.cancelFn()
		}
		return nil, context.Canceled
	}
	return s.MockStream.Recv()
}

func TestLooperStreaming_ProcessRealPipeline(t *testing.T) {
	newPipelineFixture := func(t *testing.T) (*httptest.Server, *OpenAIRouter, *store.MemoryStore) {
		panelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Model string `json:"model"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
			switch req.Model {
			case "panel-a":
				writeFusionReplayCompletion(w, req.Model, []map[string]interface{}{{
					"index":         0,
					"message":       map[string]interface{}{"role": "assistant", "content": "panel-a answer"},
					"finish_reason": "stop",
				}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
			case "panel-b":
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte("panel-b simulated failure"))
			default:
				w.WriteHeader(http.StatusInternalServerError)
			}
		}))

		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)

		decision := &config.Decision{
			Name: "fusion-quorum-real-pipeline",
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:                  "judge",
					AnalysisModels:         []string{"panel-a", "panel-b"},
					MaxConcurrent:          2,
					MinSuccessfulResponses: 2,
					OnError:                config.FusionOnErrorSkip,
					QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
					QuorumFallbackTarget:   "backup-model",
				},
			},
		}

		routerCfg := &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{
				Enabled: true,
			},
			Looper: config.LooperConfig{
				Endpoint: panelServer.URL,
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"panel-a":      {Capabilities: []string{"chat"}},
					"panel-b":      {Capabilities: []string{"chat"}},
					"judge":        {Capabilities: []string{"chat"}},
					"backup-model": {Capabilities: []string{"chat"}},
				},
			},
			Recipes: []config.RoutingRecipe{
				{
					Name: config.DefaultRecipeName,
					Profile: config.RoutingProfile{
						Decisions: []config.Decision{*decision},
					},
				},
			},
		}

		classifier, err := classification.BuildClassifier(routerCfg, nil, nil, nil)
		require.NoError(t, err)

		router := &OpenAIRouter{
			Config:         routerCfg,
			Classifier:     classifier,
			ReplayRecorder: recorder,
			ProtocolCodecs: protocolcodec.NewBuiltinRegistry(),
		}

		return panelServer, router, memStore
	}

	makeStandardRequests := func(requestID string) (*ext_proc.ProcessingRequest, *ext_proc.ProcessingRequest, *ext_proc.ProcessingRequest) {
		headersReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: ":path", Value: "/v1/chat/completions"},
							{Key: "content-type", Value: "application/json"},
							{Key: "x-request-id", Value: requestID},
						},
					},
				},
			},
		}
		bodyReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body:        []byte(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello"}],"stream":true}`),
					EndOfStream: true,
				},
			},
		}
		respHeadersReq := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: ":status", Value: "200"},
							{Key: "content-type", Value: "text/event-stream"},
						},
					},
				},
			},
		}
		return headersReq, bodyReq, respHeadersReq
	}

	t.Run("branch A: normal stream completion settles fallback_served and lifecycle completed", func(t *testing.T) {
		server, router, memStore := newPipelineFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := makeStandardRequests("req-e2e-pipeline-branch-a")
		chunk1Req := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        []byte("data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"model\":\"backup-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n"),
					EndOfStream: false,
				},
			},
		}
		chunk2Req := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body: []byte("data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"model\":\"backup-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n" +
						"data: [DONE]\n\n"),
					EndOfStream: true,
				},
			},
		}

		mockStream := &looperRealPipelineMockStream{
			MockStream: *NewMockStream([]*ext_proc.ProcessingRequest{
				headersReq,
				bodyReq,
				respHeadersReq,
				chunk1Req,
				chunk2Req,
			}),
		}

		err := router.Process(mockStream)
		require.NoError(t, err)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.LooperStreamingActive)
		require.Equal(t, "backup-model", capturedCtx.VSRSelectedModel)
		require.NotNil(t, capturedCtx.SemanticStreamState)
		require.Nil(t, capturedCtx.SemanticStreamState.failed)
		require.True(t, capturedCtx.SemanticStreamState.terminal)
		require.False(t, capturedCtx.StreamingAborted)

		// Quorum finalized to fallback_served
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumFallbackServed), capturedCtx.VSRFusionQuorum.Disposition)
		require.Equal(t, "backup-model", capturedCtx.VSRFusionQuorum.FallbackTarget)

		// Replay storage verification
		records, err := memStore.List(context.Background())
		require.NoError(t, err)
		require.Len(t, records, 1)

		record := records[0]
		require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
		require.Equal(t, "response_complete", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "fallback_served", record.RouteDiagnostics.FusionQuorum.Disposition)
		require.Equal(t, "backup-model", record.RouteDiagnostics.FusionQuorum.FallbackTarget)
		require.Equal(t, 2, record.RouteDiagnostics.FusionQuorum.RequiredCount)
		require.Equal(t, 1, record.RouteDiagnostics.FusionQuorum.UsableCount)

		// Authoritative usage: panel-a (10 prompt, 2 completion) + synthesis (10 prompt, 5 completion) = 20, 7, 27
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.Equal(t, 20, *record.PromptTokens)
		require.Equal(t, 7, *record.CompletionTokens)
		require.Equal(t, 27, *record.TotalTokens)
	})

	t.Run("branch B: upstream stream incomplete settles response_encode_failed and lifecycle failed", func(t *testing.T) {
		server, router, memStore := newPipelineFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := makeStandardRequests("req-e2e-pipeline-branch-b")
		// Upstream disconnects after partial chunk without [DONE]
		chunk1Req := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        []byte("data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"model\":\"backup-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n"),
					EndOfStream: true,
				},
			},
		}

		mockStream := &looperRealPipelineMockStream{
			MockStream: *NewMockStream([]*ext_proc.ProcessingRequest{
				headersReq,
				bodyReq,
				respHeadersReq,
				chunk1Req,
			}),
		}

		err := router.Process(mockStream)
		require.NoError(t, err)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.LooperStreamingActive)
		require.Equal(t, "backup-model", capturedCtx.VSRSelectedModel)
		require.NotNil(t, capturedCtx.SemanticStreamState)
		require.NotNil(t, capturedCtx.SemanticStreamState.failed, "real codec must register failure event in SemanticStreamState.failed")
		require.Equal(t, "stream_incomplete", capturedCtx.SemanticStreamState.failed.Code)
		require.Contains(t, capturedCtx.SemanticStreamState.failed.Error(), "stream_incomplete")
		require.True(t, capturedCtx.StreamingAborted)

		// Quorum must be response_encode_failed and strictly NOT cancelled
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), capturedCtx.VSRFusionQuorum.Disposition)
		require.NotEqual(t, string(looper.FusionQuorumCancelled), capturedCtx.VSRFusionQuorum.Disposition, "quorum must strictly NOT be cancelled on stream disconnect")
		require.Equal(t, "backup-model", capturedCtx.VSRFusionQuorum.FallbackTarget)

		// Replay storage verification: LifecycleFailed with upstream_stream_incomplete
		records, err := memStore.List(context.Background())
		require.NoError(t, err)
		require.Len(t, records, 1)

		record := records[0]
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "response_encode_failed", record.RouteDiagnostics.FusionQuorum.Disposition)
		require.Equal(t, "backup-model", record.RouteDiagnostics.FusionQuorum.FallbackTarget)
		require.Equal(t, 2, record.RouteDiagnostics.FusionQuorum.RequiredCount)
		require.Equal(t, 1, record.RouteDiagnostics.FusionQuorum.UsableCount)

		// Estimated usage: panel-a intermediate tokens (10 prompt, 2 completion) + estimated synthesis tokens
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.GreaterOrEqual(t, *record.PromptTokens, 10)
		require.GreaterOrEqual(t, *record.CompletionTokens, 3)
		require.Equal(t, *record.PromptTokens+*record.CompletionTokens, *record.TotalTokens)
	})

	t.Run("branch C: client cancellation settles quorum cancelled and lifecycle aborted", func(t *testing.T) {
		server, router, memStore := newPipelineFixture(t)
		defer server.Close()

		var capturedCtx *RequestContext
		router.OnRequestContext = func(ctx *RequestContext) {
			capturedCtx = ctx
		}

		headersReq, bodyReq, respHeadersReq := makeStandardRequests("req-e2e-pipeline-branch-c")
		chunk1Req := &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        []byte("data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"model\":\"backup-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n"),
					EndOfStream: false,
				},
			},
		}

		ctx, cancel := context.WithCancel(context.Background())
		mockStream := &looperRealPipelineMockStream{
			MockStream: *NewMockStream([]*ext_proc.ProcessingRequest{
				headersReq,
				bodyReq,
				respHeadersReq,
				chunk1Req,
			}),
			cancelAfter: 4,
			cancelFn:    cancel,
		}
		mockStream.Ctx = ctx

		err := router.Process(mockStream)
		require.NoError(t, err)

		require.NotNil(t, capturedCtx)
		require.True(t, capturedCtx.LooperStreamingActive)
		require.Equal(t, "backup-model", capturedCtx.VSRSelectedModel)
		require.True(t, capturedCtx.StreamingAborted)

		// Quorum must be cancelled
		require.NotNil(t, capturedCtx.VSRFusionQuorum)
		require.Equal(t, string(looper.FusionQuorumCancelled), capturedCtx.VSRFusionQuorum.Disposition)
		require.Equal(t, "backup-model", capturedCtx.VSRFusionQuorum.FallbackTarget)

		// Replay storage verification: LifecycleAborted with client_canceled
		records, err := memStore.List(context.Background())
		require.NoError(t, err)
		require.Len(t, records, 1)

		record := records[0]
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.Equal(t, "client_canceled", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, "cancelled", record.RouteDiagnostics.FusionQuorum.Disposition)
		require.Equal(t, "backup-model", record.RouteDiagnostics.FusionQuorum.FallbackTarget)
		require.Equal(t, 2, record.RouteDiagnostics.FusionQuorum.RequiredCount)
		require.Equal(t, 1, record.RouteDiagnostics.FusionQuorum.UsableCount)

		// Estimated usage: panel-a intermediate tokens (10 prompt, 2 completion) + estimated synthesis tokens
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.GreaterOrEqual(t, *record.PromptTokens, 10)
		require.GreaterOrEqual(t, *record.CompletionTokens, 3)
		require.Equal(t, *record.PromptTokens+*record.CompletionTokens, *record.TotalTokens)
	})
}

// TestEmbeddingContextPriorityContract verifies the precedence rules of embeddingContext():
// 1. TraceContext takes highest priority when present (inheriting its deadline/cancellation).
// 2. Context is used as fallback when TraceContext is nil.
// 3. context.Background() is returned when both are nil or ctx itself is nil.
func TestEmbeddingContextPriorityContract(t *testing.T) {
	// Case 1: ctx == nil returns context.Background()
	var nilCtx *RequestContext
	require.Equal(t, context.Background(), nilCtx.embeddingContext())

	// Case 2: ctx.Context is valid without deadline, but ctx.TraceContext has a timeout deadline.
	// embeddingContext() must prioritize TraceContext, carrying its deadline and triggering <-done on expiration.
	rawStreamCtx := context.Background()
	timeoutDur := 50 * time.Millisecond
	attemptCtx, cancelAttempt := context.WithTimeout(rawStreamCtx, timeoutDur)
	defer cancelAttempt()

	ctx := &RequestContext{
		Context:      rawStreamCtx,
		TraceContext: attemptCtx,
	}

	embCtx := ctx.embeddingContext()
	deadline, hasDeadline := embCtx.Deadline()
	require.True(t, hasDeadline, "returned context must have deadline")
	expectedDeadline, _ := attemptCtx.Deadline()
	require.Equal(t, expectedDeadline, deadline)

	// Wait for the phase deadline to trigger
	select {
	case <-embCtx.Done():
		require.ErrorIs(t, embCtx.Err(), context.DeadlineExceeded)
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for embCtx.Done()")
	}

	// Verify the original stream context was NOT expired or canceled
	require.NoError(t, rawStreamCtx.Err(), "raw stream context must remain active")

	// Case 3: ctx.TraceContext == nil and ctx.Context != nil -> fallback to ctx.Context
	fallbackCtx := &RequestContext{
		Context:      rawStreamCtx,
		TraceContext: nil,
	}
	require.Equal(t, rawStreamCtx, fallbackCtx.embeddingContext())

	// Case 4: both ctx.TraceContext and ctx.Context are nil -> returns context.Background()
	emptyCtx := &RequestContext{
		Context:      nil,
		TraceContext: nil,
	}
	require.Equal(t, context.Background(), emptyCtx.embeddingContext())
}

// TestEmbeddingContextFallbackPhaseRemoteDetectionDeadlineExceeded verifies that under a
// fallback attempt context (where TraceContext has a short phase deadline while the original
// gRPC Context is long-lived), remote detection relying on embeddingContext() accurately
// cancels when the phase deadline expires and returns deadline exceeded, rather than hanging
// on the unconstrained original stream context.
func TestEmbeddingContextFallbackPhaseRemoteDetectionDeadlineExceeded(t *testing.T) {
	// Mock remote hallucination detector that hangs longer than the attempt deadline
	stop := make(chan struct{})
	slowServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			// Client timed out / disconnected; abort promptly
			return
		case <-stop:
			return
		case <-time.After(5 * time.Second):
			// If not canceled, return a valid detection response
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"choices": []map[string]interface{}{
					{
						"message": map[string]string{
							"content": `{"hallucinated_spans":[]}`,
						},
					},
				},
			})
		}
	}))
	t.Cleanup(func() {
		close(stop)
		slowServer.Close()
	})

	router, reqCtx := newHallucinationSignalRouter(t, slowServer, "header")

	// Simulate gRPC stream initialization: long-lived Context
	rawStreamCtx := context.Background()

	// Simulate response fallback candidate attempt: short phase deadline mounted on TraceContext
	phaseTimeout := 60 * time.Millisecond
	attemptCtx, cancelAttempt := context.WithTimeout(rawStreamCtx, phaseTimeout)
	defer cancelAttempt()

	reqCtx.Context = rawStreamCtx
	reqCtx.TraceContext = attemptCtx

	// 1. Direct detection call via classifier using ctx.embeddingContext()
	start := time.Now()
	_, err := router.Classifier.DetectHallucination(
		reqCtx.embeddingContext(),
		reqCtx.ToolResultsContext,
		reqCtx.UserContent,
		hallucinationAnswer,
	)
	elapsed := time.Since(start)
	t.Logf("DetectHallucination elapsed: %v", elapsed)

	require.Error(t, err, "remote detection must fail when phase deadline expires")
	isDeadlineExceeded := errors.Is(err, context.DeadlineExceeded) ||
		strings.Contains(err.Error(), "deadline exceeded") ||
		strings.Contains(err.Error(), "context deadline exceeded")
	require.Truef(t, isDeadlineExceeded, "expected deadline exceeded error, got: %v", err)
	require.Less(t, elapsed, 1*time.Second, "remote detection must abort within attempt deadline, not hang on raw stream context")
	require.NoError(t, rawStreamCtx.Err(), "raw stream context must remain valid and unexpired")

	// 2. Direct detection with explanations via classifier using ctx.embeddingContext()
	attemptCtx2, cancelAttempt2 := context.WithTimeout(rawStreamCtx, phaseTimeout)
	defer cancelAttempt2()
	reqCtx.TraceContext = attemptCtx2

	start = time.Now()
	_, err2 := router.Classifier.DetectHallucinationWithExplanations(
		reqCtx.embeddingContext(),
		reqCtx.ToolResultsContext,
		reqCtx.UserContent,
		hallucinationAnswer,
	)
	elapsed2 := time.Since(start)
	t.Logf("DetectHallucinationWithExplanations elapsed: %v", elapsed2)

	require.Error(t, err2, "remote detection with explanations must fail when phase deadline expires")
	isDeadlineExceeded2 := errors.Is(err2, context.DeadlineExceeded) ||
		strings.Contains(err2.Error(), "deadline exceeded") ||
		strings.Contains(err2.Error(), "context deadline exceeded")
	require.Truef(t, isDeadlineExceeded2, "expected deadline exceeded error, got: %v", err2)
	require.Less(t, elapsed2, 1*time.Second, "remote detection must abort within attempt deadline")

	// 3. Evidence detection helper via router using ctx.embeddingContext()
	attemptCtx3, cancelAttempt3 := context.WithTimeout(rawStreamCtx, phaseTimeout)
	defer cancelAttempt3()
	reqCtx.TraceContext = attemptCtx3

	start = time.Now()
	evidence, err3 := router.detectHallucinationEvidence(
		reqCtx.embeddingContext(),
		router.Classifier,
		reqCtx,
		hallucinationAnswer,
	)
	elapsed3 := time.Since(start)
	t.Logf("detectHallucinationEvidence elapsed: %v", elapsed3)

	require.Error(t, err3, "evidence detection must fail when phase deadline expires")
	require.Nil(t, evidence, "no evidence should be returned on deadline exceeded")
	isDeadlineExceeded3 := errors.Is(err3, context.DeadlineExceeded) ||
		strings.Contains(err3.Error(), "deadline exceeded") ||
		strings.Contains(err3.Error(), "context deadline exceeded")
	require.Truef(t, isDeadlineExceeded3, "expected deadline exceeded error, got: %v", err3)
	require.Less(t, elapsed3, 1*time.Second, "evidence detection must abort within attempt deadline")
}
