package extproc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

func TestNativeReliabilityBudget(t *testing.T) {
	five := 5
	one := 1

	routerWithModels := func(models map[string]int) *OpenAIRouter {
		cfg := map[string]config.ModelParams{}
		for m, retries := range models {
			cfg[m] = config.ModelParams{
				Reliability: config.ProviderReliability{
					RetryCount: retries,
				},
			}
		}
		return &OpenAIRouter{
			Config: &config.RouterConfig{
				BackendModels: config.BackendModels{
					ModelConfig: cfg,
				},
			},
		}
	}

	routerWithModelRetry := func(model string, retries int) *OpenAIRouter {
		return routerWithModels(map[string]int{model: retries})
	}

	t.Run("TestCase1_LooperRemainingBudget1_ClampsRetriesStrictlyToZero", func(t *testing.T) {
		cases := []struct {
			name          string
			modelRetries  int
			decisionRetry *int
		}{
			{
				name:          "model_configured_retries_only",
				modelRetries:  5,
				decisionRetry: nil,
			},
			{
				name:          "decision_configured_retries_only",
				modelRetries:  0,
				decisionRetry: &five,
			},
			{
				name:          "both_model_and_decision_configured_retries",
				modelRetries:  3,
				decisionRetry: &five,
			},
			{
				name:          "neither_configured_retries",
				modelRetries:  0,
				decisionRetry: nil,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				router := routerWithModelRetry("judge", tc.modelRetries)
				var decision *config.Decision
				if tc.decisionRetry != nil {
					decision = &config.Decision{
						Name: "stream-decision",
						Reliability: &config.DecisionReliability{
							RetryCount:   tc.decisionRetry,
							TotalTimeout: "10s",
						},
					}
				} else {
					decision = &config.Decision{
						Name: "stream-decision-no-rel",
					}
				}

				session := &routingSession{
					router: router,
					ctx: &RequestContext{
						VSRSelectedDecision:   decision,
						VSRSelectedModel:      "judge",
						LooperTargetModel:     "judge",
						LooperStreamingActive: true,
						LooperRemainingBudget: 1,
					},
				}

				got := session.Reliability()
				require.NotNil(t, got, "Reliability must return non-nil override when LooperStreamingActive is true")
				require.NotNil(t, got.RetryCount, "RetryCount must be set to enforce physical budget limit")
				require.Equal(t, 0, *got.RetryCount, "RetryCount must be strictly 0 when LooperRemainingBudget == 1")
			})
		}
	})

	t.Run("TestCase2_LooperRemainingBudget3_ModelRetries5_ClampsToTwo", func(t *testing.T) {
		router := routerWithModelRetry("judge", 5)
		session := &routingSession{
			router: router,
			ctx: &RequestContext{
				VSRSelectedModel:      "judge",
				LooperTargetModel:     "judge",
				LooperStreamingActive: true,
				LooperRemainingBudget: 3,
			},
		}

		got := session.Reliability()
		require.NotNil(t, got)
		require.NotNil(t, got.RetryCount)
		require.Equal(t, 2, *got.RetryCount, "1 initial attempt + 2 retries = 3 total budget calls")
	})

	t.Run("TestCase3_DecisionWithoutReliabilityOverride_ClampsFromModelParams", func(t *testing.T) {
		budgetTestCases := []struct {
			name            string
			remainingBudget int
			modelRetries    int
			wantRetries     int
		}{
			{
				name:            "remaining_budget_3_model_5_clamps_to_2",
				remainingBudget: 3,
				modelRetries:    5,
				wantRetries:     2,
			},
			{
				name:            "remaining_budget_1_model_5_clamps_to_0",
				remainingBudget: 1,
				modelRetries:    5,
				wantRetries:     0,
			},
			{
				name:            "remaining_budget_10_model_5_caps_at_configured_5",
				remainingBudget: 10,
				modelRetries:    5,
				wantRetries:     5,
			},
			{
				name:            "remaining_budget_0_clamps_to_0",
				remainingBudget: 0,
				modelRetries:    5,
				wantRetries:     0,
			},
		}

		for _, tc := range budgetTestCases {
			t.Run(tc.name, func(t *testing.T) {
				router := routerWithModelRetry("judge", tc.modelRetries)
				decision := &config.Decision{
					Name: "no-reliability-decision",
				}
				session := &routingSession{
					router: router,
					ctx: &RequestContext{
						VSRSelectedDecision:   decision,
						VSRSelectedModel:      "judge",
						LooperTargetModel:     "judge",
						LooperStreamingActive: true,
						LooperRemainingBudget: tc.remainingBudget,
					},
				}

				got := session.Reliability()
				require.NotNil(t, got)
				require.NotNil(t, got.RetryCount)
				require.Equal(t, tc.wantRetries, *got.RetryCount)
			})
		}
	})

	t.Run("DecisionOverridePrecedenceOverModelConfig", func(t *testing.T) {
		router := routerWithModelRetry("judge", 5)
		decision := &config.Decision{
			Name: "decision-with-1-retry",
			Reliability: &config.DecisionReliability{
				RetryCount: &one,
			},
		}
		session := &routingSession{
			router: router,
			ctx: &RequestContext{
				VSRSelectedDecision:   decision,
				VSRSelectedModel:      "judge",
				LooperTargetModel:     "judge",
				LooperStreamingActive: true,
				LooperRemainingBudget: 4, // max allowed is 3, but decision configured 1
			},
		}

		got := session.Reliability()
		require.NotNil(t, got)
		require.NotNil(t, got.RetryCount)
		require.Equal(t, 1, *got.RetryCount, "decision override of 1 retry must take precedence and not be amplified to 3")
	})

	t.Run("PreservesOtherDecisionReliabilityFields", func(t *testing.T) {
		router := routerWithModelRetry("judge", 5)
		decision := &config.Decision{
			Name: "decision-with-timeouts",
			Reliability: &config.DecisionReliability{
				TotalTimeout:         "30s",
				PerTryTimeout:        "5s",
				FirstByteTimeout:     "2s",
				RetryCount:           &five,
				RetryOn:              "5xx,reset",
				RetriableStatusCodes: []int{502, 503},
			},
		}
		session := &routingSession{
			router: router,
			ctx: &RequestContext{
				VSRSelectedDecision:   decision,
				VSRSelectedModel:      "judge",
				LooperTargetModel:     "judge",
				LooperStreamingActive: true,
				LooperRemainingBudget: 3,
			},
		}

		got := session.Reliability()
		require.NotNil(t, got)
		require.NotNil(t, got.RetryCount)
		require.Equal(t, 2, *got.RetryCount, "RetryCount must be clamped to 2")
		require.Equal(t, 30*time.Second, *got.TotalTimeout)
		require.Equal(t, 5*time.Second, *got.PerTryTimeout)
		require.Equal(t, 2*time.Second, *got.FirstByteTimeout)
		require.Equal(t, []string{"5xx", "reset"}, got.RetryOn)
		require.Equal(t, []int{502, 503}, got.RetriableStatusCodes)
	})

	t.Run("NonStreamingRequestsAreNotClamped", func(t *testing.T) {
		router := routerWithModelRetry("judge", 5)

		// Case A: Decision with no reliability override returns nil in non-streaming mode
		sessionA := &routingSession{
			router: router,
			ctx: &RequestContext{
				VSRSelectedModel:      "judge",
				LooperStreamingActive: false,
			},
		}
		require.Nil(t, sessionA.Reliability(), "non-streaming request without decision reliability returns nil")

		// Case B: Decision with reliability override returns original override unclamped
		sessionB := &routingSession{
			router: router,
			ctx: &RequestContext{
				VSRSelectedDecision: &config.Decision{
					Reliability: &config.DecisionReliability{
						RetryCount: &five,
					},
				},
				VSRSelectedModel:      "judge",
				LooperStreamingActive: false,
				LooperRemainingBudget: 1, // should be ignored when LooperStreamingActive is false
			},
		}
		gotB := sessionB.Reliability()
		require.NotNil(t, gotB)
		require.Equal(t, 5, *gotB.RetryCount, "non-streaming request preserves configured retry count")
	})

	t.Run("NativeGatewayEnginePlanAttachesClampedReliability", func(t *testing.T) {
		router := routerWithModelRetry("judge", 5)
		session := &routingSession{
			router: router,
			ctx: &RequestContext{
				VSRSelectedModel:      "judge",
				LooperTargetModel:     "judge",
				LooperStreamingActive: true,
				LooperRemainingBudget: 3,
			},
		}

		override := session.Reliability()
		require.NotNil(t, override)
		require.NotNil(t, override.RetryCount)
		require.Equal(t, 2, *override.RetryCount)

		// Verify plan call reliability assignment contract as performed by routing.Engine
		call := &routing.Call{
			Route:       "judge",
			Reliability: []*routing.Reliability{override},
		}
		require.Len(t, call.Reliability, 1)
		require.Equal(t, 2, *call.Reliability[0].RetryCount)
	})

	t.Run("ClientHeaderModelDoesNotAmplifySynthesisTargetRetryCount", func(t *testing.T) {
		t.Run("judge_configured_zero_retries", func(t *testing.T) {
			router := routerWithModels(map[string]int{
				"model-a": 5,
				"judge":   0,
			})
			session := &routingSession{
				router: router,
				clientHeader: routing.Header{
					{Name: routing.RouteHeader, Value: "model-a"},
				},
				ctx: &RequestContext{
					LooperStreamingActive: true,
					LooperTargetModel:     "judge",
					LooperRemainingBudget: 3,
				},
			}

			got := session.Reliability()
			require.NotNil(t, got, "Reliability must return non-nil override during looper streaming")
			require.NotNil(t, got.RetryCount, "RetryCount must be set")
			require.Equal(t, 0, *got.RetryCount, "RetryCount must be strictly 0, not amplified by clientHeader model-a")
		})

		t.Run("judge_unconfigured_retries", func(t *testing.T) {
			router := routerWithModels(map[string]int{
				"model-a": 5,
			})
			session := &routingSession{
				router: router,
				clientHeader: routing.Header{
					{Name: routing.RouteHeader, Value: "model-a"},
				},
				ctx: &RequestContext{
					LooperStreamingActive: true,
					LooperTargetModel:     "judge",
					LooperRemainingBudget: 3,
				},
			}

			got := session.Reliability()
			require.NotNil(t, got, "Reliability must return non-nil override during looper streaming")
			require.NotNil(t, got.RetryCount, "RetryCount must be set")
			require.Equal(t, 0, *got.RetryCount, "RetryCount must be strictly 0 when judge is not configured")
		})
	})

	t.Run("ClientHeaderModelDoesNotSuppressSynthesisTargetRetryCount", func(t *testing.T) {
		router := routerWithModels(map[string]int{
			"model-a": 0,
			"judge":   5,
		})
		session := &routingSession{
			router: router,
			clientHeader: routing.Header{
				{Name: routing.RouteHeader, Value: "model-a"},
			},
			ctx: &RequestContext{
				LooperStreamingActive: true,
				LooperTargetModel:     "judge",
				LooperRemainingBudget: 3,
			},
		}

		got := session.Reliability()
		require.NotNil(t, got)
		require.NotNil(t, got.RetryCount)
		require.Equal(t, 2, *got.RetryCount, "RetryCount must identify judge (5 retries clamped to budget 2), not suppressed to 0 by clientHeader model-a")
	})

	t.Run("NativeGatewayEnginePlanWithClientHeaderBindsSynthesisTargetReliability", func(t *testing.T) {
		t.Run("judge_target_clamped_to_budget", func(t *testing.T) {
			router := routerWithModels(map[string]int{
				"model-a": 0,
				"judge":   5,
			})
			rs := &routingSession{
				router: router,
				ctx: &RequestContext{
					LooperStreamingActive: true,
					LooperTargetModel:     "judge",
					LooperRemainingBudget: 3,
				},
			}
			session := &budgetEngineTestSession{
				routingSession: rs,
				targetModel:    "judge",
			}

			engine := routing.NewEngine(&budgetTestProcessor{session: session}, routing.DefaultOptions)
			plan, err := engine.Plan(context.Background(), &routing.Request{
				Header: routing.Header{
					{Name: ":method", Value: "POST"},
					{Name: routing.RouteHeader, Value: "model-a"},
				},
				Body: []byte(`{"model":"model-a"}`),
			})
			require.NoError(t, err)
			require.NotNil(t, plan.Call)
			require.Equal(t, "judge", plan.Call.Route)
			require.Len(t, plan.Call.Reliability, 1)
			require.NotNil(t, plan.Call.Reliability[0].RetryCount)
			require.Equal(t, 2, *plan.Call.Reliability[0].RetryCount, "plan Call must bind clamped retries for actual judge target")
		})

		t.Run("judge_zero_retries_not_amplified_by_client_header", func(t *testing.T) {
			router := routerWithModels(map[string]int{
				"model-a": 5,
				"judge":   0,
			})
			rs := &routingSession{
				router: router,
				ctx: &RequestContext{
					LooperStreamingActive: true,
					LooperTargetModel:     "judge",
					LooperRemainingBudget: 3,
				},
			}
			session := &budgetEngineTestSession{
				routingSession: rs,
				targetModel:    "judge",
			}

			engine := routing.NewEngine(&budgetTestProcessor{session: session}, routing.DefaultOptions)
			plan, err := engine.Plan(context.Background(), &routing.Request{
				Header: routing.Header{
					{Name: ":method", Value: "POST"},
					{Name: routing.RouteHeader, Value: "model-a"},
				},
				Body: []byte(`{"model":"model-a"}`),
			})
			require.NoError(t, err)
			require.NotNil(t, plan.Call)
			require.Equal(t, "judge", plan.Call.Route)
			require.Len(t, plan.Call.Reliability, 1)
			require.NotNil(t, plan.Call.Reliability[0].RetryCount)
			require.Equal(t, 0, *plan.Call.Reliability[0].RetryCount, "plan Call must remain 0 retries and not be amplified by client header model-a")
		})
	})

	t.Run("DecisionRetryConditionsWithoutExplicitRetryCount", func(t *testing.T) {
		zero := 0

		envoyHeaderMap := func(router *OpenAIRouter, ctx *RequestContext, model string) map[string]string {
			mutation := &ext_proc.HeaderMutation{
				SetHeaders: []*core.HeaderValueOption{
					{
						Header: &core.HeaderValue{Key: routing.RouteHeader, RawValue: []byte(model)},
					},
				},
			}
			buffered := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							HeaderMutation: mutation,
						},
					},
				},
			}
			router.addEnvoyReliabilityHeaders(buffered, ctx)
			out := map[string]string{}
			for _, opt := range mutation.GetSetHeaders() {
				out[opt.GetHeader().GetKey()] = string(opt.GetHeader().GetRawValue())
			}
			return out
		}

		t.Run("budget_2_decision_retry_on_5xx_no_base_policy_allows_1_retry", func(t *testing.T) {
			router := routerWithModels(map[string]int{"judge": 0})
			decision := &config.Decision{
				Name: "judge-retry-5xx",
				Reliability: &config.DecisionReliability{
					RetryOn: "5xx",
				},
			}
			ctx := &RequestContext{
				VSRSelectedDecision:   decision,
				VSRSelectedModel:      "judge",
				LooperTargetModel:     "judge",
				LooperStreamingActive: true,
				LooperRemainingBudget: 2,
			}
			rs := &routingSession{
				router: router,
				ctx:    ctx,
			}

			// Native check
			got := rs.Reliability()
			require.NotNil(t, got)
			require.NotNil(t, got.RetryCount)
			require.Equal(t, 1, *got.RetryCount, "budget 2 allows 1 retry for conditions-only decision without base retry policy")
			require.Equal(t, []string{"5xx"}, got.RetryOn)

			// Envoy check
			headers := envoyHeaderMap(router, ctx, "judge")
			require.Equal(t, "1", headers[envoyMaxRetriesHeader])
			require.Equal(t, "5xx", headers[envoyRetryOnHeader])
		})

		t.Run("budget_1_decision_retry_on_5xx_strictly_clamped_to_zero", func(t *testing.T) {
			router := routerWithModels(map[string]int{"judge": 0})
			decision := &config.Decision{
				Name: "judge-retry-5xx",
				Reliability: &config.DecisionReliability{
					RetryOn: "5xx",
				},
			}
			ctx := &RequestContext{
				VSRSelectedDecision:   decision,
				VSRSelectedModel:      "judge",
				LooperTargetModel:     "judge",
				LooperStreamingActive: true,
				LooperRemainingBudget: 1,
			}
			rs := &routingSession{
				router: router,
				ctx:    ctx,
			}

			// Native check
			got := rs.Reliability()
			require.NotNil(t, got)
			require.NotNil(t, got.RetryCount)
			require.Equal(t, 0, *got.RetryCount, "budget 1 must strictly clamp retries to 0")

			// Envoy check
			headers := envoyHeaderMap(router, ctx, "judge")
			require.Equal(t, "0", headers[envoyMaxRetriesHeader])
		})

		t.Run("decision_explicit_retry_count_zero_strictly_zero_even_with_retry_on", func(t *testing.T) {
			router := routerWithModels(map[string]int{"judge": 0})
			decision := &config.Decision{
				Name: "judge-explicit-zero",
				Reliability: &config.DecisionReliability{
					RetryCount: &zero,
					RetryOn:    "5xx",
				},
			}
			ctx := &RequestContext{
				VSRSelectedDecision:   decision,
				VSRSelectedModel:      "judge",
				LooperTargetModel:     "judge",
				LooperStreamingActive: true,
				LooperRemainingBudget: 5,
			}
			rs := &routingSession{
				router: router,
				ctx:    ctx,
			}

			// Native check
			got := rs.Reliability()
			require.NotNil(t, got)
			require.NotNil(t, got.RetryCount)
			require.Equal(t, 0, *got.RetryCount, "explicit retry_count 0 must strictly override even with remaining budget")

			// Envoy check
			headers := envoyHeaderMap(router, ctx, "judge")
			require.Equal(t, "0", headers[envoyMaxRetriesHeader])
		})

		t.Run("target_model_zero_retries_and_decision_no_retry_conditions_strictly_zero", func(t *testing.T) {
			router := routerWithModels(map[string]int{"judge": 0})
			decision := &config.Decision{
				Name: "judge-no-retry-conditions",
				Reliability: &config.DecisionReliability{
					TotalTimeout: "10s",
				},
			}
			ctx := &RequestContext{
				VSRSelectedDecision:   decision,
				VSRSelectedModel:      "judge",
				LooperTargetModel:     "judge",
				LooperStreamingActive: true,
				LooperRemainingBudget: 5,
			}
			rs := &routingSession{
				router: router,
				ctx:    ctx,
			}

			// Native check
			got := rs.Reliability()
			require.NotNil(t, got)
			require.NotNil(t, got.RetryCount)
			require.Equal(t, 0, *got.RetryCount, "model configured 0 and decision without retry conditions must remain 0")

			// Envoy check
			headers := envoyHeaderMap(router, ctx, "judge")
			require.Equal(t, "0", headers[envoyMaxRetriesHeader])
		})

		t.Run("target_model_per_try_timeout_only_base_zero_retries_remains_zero", func(t *testing.T) {
			router := &OpenAIRouter{
				Config: &config.RouterConfig{
					BackendModels: config.BackendModels{
						ModelConfig: map[string]config.ModelParams{
							"judge": {
								Reliability: config.ProviderReliability{
									PerTryTimeout: "5s",
									RetryCount:    0,
								},
							},
						},
					},
				},
			}
			decision := &config.Decision{
				Name: "judge-conditions",
				Reliability: &config.DecisionReliability{
					RetryOn: "5xx",
				},
			}
			ctx := &RequestContext{
				VSRSelectedDecision:   decision,
				VSRSelectedModel:      "judge",
				LooperTargetModel:     "judge",
				LooperStreamingActive: true,
				LooperRemainingBudget: 3,
			}
			rs := &routingSession{
				router: router,
				ctx:    ctx,
			}

			got := rs.Reliability()
			require.NotNil(t, got)
			require.NotNil(t, got.RetryCount)
			require.Equal(t, 0, *got.RetryCount, "provider with per_try_timeout rendered zero-retry policy must stay 0")

			headers := envoyHeaderMap(router, ctx, "judge")
			require.Equal(t, "0", headers[envoyMaxRetriesHeader])
		})
	})

	t.Run("EnginePlanToUpstreamExecuteSimulation_Budget2_Judge503ThenSSE200", func(t *testing.T) {
		t.Run("budget_2_retries_once_and_succeeds_with_2_physical_requests", func(t *testing.T) {
			var attempts atomic.Int32
			backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				att := attempts.Add(1)
				if att == 1 {
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = w.Write([]byte(`{"error":"temporarily unavailable"}`))
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello from judge\"}}]}\n\ndata: [DONE]\n\n"))
			}))
			defer backendServer.Close()

			u, err := url.Parse(backendServer.URL)
			require.NoError(t, err)
			port, err := strconv.Atoi(u.Port())
			require.NoError(t, err)

			topology := upstream.Topology{
				Clusters: []upstream.ClusterSpec{
					{
						Name:     "judge",
						LBPolicy: upstream.LBRoundRobin,
						Endpoints: []upstream.EndpointSpec{
							{Name: "judge-endpoint", Scheme: "http", Host: u.Hostname(), Port: port, Weight: 1},
						},
					},
				},
				DefaultCluster: "judge",
			}
			upstreamSet, err := upstream.New(topology, upstream.Options{})
			require.NoError(t, err)
			defer func() { _ = upstreamSet.Close(context.Background()) }()

			router := &OpenAIRouter{
				Config: &config.RouterConfig{
					BackendModels: config.BackendModels{
						ModelConfig: map[string]config.ModelParams{
							"judge": {},
						},
					},
				},
			}
			decision := &config.Decision{
				Name: "fusion-judge-decision",
				Reliability: &config.DecisionReliability{
					RetryOn: "5xx",
				},
			}
			rs := &routingSession{
				router: router,
				ctx: &RequestContext{
					VSRSelectedDecision:   decision,
					VSRSelectedModel:      "judge",
					LooperTargetModel:     "judge",
					LooperStreamingActive: true,
					LooperRemainingBudget: 2,
				},
			}
			session := &budgetEngineTestSession{
				routingSession: rs,
				targetModel:    "judge",
			}

			engine := routing.NewEngine(&budgetTestProcessor{session: session}, routing.DefaultOptions)
			plan, err := engine.Plan(context.Background(), &routing.Request{
				Header: routing.Header{
					{Name: ":method", Value: "POST"},
					{Name: ":path", Value: "/v1/chat/completions"},
					{Name: routing.RouteHeader, Value: "initial-model"},
				},
				Body: []byte(`{"model":"initial-model","stream":true}`),
			})
			require.NoError(t, err)
			require.NotNil(t, plan.Call)
			require.Equal(t, "judge", plan.Call.Route)
			require.Len(t, plan.Call.Reliability, 1)
			require.NotNil(t, plan.Call.Reliability[0].RetryCount)
			require.Equal(t, 1, *plan.Call.Reliability[0].RetryCount, "plan Call must have clamped effective retry count of 1")

			result, err := upstreamSet.Execute(context.Background(), plan.Call, "")
			require.NoError(t, err)
			require.NotNil(t, result)
			require.NotNil(t, result.Response)
			require.Equal(t, http.StatusOK, result.Response.StatusCode, "response must succeed on retry")

			bodyBytes, err := io.ReadAll(result.Response.Body)
			require.NoError(t, err)
			_ = result.Response.Body.Close()
			require.Contains(t, string(bodyBytes), "hello from judge")
			require.Contains(t, string(bodyBytes), "data: [DONE]")

			require.Equal(t, int32(2), attempts.Load(), "physical requests must be sent exactly twice (1 initial + 1 retry)")
		})

		t.Run("budget_1_no_retry_on_5xx_strictly_1_physical_request", func(t *testing.T) {
			var attempts atomic.Int32
			backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempts.Add(1)
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte(`{"error":"temporarily unavailable"}`))
			}))
			defer backendServer.Close()

			u, err := url.Parse(backendServer.URL)
			require.NoError(t, err)
			port, err := strconv.Atoi(u.Port())
			require.NoError(t, err)

			topology := upstream.Topology{
				Clusters: []upstream.ClusterSpec{
					{
						Name:     "judge",
						LBPolicy: upstream.LBRoundRobin,
						Endpoints: []upstream.EndpointSpec{
							{Name: "judge-endpoint", Scheme: "http", Host: u.Hostname(), Port: port, Weight: 1},
						},
					},
				},
				DefaultCluster: "judge",
			}
			upstreamSet, err := upstream.New(topology, upstream.Options{})
			require.NoError(t, err)
			defer func() { _ = upstreamSet.Close(context.Background()) }()

			router := &OpenAIRouter{
				Config: &config.RouterConfig{
					BackendModels: config.BackendModels{
						ModelConfig: map[string]config.ModelParams{
							"judge": {},
						},
					},
				},
			}
			decision := &config.Decision{
				Name: "fusion-judge-decision",
				Reliability: &config.DecisionReliability{
					RetryOn: "5xx",
				},
			}
			rs := &routingSession{
				router: router,
				ctx: &RequestContext{
					VSRSelectedDecision:   decision,
					VSRSelectedModel:      "judge",
					LooperTargetModel:     "judge",
					LooperStreamingActive: true,
					LooperRemainingBudget: 1, // Only 1 budget -> 0 retries
				},
			}
			session := &budgetEngineTestSession{
				routingSession: rs,
				targetModel:    "judge",
			}

			engine := routing.NewEngine(&budgetTestProcessor{session: session}, routing.DefaultOptions)
			plan, err := engine.Plan(context.Background(), &routing.Request{
				Header: routing.Header{
					{Name: ":method", Value: "POST"},
					{Name: ":path", Value: "/v1/chat/completions"},
					{Name: routing.RouteHeader, Value: "initial-model"},
				},
				Body: []byte(`{"model":"initial-model","stream":true}`),
			})
			require.NoError(t, err)
			require.NotNil(t, plan.Call)
			require.Equal(t, 0, *plan.Call.Reliability[0].RetryCount)

			result, err := upstreamSet.Execute(context.Background(), plan.Call, "")
			require.NoError(t, err)
			require.NotNil(t, result)
			require.NotNil(t, result.Response)
			require.Equal(t, http.StatusServiceUnavailable, result.Response.StatusCode)
			_ = result.Response.Body.Close()

			require.Equal(t, int32(1), attempts.Load(), "physical request must be sent exactly once when budget is 1")
		})
	})
}

var (
	_ routing.ReliabilitySession = (*budgetEngineTestSession)(nil)
	_ routing.Processor          = (*budgetTestProcessor)(nil)
)

type budgetTestProcessor struct {
	session routing.Session
}

func (p *budgetTestProcessor) Open(context.Context) (routing.Session, error) {
	return p.session, nil
}

type budgetEngineTestSession struct {
	*routingSession
	targetModel string
}

func (s *budgetEngineTestSession) RequestHeaders(header routing.Header, endOfStream bool) (*routing.Effect, error) {
	s.clientHeader = header.Clone()
	return &routing.Effect{}, nil
}

func (s *budgetEngineTestSession) RequestBody(body []byte, endOfStream bool) (*routing.Effect, error) {
	return &routing.Effect{
		Header: &routing.HeaderMutation{
			Set: []routing.HeaderOption{
				{Name: routing.RouteHeader, Value: s.targetModel},
			},
		},
		ClearRouteCache: true,
	}, nil
}

func (s *budgetEngineTestSession) ResponseHeaders(header routing.Header, endOfStream bool) (*routing.Effect, error) {
	return &routing.Effect{}, nil
}

func (s *budgetEngineTestSession) ResponseBody(body []byte, endOfStream bool) (*routing.Effect, error) {
	return &routing.Effect{}, nil
}

func (s *budgetEngineTestSession) Evidence() routing.Evidence {
	return routing.Evidence{}
}

func (s *budgetEngineTestSession) Close(err error) {}

type remotePIITestDetector struct {
	client *connector.Client
}

func (d *remotePIITestDetector) ClassifyPIIWithDetailsAndThreshold(
	ctx context.Context, text string, threshold float32,
) ([]classification.PIIDetection, error) {
	op := connector.Operation{
		Name:      "token_spans",
		Method:    http.MethodPost,
		Path:      "/classify",
		RetrySafe: true,
	}
	body, _ := json.Marshal(map[string]string{"inputs": text})
	res, err := d.client.Do(ctx, op, body)
	if err != nil {
		return nil, err
	}
	var spans []struct {
		EntityGroup string  `json:"entity_group"`
		Start       int     `json:"start"`
		End         int     `json:"end"`
		Score       float32 `json:"score"`
		Word        string  `json:"word"`
	}
	if err := json.Unmarshal(res, &spans); err != nil {
		return nil, err
	}
	var detections []classification.PIIDetection
	for _, s := range spans {
		detections = append(detections, classification.PIIDetection{
			EntityType: s.EntityGroup,
			Start:      s.Start,
			End:        s.End,
			Text:       s.Word,
			Confidence: s.Score,
		})
	}
	return detections, nil
}

func TestLooperStreaming_RemoteMaskingBudgetInheritance(t *testing.T) {
	five := 5

	setupServersAndRouter := func(t *testing.T) (*atomic.Int32, *atomic.Int32, *atomic.Int32, *OpenAIRouter, *RequestContext, *llmprotocol.Request) {
		t.Helper()
		var panelCalls, judgeCalls, piiCalls atomic.Int32

		upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var payload struct {
				Model string `json:"model"`
			}
			_ = json.NewDecoder(r.Body).Decode(&payload)
			switch payload.Model {
			case "panel-a":
				panelCalls.Add(1)
				writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Panel analysis output from panel-a",
					},
					"finish_reason": "stop",
				}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
			case "judge":
				judgeCalls.Add(1)
				writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Judge final response",
					},
					"finish_reason": "stop",
				}}, map[string]int64{"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
			default:
				w.WriteHeader(http.StatusNotFound)
			}
		}))
		t.Cleanup(upstreamServer.Close)

		piiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			piiCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]map[string]interface{}{{
				"entity_group": "EMAIL_ADDRESS",
				"start":        12,
				"end":          28,
				"score":        0.99,
				"word":         "alice@example.com",
			}})
		}))
		t.Cleanup(piiServer.Close)

		client, err := connector.New(piiServer.URL, nil, connector.Options{
			AttemptTimeout:   5 * time.Second,
			MaxRequestBytes:  1024 * 1024,
			MaxResponseBytes: 1024 * 1024,
			MaxErrorBytes:    1024 * 1024,
		})
		require.NoError(t, err)
		useMaskingDetector(t, &remotePIITestDetector{client: client})

		router, reqCtx, protoReq := quorumFallbackRouter(upstreamServer.URL)
		router.Config.ClearRouteCache = true
		reqCtx.ExpectStreamingResponse = true
		protoReq.Stream = true
		protoReq.Messages = []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "email me at alice@example.com please"}},
		}}

		modelCfg := router.Config.BackendModels.ModelConfig["judge"]
		modelCfg.Reliability = config.ProviderReliability{RetryCount: 5}
		router.Config.BackendModels.ModelConfig["judge"] = modelCfg

		return &panelCalls, &judgeCalls, &piiCalls, router, reqCtx, protoReq
	}

	newDecision := func() *config.Decision {
		return &config.Decision{
			Name: "fusion-stream-remote-masking",
			Plugins: []config.DecisionPlugin{{
				Type:          config.DecisionPluginMasking,
				Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
			}},
			Reliability: &config.DecisionReliability{
				RetryCount: &five,
			},
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}
	}

	t.Run("remotemasking_exhausts_last_budget_intercepts_dispatch", func(t *testing.T) {
		panelCalls, judgeCalls, piiCalls, router, reqCtx, protoReq := setupServersAndRouter(t)
		decision := newDecision()
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Budget limit = 3: 1 used by panel-a, 2 consumed by remote PII detector (1 per message).
		// Remaining budget becomes 0 (< 1), so final judge dispatch must be intercepted!
		execCtx, ledger := budget.WithLimit(context.Background(), 3)
		streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.NotNil(t, streamResp.GetImmediateResponse(), "must return immediate error response")

		immediate := streamResp.GetImmediateResponse()
		require.Equal(t, 400, int(immediate.GetStatus().GetCode()), "status code must be 400")
		require.Contains(t, string(immediate.GetBody()), "budget_exhausted")

		require.Equal(t, int32(1), panelCalls.Load(), "panel-a must be called exactly once")
		require.Equal(t, int32(2), piiCalls.Load(), "remote PII detector must be called for each message and consume budget")
		require.Equal(t, int32(0), judgeCalls.Load(), "upstream judge must receive ZERO dispatches")

		require.Equal(t, 3, ledger.Used(), "ledger must reflect panel and masking calls")
		require.Equal(t, 0, ledger.Remaining(), "ledger remaining must be 0")

		require.Nil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must remain nil when budget is exhausted after quorum was achieved")
		require.Equal(t, "looper_budget_exhausted", reqCtx.TerminalLifecycleReason)
		record, found := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, "looper_budget_exhausted", record.TerminalReason)
		if record.RouteDiagnostics != nil {
			require.Nil(t, record.RouteDiagnostics.FusionQuorum, "replay record FusionQuorum must remain nil")
		}
	})

	t.Run("remotemasking_ample_budget_clamps_retries", func(t *testing.T) {
		panelCalls, judgeCalls, piiCalls, router, reqCtx, protoReq := setupServersAndRouter(t)
		decision := newDecision()
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Budget limit = 4: 1 used by panel-a, 2 consumed by remote PII detector.
		// Remaining budget becomes 1 (>= 1), so final judge dispatch proceeds, but retries clamped to 0.
		execCtx, ledger := budget.WithLimit(context.Background(), 4)
		streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.Nil(t, streamResp.GetImmediateResponse(), "dispatch should succeed without immediate error")

		require.Equal(t, int32(1), panelCalls.Load(), "panel-a must be called once")
		require.Equal(t, int32(2), piiCalls.Load(), "remote PII detector must be called twice (for 2 messages)")
		require.Equal(t, int32(0), judgeCalls.Load(), "upstream judge not yet executed by envoy")

		require.Equal(t, 3, ledger.Used())
		require.Equal(t, 1, ledger.Remaining())
		require.Equal(t, 1, reqCtx.LooperRemainingBudget, "LooperRemainingBudget must be updated to 1 after masking")

		router.addEnvoyReliabilityHeaders(streamResp, reqCtx)
		headers := headerValuesByName(streamResp.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders())
		require.Equal(t, "0", headers["x-envoy-max-retries"],
			"x-envoy-max-retries must be clamped to 0 when remaining budget is 1")
	})

	t.Run("remotemasking_sufficient_budget_for_retries", func(t *testing.T) {
		panelCalls, judgeCalls, piiCalls, router, reqCtx, protoReq := setupServersAndRouter(t)
		decision := newDecision()
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Budget limit = 5: 1 used by panel-a, 2 consumed by remote PII detector.
		// Remaining budget becomes 2 (>= 1), so final judge dispatch proceeds and retries clamped to 1.
		execCtx, ledger := budget.WithLimit(context.Background(), 5)
		streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.Nil(t, streamResp.GetImmediateResponse())

		require.Equal(t, int32(1), panelCalls.Load())
		require.Equal(t, int32(2), piiCalls.Load())
		require.Equal(t, int32(0), judgeCalls.Load())

		require.Equal(t, 3, ledger.Used())
		require.Equal(t, 2, ledger.Remaining())
		require.Equal(t, 2, reqCtx.LooperRemainingBudget, "LooperRemainingBudget must be updated to 2 after masking")

		router.addEnvoyReliabilityHeaders(streamResp, reqCtx)
		headers := headerValuesByName(streamResp.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders())
		require.Equal(t, "1", headers["x-envoy-max-retries"],
			"x-envoy-max-retries must be clamped to 1 when remaining budget is 2")
	})

	t.Run("remotemasking_exceeds_budget_during_scan_intercepts", func(t *testing.T) {
		panelCalls, judgeCalls, piiCalls, router, reqCtx, protoReq := setupServersAndRouter(t)
		decision := newDecision()
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Budget limit = 2: 1 used by panel-a.
		// PrepareStreamingSynthesis succeeds with 1 call remaining.
		// During masking, message 0 consumes the 1 remaining call.
		// Then message 1 attempts to consume and gets budget.ErrExhausted!
		execCtx, ledger := budget.WithLimit(context.Background(), 2)
		streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.NotNil(t, streamResp.GetImmediateResponse())

		immediate := streamResp.GetImmediateResponse()
		require.Equal(t, 400, int(immediate.GetStatus().GetCode()), "status code must be 400")
		require.Contains(t, string(immediate.GetBody()), "budget_exhausted")

		require.Equal(t, int32(1), panelCalls.Load(), "panel-a consumed 1 call")
		require.Equal(t, int32(1), piiCalls.Load(), "message 0 PII consumed 1 call, message 1 PII blocked by budget")
		require.Equal(t, int32(0), judgeCalls.Load(), "upstream judge received 0 calls")

		require.Equal(t, 2, ledger.Used())
		require.Equal(t, 0, ledger.Remaining())

		require.Nil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must remain nil when budget is exhausted after quorum was achieved")
		require.Equal(t, "looper_budget_exhausted", reqCtx.TerminalLifecycleReason)
		record, found := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, "looper_budget_exhausted", record.TerminalReason)
		if record.RouteDiagnostics != nil {
			require.Nil(t, record.RouteDiagnostics.FusionQuorum, "replay record FusionQuorum must remain nil")
		}
	})

	t.Run("intermediate_prepare_exhausts_budget_intercepts", func(t *testing.T) {
		panelCalls, judgeCalls, piiCalls, router, reqCtx, protoReq := setupServersAndRouter(t)
		decision := newDecision()
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Budget limit = 1: 1 used by panel-a, intermediate prepare exhausts budget.
		execCtx, ledger := budget.WithLimit(context.Background(), 1)
		streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.NotNil(t, streamResp.GetImmediateResponse())

		immediate := streamResp.GetImmediateResponse()
		require.Equal(t, 400, int(immediate.GetStatus().GetCode()), "status code must be 400")
		require.Contains(t, string(immediate.GetBody()), "budget_exhausted")

		require.Equal(t, int32(1), panelCalls.Load(), "panel-a consumed the 1 allowed call")
		require.Equal(t, int32(0), piiCalls.Load(), "remote PII detector was never called")
		require.Equal(t, int32(0), judgeCalls.Load(), "upstream judge received 0 calls")

		require.Equal(t, 1, ledger.Used())
		require.Equal(t, 0, ledger.Remaining())

		require.Nil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must remain nil when budget is exhausted after quorum was achieved")
		require.Equal(t, "looper_budget_exhausted", reqCtx.TerminalLifecycleReason)
		record, found := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, "looper_budget_exhausted", record.TerminalReason)
		if record.RouteDiagnostics != nil {
			require.Nil(t, record.RouteDiagnostics.FusionQuorum, "replay record FusionQuorum must remain nil")
		}
	})

	t.Run("quorum_fallback_exhausts_budget_records_quorum_budget_exhausted", func(t *testing.T) {
		panelCalls, judgeCalls, piiCalls, router, reqCtx, protoReq := setupServersAndRouter(t)
		decision := &config.Decision{
			Name: "fusion-stream-quorum-fallback-budget-exhausted",
			Plugins: []config.DecisionPlugin{{
				Type:          config.DecisionPluginMasking,
				Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
			}},
			Reliability: &config.DecisionReliability{
				RetryCount: &five,
			},
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:                  "judge",
					AnalysisModels:         []string{"panel-a", "panel-b"},
					MinSuccessfulResponses: 2,
					OnError:                config.FusionOnErrorSkip,
					QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
					QuorumFallbackTarget:   "backup-model",
					AnalysisMode:           config.FusionAnalysisModeOneCall,
				},
			},
		}
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Budget limit = 2: 1 used by panel-a (succeeds), 1 used by panel-b (returns 404).
		// Panel quorum (min 2) fails, so fallback plan to "backup-model" is prepared carrying
		// FusionQuorumFallbackReady.
		// Remaining budget is now 0 (< 1), so remote masking gets budget.ErrExhausted during dispatch.
		// Since plan.QuorumOutcome was already populated from the real below-quorum outcome,
		// budget exhaustion must update its disposition to FusionQuorumBudgetExhausted and finalize it.
		execCtx, ledger := budget.WithLimit(context.Background(), 2)
		streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.NotNil(t, streamResp.GetImmediateResponse())

		immediate := streamResp.GetImmediateResponse()
		require.Equal(t, 400, int(immediate.GetStatus().GetCode()), "status code must be 400")
		require.Contains(t, string(immediate.GetBody()), "budget_exhausted")

		require.Equal(t, int32(1), panelCalls.Load(), "panel-a consumed 1 call")
		require.Equal(t, int32(0), piiCalls.Load(), "PII was blocked by budget exhaustion")
		require.Equal(t, int32(0), judgeCalls.Load(), "upstream judge received 0 calls")

		require.Equal(t, 2, ledger.Used())
		require.Equal(t, 0, ledger.Remaining())

		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated for true quorum failure")
		require.Equal(t, string(looper.FusionQuorumBudgetExhausted), reqCtx.VSRFusionQuorum.Disposition)
		require.Equal(t, 2, reqCtx.VSRFusionQuorum.RequiredCount)
		require.Equal(t, 1, reqCtx.VSRFusionQuorum.UsableCount)
		require.Equal(t, "backup-model", reqCtx.VSRFusionQuorum.FallbackTarget)

		require.Equal(t, "looper_budget_exhausted", reqCtx.TerminalLifecycleReason)
		record, found := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, "looper_budget_exhausted", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, string(looper.FusionQuorumBudgetExhausted), record.RouteDiagnostics.FusionQuorum.Disposition)
	})

	t.Run("quorum_fallback_post_masking_exhausts_budget_records_quorum_budget_exhausted", func(t *testing.T) {
		panelCalls, judgeCalls, piiCalls, router, reqCtx, protoReq := setupServersAndRouter(t)
		decision := &config.Decision{
			Name: "fusion-stream-quorum-fallback-post-masking-budget-exhausted",
			Plugins: []config.DecisionPlugin{{
				Type:          config.DecisionPluginMasking,
				Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
			}},
			Reliability: &config.DecisionReliability{
				RetryCount: &five,
			},
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:                  "judge",
					AnalysisModels:         []string{"panel-a", "panel-b"},
					MinSuccessfulResponses: 2,
					OnError:                config.FusionOnErrorSkip,
					QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
					QuorumFallbackTarget:   "backup-model",
					AnalysisMode:           config.FusionAnalysisModeOneCall,
				},
			},
		}
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		// Budget limit = 3: 1 used by panel-a, 1 by panel-b.
		// Fallback plan prepared carrying FusionQuorumFallbackReady.
		// 1 remaining budget is consumed by remote PII detector for the fallback request.
		// Post-masking remaining budget is 0 (< 1), intercepting dispatch before judge.
		// plan.QuorumOutcome must be updated to FusionQuorumBudgetExhausted.
		execCtx, ledger := budget.WithLimit(context.Background(), 3)
		streamResp, err := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		require.NoError(t, err)
		require.NotNil(t, streamResp)
		require.NotNil(t, streamResp.GetImmediateResponse())

		immediate := streamResp.GetImmediateResponse()
		require.Equal(t, 400, int(immediate.GetStatus().GetCode()), "status code must be 400")
		require.Contains(t, string(immediate.GetBody()), "budget_exhausted")

		require.Equal(t, int32(1), panelCalls.Load(), "panel-a consumed 1 call")
		require.Equal(t, int32(1), piiCalls.Load(), "PII consumed 1 call")
		require.Equal(t, int32(0), judgeCalls.Load(), "upstream judge received 0 calls")

		require.Equal(t, 3, ledger.Used())
		require.Equal(t, 0, ledger.Remaining())

		require.NotNil(t, reqCtx.VSRFusionQuorum, "VSRFusionQuorum must be populated for true quorum failure")
		require.Equal(t, string(looper.FusionQuorumBudgetExhausted), reqCtx.VSRFusionQuorum.Disposition)
		require.Equal(t, 2, reqCtx.VSRFusionQuorum.RequiredCount)
		require.Equal(t, 1, reqCtx.VSRFusionQuorum.UsableCount)
		require.Equal(t, "backup-model", reqCtx.VSRFusionQuorum.FallbackTarget)

		require.Equal(t, "looper_budget_exhausted", reqCtx.TerminalLifecycleReason)
		record, found := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, "looper_budget_exhausted", record.TerminalReason)
		require.NotNil(t, record.RouteDiagnostics)
		require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
		require.Equal(t, string(looper.FusionQuorumBudgetExhausted), record.RouteDiagnostics.FusionQuorum.Disposition)
	})
}

func TestLooperStreaming_RemoteMaskingCancelAndTimeout(t *testing.T) {
	five := 5

	newDecision := func() *config.Decision {
		return &config.Decision{
			Name: "fusion-stream-remote-masking",
			Plugins: []config.DecisionPlugin{{
				Type:          config.DecisionPluginMasking,
				Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
			}},
			Reliability: &config.DecisionReliability{
				RetryCount: &five,
			},
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"panel-a"},
					AnalysisMode:   config.FusionAnalysisModeOneCall,
				},
			},
		}
	}

	setupRouterAndContext := func(t *testing.T, upstreamURL string) (*OpenAIRouter, *RequestContext, *llmprotocol.Request, *looper.Request) {
		t.Helper()
		router, reqCtx, protoReq := quorumFallbackRouter(upstreamURL)
		router.Config.ClearRouteCache = true
		reqCtx.ExpectStreamingResponse = true
		protoReq.Stream = true
		protoReq.Messages = []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "email me at alice@example.com please"}},
		}}

		modelCfg := router.Config.BackendModels.ModelConfig["judge"]
		modelCfg.Reliability = config.ProviderReliability{RetryCount: 5}
		router.Config.BackendModels.ModelConfig["judge"] = modelCfg

		decision := newDecision()
		reqCtx.VSRSelectedDecision = decision

		looperReq, errResp := router.buildLooperRequest(protoReq, decision, reqCtx)
		require.Nil(t, errResp)
		require.NotNil(t, looperReq)

		return router, reqCtx, protoReq, looperReq
	}

	t.Run("test_case_a_cancel_during_pii_masking_returns_500_aborted", func(t *testing.T) {
		var panelCalls, judgeCalls, piiCalls atomic.Int32
		upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var payload struct {
				Model string `json:"model"`
			}
			_ = json.NewDecoder(r.Body).Decode(&payload)
			switch payload.Model {
			case "panel-a":
				panelCalls.Add(1)
				writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Panel analysis output from panel-a",
					},
					"finish_reason": "stop",
				}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
			case "judge":
				judgeCalls.Add(1)
				writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Judge final response",
					},
					"finish_reason": "stop",
				}}, map[string]int64{"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
			default:
				w.WriteHeader(http.StatusNotFound)
			}
		}))
		defer upstreamServer.Close()

		piiUnblock := make(chan struct{})
		piiStarted := make(chan struct{}, 1)

		piiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			piiCalls.Add(1)
			select {
			case piiStarted <- struct{}{}:
			default:
			}
			select {
			case <-r.Context().Done():
			case <-piiUnblock:
			}
		}))
		defer func() {
			close(piiUnblock)
			piiServer.Close()
		}()

		client, err := connector.New(piiServer.URL, nil, connector.Options{
			AttemptTimeout:   5 * time.Second,
			MaxRetries:       0,
			MaxRequestBytes:  1024 * 1024,
			MaxResponseBytes: 1024 * 1024,
			MaxErrorBytes:    1024 * 1024,
		})
		require.NoError(t, err)
		useMaskingDetector(t, &remotePIITestDetector{client: client})

		router, reqCtx, protoReq, looperReq := setupRouterAndContext(t, upstreamServer.URL)
		decision := newDecision()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		reqCtx.TraceContext = ctx

		execCtx, _ := budget.WithLimit(ctx, 10)

		done := make(chan struct{})
		var streamResp *ext_proc.ProcessingResponse
		var dispatchErr error

		go func() {
			defer close(done)
			streamResp, dispatchErr = router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)
		}()

		select {
		case <-piiStarted:
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for PII request to start")
		}

		// Cancel client context during remote PII masking
		cancel()

		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for handleLooperStreamingDispatch to return")
		}

		require.NoError(t, dispatchErr)
		require.NotNil(t, streamResp)
		immediate := streamResp.GetImmediateResponse()
		require.NotNil(t, immediate, "must return immediate error response")
		require.Equal(t, 500, int(immediate.GetStatus().GetCode()), "must return HTTP 500 on client cancellation")

		record, found := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, 500, record.ResponseStatus)
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState, "lifecycle must be LifecycleAborted")
		require.Equal(t, "client_cancelled_during_looper_dispatch", record.TerminalReason, "terminal reason must be client_cancelled_during_looper_dispatch, not masking failure")

		require.Equal(t, int32(1), panelCalls.Load(), "panel-a must be called once")
		require.Equal(t, int32(1), piiCalls.Load(), "remote PII detector must be called once")
		require.Equal(t, int32(0), judgeCalls.Load(), "upstream judge must receive ZERO dispatches")
	})

	t.Run("test_case_b_deadline_exceeded_during_pii_masking_returns_504", func(t *testing.T) {
		var panelCalls, judgeCalls, piiCalls atomic.Int32
		upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var payload struct {
				Model string `json:"model"`
			}
			_ = json.NewDecoder(r.Body).Decode(&payload)
			switch payload.Model {
			case "panel-a":
				panelCalls.Add(1)
				writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Panel analysis output from panel-a",
					},
					"finish_reason": "stop",
				}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
			case "judge":
				judgeCalls.Add(1)
				writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Judge final response",
					},
					"finish_reason": "stop",
				}}, map[string]int64{"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
			default:
				w.WriteHeader(http.StatusNotFound)
			}
		}))
		defer upstreamServer.Close()

		piiUnblock := make(chan struct{})
		piiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			piiCalls.Add(1)
			select {
			case <-r.Context().Done():
			case <-piiUnblock:
			}
		}))
		defer func() {
			close(piiUnblock)
			piiServer.Close()
		}()

		client, err := connector.New(piiServer.URL, nil, connector.Options{
			AttemptTimeout:   2 * time.Second,
			MaxRetries:       0,
			MaxRequestBytes:  1024 * 1024,
			MaxResponseBytes: 1024 * 1024,
			MaxErrorBytes:    1024 * 1024,
		})
		require.NoError(t, err)
		useMaskingDetector(t, &remotePIITestDetector{client: client})

		router, reqCtx, protoReq, looperReq := setupRouterAndContext(t, upstreamServer.URL)
		decision := newDecision()

		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		reqCtx.TraceContext = ctx

		execCtx, _ := budget.WithLimit(ctx, 10)

		streamResp, dispatchErr := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)

		require.NoError(t, dispatchErr)
		require.NotNil(t, streamResp)
		immediate := streamResp.GetImmediateResponse()
		require.NotNil(t, immediate, "must return immediate error response")
		require.Equal(t, 504, int(immediate.GetStatus().GetCode()), "must return HTTP 504 on deadline exceeded")

		record, found := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, 504, record.ResponseStatus)
		require.Equal(t, "looper_stream_dispatch_deadline_exceeded", record.TerminalReason, "terminal reason must be looper_stream_dispatch_deadline_exceeded, not masking failure")

		require.Equal(t, int32(1), panelCalls.Load(), "panel-a must be called once")
		require.Equal(t, int32(1), piiCalls.Load(), "remote PII detector must be called once")
		require.Equal(t, int32(0), judgeCalls.Load(), "upstream judge must receive ZERO dispatches")
	})

	t.Run("test_case_c_genuine_pii_unavailable_error_returns_503", func(t *testing.T) {
		var panelCalls, judgeCalls, piiCalls atomic.Int32
		upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var payload struct {
				Model string `json:"model"`
			}
			_ = json.NewDecoder(r.Body).Decode(&payload)
			switch payload.Model {
			case "panel-a":
				panelCalls.Add(1)
				writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Panel analysis output from panel-a",
					},
					"finish_reason": "stop",
				}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
			case "judge":
				judgeCalls.Add(1)
				writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Judge final response",
					},
					"finish_reason": "stop",
				}}, map[string]int64{"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
			default:
				w.WriteHeader(http.StatusNotFound)
			}
		}))
		defer upstreamServer.Close()

		piiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			piiCalls.Add(1)
			http.Error(w, "PII service temporarily unavailable", http.StatusServiceUnavailable)
		}))
		defer piiServer.Close()

		client, err := connector.New(piiServer.URL, nil, connector.Options{
			AttemptTimeout:   5 * time.Second,
			MaxRetries:       0,
			MaxRequestBytes:  1024 * 1024,
			MaxResponseBytes: 1024 * 1024,
			MaxErrorBytes:    1024 * 1024,
		})
		require.NoError(t, err)
		useMaskingDetector(t, &remotePIITestDetector{client: client})

		router, reqCtx, protoReq, looperReq := setupRouterAndContext(t, upstreamServer.URL)
		decision := newDecision()

		execCtx, _ := budget.WithLimit(context.Background(), 10)

		streamResp, dispatchErr := router.handleLooperStreamingDispatch(execCtx, looperReq, decision, protoReq.Model, reqCtx)

		require.NoError(t, dispatchErr)
		require.NotNil(t, streamResp)
		immediate := streamResp.GetImmediateResponse()
		require.NotNil(t, immediate, "must return immediate error response")
		require.Equal(t, 503, int(immediate.GetStatus().GetCode()), "genuine PII service failure must return 503")

		record, found := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, found, "replay record must exist")
		require.Equal(t, 503, record.ResponseStatus)
		require.Equal(t, "looper_stream_dispatch_masking_unavailable", record.TerminalReason, "terminal reason must be looper_stream_dispatch_masking_unavailable")

		require.Equal(t, int32(1), panelCalls.Load(), "panel-a must be called once")
		require.Equal(t, int32(1), piiCalls.Load(), "remote PII detector must be called once")
		require.Equal(t, int32(0), judgeCalls.Load(), "upstream judge must receive ZERO dispatches")
	})
}
