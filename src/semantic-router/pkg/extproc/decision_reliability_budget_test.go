package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
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
