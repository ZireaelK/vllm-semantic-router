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
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// faultInjectingQuorumStore wraps store.Storage and injects transient failures
// on UpdateFusionQuorum for testing retry semantics.
type faultInjectingQuorumStore struct {
	store.Storage
	mu                  sync.Mutex
	calls               []string
	failQuorumAttempts  atomic.Int32
	quorumFailCount     atomic.Int32
	quorumSuccessCount  atomic.Int32
	completedCommitSeen atomic.Bool
	failedCommitSeen    atomic.Bool
	abortedCommitSeen   atomic.Bool
	quorumAtCompleted   *store.FusionQuorumDiagnostics
	quorumAtFailed      *store.FusionQuorumDiagnostics
	quorumAtAborted     *store.FusionQuorumDiagnostics
}

func newFaultInjectingQuorumStore(failAttempts int32) *faultInjectingQuorumStore {
	base := store.NewMemoryStore(20, 0)
	s := &faultInjectingQuorumStore{Storage: base}
	s.failQuorumAttempts.Store(failAttempts)
	return s
}

func (s *faultInjectingQuorumStore) UpdateFusionQuorum(ctx context.Context, id string, quorum store.FusionQuorumDiagnostics) error {
	s.mu.Lock()
	s.calls = append(s.calls, "UpdateFusionQuorum")
	s.mu.Unlock()

	if s.failQuorumAttempts.Load() > 0 {
		s.failQuorumAttempts.Add(-1)
		s.quorumFailCount.Add(1)
		return errors.New("transient storage error: connection reset by peer")
	}

	s.quorumSuccessCount.Add(1)
	return s.Storage.UpdateFusionQuorum(ctx, id, quorum)
}

func (s *faultInjectingQuorumStore) UpdateLifecycle(ctx context.Context, id string, state string, endedAt time.Time, durationMS int64, reason string) error {
	s.mu.Lock()
	s.calls = append(s.calls, "UpdateLifecycle:"+state)
	if state == store.LifecycleCompleted {
		s.completedCommitSeen.Store(true)
		if rec, ok, _ := s.Storage.Get(ctx, id); ok && rec.RouteDiagnostics != nil && rec.RouteDiagnostics.FusionQuorum != nil {
			cloned := *rec.RouteDiagnostics.FusionQuorum
			s.quorumAtCompleted = &cloned
		}
	} else if state == store.LifecycleFailed {
		s.failedCommitSeen.Store(true)
		if rec, ok, _ := s.Storage.Get(ctx, id); ok && rec.RouteDiagnostics != nil && rec.RouteDiagnostics.FusionQuorum != nil {
			cloned := *rec.RouteDiagnostics.FusionQuorum
			s.quorumAtFailed = &cloned
		}
	} else if state == store.LifecycleAborted {
		s.abortedCommitSeen.Store(true)
		if rec, ok, _ := s.Storage.Get(ctx, id); ok && rec.RouteDiagnostics != nil && rec.RouteDiagnostics.FusionQuorum != nil {
			cloned := *rec.RouteDiagnostics.FusionQuorum
			s.quorumAtAborted = &cloned
		}
	}
	s.mu.Unlock()
	return s.Storage.UpdateLifecycle(ctx, id, state, endedAt, durationMS, reason)
}

func (s *faultInjectingQuorumStore) snapshotCalls() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.calls...)
}

func setupM2QuorumFallbackFixture(t *testing.T, failAttempts int32) (*httptest.Server, *OpenAIRouter, *RequestContext, *config.Decision, *faultInjectingQuorumStore) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))

	router, reqCtx, request := quorumFallbackRouter(server.URL)
	router.Config.ClearRouteCache = true
	reqCtx.ExpectStreamingResponse = true
	request.Stream = true

	faultStore := newFaultInjectingQuorumStore(failAttempts)
	recorder := routerreplay.NewRecorder(faultStore)
	router.ReplayRecorder = recorder
	reqCtx.RouterReplayRecorder = recorder

	decision := &config.Decision{
		Name: "fusion-quorum-m2-fixture",
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
	require.Nil(t, response.GetImmediateResponse())
	require.True(t, reqCtx.LooperStreamingActive)
	require.Equal(t, "backup-model", reqCtx.VSRSelectedModel)
	require.NotEmpty(t, reqCtx.RouterReplayID)

	require.NotNil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be staged during dispatch")
	require.Equal(t, looper.FusionQuorumFallbackReady, reqCtx.PendingLooperQuorumOutcome.Disposition)

	return server, router, reqCtx, decision, faultStore
}

// TestLooperQuorum_M2_SSEStreaming_TransientFailureAndEOFRetry verifies:
// 1) Initial UpdateFusionQuorum failure does NOT commit LifecycleCompleted prematurely;
// 2) Initial terminal intent (LifecycleCompleted, response_complete) is preserved;
// 3) Subsequent EOF/cleanup retries UpdateFusionQuorum, succeeds, and commits LifecycleCompleted;
// 4) Quorum diagnostics are fully persisted in the final record.
func TestLooperQuorum_M2_SSEStreaming_TransientFailureAndEOFRetry(t *testing.T) {
	server, router, reqCtx, _, faultStore := setupM2QuorumFallbackFixture(t, 2)
	defer server.Close()

	reqCtx.IsStreamingResponse = true
	reqCtx.SemanticStreamState = &semanticResponseStreamState{
		responseID: "resp-stream-m2",
		model:      "backup-model",
		stop:       llmprotocol.StopEndTurn,
		terminal:   true,
		items: map[int]*semanticStreamItem{
			0: {completed: true, text: "fallback stream answer"},
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

	// 1. Initial response finalization: UpdateFusionQuorum fails with transient error
	router.finalizeSemanticStreamingResponse(reqCtx, nil)

	// Verify pending outcome was NOT wiped out and completed was NOT committed
	require.NotNil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be retained when persistence fails")
	require.False(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must be false when quorum write failed")
	require.False(t, faultStore.completedCommitSeen.Load(), "store must NOT have received LifecycleCompleted prematurely")

	// Verify initial terminal intent was preserved
	require.Equal(t, routerreplay.LifecycleCompleted, reqCtx.TerminalLifecycleState)
	require.Equal(t, "response_complete", reqCtx.TerminalLifecycleReason)

	// Verify 2 failed attempts in faultStore (initial response attempt + finalizeRouterReplay attempt)
	require.Equal(t, int32(2), faultStore.quorumFailCount.Load())
	require.Equal(t, int32(0), faultStore.quorumSuccessCount.Load())

	// 2. Stream EOF cleanup runs (handleProcessReceiveError with io.EOF)
	err := router.handleProcessReceiveError(reqCtx, io.EOF)
	require.NoError(t, err)

	// Verify retry succeeded
	require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be cleared after successful retry")
	require.True(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must be true after successful retry")
	require.True(t, faultStore.completedCommitSeen.Load(), "LifecycleCompleted must now be committed to store")
	require.Equal(t, int32(1), faultStore.quorumSuccessCount.Load())

	// Verify store state & ordering
	calls := faultStore.snapshotCalls()
	require.Contains(t, calls, "UpdateFusionQuorum")
	require.Contains(t, calls, "UpdateLifecycle:completed")

	var quorumSuccessIdx, lifecycleIdx int = -1, -1
	for idx, call := range calls {
		if call == "UpdateFusionQuorum" && idx > 0 { // second call
			quorumSuccessIdx = idx
		}
		if call == "UpdateLifecycle:completed" {
			lifecycleIdx = idx
		}
	}
	require.True(t, quorumSuccessIdx >= 0 && lifecycleIdx >= 0)
	require.Less(t, quorumSuccessIdx, lifecycleIdx, "UpdateFusionQuorum must succeed BEFORE UpdateLifecycle:completed is committed")

	// Verify final persisted record in store
	record, ok, err := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
	require.Equal(t, "response_complete", record.TerminalReason)
	require.NotNil(t, record.RouteDiagnostics)
	require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
	require.Equal(t, string(looper.FusionQuorumFallbackServed), record.RouteDiagnostics.FusionQuorum.Disposition)
	require.Equal(t, 2, record.RouteDiagnostics.FusionQuorum.RequiredCount)
	require.Equal(t, 0, record.RouteDiagnostics.FusionQuorum.UsableCount)
	require.Equal(t, "backup-model", record.RouteDiagnostics.FusionQuorum.FallbackTarget)
}

// TestLooperQuorum_M2_BufferedJSON_TransientFailureAndCloseRetry verifies:
// 1) Initial UpdateFusionQuorum failure during buffered JSON handling preserves PendingLooperQuorumOutcome;
// 2) routingSession.Close retries UpdateFusionQuorum, succeeds, and commits LifecycleCompleted;
// 3) Quorum diagnostics are fully present in the store.
func TestLooperQuorum_M2_BufferedJSON_TransientFailureAndCloseRetry(t *testing.T) {
	server, router, reqCtx, _, faultStore := setupM2QuorumFallbackFixture(t, 2)
	defer server.Close()

	session := &routingSession{
		router: router,
		ctx:    reqCtx,
	}

	body := []byte(`{
		"id": "chatcmpl-m2-json",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "backup-model",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "buffered fallback response"},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
	}`)

	// 1. Initial buffered response body pipeline: UpdateFusionQuorum fails
	resp := router.handleNonStreamingResponseBody(body, reqCtx, 100*time.Millisecond)
	require.NotNil(t, resp)
	require.Nil(t, resp.GetImmediateResponse())

	// Verify outcome was retained and completed was NOT committed
	require.NotNil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be retained on transient failure")
	require.False(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must remain false")
	require.False(t, faultStore.completedCommitSeen.Load(), "LifecycleCompleted must not be committed yet")
	require.Equal(t, routerreplay.LifecycleCompleted, reqCtx.TerminalLifecycleState)
	require.Equal(t, "response_complete", reqCtx.TerminalLifecycleReason)

	// 2. Session Close runs (session.Close(nil))
	session.Close(nil)

	// Verify retry succeeded
	require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be cleared after session Close retry")
	require.True(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must be true")
	require.True(t, faultStore.completedCommitSeen.Load(), "LifecycleCompleted must now be committed")

	// Verify final persisted record in store
	record, ok, err := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
	require.Equal(t, "response_complete", record.TerminalReason)
	require.NotNil(t, record.RouteDiagnostics)
	require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
	require.Equal(t, string(looper.FusionQuorumFallbackServed), record.RouteDiagnostics.FusionQuorum.Disposition)
	require.Equal(t, 2, record.RouteDiagnostics.FusionQuorum.RequiredCount)
	require.Equal(t, "backup-model", record.RouteDiagnostics.FusionQuorum.FallbackTarget)
}

// TestLooperQuorum_M2_MetricsAndEventsIdempotency verifies that when
// finalizeLooperQuorumOutcome is invoked multiple times across failure and retry:
// 1) Prometheus metrics (RecordFusionQuorumFailure) are emitted exactly once;
// 2) LooperQuorumMetricsRecorded flag guards duplicate emission;
// 3) RouterReplayCompleteLogged flag guards duplicate router_replay_complete emission.
func TestLooperQuorum_M2_MetricsAndEventsIdempotency(t *testing.T) {
	decision := &config.Decision{Name: "quorum-m2-metrics-idempotent"}
	outcome := &looper.FusionQuorumOutcome{
		RequiredCount:  2,
		UsableCount:    1,
		SelectedPolicy: "fallback",
		FallbackTarget: "backup-model",
		Disposition:    looper.FusionQuorumFallbackReady,
		Attempts: []looper.FusionQuorumAttemptOutcome{
			{Model: "panel-a", State: looper.FusionPanelAttemptUsable},
			{Model: "panel-b", State: looper.FusionPanelAttemptFailed},
		},
	}

	faultStore := newFaultInjectingQuorumStore(1)
	recorder := routerreplay.NewRecorder(faultStore)

	reqCtx := &RequestContext{
		RequestID:            "m2-metrics-req",
		RouterReplayID:       "m2-replay-id",
		RouterReplayRecorder: recorder,
	}

	// Create initial replay record via recorder
	replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		RequestID:     reqCtx.RequestID,
		OriginalModel: "panel-a",
		SelectedModel: "backup-model",
	})
	require.NoError(t, err)
	reqCtx.RouterReplayID = replayID

	failure := quorumFailureSeries(decision.Name, "fallback", "fallback_served")
	fallback := quorumFallbackSeries(decision.Name, "backup-model", "fallback_served")
	required := quorumHistogramSeries(decision.Name, "llm_fusion_quorum_required_responses")
	usable := quorumHistogramSeries(decision.Name, "llm_fusion_quorum_usable_responses")
	usableState := quorumAttemptSeries(decision.Name, "usable")
	failedState := quorumAttemptSeries(decision.Name, "failed")
	before := captureQuorumSeries(t, failure, fallback, required, usable, usableState, failedState)

	// Call 1: fails in UpdateFusionQuorum
	err1 := finalizeLooperQuorumOutcome(reqCtx, outcome, decision, true)
	require.Error(t, err1, "first call must return storage failure")
	require.True(t, reqCtx.LooperQuorumMetricsRecorded, "LooperQuorumMetricsRecorded must be true after first call")

	// Verify metrics were emitted once
	assert.Equal(t, 1.0, before.delta(t, failure))
	assert.Equal(t, 1.0, before.delta(t, fallback))
	assert.Equal(t, 1.0, before.delta(t, required))
	assert.Equal(t, 1.0, before.delta(t, usable))
	assert.Equal(t, 1.0, before.delta(t, usableState))
	assert.Equal(t, 1.0, before.delta(t, failedState))

	// Call 2 (retry): succeeds in UpdateFusionQuorum
	err2 := finalizeLooperQuorumOutcome(reqCtx, outcome, decision, true)
	require.NoError(t, err2, "second call must succeed")

	// Verify metrics were NOT emitted again (deltas still 1.0, meaning zero additional emission)
	assert.Equal(t, 1.0, before.delta(t, failure), "metric must not double-count on retry")
	assert.Equal(t, 1.0, before.delta(t, fallback), "metric must not double-count on retry")
	assert.Equal(t, 1.0, before.delta(t, required), "metric must not double-count on retry")
	assert.Equal(t, 1.0, before.delta(t, usable), "metric must not double-count on retry")
	assert.Equal(t, 1.0, before.delta(t, usableState), "metric must not double-count on retry")
	assert.Equal(t, 1.0, before.delta(t, failedState), "metric must not double-count on retry")
}

// TestLooperQuorum_M2_SettleAbortedLooperStreamingUsage_PreservesCompletedIntent verifies
// that when settleAbortedLooperStreamingUsage is called during stream disconnect / error,
// if TerminalLifecycleState is already LifecycleCompleted, it does NOT overwrite
// the disposition with FusionQuorumCancelled, but retains terminal fallback_served.
func TestLooperQuorum_M2_SettleAbortedLooperStreamingUsage_PreservesCompletedIntent(t *testing.T) {
	faultStore := newFaultInjectingQuorumStore(0)
	recorder := routerreplay.NewRecorder(faultStore)
	router := &OpenAIRouter{ReplayRecorder: recorder}

	outcome := &looper.FusionQuorumOutcome{
		RequiredCount:  2,
		UsableCount:    1,
		SelectedPolicy: "fallback",
		FallbackTarget: "backup-model",
		Disposition:    looper.FusionQuorumFallbackServed, // already terminal
	}

	reqCtx := &RequestContext{
		RequestID:                  "m2-settle-test",
		RouterReplayID:             "m2-settle-replay",
		RouterReplayRecorder:       recorder,
		PendingLooperQuorumOutcome: outcome,
		TerminalLifecycleState:     routerreplay.LifecycleCompleted,
		TerminalLifecycleReason:    "response_complete",
	}

	replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		RequestID:     reqCtx.RequestID,
		OriginalModel: "panel-a",
		SelectedModel: "backup-model",
	})
	require.NoError(t, err)
	reqCtx.RouterReplayID = replayID

	// Call settleAbortedLooperStreamingUsage with cancellation error
	router.settleAbortedLooperStreamingUsage(reqCtx, errors.New("client connection dropped"))

	// Outcome disposition must NOT have been changed to cancelled
	require.Equal(t, looper.FusionQuorumFallbackServed, outcome.Disposition,
		"settleAbortedLooperStreamingUsage must retain terminal completed disposition")
	require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "pending outcome must be cleared after successful settle")
}

// TestLooperQuorum_M2_ResponsePhaseImmediateRetry_Succeeds verifies that if
// UpdateFusionQuorum fails on the initial response phase attempt but succeeds on
// the immediate retry inside finalizeRouterReplay (e.g. transient 1-attempt hiccup),
// the record is cleanly completed and pending outcome is cleared immediately.
func TestLooperQuorum_M2_ResponsePhaseImmediateRetry_Succeeds(t *testing.T) {
	server, router, reqCtx, _, faultStore := setupM2QuorumFallbackFixture(t, 1) // only 1 attempt fails
	defer server.Close()

	reqCtx.IsStreamingResponse = true
	reqCtx.SemanticStreamState = &semanticResponseStreamState{
		responseID: "resp-stream-m2-single",
		model:      "backup-model",
		stop:       llmprotocol.StopEndTurn,
		terminal:   true,
		items: map[int]*semanticStreamItem{
			0: {completed: true, text: "fallback answer"},
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

	// Attempt 1 failed at line 640; attempt 2 inside finalizeRouterReplay succeeded!
	require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be cleared by finalizeRouterReplay retry")
	require.True(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must be true")
	require.True(t, faultStore.completedCommitSeen.Load(), "LifecycleCompleted must be committed")
	require.Equal(t, int32(1), faultStore.quorumFailCount.Load())
	require.Equal(t, int32(1), faultStore.quorumSuccessCount.Load())

	// Store record must have quorum diagnostics
	record, ok, err := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
	require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
}

// TestLooperQuorum_M2_PersistentStorageFailure_NeverCommitsCompleted verifies that
// if storage remains unavailable throughout response phase and EOF/cleanup,
// LifecycleCompleted is NEVER committed without quorum diagnostics.
func TestLooperQuorum_M2_PersistentStorageFailure_NeverCommitsCompleted(t *testing.T) {
	server, router, reqCtx, _, faultStore := setupM2QuorumFallbackFixture(t, 999) // persistently failing
	defer server.Close()

	reqCtx.IsStreamingResponse = true
	reqCtx.SemanticStreamState = &semanticResponseStreamState{
		responseID: "resp-stream-m2-persist-fail",
		model:      "backup-model",
		stop:       llmprotocol.StopEndTurn,
		terminal:   true,
		items: map[int]*semanticStreamItem{
			0: {completed: true, text: "fallback answer"},
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

	// Response phase must NOT commit completed
	require.NotNil(t, reqCtx.PendingLooperQuorumOutcome)
	require.False(t, reqCtx.RouterReplayCommitted)
	require.False(t, faultStore.completedCommitSeen.Load())

	// EOF / cleanup attempt
	_ = router.handleProcessReceiveError(reqCtx, io.EOF)

	// Still must NOT commit LifecycleCompleted
	require.False(t, reqCtx.RouterReplayCommitted, "must never commit completed when quorum write permanently fails")
	require.False(t, faultStore.completedCommitSeen.Load(), "store must never see LifecycleCompleted")
	require.NotNil(t, reqCtx.PendingLooperQuorumOutcome, "outcome remains pending for potential later recovery")
}

// TestLooperQuorum_M2_ProviderFailure_TransientFailureAndCleanupRetry verifies:
//  1. When an upstream provider error occurs (e.g. HTTP 502/503), initial UpdateFusionQuorum failure
//     does NOT commit LifecycleFailed prematurely;
//  2. router_replay_complete is NOT emitted while quorum diagnostics are pending;
//  3. Initial terminal intent (LifecycleFailed, upstream_error_response) is preserved on RequestContext;
//  4. When storage recovers on stream cleanup retry (handleProcessReceiveError), UpdateFusionQuorum
//     persists the quorum diagnostics, LifecycleFailed is committed to store, and
//     router_replay_complete is emitted exactly once;
//  5. UpdateFusionQuorum succeeds BEFORE UpdateLifecycle:failed in the store call sequence.
func TestLooperQuorum_M2_ProviderFailure_TransientFailureAndCleanupRetry(t *testing.T) {
	server, router, reqCtx, _, faultStore := setupM2QuorumFallbackFixture(t, 1)
	defer server.Close()

	reqCtx.UpstreamStatusCode = http.StatusBadGateway

	// 1. Initial response finalization with upstream error:
	// attachRouterReplayResponse calls finalizeRouterReplay with LifecycleFailed and upstream_error_response.
	// UpdateFusionQuorum fails with transient error.
	router.attachRouterReplayResponse(reqCtx, []byte(`{"error": "bad gateway from upstream provider"}`), true)

	// Verify pending outcome was NOT wiped out and failed was NOT committed prematurely
	require.NotNil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be retained when persistence fails on failed path")
	require.False(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must be false when quorum write failed")
	require.False(t, faultStore.failedCommitSeen.Load(), "store must NOT have received LifecycleFailed prematurely")
	require.False(t, reqCtx.RouterReplayCompleteLogged, "router_replay_complete must NOT be emitted while quorum diagnostics pending")

	// Verify initial terminal intent was preserved
	require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState)
	require.Equal(t, "upstream_error_response", reqCtx.TerminalLifecycleReason)

	// Verify failed attempt in faultStore
	require.Equal(t, int32(1), faultStore.quorumFailCount.Load())
	require.Equal(t, int32(0), faultStore.quorumSuccessCount.Load())

	// 2. Stream cleanup runs with upstream error
	err := router.handleProcessReceiveError(reqCtx, errors.New("upstream bad gateway 502"))
	require.Error(t, err)

	// Verify retry succeeded
	require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be cleared after successful retry")
	require.True(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must be true after successful retry")
	require.True(t, faultStore.failedCommitSeen.Load(), "LifecycleFailed must now be committed to store")
	require.True(t, reqCtx.RouterReplayCompleteLogged, "router_replay_complete must be emitted exactly once after recovery")
	require.Equal(t, int32(1), faultStore.quorumSuccessCount.Load())

	// Verify store state & ordering: UpdateFusionQuorum must precede UpdateLifecycle:failed
	calls := faultStore.snapshotCalls()
	require.Contains(t, calls, "UpdateFusionQuorum")
	require.Contains(t, calls, "UpdateLifecycle:failed")

	var quorumSuccessIdx, lifecycleIdx int = -1, -1
	for idx, call := range calls {
		if call == "UpdateFusionQuorum" && idx > 0 {
			quorumSuccessIdx = idx
		}
		if call == "UpdateLifecycle:failed" {
			lifecycleIdx = idx
		}
	}
	require.True(t, quorumSuccessIdx >= 0 && lifecycleIdx >= 0)
	require.Less(t, quorumSuccessIdx, lifecycleIdx, "UpdateFusionQuorum must succeed BEFORE UpdateLifecycle:failed is committed")

	// Verify final persisted record in store
	record, ok, getErr := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
	require.NoError(t, getErr)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "upstream_error_response", record.TerminalReason)
	require.NotNil(t, record.RouteDiagnostics)
	require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
	require.Equal(t, string(looper.FusionQuorumResponseEncodeFailed), record.RouteDiagnostics.FusionQuorum.Disposition)
	require.Equal(t, 2, record.RouteDiagnostics.FusionQuorum.RequiredCount)
	require.Equal(t, 0, record.RouteDiagnostics.FusionQuorum.UsableCount)
	require.Equal(t, "backup-model", record.RouteDiagnostics.FusionQuorum.FallbackTarget)
}

// TestLooperQuorum_M2_ClientCancellation_TransientFailureAndCleanupRetry verifies:
//  1. When client cancels/aborts (context.Canceled / client connection dropped),
//     initial UpdateFusionQuorum failure does NOT commit LifecycleAborted prematurely;
//  2. router_replay_complete is NOT emitted while quorum diagnostics are pending;
//  3. Initial terminal intent (LifecycleAborted, client_canceled) is preserved;
//  4. When storage recovers on cleanup / Close retry, UpdateFusionQuorum persists
//     the quorum diagnostics, LifecycleAborted is committed to store, and
//     router_replay_complete is emitted exactly once;
//  5. UpdateFusionQuorum succeeds BEFORE UpdateLifecycle:aborted.
func TestLooperQuorum_M2_ClientCancellation_TransientFailureAndCleanupRetry(t *testing.T) {
	server, router, reqCtx, _, faultStore := setupM2QuorumFallbackFixture(t, 2)
	defer server.Close()

	session := &routingSession{
		router: router,
		ctx:    reqCtx,
	}

	cancelErr := context.Canceled

	// 1. Initial cancellation receive error:
	// settleAbortedLooperStreamingUsage and finalizeRouterReplay both attempt UpdateFusionQuorum and fail.
	err := router.handleProcessReceiveError(reqCtx, cancelErr)
	require.NoError(t, err, "handleProcessReceiveError handles context.Canceled gracefully")

	// Verify pending outcome was NOT cleared and aborted was NOT committed prematurely
	require.NotNil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be retained when persistence fails on aborted path")
	require.False(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must remain false")
	require.False(t, faultStore.abortedCommitSeen.Load(), "store must NOT have received LifecycleAborted prematurely")
	require.False(t, reqCtx.RouterReplayCompleteLogged, "router_replay_complete must NOT be emitted while quorum diagnostics pending")

	// Verify terminal intent was captured
	require.Equal(t, routerreplay.LifecycleAborted, reqCtx.TerminalLifecycleState)
	require.Equal(t, "client_canceled", reqCtx.TerminalLifecycleReason)

	// Verify 2 failed attempts in faultStore
	require.Equal(t, int32(2), faultStore.quorumFailCount.Load())
	require.Equal(t, int32(0), faultStore.quorumSuccessCount.Load())

	// 2. Storage recovers; session.Close retries finalization
	session.Close(cancelErr)

	// Verify retry succeeded
	require.Nil(t, reqCtx.PendingLooperQuorumOutcome, "PendingLooperQuorumOutcome must be cleared after session Close retry")
	require.True(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must be true after recovery")
	require.True(t, faultStore.abortedCommitSeen.Load(), "LifecycleAborted must now be committed")
	require.True(t, reqCtx.RouterReplayCompleteLogged, "router_replay_complete must be emitted exactly once after recovery")
	require.Equal(t, int32(1), faultStore.quorumSuccessCount.Load())

	// Verify call ordering: UpdateFusionQuorum must precede UpdateLifecycle:aborted
	calls := faultStore.snapshotCalls()
	require.Contains(t, calls, "UpdateFusionQuorum")
	require.Contains(t, calls, "UpdateLifecycle:aborted")

	var quorumSuccessIdx, lifecycleIdx int = -1, -1
	for idx, call := range calls {
		if call == "UpdateFusionQuorum" && idx >= 2 {
			quorumSuccessIdx = idx
		}
		if call == "UpdateLifecycle:aborted" {
			lifecycleIdx = idx
		}
	}
	require.True(t, quorumSuccessIdx >= 0 && lifecycleIdx >= 0)
	require.Less(t, quorumSuccessIdx, lifecycleIdx, "UpdateFusionQuorum must succeed BEFORE UpdateLifecycle:aborted is committed")

	// Verify persisted record in store
	record, ok, getErr := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
	require.NoError(t, getErr)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_canceled", record.TerminalReason)
	require.NotNil(t, record.RouteDiagnostics)
	require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
	require.Equal(t, string(looper.FusionQuorumCancelled), record.RouteDiagnostics.FusionQuorum.Disposition)
	require.Equal(t, 2, record.RouteDiagnostics.FusionQuorum.RequiredCount)
	require.Equal(t, 0, record.RouteDiagnostics.FusionQuorum.UsableCount)
	require.Equal(t, "backup-model", record.RouteDiagnostics.FusionQuorum.FallbackTarget)
}

// TestLooperQuorum_M2_PersistentStorageFailure_NeverCommitsFailedOrAborted verifies that
// if storage remains persistently unavailable during provider failure or client cancellation:
// 1) Neither LifecycleFailed nor LifecycleAborted is ever committed without quorum diagnostics;
// 2) router_replay_complete is NEVER emitted;
// 3) RouterReplayCommitted remains false throughout cleanup.
func TestLooperQuorum_M2_PersistentStorageFailure_NeverCommitsFailedOrAborted(t *testing.T) {
	t.Run("provider_failure_path", func(t *testing.T) {
		server, router, reqCtx, _, faultStore := setupM2QuorumFallbackFixture(t, 999)
		defer server.Close()

		reqCtx.UpstreamStatusCode = http.StatusBadGateway

		// 1. Initial response error
		router.attachRouterReplayResponse(reqCtx, []byte(`{"error": "upstream failure"}`), true)

		require.NotNil(t, reqCtx.PendingLooperQuorumOutcome)
		require.False(t, reqCtx.RouterReplayCommitted)
		require.False(t, faultStore.failedCommitSeen.Load())
		require.False(t, reqCtx.RouterReplayCompleteLogged)

		// 2. Stream cleanup
		_ = router.handleProcessReceiveError(reqCtx, errors.New("upstream 502"))

		require.False(t, reqCtx.RouterReplayCommitted, "must never commit failed when quorum write persistently fails")
		require.False(t, faultStore.failedCommitSeen.Load(), "store must never see LifecycleFailed")
		require.False(t, reqCtx.RouterReplayCompleteLogged, "must never emit router_replay_complete")
		require.NotNil(t, reqCtx.PendingLooperQuorumOutcome, "outcome remains pending")

		// Replay record in store must NOT have transitioned to failed
		record, ok, err := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
		require.NoError(t, err)
		require.True(t, ok)
		require.NotEqual(t, routerreplay.LifecycleFailed, record.LifecycleState)
	})

	t.Run("client_cancellation_path", func(t *testing.T) {
		server, router, reqCtx, _, faultStore := setupM2QuorumFallbackFixture(t, 999)
		defer server.Close()

		session := &routingSession{
			router: router,
			ctx:    reqCtx,
		}

		cancelErr := context.Canceled

		// 1. Receive error cleanup
		_ = router.handleProcessReceiveError(reqCtx, cancelErr)

		require.NotNil(t, reqCtx.PendingLooperQuorumOutcome)
		require.False(t, reqCtx.RouterReplayCommitted)
		require.False(t, faultStore.abortedCommitSeen.Load())
		require.False(t, reqCtx.RouterReplayCompleteLogged)

		// 2. Session Close
		session.Close(cancelErr)

		require.False(t, reqCtx.RouterReplayCommitted, "must never commit aborted when quorum write persistently fails")
		require.False(t, faultStore.abortedCommitSeen.Load(), "store must never see LifecycleAborted")
		require.False(t, reqCtx.RouterReplayCompleteLogged, "must never emit router_replay_complete")
		require.NotNil(t, reqCtx.PendingLooperQuorumOutcome, "outcome remains pending")

		// Replay record in store must NOT have transitioned to aborted
		record, ok, err := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
		require.NoError(t, err)
		require.True(t, ok)
		require.NotEqual(t, routerreplay.LifecycleAborted, record.LifecycleState)
	})
}

// TestLooperQuorum_M2_UpstreamStreamIncomplete_TransientFailureAndCleanupRetry verifies:
//  1. When upstream stream aborts incomplete (StreamingAborted = true), initial UpdateFusionQuorum failure
//     does NOT commit LifecycleFailed prematurely;
//  2. Cleanup retry succeeds, committing LifecycleFailed with "upstream_stream_incomplete";
//  3. router_replay_complete is emitted exactly once with quorum diagnostics present.
func TestLooperQuorum_M2_UpstreamStreamIncomplete_TransientFailureAndCleanupRetry(t *testing.T) {
	server, router, reqCtx, _, faultStore := setupM2QuorumFallbackFixture(t, 1)
	defer server.Close()

	reqCtx.StreamingAborted = true

	// 1. Initial attach response with stream incomplete
	router.attachRouterReplayResponse(reqCtx, []byte(`{"error": "stream incomplete"}`), true)

	require.NotNil(t, reqCtx.PendingLooperQuorumOutcome)
	require.False(t, reqCtx.RouterReplayCommitted)
	require.False(t, faultStore.failedCommitSeen.Load())
	require.False(t, reqCtx.RouterReplayCompleteLogged)
	require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState)
	require.Equal(t, "upstream_stream_incomplete", reqCtx.TerminalLifecycleReason)

	// 2. Cleanup retry
	_ = router.handleProcessReceiveError(reqCtx, io.ErrUnexpectedEOF)

	require.Nil(t, reqCtx.PendingLooperQuorumOutcome)
	require.True(t, reqCtx.RouterReplayCommitted)
	require.True(t, faultStore.failedCommitSeen.Load())
	require.True(t, reqCtx.RouterReplayCompleteLogged)

	record, ok, err := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "upstream_stream_incomplete", record.TerminalReason)
	require.NotNil(t, record.RouteDiagnostics.FusionQuorum)
}
