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
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// faultInjectingPhaseReplayStore wraps store.Storage and injects transient failures
// on UpdateLifecycle to test replay finalization retry semantics on non-quorum paths.
type faultInjectingPhaseReplayStore struct {
	store.Storage
	mu                   sync.Mutex
	calls                []string
	failFinalizeAttempts atomic.Int32
	finalizeFailCount    atomic.Int32
	finalizeSuccessCount atomic.Int32
}

func newFaultInjectingPhaseReplayStore(failAttempts int32) *faultInjectingPhaseReplayStore {
	base := store.NewMemoryStore(20, 0)
	s := &faultInjectingPhaseReplayStore{Storage: base}
	s.failFinalizeAttempts.Store(failAttempts)
	return s
}

func (s *faultInjectingPhaseReplayStore) Add(ctx context.Context, rec store.Record) (string, error) {
	s.mu.Lock()
	s.calls = append(s.calls, "Add")
	s.mu.Unlock()
	return s.Storage.Add(ctx, rec)
}

func (s *faultInjectingPhaseReplayStore) UpdateLifecycle(ctx context.Context, id string, state string, endedAt time.Time, durationMS int64, reason string) error {
	s.mu.Lock()
	s.calls = append(s.calls, fmt.Sprintf("UpdateLifecycle:%s:%s", state, reason))
	s.mu.Unlock()

	if s.failFinalizeAttempts.Load() > 0 {
		s.failFinalizeAttempts.Add(-1)
		s.finalizeFailCount.Add(1)
		return errors.New("transient replay storage failure: disk busy")
	}

	s.finalizeSuccessCount.Add(1)
	return s.Storage.UpdateLifecycle(ctx, id, state, endedAt, durationMS, reason)
}

func (s *faultInjectingPhaseReplayStore) snapshotCalls() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.calls...)
}

func setupPhaseErrorRetryFixture(t *testing.T, failAttempts int32) (*OpenAIRouter, *RequestContext, *faultInjectingPhaseReplayStore) {
	t.Helper()
	faultStore := newFaultInjectingPhaseReplayStore(failAttempts)
	recorder := routerreplay.NewRecorder(faultStore)

	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model-a": {Capabilities: []string{"chat"}},
				},
			},
		},
		ReplayRecorder: recorder,
	}

	reqCtx := &RequestContext{
		RequestID:                "test-phase-error-retry",
		Headers:                  map[string]string{},
		RouterReplayPluginConfig: &replayConfig,
		RouterReplayRecorder:     recorder,
	}

	record := routerreplay.RoutingRecord{
		RequestID:      reqCtx.RequestID,
		LifecycleState: store.LifecycleInProgress,
		Timestamp:      time.Now().UTC(),
		OriginalModel:  "model-a",
		SelectedModel:  "model-a",
		Decision:       "test-decision",
	}
	id, err := recorder.AddRecord(record)
	require.NoError(t, err)
	reqCtx.RouterReplayID = id

	return router, reqCtx, faultStore
}

func TestRoutingSession_PhaseErrorReplayRetry(t *testing.T) {
	t.Run("phase error on non-quorum request retries FinalizeLifecycle during Close", func(t *testing.T) {
		router, reqCtx, faultStore := setupPhaseErrorRetryFixture(t, 1)

		session := &routingSession{
			router: router,
			ctx:    reqCtx,
		}

		// Ensure this is a non-quorum request
		require.Nil(t, reqCtx.PendingLooperQuorumOutcome)

		phaseErr := errors.New("codec error: invalid request payload")
		effect, err := session.phase(func() (*ext_proc.ProcessingResponse, error) {
			return nil, phaseErr
		}, func(response *ext_proc.ProcessingResponse) {
			t.Fatal("sent callback must not be invoked on phase error")
		})

		// Assert phase returned error
		require.Error(t, err)
		require.Equal(t, phaseErr, err)
		require.Nil(t, effect)

		// Assert state immediately after phase error:
		// Replay record is still in_progress / uncommitted because attempt 1 failed.
		require.False(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must remain false when FinalizeLifecycle fails")
		require.False(t, reqCtx.RouterReplayCompleteLogged, "router_replay_complete must NOT be emitted while uncommitted")
		require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState, "TerminalLifecycleState must be LifecycleFailed")
		require.Equal(t, "request_processing_failed", reqCtx.TerminalLifecycleReason, "TerminalLifecycleReason must be preserved")
		require.True(t, session.ended, "session must be ended")

		// Assert record in store is still in_progress
		rec, ok, getErr := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
		require.NoError(t, getErr)
		require.True(t, ok)
		require.Equal(t, store.LifecycleInProgress, rec.LifecycleState, "store record must still be in_progress")
		require.Equal(t, int32(1), faultStore.finalizeFailCount.Load())
		require.Equal(t, int32(0), faultStore.finalizeSuccessCount.Load())

		// Now call session.Close(phaseErr)
		session.Close(phaseErr)

		// Assert:
		// 1) Close successfully retries FinalizeLifecycle on the store
		require.Equal(t, int32(1), faultStore.finalizeSuccessCount.Load(), "FinalizeLifecycle retry must succeed")

		// 2) RouterReplayCommitted is set to true
		require.True(t, reqCtx.RouterReplayCommitted, "RouterReplayCommitted must be true after Close retry")

		// 3) TerminalLifecycleState is LifecycleFailed and original reason is preserved
		require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState)
		require.Equal(t, "request_processing_failed", reqCtx.TerminalLifecycleReason)

		// 4) router_replay_complete is emitted exactly once
		require.True(t, reqCtx.RouterReplayCompleteLogged, "router_replay_complete must be emitted after recovery")

		// 5) Store call order is verified
		calls := faultStore.snapshotCalls()
		expectedCalls := []string{
			"Add",
			"UpdateLifecycle:failed:request_processing_failed",
			"UpdateLifecycle:failed:request_processing_failed",
		}
		require.Equal(t, expectedCalls, calls, "Store call order must show Add followed by two UpdateLifecycle attempts")

		// Verify persisted state in store
		rec, ok, getErr = faultStore.Get(context.Background(), reqCtx.RouterReplayID)
		require.NoError(t, getErr)
		require.True(t, ok)
		require.Equal(t, store.LifecycleFailed, rec.LifecycleState)
		require.Equal(t, "request_processing_failed", rec.TerminalReason)

		// Calling Close again is a no-op (idempotent)
		session.Close(phaseErr)
		require.Equal(t, expectedCalls, faultStore.snapshotCalls(), "Repeated Close must be a no-op")
	})

	t.Run("panic recovery on non-quorum request retries FinalizeLifecycle during Close", func(t *testing.T) {
		router, reqCtx, faultStore := setupPhaseErrorRetryFixture(t, 1)

		session := &routingSession{
			router: router,
			ctx:    reqCtx,
		}

		require.Nil(t, reqCtx.PendingLooperQuorumOutcome)

		effect, err := session.phase(func() (*ext_proc.ProcessingResponse, error) {
			panic("processor internal crash")
		}, func(response *ext_proc.ProcessingResponse) {
			t.Fatal("sent callback must not be invoked on panic")
		})

		// Assert panic was recovered and internal status error returned
		require.Error(t, err)
		require.Nil(t, effect)
		require.Equal(t, codes.Internal, status.Code(err))
		require.Contains(t, err.Error(), "processor internal crash")

		// Assert state immediately after panic recovery:
		// Attempt 1 failed, uncommitted, reason preserved as processor_panic
		require.False(t, reqCtx.RouterReplayCommitted)
		require.False(t, reqCtx.RouterReplayCompleteLogged)
		require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState)
		require.Equal(t, "processor_panic", reqCtx.TerminalLifecycleReason)
		require.True(t, session.ended)

		// Store record is still in_progress
		rec, ok, getErr := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
		require.NoError(t, getErr)
		require.True(t, ok)
		require.Equal(t, store.LifecycleInProgress, rec.LifecycleState)
		require.Equal(t, int32(1), faultStore.finalizeFailCount.Load())
		require.Equal(t, int32(0), faultStore.finalizeSuccessCount.Load())

		// Now call session.Close(err)
		session.Close(err)

		// Assert:
		// 1) Close successfully retries FinalizeLifecycle
		require.Equal(t, int32(1), faultStore.finalizeSuccessCount.Load())

		// 2) RouterReplayCommitted is set to true
		require.True(t, reqCtx.RouterReplayCommitted)

		// 3) TerminalLifecycleState is LifecycleFailed and reason is processor_panic
		require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState)
		require.Equal(t, "processor_panic", reqCtx.TerminalLifecycleReason)

		// 4) router_replay_complete is emitted exactly once
		require.True(t, reqCtx.RouterReplayCompleteLogged)

		// 5) Store call order is verified
		calls := faultStore.snapshotCalls()
		expectedCalls := []string{
			"Add",
			"UpdateLifecycle:failed:processor_panic",
			"UpdateLifecycle:failed:processor_panic",
		}
		require.Equal(t, expectedCalls, calls)

		rec, ok, getErr = faultStore.Get(context.Background(), reqCtx.RouterReplayID)
		require.NoError(t, getErr)
		require.True(t, ok)
		require.Equal(t, store.LifecycleFailed, rec.LifecycleState)
		require.Equal(t, "processor_panic", rec.TerminalReason)
	})

	t.Run("preserves original terminal intent invalid_request across failure and retry", func(t *testing.T) {
		router, reqCtx, faultStore := setupPhaseErrorRetryFixture(t, 1)

		// Pre-populate terminal intent
		reqCtx.TerminalLifecycleState = routerreplay.LifecycleFailed
		reqCtx.TerminalLifecycleReason = "invalid_request"

		session := &routingSession{
			router: router,
			ctx:    reqCtx,
		}

		err := errors.New("client sent malformed parameters")
		_, phaseErr := session.phase(func() (*ext_proc.ProcessingResponse, error) {
			return nil, err
		}, func(response *ext_proc.ProcessingResponse) {})
		require.Error(t, phaseErr)

		// Attempt 1 failed; intent preserved
		require.False(t, reqCtx.RouterReplayCommitted)
		require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState)
		require.Equal(t, "invalid_request", reqCtx.TerminalLifecycleReason)

		// Close retries and commits with preserved reason
		session.Close(phaseErr)

		require.True(t, reqCtx.RouterReplayCommitted)
		require.True(t, reqCtx.RouterReplayCompleteLogged)
		require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState)
		require.Equal(t, "invalid_request", reqCtx.TerminalLifecycleReason)

		rec, ok, getErr := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
		require.NoError(t, getErr)
		require.True(t, ok)
		require.Equal(t, store.LifecycleFailed, rec.LifecycleState)
		require.Equal(t, "invalid_request", rec.TerminalReason)

		calls := faultStore.snapshotCalls()
		expectedCalls := []string{
			"Add",
			"UpdateLifecycle:failed:invalid_request",
			"UpdateLifecycle:failed:invalid_request",
		}
		require.Equal(t, expectedCalls, calls)
	})

	t.Run("preserves original terminal intent upstream_response_failed across failure and retry", func(t *testing.T) {
		router, reqCtx, faultStore := setupPhaseErrorRetryFixture(t, 1)

		reqCtx.TerminalLifecycleState = routerreplay.LifecycleFailed
		reqCtx.TerminalLifecycleReason = "upstream_response_failed"

		session := &routingSession{
			router: router,
			ctx:    reqCtx,
		}

		err := errors.New("upstream gateway timeout")
		_, phaseErr := session.phase(func() (*ext_proc.ProcessingResponse, error) {
			return nil, err
		}, func(response *ext_proc.ProcessingResponse) {})
		require.Error(t, phaseErr)

		require.False(t, reqCtx.RouterReplayCommitted)
		require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState)
		require.Equal(t, "upstream_response_failed", reqCtx.TerminalLifecycleReason)

		session.Close(phaseErr)

		require.True(t, reqCtx.RouterReplayCommitted)
		require.True(t, reqCtx.RouterReplayCompleteLogged)
		require.Equal(t, routerreplay.LifecycleFailed, reqCtx.TerminalLifecycleState)
		require.Equal(t, "upstream_response_failed", reqCtx.TerminalLifecycleReason)

		rec, ok, getErr := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
		require.NoError(t, getErr)
		require.True(t, ok)
		require.Equal(t, store.LifecycleFailed, rec.LifecycleState)
		require.Equal(t, "upstream_response_failed", rec.TerminalReason)
	})

	t.Run("persistent storage failure on retry leaves uncommitted without panic", func(t *testing.T) {
		// Both attempt 1 and attempt 2 fail
		router, reqCtx, faultStore := setupPhaseErrorRetryFixture(t, 2)

		session := &routingSession{
			router: router,
			ctx:    reqCtx,
		}

		phaseErr := errors.New("upstream connection reset")
		_, err := session.phase(func() (*ext_proc.ProcessingResponse, error) {
			return nil, phaseErr
		}, func(response *ext_proc.ProcessingResponse) {})
		require.Error(t, err)

		require.False(t, reqCtx.RouterReplayCommitted)
		require.Equal(t, int32(1), faultStore.finalizeFailCount.Load())

		// Close retries, but attempt 2 also fails
		session.Close(phaseErr)

		require.False(t, reqCtx.RouterReplayCommitted, "Must remain uncommitted when retry also fails")
		require.False(t, reqCtx.RouterReplayCompleteLogged, "Must NOT emit router_replay_complete when retry fails")
		require.Equal(t, int32(2), faultStore.finalizeFailCount.Load())
		require.Equal(t, int32(0), faultStore.finalizeSuccessCount.Load())

		rec, ok, getErr := faultStore.Get(context.Background(), reqCtx.RouterReplayID)
		require.NoError(t, getErr)
		require.True(t, ok)
		require.Equal(t, store.LifecycleInProgress, rec.LifecycleState)
	})

	t.Run("already committed replay record skips retry in Close", func(t *testing.T) {
		router, reqCtx, faultStore := setupPhaseErrorRetryFixture(t, 0)
		reqCtx.RouterReplayCommitted = true
		reqCtx.RouterReplayCompleteLogged = true
		reqCtx.TerminalLifecycleState = routerreplay.LifecycleCompleted
		reqCtx.TerminalLifecycleReason = "response_complete"

		session := &routingSession{
			router: router,
			ctx:    reqCtx,
			ended:  true,
		}

		session.Close(nil)

		// Verify no UpdateLifecycle was called
		calls := faultStore.snapshotCalls()
		require.Equal(t, []string{"Add"}, calls)
	})
}
