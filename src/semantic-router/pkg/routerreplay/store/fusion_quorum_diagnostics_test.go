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

package store

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	milvusclient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/qdrant/go-client/qdrant"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
)

func TestFusionQuorumDiagnosticsJSONRoundTrip(t *testing.T) {
	record := Record{
		ID: "fusion-quorum",
		RouteDiagnostics: &RouteDiagnostics{
			SelectionMethod: "fusion",
			FusionQuorum: &FusionQuorumDiagnostics{
				RequiredCount: 2,
				UsableCount:   1,
				Attempts: []FusionPanelAttemptDiagnostics{
					{Model: "panel-a", State: "usable", PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12},
					{Model: "panel-b", State: "unusable", PromptTokens: 20, CompletionTokens: 3, TotalTokens: 23},
					{Model: "panel-c", State: "failed"},
				},
			},
		},
	}

	decoded := roundTripRecord(t, record)
	assertFusionQuorumDiagnosticsRoundTrip(t, decoded.RouteDiagnostics)
}

func assertFusionQuorumDiagnosticsRoundTrip(t *testing.T, diagnostics *RouteDiagnostics) {
	t.Helper()
	if diagnostics == nil || diagnostics.FusionQuorum == nil {
		t.Fatalf("Fusion quorum diagnostics missing after round trip: %+v", diagnostics)
	}
	quorum := diagnostics.FusionQuorum
	if quorum.RequiredCount != 2 || quorum.UsableCount != 1 || len(quorum.Attempts) != 3 {
		t.Fatalf("Fusion quorum diagnostics changed after round trip: %+v", quorum)
	}
	assertFusionAttemptRoundTrip(t, quorum.Attempts[1])
}

func assertFusionAttemptRoundTrip(t *testing.T, got FusionPanelAttemptDiagnostics) {
	t.Helper()
	if got.Model != "panel-b" || got.State != "unusable" || got.PromptTokens != 20 ||
		got.CompletionTokens != 3 || got.TotalTokens != 23 {
		t.Fatalf("Fusion attempt changed after round trip: %+v", got)
	}
}

func TestCloneRecordDeepCopiesFusionQuorumDiagnostics(t *testing.T) {
	record := Record{RouteDiagnostics: &RouteDiagnostics{
		FusionQuorum: &FusionQuorumDiagnostics{
			RequiredCount: 2,
			UsableCount:   1,
			Attempts: []FusionPanelAttemptDiagnostics{{
				Model: "panel-a", State: "usable", TotalTokens: 12,
			}},
		},
	}}

	cloned := cloneRecord(record)
	if cloned.RouteDiagnostics == nil || cloned.RouteDiagnostics.FusionQuorum == nil {
		t.Fatal("cloned record lost Fusion quorum diagnostics")
	}
	if cloned.RouteDiagnostics.FusionQuorum == record.RouteDiagnostics.FusionQuorum {
		t.Fatal("Fusion quorum diagnostics pointer was not cloned")
	}
	if &cloned.RouteDiagnostics.FusionQuorum.Attempts[0] == &record.RouteDiagnostics.FusionQuorum.Attempts[0] {
		t.Fatal("Fusion quorum attempt slice was not cloned")
	}

	cloned.RouteDiagnostics.FusionQuorum.RequiredCount = 9
	cloned.RouteDiagnostics.FusionQuorum.Attempts[0].Model = "mutated"
	if record.RouteDiagnostics.FusionQuorum.RequiredCount != 2 ||
		record.RouteDiagnostics.FusionQuorum.Attempts[0].Model != "panel-a" {
		t.Fatalf("clone mutation changed original Fusion diagnostics: %+v", record.RouteDiagnostics.FusionQuorum)
	}
}

func TestFusionQuorumDiagnosticsOldRecordCompatibility(t *testing.T) {
	var record Record
	if err := json.Unmarshal([]byte(`{
		"id":"legacy-record",
		"route_diagnostics":{"selection_method":"fusion"}
	}`), &record); err != nil {
		t.Fatalf("unmarshal legacy record: %v", err)
	}
	if record.RouteDiagnostics == nil {
		t.Fatal("legacy route diagnostics missing")
	}
	if record.RouteDiagnostics.FusionQuorum != nil {
		t.Fatalf("legacy record invented Fusion quorum diagnostics: %+v", record.RouteDiagnostics.FusionQuorum)
	}
}

func TestUpdateFusionQuorumBackends(t *testing.T) {
	stores := []struct {
		name string
		init func(t *testing.T) Storage
	}{
		{name: "memory", init: createTestMemoryStore},
		{name: "redis", init: createTestRedisStore},
		{name: "milvus", init: createTestMilvusStore},
		{name: "qdrant", init: createTestQdrantStore},
	}

	quorumPayload := FusionQuorumDiagnostics{
		RequiredCount:  3,
		UsableCount:    1,
		SelectedPolicy: "fallback_model",
		FallbackTarget: "backup-vllm",
		Disposition:    "quorum_failed_fallback",
		Attempts: []FusionPanelAttemptDiagnostics{
			{Model: "panel-1", State: "usable", PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
			{Model: "panel-2", State: "error", PromptTokens: 5, CompletionTokens: 0, TotalTokens: 5},
		},
	}

	for _, backend := range stores {
		t.Run(backend.name, func(t *testing.T) {
			store := backend.init(t)
			ctx := context.Background()

			t.Run("uninitialized RouteDiagnostics", func(t *testing.T) {
				rec := Record{
					ID:        fmt.Sprintf("quorum-uninit-%s", backend.name),
					Timestamp: time.Now().UTC().Truncate(time.Second),
				}
				if _, err := store.Add(ctx, rec); err != nil {
					t.Fatalf("Add: %v", err)
				}

				if err := store.UpdateFusionQuorum(ctx, rec.ID, quorumPayload); err != nil {
					t.Fatalf("UpdateFusionQuorum: %v", err)
				}

				got, found, err := store.Get(ctx, rec.ID)
				if err != nil {
					t.Fatalf("Get: %v", err)
				}
				if !found {
					t.Fatal("record not found after UpdateFusionQuorum")
				}
				if got.RouteDiagnostics == nil {
					t.Fatal("RouteDiagnostics nil after UpdateFusionQuorum")
				}
				assertFusionQuorumEqual(t, got.RouteDiagnostics.FusionQuorum, &quorumPayload)
			})

			t.Run("existing RouteDiagnostics preserved", func(t *testing.T) {
				zero := 0.0
				accepted := true
				rec := Record{
					ID:        fmt.Sprintf("quorum-existing-%s", backend.name),
					Timestamp: time.Now().UTC().Truncate(time.Second),
					RouteDiagnostics: &RouteDiagnostics{
						SelectionMethod: "fusion",
						SelectionTrace: &selectiontrace.MultiFactorObjective{
							FinalSurvivors: []config.ModelRef{{Model: "survivor-1"}},
							Stages: []selectiontrace.ObjectiveStage{{
								Factor: "latency", Metric: "ttft", Action: selectiontrace.StageSkipped,
								Candidates: []selectiontrace.CandidateValue{
									{Candidate: config.ModelRef{Model: "candidate-1"}, Value: &zero},
								},
							}},
						},
						Looper: &LooperDiagnostics{
							Version:             1,
							TraceID:             "looper-trace-xyz",
							Algorithm:           "confidence",
							FinalAttemptOrdinal: 1,
							Attempts: []LooperAttempt{{
								Ordinal:  1,
								Stage:    "candidate",
								Status:   "succeeded",
								Accepted: &accepted,
								Selected: true,
								Model:    "looper-model-a",
								Usage:    LooperUsage{PromptTokens: 15, CompletionTokens: 25, TotalTokens: 40},
							}},
						},
					},
				}

				if _, err := store.Add(ctx, rec); err != nil {
					t.Fatalf("Add: %v", err)
				}

				if err := store.UpdateFusionQuorum(ctx, rec.ID, quorumPayload); err != nil {
					t.Fatalf("UpdateFusionQuorum: %v", err)
				}

				got, found, err := store.Get(ctx, rec.ID)
				if err != nil {
					t.Fatalf("Get: %v", err)
				}
				if !found {
					t.Fatal("record not found")
				}
				if got.RouteDiagnostics == nil {
					t.Fatal("RouteDiagnostics nil")
				}

				// Check quorum fields
				assertFusionQuorumEqual(t, got.RouteDiagnostics.FusionQuorum, &quorumPayload)

				// Check that other diagnostics were not corrupted
				if got.RouteDiagnostics.SelectionMethod != "fusion" {
					t.Errorf("SelectionMethod = %q, want %q", got.RouteDiagnostics.SelectionMethod, "fusion")
				}
				if got.RouteDiagnostics.SelectionTrace == nil || len(got.RouteDiagnostics.SelectionTrace.FinalSurvivors) != 1 ||
					got.RouteDiagnostics.SelectionTrace.FinalSurvivors[0].Model != "survivor-1" {
					t.Errorf("SelectionTrace corrupted: %+v", got.RouteDiagnostics.SelectionTrace)
				}
				if got.RouteDiagnostics.Looper == nil || got.RouteDiagnostics.Looper.TraceID != "looper-trace-xyz" ||
					len(got.RouteDiagnostics.Looper.Attempts) != 1 || got.RouteDiagnostics.Looper.Attempts[0].Model != "looper-model-a" {
					t.Errorf("Looper diagnostics corrupted: %+v", got.RouteDiagnostics.Looper)
				}
			})

			t.Run("non-existent record returns error", func(t *testing.T) {
				if err := store.UpdateFusionQuorum(ctx, "non-existent-id", quorumPayload); err == nil {
					t.Fatal("expected error updating non-existent record, got nil")
				}
			})
		})
	}
}

func assertFusionQuorumEqual(t *testing.T, got, want *FusionQuorumDiagnostics) {
	t.Helper()
	if got == nil {
		t.Fatal("FusionQuorum is nil")
	}
	if got.RequiredCount != want.RequiredCount {
		t.Errorf("RequiredCount = %d, want %d", got.RequiredCount, want.RequiredCount)
	}
	if got.UsableCount != want.UsableCount {
		t.Errorf("UsableCount = %d, want %d", got.UsableCount, want.UsableCount)
	}
	if got.SelectedPolicy != want.SelectedPolicy {
		t.Errorf("SelectedPolicy = %q, want %q", got.SelectedPolicy, want.SelectedPolicy)
	}
	if got.FallbackTarget != want.FallbackTarget {
		t.Errorf("FallbackTarget = %q, want %q", got.FallbackTarget, want.FallbackTarget)
	}
	if got.Disposition != want.Disposition {
		t.Errorf("Disposition = %q, want %q", got.Disposition, want.Disposition)
	}
	if len(got.Attempts) != len(want.Attempts) {
		t.Fatalf("len(Attempts) = %d, want %d", len(got.Attempts), len(want.Attempts))
	}
	for i := range want.Attempts {
		if got.Attempts[i] != want.Attempts[i] {
			t.Errorf("Attempts[%d] = %+v, want %+v", i, got.Attempts[i], want.Attempts[i])
		}
	}
}

func createTestMemoryStore(t *testing.T) Storage {
	return NewMemoryStore(100, 0)
}

func createTestRedisStore(t *testing.T) Storage {
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	return &RedisStore{
		client:    client,
		keyPrefix: "router_replay:",
		lifecycle: newStorageLifecycle(),
	}
}

type mockMilvusClient struct {
	milvusclient.Client
	mu      sync.Mutex
	records map[string]string
}

func (m *mockMilvusClient) Insert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var id, data string
	for _, col := range columns {
		if col.Name() == "id" {
			if vc, ok := col.(*entity.ColumnVarChar); ok && vc.Len() > 0 {
				id, _ = vc.ValueByIdx(0)
			}
		} else if col.Name() == "data" {
			if vc, ok := col.(*entity.ColumnVarChar); ok && vc.Len() > 0 {
				data, _ = vc.ValueByIdx(0)
			}
		}
	}
	if id != "" {
		m.records[id] = data
	}
	return nil, nil
}

func (m *mockMilvusClient) Upsert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
	return m.Insert(ctx, collName, partitionName, columns...)
}

func (m *mockMilvusClient) Query(ctx context.Context, collName string, partitions []string, expr string, outputFields []string, opts ...milvusclient.SearchQueryOptionFunc) (milvusclient.ResultSet, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var id string
	if strings.HasPrefix(expr, "id == '") && strings.HasSuffix(expr, "'") {
		id = strings.TrimSuffix(strings.TrimPrefix(expr, "id == '"), "'")
	} else if strings.HasPrefix(expr, `id == "`) && strings.HasSuffix(expr, `"`) {
		id = strings.TrimSuffix(strings.TrimPrefix(expr, `id == "`), `"`)
	}
	data, ok := m.records[id]
	if !ok {
		return milvusclient.ResultSet{}, nil
	}
	return milvusclient.ResultSet{entity.NewColumnVarChar("data", []string{data})}, nil
}

func (m *mockMilvusClient) Flush(ctx context.Context, collName string, async bool, opts ...milvusclient.FlushOption) error {
	return nil
}

func (m *mockMilvusClient) Close() error {
	return nil
}

func createTestMilvusStore(t *testing.T) Storage {
	fake := &mockMilvusClient{records: make(map[string]string)}
	return &MilvusStore{
		client:           fake,
		collectionName:   "router_replay",
		consistencyLevel: entity.ClStrong,
		lifecycle:        newStorageLifecycle(),
	}
}

type mockPointsServer struct {
	qdrant.UnimplementedPointsServer
	mu     sync.Mutex
	points map[string]*qdrant.PointStruct
}

func (s *mockPointsServer) Upsert(ctx context.Context, req *qdrant.UpsertPoints) (*qdrant.PointsOperationResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, pt := range req.Points {
		if pt.Id != nil {
			s.points[pt.Id.GetUuid()] = pt
		}
	}
	return &qdrant.PointsOperationResponse{
		Result: &qdrant.UpdateResult{Status: qdrant.UpdateStatus_Completed},
	}, nil
}

func (s *mockPointsServer) Get(ctx context.Context, req *qdrant.GetPoints) (*qdrant.GetResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var pts []*qdrant.RetrievedPoint
	for _, id := range req.Ids {
		if pt, ok := s.points[id.GetUuid()]; ok {
			pts = append(pts, &qdrant.RetrievedPoint{
				Id:      pt.Id,
				Payload: pt.Payload,
			})
		}
	}
	return &qdrant.GetResponse{Result: pts}, nil
}

type mockCollectionsServer struct {
	qdrant.UnimplementedCollectionsServer
}

func (s *mockCollectionsServer) CollectionExists(ctx context.Context, req *qdrant.CollectionExistsRequest) (*qdrant.CollectionExistsResponse, error) {
	return &qdrant.CollectionExistsResponse{
		Result: &qdrant.CollectionExists{Exists: true},
	}, nil
}

func createTestQdrantStore(t *testing.T) Storage {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	ptsSrv := &mockPointsServer{points: make(map[string]*qdrant.PointStruct)}
	colSrv := &mockCollectionsServer{}
	qdrant.RegisterPointsServer(grpcServer, ptsSrv)
	qdrant.RegisterCollectionsServer(grpcServer, colSrv)
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		_ = lis.Close()
	})

	port := lis.Addr().(*net.TCPAddr).Port
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:                   "127.0.0.1",
		Port:                   port,
		UseTLS:                 false,
		SkipCompatibilityCheck: true,
	})
	if err != nil {
		t.Fatalf("qdrant.NewClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })

	return &QdrantStore{
		client:         client,
		collectionName: "router_replay",
		lifecycle:      newStorageLifecycle(),
	}
}
