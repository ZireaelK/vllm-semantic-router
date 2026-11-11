package budget

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

func TestConcurrentConsumersCannotExceedRequestBudget(t *testing.T) {
	ctx, ledger := WithLimit(context.Background(), 7)
	var successes atomic.Int64
	var group sync.WaitGroup
	for range 100 {
		group.Go(func() {
			if err := Consume(ctx); err == nil {
				successes.Add(1)
			} else if !errors.Is(err, ErrExhausted) {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
	group.Wait()
	if successes.Load() != 7 || ledger.Used() != 7 || ledger.Remaining() != 0 {
		t.Fatalf("successes=%d used=%d remaining=%d", successes.Load(), ledger.Used(), ledger.Remaining())
	}
}

func TestNestedAndCanceledContextsCannotReplenishBudget(t *testing.T) {
	ctx, ledger := WithLimit(context.Background(), 1)
	child, other := WithLimit(ctx, 100)
	if other != ledger {
		t.Fatal("nested request acquired a new budget")
	}
	if err := Consume(child); err != nil {
		t.Fatal(err)
	}
	if err := Consume(ctx); !errors.Is(err, ErrExhausted) {
		t.Fatalf("expected exhaustion, got %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Consume(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
}

func TestLedgerRelease(t *testing.T) {
	ctx, ledger := WithLimit(context.Background(), 5)
	for range 5 {
		if err := Consume(ctx); err != nil {
			t.Fatalf("unexpected consume error: %v", err)
		}
	}
	if ledger.Remaining() != 0 || ledger.Used() != 5 {
		t.Fatalf("used=%d remaining=%d want 5, 0", ledger.Used(), ledger.Remaining())
	}
	if err := Consume(ctx); !errors.Is(err, ErrExhausted) {
		t.Fatalf("expected ErrExhausted, got %v", err)
	}

	// Release 2 unused attempts
	Release(ctx, 2)
	if ledger.Remaining() != 2 || ledger.Used() != 3 {
		t.Fatalf("after Release(2): used=%d remaining=%d want 3, 2", ledger.Used(), ledger.Remaining())
	}

	// Consume 1 should succeed
	if err := Consume(ctx); err != nil {
		t.Fatalf("expected consume to succeed after release, got %v", err)
	}
	if ledger.Remaining() != 1 || ledger.Used() != 4 {
		t.Fatalf("used=%d remaining=%d want 4, 1", ledger.Used(), ledger.Remaining())
	}

	// Releasing more than used clamps to 0 used
	ledger.Release(10)
	if ledger.Used() != 0 || ledger.Remaining() != 5 {
		t.Fatalf("after ledger.Release(10): used=%d remaining=%d want 0, 5", ledger.Used(), ledger.Remaining())
	}

	// Releasing <= 0 or on nil context does nothing
	Release(ctx, 0)
	Release(ctx, -1)
	Release(nil, 2)
	(*Ledger)(nil).Release(1)
	if ledger.Used() != 0 || ledger.Remaining() != 5 {
		t.Fatalf("no-op releases changed state: used=%d remaining=%d", ledger.Used(), ledger.Remaining())
	}
}
