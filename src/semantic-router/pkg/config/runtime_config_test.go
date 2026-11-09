package config

import "testing"

// LooperConfig.GetMaxResponseBytes bounds a single upstream model response so
// an oversized body cannot exhaust router memory. It mirrors the existing
// GetGRPCMaxMsgSize pattern: an MB knob with a safe default when unset.

func TestLooperConfigGetMaxResponseBytes_DefaultWhenUnset(t *testing.T) {
	cfg := &LooperConfig{}

	if got := cfg.GetMaxResponseBytes(); got != DefaultMaxResponseBytes {
		t.Errorf("GetMaxResponseBytes() = %d, want default %d", got, DefaultMaxResponseBytes)
	}
}

func TestLooperConfigGetMaxResponseBytes_ExplicitMB(t *testing.T) {
	cfg := &LooperConfig{MaxResponseBytesMB: 4}

	want := int64(4) * 1024 * 1024
	if got := cfg.GetMaxResponseBytes(); got != want {
		t.Errorf("GetMaxResponseBytes() = %d, want %d", got, want)
	}
}

func TestLooperConfigGetMaxResponseBytes_NonPositiveFallsBackToDefault(t *testing.T) {
	cfg := &LooperConfig{MaxResponseBytesMB: -5}

	if got := cfg.GetMaxResponseBytes(); got != DefaultMaxResponseBytes {
		t.Errorf("GetMaxResponseBytes() = %d, want default %d for non-positive input", got, DefaultMaxResponseBytes)
	}
}

func TestLooperConfigIsStreamingEnabled(t *testing.T) {
	t.Run("nil receiver defaults to true", func(t *testing.T) {
		var cfg *LooperConfig
		if !cfg.IsStreamingEnabled() {
			t.Errorf("IsStreamingEnabled() = false, want true for nil LooperConfig")
		}
	})

	t.Run("nil Enabled defaults to true (Milestone 2 default)", func(t *testing.T) {
		cfg := &LooperConfig{}
		if !cfg.IsStreamingEnabled() {
			t.Errorf("IsStreamingEnabled() = false, want true when Streaming.Enabled is nil")
		}
	})

	t.Run("explicitly enabled true", func(t *testing.T) {
		enabled := true
		cfg := &LooperConfig{Streaming: LooperStreamingConfig{Enabled: &enabled}}
		if !cfg.IsStreamingEnabled() {
			t.Errorf("IsStreamingEnabled() = false, want true when Streaming.Enabled is true")
		}
	})

	t.Run("explicitly disabled false falls back to buffering", func(t *testing.T) {
		disabled := false
		cfg := &LooperConfig{Streaming: LooperStreamingConfig{Enabled: &disabled}}
		if cfg.IsStreamingEnabled() {
			t.Errorf("IsStreamingEnabled() = true, want false when Streaming.Enabled is false")
		}
	})
}
