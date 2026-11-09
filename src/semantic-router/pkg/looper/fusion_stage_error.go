package looper

import (
	"fmt"
)

// FusionStageFailureError represents a failure during post-panel fusion stages
// (e.g. grounding, analysis, stage context validation) where quorum was met
// but a subsequent step failed. It carries intermediate usage for accounting
// without falsely triggering quorum failure alarms.
type FusionStageFailureError struct {
	Cause      error
	ModelsUsed []string
	Iterations int
	Usage      TokenUsage
}

func (e *FusionStageFailureError) Error() string {
	if e == nil {
		return ""
	}
	if e.Cause != nil {
		return fmt.Sprintf("fusion stage failure: %v", e.Cause)
	}
	return "fusion stage failure"
}

func (e *FusionStageFailureError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

func NewFusionStageFailureError(cause error, modelsUsed []string, iterations int, usage TokenUsage) *FusionStageFailureError {
	return &FusionStageFailureError{
		Cause:      cause,
		ModelsUsed: append([]string(nil), modelsUsed...),
		Iterations: iterations,
		Usage:      usage,
	}
}
