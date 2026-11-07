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

package looper

import (
	"context"

	"github.com/openai/openai-go"
)

// StreamingSynthesisPlan contains the prepared parameters and intermediate
// evidence needed to dispatch a native streaming final stage through Envoy.
type StreamingSynthesisPlan struct {
	SynthesisParams   *openai.ChatCompletionNewParams
	TargetModel       string               // Upstream model target for dispatch
	IntermediateUsage TokenUsage           // Aggregated usage from intermediate stages (panel + analysis)
	Trace             any                  // Intermediate execution evidence for replay (e.g., *FusionTrace, *workflowTrace, []RoundResponse)
	ModelsUsed        []string             // Unique models that contributed to intermediate stages
	Iterations        int                  // Number of intermediate iterations executed
	QuorumOutcome     *FusionQuorumOutcome // Quorum decision evidence (if fallback occurred)
}

// StreamingLooper is optionally implemented by loopers that support preparing
// native streaming execution for eligible final stages.
type StreamingLooper interface {
	Looper
	PrepareStreamingSynthesis(ctx context.Context, req *Request) (*StreamingSynthesisPlan, error)
}
