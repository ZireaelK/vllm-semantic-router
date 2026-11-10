package extproc

import (
	"slices"
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

var _ routing.ReliabilitySession = (*routingSession)(nil)

// Reliability is the planned call's override: the matched decision's
// reliability block, bounded by the remaining physical call budget during
// looper streaming.
func (s *routingSession) Reliability() *routing.Reliability {
	if s == nil || s.ctx == nil {
		return nil
	}
	override := callReliability(s.ctx.VSRSelectedDecision)
	if !s.ctx.LooperStreamingActive {
		return override
	}

	route := s.targetRoute()
	configuredRetries := configuredRetriesFromDecisionOrModel(s.router, s.ctx, override, route)
	maxAllowedRetries := max(0, s.ctx.LooperRemainingBudget-1)
	effectiveRetries := min(configuredRetries, maxAllowedRetries)

	out := override
	if out == nil {
		out = &routing.Reliability{}
	}
	out.RetryCount = &effectiveRetries
	return out
}

func (s *routingSession) targetRoute() string {
	if s == nil {
		return ""
	}
	if s.ctx != nil && s.ctx.LooperStreamingActive {
		candidate := firstNonEmpty(s.ctx.LooperTargetModel, s.ctx.VSRSelectedModel, s.ctx.RequestModel)
		route := s.ctx.backendModelForCandidate(candidate)
		if route != "" {
			return route
		}
		return candidate
	}
	var route string
	if s.clientHeader != nil {
		route = s.clientHeader.Get(routing.RouteHeader)
	}
	if route == "" && s.ctx != nil {
		candidate := firstNonEmpty(s.ctx.LooperTargetModel, s.ctx.VSRSelectedModel, s.ctx.RequestModel)
		route = s.ctx.backendModelForCandidate(candidate)
	}
	return route
}

// configuredRetriesFromDecisionOrModel determines the configured retry count
// prioritizing decision reliability overrides, then falling back to backend model config.
func configuredRetriesFromDecisionOrModel(r *OpenAIRouter, ctx *RequestContext, override *routing.Reliability, route string) int {
	if override != nil && override.RetryCount != nil {
		return max(0, *override.RetryCount)
	}
	if r == nil || r.Config == nil {
		return 0
	}
	target := route
	if target == "" && ctx != nil {
		candidate := firstNonEmpty(ctx.LooperTargetModel, ctx.VSRSelectedModel, ctx.RequestModel)
		target = ctx.backendModelForCandidate(candidate)
		if target == "" {
			target = candidate
		}
	}
	if target != "" {
		if modelCfg, ok := r.Config.ModelConfig[target]; ok {
			return max(0, modelCfg.Reliability.RetryCount)
		}
	}
	if ctx != nil && ctx.LooperStreamingActive {
		// During looper streaming, targetRoute already resolved the target model candidate.
		// If route or LooperTargetModel is not configured in ModelConfig, retries default
		// to 0 and must never fall back to unrelated models (e.g. from client header or prior turns).
		if ctx.LooperTargetModel != "" && ctx.LooperTargetModel != target {
			if modelCfg, ok := r.Config.ModelConfig[ctx.LooperTargetModel]; ok {
				return max(0, modelCfg.Reliability.RetryCount)
			}
		}
		return 0
	}
	if ctx != nil && ctx.LooperTargetModel != "" {
		if modelCfg, ok := r.Config.ModelConfig[ctx.LooperTargetModel]; ok {
			return max(0, modelCfg.Reliability.RetryCount)
		}
	} else if ctx != nil && ctx.VSRSelectedModel != "" {
		if modelCfg, ok := r.Config.ModelConfig[ctx.VSRSelectedModel]; ok {
			return max(0, modelCfg.Reliability.RetryCount)
		}
	}
	return 0
}

// callReliability converts a decision's reliability block to the routing
// core's neutral override; nil when the decision sets none.
func callReliability(decision *config.Decision) *routing.Reliability {
	if decision == nil || decision.Reliability == nil {
		return nil
	}
	r := decision.Reliability
	out := &routing.Reliability{
		TotalTimeout:         reliabilityDuration(r.TotalTimeout),
		PerTryTimeout:        reliabilityDuration(r.PerTryTimeout),
		IdleTimeout:          reliabilityDuration(r.IdleTimeout),
		FirstByteTimeout:     reliabilityDuration(r.FirstByteTimeout),
		RetriableStatusCodes: slices.Clone(r.RetriableStatusCodes),
		RetryBackOffBase:     reliabilityDuration(r.RetryBackOffBase),
		RetryBackOffMax:      reliabilityDuration(r.RetryBackOffMax),
		RetryAfterMax:        reliabilityDuration(r.RetryAfterMax),
	}
	if r.RetryCount != nil {
		count := *r.RetryCount
		out.RetryCount = &count
	}
	for _, token := range strings.Split(r.RetryOn, ",") {
		if token = strings.TrimSpace(token); token != "" {
			out.RetryOn = append(out.RetryOn, token)
		}
	}
	return out
}

// reliabilityDuration reads a duration the config loader validated; empty is
// unset.
func reliabilityDuration(value string) *time.Duration {
	d, err := time.ParseDuration(strings.TrimSpace(value))
	if err != nil {
		return nil
	}
	return &d
}

// The per-request headers Envoy's router reads after ext_proc's mutations.
const (
	envoyRouteTimeoutHeader         = "x-envoy-upstream-rq-timeout-ms"
	envoyPerTryTimeoutHeader        = "x-envoy-upstream-rq-per-try-timeout-ms"
	envoyMaxRetriesHeader           = "x-envoy-max-retries"
	envoyRetryOnHeader              = "x-envoy-retry-on"
	envoyRetriableStatusCodesHeader = "x-envoy-retriable-status-codes"
)

// addEnvoyReliabilityHeaders asks Envoy's router to apply the matched
// decision's override to this request. It runs only where the Router answers
// Envoy over gRPC: the native gateway takes the same override from
// routing.Call.Reliability, so its requests never carry these headers.
func (r *OpenAIRouter) addEnvoyReliabilityHeaders(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if ctx == nil || response == nil || response.GetImmediateResponse() != nil {
		return
	}
	override := callReliability(ctx.VSRSelectedDecision)
	if override == nil && !ctx.LooperStreamingActive {
		return
	}
	// The route mutation travels with the held header reply in full-duplex
	// mode, and with the body reply otherwise.
	var mutation *ext_proc.HeaderMutation
	if hold := ctx.fullDuplexHold; hold != nil {
		if hold.routeMutation == nil {
			hold.routeMutation = &ext_proc.HeaderMutation{}
		}
		mutation = hold.routeMutation
	} else if body := response.GetRequestBody(); body != nil {
		if body.Response == nil {
			body.Response = &ext_proc.CommonResponse{}
		}
		if body.Response.HeaderMutation == nil {
			body.Response.HeaderMutation = &ext_proc.HeaderMutation{}
		}
		mutation = body.Response.HeaderMutation
	} else {
		return
	}
	route := routeKeyOf(mutation)
	if route == "" {
		route = ctx.backendModelForCandidate(firstNonEmpty(ctx.LooperTargetModel, ctx.VSRSelectedModel))
	}

	configuredRetries := configuredRetriesFromDecisionOrModel(r, ctx, override, route)

	var finalRetries int
	if ctx.LooperStreamingActive {
		maxAllowed := max(0, ctx.LooperRemainingBudget-1)
		finalRetries = min(configuredRetries, maxAllowed)
		if override != nil {
			override.RetryCount = &finalRetries
		}
	}

	if override != nil {
		mutation.SetHeaders = append(mutation.SetHeaders, envoyReliabilityHeaders(override, r.routeHasRetryPolicy(route))...)
	}
	if ctx.LooperStreamingActive {
		setEnvoyHeader(mutation, envoyMaxRetriesHeader, strconv.Itoa(finalRetries))
	}
}

func setEnvoyHeader(mutation *ext_proc.HeaderMutation, key, value string) {
	for _, option := range mutation.SetHeaders {
		if option.Header != nil && strings.EqualFold(option.Header.Key, key) {
			option.Header.RawValue = []byte(value)
			option.Header.Value = ""
			option.AppendAction = core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD
			return
		}
	}
	mutation.SetHeaders = append(mutation.SetHeaders, &core.HeaderValueOption{
		Header:       &core.HeaderValue{Key: key, RawValue: []byte(value)},
		AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
	})
}

// envoyReliabilityHeaders renders an override as Envoy's per-request
// headers: the route and per-try timeouts in milliseconds (0 disables them),
// the retry count, and the retry conditions and statuses Envoy adds to the
// route's. A route without a retry policy has no conditions for the count to
// apply to, so it gets the default ones, as the native merge does. Envoy has
// no per-request header for the other fields; validation keeps them to the
// native gateway.
func envoyReliabilityHeaders(r *routing.Reliability, routeRetries bool) []*core.HeaderValueOption {
	var out []*core.HeaderValueOption
	set := func(name, value string) {
		out = append(out, &core.HeaderValueOption{
			Header:       &core.HeaderValue{Key: name, RawValue: []byte(value)},
			AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
		})
	}
	if r.TotalTimeout != nil {
		set(envoyRouteTimeoutHeader, strconv.FormatInt(r.TotalTimeout.Milliseconds(), 10))
	}
	if r.PerTryTimeout != nil {
		set(envoyPerTryTimeoutHeader, strconv.FormatInt(r.PerTryTimeout.Milliseconds(), 10))
	}
	if r.RetryCount != nil {
		set(envoyMaxRetriesHeader, strconv.Itoa(*r.RetryCount))
	}
	retryOn := r.RetryOn
	if len(retryOn) == 0 && !routeRetries && (r.RetryCount != nil || len(r.RetriableStatusCodes) > 0) {
		retryOn = []string{config.DefaultProviderRetryOn}
	}
	if len(retryOn) > 0 {
		set(envoyRetryOnHeader, strings.Join(retryOn, ","))
	}
	if len(r.RetriableStatusCodes) > 0 {
		codes := make([]string, len(r.RetriableStatusCodes))
		for i, code := range r.RetriableStatusCodes {
			codes[i] = strconv.Itoa(code)
		}
		set(envoyRetriableStatusCodesHeader, strings.Join(codes, ","))
	}
	return out
}

// routeHasRetryPolicy reports whether the Envoy route for a provider model
// has a retry policy: the template renders one for retries or a per-try
// timeout.
func (r *OpenAIRouter) routeHasRetryPolicy(model string) bool {
	if r == nil || r.Config == nil {
		return false
	}
	reliability := r.Config.ModelConfig[model].Reliability
	return reliability.RetryCount > 0 || strings.TrimSpace(reliability.PerTryTimeout) != ""
}

// routeKeyOf is the route key a header mutation sets, if any.
func routeKeyOf(mutation *ext_proc.HeaderMutation) string {
	for _, option := range mutation.GetSetHeaders() {
		if strings.EqualFold(option.GetHeader().GetKey(), routing.RouteHeader) {
			return string(option.GetHeader().GetRawValue())
		}
	}
	return ""
}
