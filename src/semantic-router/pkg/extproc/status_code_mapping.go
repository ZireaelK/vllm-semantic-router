package extproc

import (
	"net/http"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func statusCodeToImmediateResponseCode(statusCode int) typev3.StatusCode {
	// Immediate responses are final HTTP responses. Check the range before
	// narrowing to int32, and use Envoy's enum rather than a partial local list.
	if statusCode >= http.StatusOK && statusCode <= 599 {
		if _, defined := typev3.StatusCode_name[int32(statusCode)]; defined {
			return typev3.StatusCode(statusCode)
		}
	}
	return typev3.StatusCode_InternalServerError
}

// An error body must never be paired with a success or redirect status.
func errorResponseStatusCode(statusCode int) int {
	code := int(statusCodeToImmediateResponseCode(statusCode))
	if code < http.StatusBadRequest {
		return http.StatusInternalServerError
	}
	return code
}

func protocolErrorToHTTPStatus(protocolError *llmprotocol.ProtocolError) int {
	if protocolError == nil {
		return http.StatusBadRequest
	}
	switch protocolError.Category {
	case llmprotocol.ErrorNotFound:
		return http.StatusNotFound
	case llmprotocol.ErrorAuthentication:
		return http.StatusUnauthorized
	case llmprotocol.ErrorPermission:
		return http.StatusForbidden
	case llmprotocol.ErrorConflict:
		return http.StatusConflict
	case llmprotocol.ErrorRateLimited:
		return http.StatusTooManyRequests
	case llmprotocol.ErrorUpstreamTimeout:
		return http.StatusGatewayTimeout
	case llmprotocol.ErrorUpstreamUnavailable:
		return http.StatusBadGateway
	case llmprotocol.ErrorInternal:
		return http.StatusInternalServerError
	case llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorUnsupportedFeature:
		return http.StatusBadRequest
	default:
		return http.StatusBadRequest
	}
}
