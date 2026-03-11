package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/geminicli"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

// ForwardOpenAIChat proxies an OpenAI chat/completions request directly to Google's OpenAI compat endpoint.
func (s *GeminiMessagesCompatService) ForwardOpenAIChat(ctx context.Context, c *gin.Context, account *Account, originalModel string, stream bool, body []byte) (*ForwardResult, error) {
	startTime := time.Now()

	mappedModel := account.GetMappedModel(originalModel)

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	var requestIDHeader string
	var buildReq func(ctx context.Context) (*http.Request, string, error)

	switch account.Type {
	case AccountTypeAPIKey:
		buildReq = func(ctx context.Context) (*http.Request, string, error) {
			apiKey := account.GetCredential("api_key")
			if strings.TrimSpace(apiKey) == "" {
				return nil, "", errors.New("gemini api_key not configured")
			}

			baseURL := account.GetGeminiBaseURL(geminicli.AIStudioBaseURL)
			normalizedBaseURL, err := s.validateUpstreamBaseURL(baseURL)
			if err != nil {
				return nil, "", err
			}

			fullURL := fmt.Sprintf("%s/v1beta/openai/chat/completions", strings.TrimRight(normalizedBaseURL, "/"))

			// Inject the mapped model into the body
			var reqMap map[string]interface{}
			if err := json.Unmarshal(body, &reqMap); err == nil {
				reqMap["model"] = mappedModel
				if newBody, err := json.Marshal(reqMap); err == nil {
					body = newBody
				}
			}

			upstreamReq, err := http.NewRequestWithContext(ctx, http.MethodPost, fullURL, bytes.NewReader(body))
			if err != nil {
				return nil, "", err
			}
			upstreamReq.Header.Set("Content-Type", "application/json")
			upstreamReq.Header.Set("x-goog-api-key", apiKey)
			return upstreamReq, "x-request-id", nil
		}
		requestIDHeader = "x-request-id"

	case AccountTypeOAuth:
		buildReq = func(ctx context.Context) (*http.Request, string, error) {
			if s.tokenProvider == nil {
				return nil, "", errors.New("gemini token provider not configured")
			}
			accessToken, err := s.tokenProvider.GetAccessToken(ctx, account)
			if err != nil {
				return nil, "", err
			}

			baseURL := account.GetGeminiBaseURL(geminicli.AIStudioBaseURL)
			normalizedBaseURL, err := s.validateUpstreamBaseURL(baseURL)
			if err != nil {
				return nil, "", err
			}

			fullURL := fmt.Sprintf("%s/v1beta/openai/chat/completions", strings.TrimRight(normalizedBaseURL, "/"))

			// Inject the mapped model into the body
			var reqMap map[string]interface{}
			if err := json.Unmarshal(body, &reqMap); err == nil {
				reqMap["model"] = mappedModel
				if newBody, err := json.Marshal(reqMap); err == nil {
					body = newBody
				}
			}

			upstreamReq, err := http.NewRequestWithContext(ctx, http.MethodPost, fullURL, bytes.NewReader(body))
			if err != nil {
				return nil, "", err
			}
			upstreamReq.Header.Set("Content-Type", "application/json")
			upstreamReq.Header.Set("Authorization", "Bearer "+accessToken)
			return upstreamReq, "x-request-id", nil
		}
		requestIDHeader = "x-request-id"

	default:
		return nil, fmt.Errorf("unsupported account type for openai compat: %s", account.Type)
	}

	var resp *http.Response
	for attempt := 1; attempt <= geminiMaxRetries; attempt++ {
		upstreamReq, idHeader, err := buildReq(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			return nil, s.writeGoogleError(c, http.StatusBadGateway, err.Error())
		}
		requestIDHeader = idHeader

		if c != nil {
			c.Set(OpsUpstreamRequestBodyKey, string(body))
		}

		resp, err = s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
		if err != nil {
			safeErr := sanitizeUpstreamErrorMessage(err.Error())
			appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
				Platform:           account.Platform,
				AccountID:          account.ID,
				AccountName:        account.Name,
				UpstreamStatusCode: 0,
				Kind:               "request_error",
				Message:            safeErr,
			})
			if attempt < geminiMaxRetries && s.shouldRetryGeminiUpstreamError(account, 0) {
				logger.LegacyPrintf("service.gemini_openai_compat", "Gemini account %d: upstream request failed, retry %d/%d: %v", account.ID, attempt, geminiMaxRetries, err)
				sleepGeminiBackoff(attempt)
				continue
			}
			setOpsUpstreamError(c, 0, safeErr, "")
			return nil, s.writeGoogleError(c, http.StatusBadGateway, "Upstream request failed after retries: "+safeErr)
		}

		if resp.StatusCode >= 400 {
			respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
			_ = resp.Body.Close()

			s.handleGeminiUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)

			if attempt < geminiMaxRetries && s.shouldRetryGeminiUpstreamError(account, resp.StatusCode) {
				logger.LegacyPrintf("service.gemini_openai_compat", "Gemini account %d: upstream returned %d, retry %d/%d", account.ID, resp.StatusCode, attempt, geminiMaxRetries)
				sleepGeminiBackoff(attempt)
				continue
			}

			if s.shouldFailoverGeminiUpstreamError(resp.StatusCode) {
				return nil, &UpstreamFailoverError{
					StatusCode:        resp.StatusCode,
					ResponseBody:      respBody,
					ForceCacheBilling: false,
				}
			}

			requestID := ""
			if requestIDHeader != "" {
				requestID = resp.Header.Get(requestIDHeader)
			}
			return nil, s.writeGeminiMappedError(c, account, resp.StatusCode, requestID, respBody)
		}

		break
	}

	defer func() {
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}
	}()

	requestID := ""
	if requestIDHeader != "" {
		requestID = resp.Header.Get(requestIDHeader)
	}

	var usage ClaudeUsage
	var firstTokenMs *int

	if stream {
		if s.responseHeaderFilter != nil {
			responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
		}

		c.Status(resp.StatusCode)
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("X-Accel-Buffering", "no")

		contentType := resp.Header.Get("Content-Type")
		if contentType == "" {
			contentType = "text/event-stream; charset=utf-8"
		}
		c.Header("Content-Type", contentType)

		flusher, ok := c.Writer.(http.Flusher)
		if !ok {
			return nil, errors.New("streaming not supported")
		}

		reader := bufio.NewReader(resp.Body)
		for {
			line, err := reader.ReadString('\n')
			if len(line) > 0 {
				_, _ = io.WriteString(c.Writer, line)
				flusher.Flush()

				if firstTokenMs == nil {
					ms := int(time.Since(startTime).Milliseconds())
					firstTokenMs = &ms
				}

				if strings.HasPrefix(line, "data: ") {
					payload := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
					if payload != "" && payload != "[DONE]" {
						if gjson.Get(payload, "usage").Exists() {
							usage = ClaudeUsage{
								InputTokens:  int(gjson.Get(payload, "usage.prompt_tokens").Int()),
								OutputTokens: int(gjson.Get(payload, "usage.completion_tokens").Int()),
							}
						}
					}
				}
			}

			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return nil, err
			}
		}
	} else {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}

		if gjson.Get(string(bodyBytes), "usage").Exists() {
			usage = ClaudeUsage{
				InputTokens:  int(gjson.Get(string(bodyBytes), "usage.prompt_tokens").Int()),
				OutputTokens: int(gjson.Get(string(bodyBytes), "usage.completion_tokens").Int()),
			}
		}

		if s.responseHeaderFilter != nil {
			responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
		}
		contentType := resp.Header.Get("Content-Type")
		if contentType == "" {
			contentType = "application/json"
		}
		c.Data(resp.StatusCode, contentType, bodyBytes)
	}

	return &ForwardResult{
		RequestID:    requestID,
		Model:        originalModel,
		Stream:       stream,
		Duration:     time.Since(startTime),
		FirstTokenMs: firstTokenMs,
		Usage:        usage,
	}, nil
}
