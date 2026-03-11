package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"github.com/Wei-Shaw/sub2api/internal/pkg/ip"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func (h *GatewayHandler) GeminiOpenAIChatCompletions(c *gin.Context) {
	apiKey, ok := middleware.GetAPIKeyFromContext(c)
	if !ok || apiKey == nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid API key"})
		return
	}
	authSubject, ok := middleware.GetAuthSubjectFromContext(c)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "User context not found"})
		return
	}

	reqLog := requestLogger(
		c,
		"handler.gemini_openai.chat_completions",
		zap.Int64("user_id", authSubject.UserID),
		zap.Int64("api_key_id", apiKey.ID),
		zap.Any("group_id", apiKey.GroupID),
	)

	if apiKey.Group == nil || apiKey.Group.Platform != service.PlatformGemini {
		c.JSON(http.StatusBadRequest, gin.H{"error": "API key group platform is not gemini"})
		return
	}

	body, err := httputil.ReadRequestBodyWithPrealloc(c.Request)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request body"})
		return
	}

	var req struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON"})
		return
	}

	if strings.TrimSpace(req.Model) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Missing model"})
		return
	}

	stream := req.Stream
	modelName := req.Model
	reqLog = reqLog.With(zap.String("model", modelName), zap.Bool("stream", stream))

	setOpsRequestContext(c, modelName, stream, body)

	subscription, _ := middleware.GetSubscriptionFromContext(c)

	geminiConcurrency := NewConcurrencyHelper(h.concurrencyHelper.concurrencyService, SSEPingFormatNone, 0)
	maxWait := service.CalculateMaxWait(authSubject.Concurrency)
	canWait, err := geminiConcurrency.IncrementWaitCount(c.Request.Context(), authSubject.UserID, maxWait)
	waitCounted := false
	if err == nil && canWait {
		waitCounted = true
	} else if !canWait {
		c.JSON(http.StatusTooManyRequests, gin.H{"error": "Too many pending requests, please retry later"})
		return
	}

	defer func() {
		if waitCounted {
			geminiConcurrency.DecrementWaitCount(c.Request.Context(), authSubject.UserID)
		}
	}()

	streamStarted := false
	if h.errorPassthroughService != nil {
		service.BindErrorPassthroughService(c, h.errorPassthroughService)
	}

	userReleaseFunc, err := geminiConcurrency.AcquireUserSlotWithWait(c, authSubject.UserID, authSubject.Concurrency, stream, &streamStarted)
	if err != nil {
		c.JSON(http.StatusTooManyRequests, gin.H{"error": err.Error()})
		return
	}
	if waitCounted {
		geminiConcurrency.DecrementWaitCount(c.Request.Context(), authSubject.UserID)
		waitCounted = false
	}
	userReleaseFunc = wrapReleaseOnDone(c.Request.Context(), userReleaseFunc)
	if userReleaseFunc != nil {
		defer userReleaseFunc()
	}

	if err := h.billingCacheService.CheckBillingEligibility(c.Request.Context(), apiKey.User, apiKey, apiKey.Group, subscription); err != nil {
		c.JSON(http.StatusPaymentRequired, gin.H{"error": err.Error()})
		return
	}

	// Calculate session hash
	parsedReq := &service.ParsedRequest{
		SessionContext: &service.SessionContext{
			ClientIP:  ip.GetClientIP(c),
			UserAgent: c.GetHeader("User-Agent"),
			APIKeyID:  apiKey.ID,
		},
		Model: modelName,
	}
	sessionHash := h.gatewayService.GenerateSessionHash(parsedReq)

	// We use maxAccountSwitchesGemini
	fs := NewFailoverState(h.maxAccountSwitchesGemini, false)

	for {
		selection, err := h.geminiCompatService.SelectAccountForModel(c.Request.Context(), apiKey.GroupID, sessionHash, modelName)
		if err != nil {
			if !streamStarted {
				c.JSON(http.StatusServiceUnavailable, gin.H{"error": "No available accounts: " + err.Error()})
			}
			return
		}
		account := selection

		result, err := h.geminiCompatService.ForwardOpenAIChat(c.Request.Context(), c, account, modelName, stream, body)
		if err != nil {
			var failoverErr *service.UpstreamFailoverError
			if errors.As(err, &failoverErr) {
				action := fs.HandleFailoverError(c.Request.Context(), h.gatewayService, account.ID, account.Platform, failoverErr)
				switch action {
				case FailoverContinue:
					continue
				case FailoverExhausted:
					if !streamStarted {
						c.JSON(http.StatusBadGateway, gin.H{"error": "Upstream request failed after failover"})
					}
					return
				case FailoverCanceled:
					return
				}
			}
			if !streamStarted {
				c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			}
			return
		}

		// Forward success, record usage
		userAgent := c.GetHeader("User-Agent")
		clientIP := c.ClientIP()

		h.submitUsageRecordTask(func(ctx context.Context) {
			if err := h.gatewayService.RecordUsage(ctx, &service.RecordUsageInput{
				Result:            result,
				APIKey:            apiKey,
				User:              apiKey.User,
				Account:           account,
				Subscription:      subscription,
				UserAgent:         userAgent,
				IPAddress:         clientIP,
				ForceCacheBilling: false,
				APIKeyService:     h.apiKeyService,
			}); err != nil {
				logger.L().Error("gemini_openai.record_usage_failed", zap.Error(err))
			}
		})
		return
	}
}
