# Scopus Search API quota is weekly, not per-second — see rate_limiter.EvenSpacing.
WEEKLY_QUOTA = 20_000  # requests/week; adjust to match your Elsevier entitlement
QUOTA_PERIOD_SECONDS = 7 * 24 * 3600
LOG_LEVEL = "INFO"
