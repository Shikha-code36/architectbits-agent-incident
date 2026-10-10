"""A fixed, simulated production incident. Pure data, no behavior.

Scenario: after a deploy, checkout-api latency jumps from ~200ms to ~4s.
The true cause is a config change that shrank the database connection pool.
"""

INCIDENT_ALERT = (
    "ALERT: checkout-api p99 latency rose from ~200ms to ~4s starting around "
    "14:02Z. Investigate and report the most likely root cause."
)

INCIDENT_LOGS = (
    "2026-03-14T13:58:02Z INFO  checkout-api request completed path=/checkout status=200 duration_ms=184",
    "2026-03-14T13:58:41Z INFO  checkout-api request completed path=/checkout status=200 duration_ms=203",
    "2026-03-14T13:59:30Z INFO  cache hit_rate=0.94 cache_latency_ms=2 status=healthy",
    "2026-03-14T14:00:00Z INFO  deploy checkout-api version=v2.14.0 started",
    "2026-03-14T14:00:45Z INFO  deploy checkout-api version=v2.14.0 completed",
    "2026-03-14T14:01:03Z INFO  config diff checkout-api DB_POOL_SIZE 50 -> 5",
    "2026-03-14T14:02:10Z INFO  checkout-api request completed path=/checkout status=200 duration_ms=910",
    "2026-03-14T14:02:35Z WARN  db-pool in_use=5/5 waiters=12 acquire_wait_ms=1850",
    "2026-03-14T14:02:50Z INFO  dns resolution_latency_ms=3 host=payments.internal status=ok",
    "2026-03-14T14:03:12Z WARN  checkout-api request slow path=/checkout duration_ms=3920",
    "2026-03-14T14:03:12Z WARN  db-pool in_use=5/5 waiters=31 acquire_wait_ms=3700",
    "2026-03-14T14:03:40Z ERROR checkout-api request timeout path=/checkout duration_ms=5000 reason=db_acquire_timeout",
    "2026-03-14T14:03:41Z INFO  cpu usage=23% host=checkout-api-1 status=healthy",
    "2026-03-14T14:03:55Z INFO  memory usage=61% host=checkout-api-1 status=healthy",
    "2026-03-14T14:04:02Z WARN  checkout-api request slow path=/checkout duration_ms=4110",
    "2026-03-14T14:04:20Z INFO  network network_latency_ms=1 host=checkout-api-1 status=ok",
    "2026-03-14T14:04:31Z ERROR db-pool acquire timeout waiters=47 in_use=5/5",
    "2026-03-14T14:04:50Z INFO  db query_time_ms=6 query=select_cart status=ok",
    "2026-03-14T14:05:10Z WARN  checkout-api request slow path=/checkout duration_ms=4530",
    "2026-03-14T14:05:30Z ERROR checkout-api request timeout path=/checkout duration_ms=5000 reason=db_acquire_timeout",
)

# Ground truth for evaluating answers later. Never shown to the model.
GROUND_TRUTH_ROOT_CAUSE = (
    "Deploy v2.14.0 reduced DB_POOL_SIZE from 50 to 5, exhausting the "
    "database connection pool."
)
