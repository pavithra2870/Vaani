"""Gateway: every model call goes through here. This is where the load/limit policy lives.

Policy (numbers are set from the measurements in runs/probe_*.jsonl, see README / report):
  * PACING       client-side sliding windows for requests/min and tokens/min, set a little BELOW the provider's
                 limits. A call that would not fit waits its turn (FIFO) instead of being sent and rejected:
                 a rejected call still counts against requests-per-minute and wastes a round trip.
  * DEADLINE     each user request has a patience budget. Work that can no longer finish in time is dropped early.
  * SHEDDING     if too many calls are already waiting, new requests are turned away at once with an honest
                 "busy, retry in N s" instead of joining a queue they cannot survive.
  * RETRY        on 429 honour retry-after (+ small jitter); on 5xx/timeout exponential backoff with full jitter;
                 retry only if the wait still fits inside the deadline. retry_mode="naive" (immediate, 6x, no
                 pacing) exists only to reproduce the retry-storm case study.
  * DAILY LIMIT  a 429 that names a per-day limit (or retry-after > 120 s) flips a flag: no more calls are sent
                 until it expires; the pipeline falls back to showing retrieved passages (no model).
Per request, time is split into: queue wait, working (inside HTTP calls), retry sleeping.
"""
import asyncio
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
import httpx
from . import budget, llm


@dataclass
class GatewayConfig:
    max_concurrency: int = 8
    max_retries: int = 3
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 8.0
    deadline_s: float = 30.0          # total patience per user request
    call_timeout_s: float = 20.0      # per-attempt timeout
    rpm_limit: int = 30               # provider limits as we believe them (Table 1)
    tpm_limit: int = 8000
    pace_margin: float = 0.9          # send at most 90% of the believed limit
    pace: bool = True
    max_queue: int = 3                # waiting calls before new requests are shed
    retry_mode: str = "smart"         # smart | naive
    est_out_tokens: dict = field(default_factory=lambda: {"understand": 60, "draft": 220, "check": 40})
    chars_per_token: float = 3.5
    # admission control (auto mode): admit a request in the richest mode whose token cost can be served within
    # patience_frac * deadline given the tokens already committed to requests in flight; else turn it away.
    admission: bool = True
    cost_full: int = 3400             # tokens one full-mode answer needs  (measured, see report)
    cost_lean: int = 1700             # tokens one lean-mode answer needs   (measured, see report)
    patience_frac: float = 0.5


@dataclass
class ReqCtx:
    """Per-user-request accounting, shared by all model calls that request makes."""
    t_start: float = field(default_factory=time.perf_counter)
    deadline_s: float = 30.0
    queue_ms: float = 0.0
    work_ms: float = 0.0
    retry_wait_ms: float = 0.0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    first_token_ms: float | None = None   # since t_start; first token of the first successful DRAFT stream
    rate_limited_hits: int = 0
    calls: list = field(default_factory=list)
    left: float = 0.0                         # tokens committed to this request but not yet reserved
    retry_after_hint_s: float | None = None   # set when we give up because of a limit

    def remaining(self) -> float:
        return self.deadline_s - (time.perf_counter() - self.t_start)


class Gateway:
    def __init__(self, cfg: GatewayConfig | None = None):
        self.cfg = cfg or GatewayConfig()
        self.sem = asyncio.Semaphore(self.cfg.max_concurrency)
        self.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=1000))
        self.pace_lock = asyncio.Lock()
        self.req_win = deque()            # [t, 1]   sliding 60 s window for requests/min
        self.tok_cap = self.cfg.tpm_limit * self.cfg.pace_margin
        self.tok_rate = self.tok_cap / 60.0        # tokens refilled per second (Groq refills continuously)
        self.tok_level = self.tok_cap              # token bucket level
        self.tok_t = time.perf_counter()
        self.waiting = 0
        self.peak_queue = 0
        self.committed = 0.0              # tokens promised to admitted requests that have not been reserved yet
        self.daily_block_until = 0.0
        self.daily_block_msg = ""

    async def close(self):
        await self.client.aclose()

    # ---------- state used by the pipeline ----------
    def _prune(self, now):
        while self.req_win and now - self.req_win[0][0] >= 60:
            self.req_win.popleft()

    def _refill(self, now):
        self.tok_level = min(self.tok_cap, self.tok_level + (now - self.tok_t) * self.tok_rate)
        self.tok_t = now

    def pressure(self) -> float:
        """Fraction of the (margin-adjusted) token bucket already spent or reserved, 0..1."""
        self._refill(time.perf_counter())
        return min(1.0, max(0.0, 1 - self.tok_level / self.tok_cap))

    def daily_blocked(self) -> float:
        """Seconds until the daily block expires (0 if not blocked)."""
        return max(0.0, self.daily_block_until - time.perf_counter())

    def should_shed(self) -> bool:
        return self.waiting >= self.cfg.max_queue

    def admit(self, ctx: "ReqCtx", mode: str):
        """Admission control. Returns (mode, 0) if admitted, or (None, seconds_to_wait_before_retrying)."""
        cfg = self.cfg
        self._refill(time.perf_counter())
        avail = max(0.0, self.tok_level)
        limit = cfg.patience_frac * ctx.deadline_s
        def wait(cost):
            return max(0.0, (self.committed + cost - avail) / self.tok_rate)
        for m in (["full", "lean"] if mode == "auto" else [mode]):
            cost = cfg.cost_full if m == "full" else cfg.cost_lean
            if wait(cost) <= limit:
                self.committed += cost
                ctx.left = cost
                return m, 0.0
        return None, max(1.0, wait(cfg.cost_lean) - limit)

    def release(self, ctx: "ReqCtx"):
        """Give back whatever part of the commitment the request never used."""
        self.committed = max(0.0, self.committed - ctx.left)
        ctx.left = 0.0

    def wait_estimate_s(self, need_tokens: float = 1700) -> float:
        """Rough time until a lean answer's worth of tokens is available (used in the 'busy' message)."""
        self._refill(time.perf_counter())
        return max(2.0, (need_tokens - self.tok_level) / self.tok_rate)

    def _sync_with_provider(self, res):
        """The provider's own counters are authoritative. Without this the pacer only knows about ITS OWN usage and
        stays optimistic forever if the real bucket was already partly drained (found in the first real evaluation)."""
        rem = res.headers.get("x-ratelimit-remaining-tokens")
        if res.status == "ok" and rem is not None:
            try:
                # header is read when the response starts; output tokens are charged at completion
                self.tok_level = min(self.tok_level, float(rem) - res.tokens_out)
            except ValueError:
                pass
        elif res.status == "rate_limited":
            m = re.search(r"tokens per minute.*?Limit (\d+), Used (\d+)", res.error)
            if m:
                self.tok_level = min(self.tok_level, float(m.group(1)) - float(m.group(2)))
            elif "tokens per minute" in res.error:
                self.tok_level = 0.0

    # ---------- pacing ----------
    def _pace_wait(self, now, est):
        cfg = self.cfg
        self._prune(now)
        self._refill(now)
        wait = 0.0
        rpm = int(cfg.rpm_limit * cfg.pace_margin)
        if len(self.req_win) >= rpm:
            wait = max(wait, self.req_win[len(self.req_win) - rpm][0] + 60 - now)
        if self.tok_level < est:
            wait = max(wait, (est - self.tok_level) / self.tok_rate)
        return max(0.0, wait)

    def _est_tokens(self, messages, step, max_tokens):
        chars = sum(len(m["content"]) for m in messages)
        out = min(max_tokens, self.cfg.est_out_tokens.get(step, 200))
        return int(chars / self.cfg.chars_per_token) + out

    # ---------- the call ----------
    async def call(self, ctx: ReqCtx, messages, step: str, max_tokens=400, is_draft=False) -> llm.LLMResult:
        cfg = self.cfg
        attempt = 0
        est = self._est_tokens(messages, step, max_tokens)
        while True:
            if budget.exhausted():
                r = llm.LLMResult()
                r.status, r.error = "budget_exhausted", "local daily cap"
                return r
            if self.daily_blocked() > 0:
                r = llm.LLMResult()
                r.status, r.error = "budget_exhausted", self.daily_block_msg
                ctx.retry_after_hint_s = self.daily_blocked()
                return r
            if ctx.remaining() <= 0:
                r = llm.LLMResult()
                r.status, r.error = "timeout", "deadline before attempt"
                return r

            # --- wait for our turn (pacing, FIFO) and a slot (queue) ---
            tq = time.perf_counter()
            self.waiting += 1
            self.peak_queue = max(self.peak_queue, self.waiting)
            entry = None
            try:
                async def _turn():
                    nonlocal entry
                    async with self.pace_lock:
                        if cfg.pace:
                            w = self._pace_wait(time.perf_counter(), est)
                            if w > ctx.remaining():
                                ctx.retry_after_hint_s = w
                                raise _WouldNotFit()
                            if w > 0:
                                await asyncio.sleep(w)
                        now = time.perf_counter()
                        self._refill(now)
                        self.tok_level -= est
                        take = min(est, ctx.left)          # this part of the commitment is now reserved for real
                        self.committed = max(0.0, self.committed - take)
                        ctx.left -= take
                        entry = est
                        self.req_win.append([now, 1])
                    await self.sem.acquire()
                await asyncio.wait_for(_turn(), timeout=max(0.001, ctx.remaining()))
            except (asyncio.TimeoutError, _WouldNotFit):
                self.waiting -= 1
                ctx.queue_ms += (time.perf_counter() - tq) * 1000
                r = llm.LLMResult()
                r.status, r.error = "timeout", "gave up waiting for a turn"
                return r
            self.waiting -= 1
            ctx.queue_ms += (time.perf_counter() - tq) * 1000

            # --- do the call ---
            t_call = time.perf_counter()
            try:
                res = await llm.chat(self.client, messages, step, max_tokens,
                                     timeout_s=min(cfg.call_timeout_s, max(0.5, ctx.remaining())))
            finally:
                self.sem.release()
            ctx.work_ms += (time.perf_counter() - t_call) * 1000
            if entry is not None and res.status == "ok":
                self.tok_level += entry - (res.tokens_in + res.tokens_out)   # reservation -> real usage
            elif entry is not None and res.status == "rate_limited":
                self.tok_level += entry                                      # provider rejected it: refund
            self._sync_with_provider(res)

            ctx.tokens_in += res.tokens_in
            ctx.tokens_out += res.tokens_out
            ctx.calls.append(dict(step=step, attempt=attempt, start_ms=(t_call - ctx.t_start) * 1000,
                                  status=res.status, http=res.http_status, ttft_ms=res.first_token_ms,
                                  total_ms=res.total_ms, tin=res.tokens_in, tout=res.tokens_out,
                                  retry_after=res.retry_after, error=res.error[:400],
                                  rem_tokens=res.headers.get("x-ratelimit-remaining-tokens"),
                                  local_level=round(self.tok_level)))
            if llm.config.BACKEND == "groq" and not res.cached:
                budget.add(res.tokens_in + res.tokens_out)
            if is_draft and res.first_token_ms is not None and ctx.first_token_ms is None:
                ctx.first_token_ms = (t_call - ctx.t_start) * 1000 + res.first_token_ms

            if res.status == "ok":
                return res
            if res.status == "rate_limited":
                ctx.rate_limited_hits += 1
                txt = res.error.lower()
                if " per day" in txt or (res.retry_after or 0) > 120:
                    ra = res.retry_after or 3600.0
                    self.daily_block_until = time.perf_counter() + ra
                    self.daily_block_msg = res.error[:200]
                    ctx.retry_after_hint_s = ra
                    res.status = "budget_exhausted"
                    return res

            # --- retry decision ---
            attempt += 1
            max_retries = 6 if cfg.retry_mode == "naive" else cfg.max_retries
            if attempt > max_retries:
                ctx.retry_after_hint_s = res.retry_after
                return res
            if cfg.retry_mode == "naive":
                wait = 0.0                                   # hammer immediately
            elif res.retry_after is not None:
                wait = res.retry_after + random.uniform(0, 0.5 * max(1.0, res.retry_after))
            else:
                wait = random.uniform(0, min(cfg.backoff_cap_s, cfg.backoff_base_s * 2 ** attempt))
            if wait >= ctx.remaining():          # can't finish in time anyway -> don't make it worse
                ctx.retry_after_hint_s = res.retry_after
                return res
            ctx.retries += 1
            ts = time.perf_counter()
            await asyncio.sleep(wait)
            ctx.retry_wait_ms += (time.perf_counter() - ts) * 1000


class _WouldNotFit(Exception):
    pass
