"""results.csv (the brief's exact columns) + results_detail.csv (extra columns keyed by request_id)."""
import csv
import datetime as dt
import os
from pathlib import Path
from .config import ROOT

# dev runs set RESULTS_FILE / RESULTS_DETAIL_FILE so they never touch the submitted files
RESULTS = Path(os.getenv("RESULTS_FILE", ROOT / "results.csv"))
DETAIL = Path(os.getenv("RESULTS_DETAIL_FILE", ROOT / "results_detail.csv"))
COLUMNS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
           "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]
DETAIL_COLUMNS = ["request_id", "scenario", "question_id", "model", "mode", "detail", "queue_ms", "work_ms",
                  "retry_wait_ms", "rate_limited_hits", "model_calls", "queue_len_at_end", "wall_start_s"]


def _append(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def record(request_id, scenario, source, start_epoch, res, grade="", question_id="", model="", queue_len=0,
           wall_start_s=0.0):
    c = res.ctx
    _append(RESULTS, COLUMNS, dict(
        request_id=request_id, scenario=scenario, source=source,
        start_time=dt.datetime.fromtimestamp(start_epoch, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        first_token_ms="" if c.first_token_ms is None else round(c.first_token_ms, 1),
        total_ms=round(res.total_ms, 1), status=res.status, retries=c.retries,
        tokens_in=c.tokens_in, tokens_out=c.tokens_out, retrieval_setting=res.setting, answer_grade=grade))
    _append(DETAIL, DETAIL_COLUMNS, dict(
        request_id=request_id, scenario=scenario, question_id=question_id, model=model, mode=res.mode,
        detail=res.detail, queue_ms=round(c.queue_ms, 1), work_ms=round(c.work_ms, 1),
        retry_wait_ms=round(c.retry_wait_ms, 1), rate_limited_hits=c.rate_limited_hits,
        model_calls=len(c.calls), queue_len_at_end=queue_len, wall_start_s=round(wall_start_s, 3)))
