# Element 18 code

Reference implementation of the pipeline described in the Element 18 report: DO risk-onset forecasting, SHAP explanation, SOP retrieval, single-call grounded advisory with deterministic validation, alert deduplication, and CPU/GPU benchmark harness.

## Status (read this first)

- The 21 unit tests pass on CPU.
- `scripts/run_pipeline.py` runs end to end on **synthetic** data. Numbers from synthetic data only prove the code runs. **Do not report them as results.**
- **Not tested here:** cuDF, CUDA XGBoost, Triton, Ollama and Llama 3.1 8B. Those paths are written but must be run and verified on a CUDA machine. The benchmark harness skips the GPU parts if cuDF or CUDA are missing and says so.
- `knowledge_base/sops_example.yaml` contains placeholders only. Real SOP actions must come from an approved source and a specialist.
- Event thresholds in `configs/exp_001.yaml` are placeholders to be set by a sensitivity analysis.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest -q tests

GPU (optional): install RAPIDS cuDF and a CUDA build of XGBoost per their official docs for your CUDA version, then `--device cuda`.

## Run

    python scripts/run_pipeline.py --config configs/exp_001.yaml                 # synthetic smoke test
    python scripts/run_pipeline.py --config configs/exp_001.yaml --csv data.csv  # real data
    python scripts/run_benchmarks.py                                             # preprocessing benchmark
    uvicorn element18.serve:app --port 8000                                      # API (needs results/exp_001/model.json)

Real CSV columns: `timestamp, pond_id, do_mg_l, temp_c, ph` (UTC, regular sampling). Rename columns from your dataset to match, and set `freq_minutes`.

## Layout

| Path | Purpose |
|---|---|
| `element18/data.py` | validation, synthetic generator |
| `element18/features.py` | trailing features (pandas and cuDF core set), ammonia fraction |
| `element18/labels.py` | onset labels, ongoing events excluded |
| `element18/split.py` | chronological split with purge gaps and leakage assertion |
| `element18/model.py` | XGBoost training, threshold, calibration |
| `element18/baselines.py` | threshold rule, persistence, logistic regression |
| `element18/metrics.py` | row and event-level metrics |
| `element18/explain.py` | SHAP in log-odds space |
| `element18/alerts.py` | alert state, cooldown, hysteresis |
| `element18/advisory.py` | SOP retrieval, one Ollama call, validation, template fallback |
| `element18/benchmark.py` | CPU/GPU and LLM latency harness |
| `element18/serve.py` | FastAPI service |
| `serving/triton_model_repo/` | Triton FIL config template (verify before use) |
| `tests/` | unit tests |

## Notes

- Features use trailing windows only (tested). Fit-on-train statistics: only `scale_pos_weight` is computed, from the training block.
- Recording versions, seeds and hardware state for each run is up to the experimenter. Benchmarks store `nvidia-smi` readings before and after.
