# Developer Notes

- Configure runtime parameters through `config.yaml`. Paths, ingestion behaviour, scoring weights, and future thresholds should live in configuration rather than code.
- Logs are emitted to the directory defined in `paths.logs_dir` and mirrored to STDOUT for quick inspection when running the CLI.
- The ingestion state file (`paths.ingestion_state`) captures metadata that later phases can use to determine incremental updates or notify stakeholders.
- When adding new metrics, update the calendar map only if new week label formats are introduced. The normalization utilities are designed to work with a broad set of header patterns using regex extraction.
- Maintain metric weights/directions centrally in `config.yaml`. The Phase 2 score engine builds a reusable metric map consumed by downstream commentary and forecasting logic, so keep naming consistent with normalized `metric` values.
- If Parquet support is desired for scorecards, install `pyarrow` or `fastparquet`. The engine falls back to CSV output when these optional dependencies are unavailable.
- Use the Phase 6 pipeline (`src/brightstar_pipeline.py`) and its `--resume` flag for production jobs to avoid reprocessing unchanged phases; logs aggregate to `logs/brightstar_pipeline.log` for simplified monitoring.
