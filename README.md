# Market Board

Static breadth / regime / theme dashboard.

- `index.html` — front end (fetches `dashboard_data.json`)
- `market_ingest.py` — nightly ingest: yfinance + Stooq -> SQLite -> JSON
- `.github/workflows/update-dashboard.yml` — runs the ingest on weekdays and commits the JSON

`market.db` is intentionally not committed; the Action caches it and rebuilds
automatically if the cache is ever missing.
