# Chinese U.S.-Listed Stocks vs TLT — Streamlit Cloud

This folder is the deployable dashboard artifact.

It must contain:

- `dashboard.py`
- `config.json`
- `requirements.txt`
- `data/streamlit_data.sqlite`

The cloud app does **not** scrape StockAnalysis.com and does **not** connect to IBKR.
Those jobs remain local.

## Streamlit Community Cloud

As of October 2026, Streamlit Community Cloud deploys from GitHub repositories.
If your primary source repository is GitLab, mirror this `cloud/` folder/repository
to GitHub, then deploy the GitHub mirror in Streamlit Community Cloud.

Entrypoint:

`dashboard.py`


Pivot dashboard entrypoint: `pivot_dashboard.py`

## Pivot dashboard

Use `pivot_dashboard.py` as the Streamlit entrypoint to deploy the TLT pivot-analysis app.

The compact cloud database includes:
- universe metadata
- Daily/Weekly/Monthly pivot tables
- reaction tables
- normalized path tables
- compact TLT daily OHLCV for candlestick/volume charts
