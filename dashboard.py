import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Chinese U.S. Stocks vs TLT", layout="wide")

cfg = json.loads(Path("config.json").read_text())
DB = cfg["database"]

TABLES = {
    "Daily": "china_tlt_relationship_day_snapshot",
    "Weekly": "china_tlt_relationship_week_snapshot",
    "Monthly": "china_tlt_relationship_month_snapshot",
}

@st.cache_data(ttl=300)
def read_sql(sql):
    con = sqlite3.connect(DB)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()

def parse_market_cap(value):
    """Convert strings such as $295.4B / 8.2B / 950M / 120K into numeric USD."""
    if value is None or pd.isna(value):
        return np.nan
    s = str(value).upper().strip().replace("$", "").replace(",", "")
    if not s or s in {"N/A", "NA", "-", "NONE", "NAN"}:
        return np.nan

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([KMBT]?)", s)
    if not m:
        return np.nan

    n = float(m.group(1))
    mult = {
        "": 1.0,
        "K": 1e3,
        "M": 1e6,
        "B": 1e9,
        "T": 1e12,
    }.get(m.group(2), 1.0)
    return n * mult

@st.cache_data(ttl=300)
def load_universe():
    u = read_sql("""
        SELECT symbol, company_name, stock_price_text, market_cap_text,
               revenue_text, scraped_at
        FROM china_universe
        WHERE active=1
        ORDER BY symbol
    """)
    if u.empty:
        return u
    u["market_cap"] = u["market_cap_text"].map(parse_market_cap)
    return u

universe = load_universe()

st.title("Chinese U.S.-Listed Stocks vs TLT")
st.caption(
    "Intermarket lead/lag relationship analysis across Daily, Weekly and Monthly timeframes. "
    "Boards are ranked by market capitalization by default."
)

with st.sidebar:
    st.header("Filters")

    if universe.empty:
        stock_options = ["All Stocks"]
        stock_map = {}
    else:
        universe["selector"] = universe.apply(
            lambda r: f"{r['symbol']} — {r['company_name']}" if str(r["company_name"]).strip()
            else str(r["symbol"]),
            axis=1
        )
        stock_options = ["All Stocks"] + universe.sort_values(
            ["market_cap", "symbol"], ascending=[False, True], na_position="last"
        )["selector"].tolist()
        stock_map = dict(zip(universe["selector"], universe["symbol"]))

    selected_label = st.selectbox(
        "Select a stock",
        stock_options,
        index=0,
        help="Choose All Stocks for the market-cap-ranked leadership boards, or isolate one company."
    )
    selected_symbol = None if selected_label == "All Stocks" else stock_map[selected_label]

    min_corr = st.slider("Minimum |best lag correlation|", 0.0, 0.9, 0.0, 0.05)
    relationship_filter = st.selectbox("Relationship", ["All", "Positive", "Inverse"])
    max_rows = st.slider("Rows per board", 10, 100, 30, 10)

    sort_mode = st.selectbox(
        "Sort boards by",
        ["Market Cap — Highest First", "Relationship Strength"],
        index=0
    )

def enrich(df):
    if df.empty:
        return df

    if not universe.empty:
        meta = universe[
            ["symbol", "market_cap_text", "market_cap", "stock_price_text", "revenue_text"]
        ].rename(columns={"symbol": "ticker"})
        df = df.merge(meta, on="ticker", how="left")
    else:
        df["market_cap_text"] = None
        df["market_cap"] = np.nan

    return df

def apply_filters(df):
    if df.empty:
        return df

    df = enrich(df)

    if selected_symbol:
        df = df[df["ticker"] == selected_symbol]

    df = df[df["best_lag_corr"].abs() >= min_corr]

    if relationship_filter == "Positive":
        df = df[df["best_lag_corr"] > 0]
    elif relationship_filter == "Inverse":
        df = df[df["best_lag_corr"] < 0]

    df = df.copy()
    df["strength"] = df["best_lag_corr"].abs()
    return df

def sorted_board(df):
    if df.empty:
        return df
    if sort_mode == "Market Cap — Highest First":
        return df.sort_values(
            ["market_cap", "strength"],
            ascending=[False, False],
            na_position="last"
        )
    return df.sort_values(
        ["strength", "market_cap"],
        ascending=[False, False],
        na_position="last"
    )

tabs = st.tabs(["Daily", "Weekly", "Monthly", "Cross-Timeframe Followers", "Universe"])

for tab, (label, table) in zip(tabs[:3], TABLES.items()):
    with tab:
        try:
            df = read_sql(f'SELECT * FROM "{table}"')
        except Exception as e:
            st.error(f"Could not load {table}: {e}")
            continue

        if df.empty:
            st.info("No analysis data.")
            continue

        df = apply_filters(df)

        if selected_symbol:
            st.subheader(f"{selected_symbol} — {label}")
            if df.empty:
                st.info("This stock has no row matching the current filters.")
            else:
                r = df.iloc[0]
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Market Cap", r.get("market_cap_text") or "—")
                m2.metric("Relationship", r.get("relationship_type", "—"))
                m3.metric("Best Lag", int(r["best_lag"]) if pd.notna(r["best_lag"]) else "—")
                m4.metric(
                    "Best Lag Corr",
                    f"{r['best_lag_corr']:.3f}" if pd.notna(r["best_lag_corr"]) else "—"
                )
                m5.metric(
                    "Score",
                    f"{r['relationship_score']:.1f}" if pd.notna(r["relationship_score"]) else "—"
                )

                detail_cols = [
                    c for c in [
                        "ticker", "company_name", "market_cap_text", "stock_price_text",
                        "revenue_text", "relationship_type", "relationship_score",
                        "best_lag", "best_lag_corr", "best_lag_beta", "follower_alert"
                    ] if c in df.columns
                ]
                st.dataframe(df[detail_cols], use_container_width=True, hide_index=True)

        else:
            left, right = st.columns(2)

            with left:
                st.subheader("Concurrent Relationships")
                x = sorted_board(df[df["best_lag"] == 0].copy())
                if x.empty:
                    st.info("None.")
                else:
                    cols = [
                        "ticker", "company_name", "market_cap_text", "relationship_type",
                        "relationship_score", "best_lag_corr", "best_lag_beta"
                    ]
                    cols = [c for c in cols if c in x.columns]
                    st.dataframe(
                        x[cols].head(max_rows),
                        use_container_width=True,
                        hide_index=True
                    )

            with right:
                st.subheader("True Followers")
                x = sorted_board(df[df["best_lag"] >= 1].copy())
                if x.empty:
                    st.info("None.")
                else:
                    cols = [
                        "ticker", "company_name", "market_cap_text", "relationship_type",
                        "relationship_score", "best_lag", "best_lag_corr",
                        "best_lag_beta", "follower_alert"
                    ]
                    cols = [c for c in cols if c in x.columns]
                    st.dataframe(
                        x[cols].head(max_rows),
                        use_container_width=True,
                        hide_index=True
                    )

        if not df.empty:
            plot = sorted_board(df).head(30).copy()
            plot["display_name"] = plot["ticker"]
            fig = px.bar(
                plot.sort_values("relationship_score"),
                x="relationship_score",
                y="display_name",
                orientation="h",
                hover_data=[
                    c for c in ["company_name", "market_cap_text", "best_lag", "best_lag_corr"]
                    if c in plot.columns
                ],
                title=(
                    f"{selected_symbol} — {label} TLT relationship"
                    if selected_symbol else
                    f"{label} TLT relationships — market-cap-ranked universe"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    frames = []
    for label, table in TABLES.items():
        try:
            d = read_sql(f'SELECT * FROM "{table}"')
        except Exception:
            continue

        if d.empty:
            continue

        d = enrich(d)
        d = d[d["best_lag"] >= 1].copy()

        if selected_symbol:
            d = d[d["ticker"] == selected_symbol]

        if d.empty:
            continue

        d["timeframe"] = label
        d["strength"] = d["best_lag_corr"].abs()
        frames.append(
            d[
                [
                    "ticker", "company_name", "market_cap_text", "market_cap",
                    "timeframe", "relationship_type", "best_lag",
                    "best_lag_corr", "strength"
                ]
            ]
        )

    if not frames:
        st.info("No cross-timeframe followers for the current selection.")
    else:
        allf = pd.concat(frames, ignore_index=True)

        summary = (
            allf.groupby(
                ["ticker", "company_name", "market_cap_text", "market_cap"],
                dropna=False,
                as_index=False
            )
            .agg(
                timeframes_following=("timeframe", "nunique"),
                combined_strength=("strength", "mean")
            )
        )

        if sort_mode == "Market Cap — Highest First":
            summary = summary.sort_values(
                ["market_cap", "timeframes_following", "combined_strength"],
                ascending=[False, False, False],
                na_position="last"
            )
        else:
            summary = summary.sort_values(
                ["timeframes_following", "combined_strength", "market_cap"],
                ascending=[False, False, False],
                na_position="last"
            )

        st.subheader(
            f"{selected_symbol} — Cross-Timeframe Followers"
            if selected_symbol else
            "Cross-Timeframe Followers"
        )
        st.dataframe(
            summary[
                [
                    "ticker", "company_name", "market_cap_text",
                    "timeframes_following", "combined_strength"
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Follower Details")
        st.dataframe(
            allf.sort_values(["ticker", "timeframe"]),
            use_container_width=True,
            hide_index=True
        )

with tabs[4]:
    if universe.empty:
        st.info("No active universe records.")
    else:
        u = universe.copy()

        if selected_symbol:
            u = u[u["symbol"] == selected_symbol]

        u = u.sort_values(
            ["market_cap", "symbol"],
            ascending=[False, True],
            na_position="last"
        )

        st.metric("Active source symbols", len(universe))
        cols = [
            "symbol", "company_name", "market_cap_text",
            "stock_price_text", "revenue_text", "scraped_at"
        ]
        st.dataframe(u[cols], use_container_width=True, hide_index=True)

st.caption(
    "Universe source: StockAnalysis.com Chinese companies on the U.S. stock market. "
    "Market-cap sorting uses the scraped market-cap values from the source universe table."
)
