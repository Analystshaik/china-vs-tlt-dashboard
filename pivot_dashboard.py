import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="TLT Pivot Analysis — Chinese U.S.-Listed Stocks",
    layout="wide"
)

cfg = json.loads(Path("config.json").read_text())
DB = cfg["database"]

TF = {
    "Daily": {
        "pivot_table": "china_tlt_pivots_daily",
        "reaction_table": "china_tlt_pivot_reactions_daily",
        "path_table": "china_tlt_pivot_paths_daily",
        "horizons": [1,2,3,5,10,20],
        "unit": "trading days",
        "rule": None
    },
    "Weekly": {
        "pivot_table": "china_tlt_pivots_weekly",
        "reaction_table": "china_tlt_pivot_reactions_weekly",
        "path_table": "china_tlt_pivot_paths_weekly",
        "horizons": [1,2,3,4,6,8],
        "unit": "weeks",
        "rule": "W-FRI"
    },
    "Monthly": {
        "pivot_table": "china_tlt_pivots_monthly",
        "reaction_table": "china_tlt_pivot_reactions_monthly",
        "path_table": "china_tlt_pivot_paths_monthly",
        "horizons": [1,2,3,4,6,9,12],
        "unit": "months",
        "rule": "ME"
    }
}

@st.cache_data(ttl=300)
def read_sql(sql, params=None):
    con = sqlite3.connect(DB)
    try:
        return pd.read_sql_query(sql, con, params=params or ())
    finally:
        con.close()

def table_exists(name):
    try:
        x = read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,)
        )
        return not x.empty
    except Exception:
        return False

def parse_market_cap(v):
    if v is None or pd.isna(v):
        return np.nan
    s = str(v).upper().replace("$","").replace(",","").strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([KMBT]?)", s)
    if not m:
        return np.nan
    mult = {"":1.0,"K":1e3,"M":1e6,"B":1e9,"T":1e12}
    return float(m.group(1)) * mult.get(m.group(2),1.0)

@st.cache_data(ttl=300)
def load_universe():
    u = read_sql("""
        SELECT symbol, company_name, market_cap_text
        FROM china_universe
        WHERE active=1
    """)
    if not u.empty:
        u["market_cap"] = u["market_cap_text"].map(parse_market_cap)
        u["selector"] = u.apply(
            lambda r: f"{r['symbol']} — {r['company_name']}"
            if str(r["company_name"]).strip() else str(r["symbol"]),
            axis=1
        )
        u = u.sort_values(["market_cap","symbol"], ascending=[False,True], na_position="last")
    return u

@st.cache_data(ttl=300)
def load_tlt_daily():
    # Local DB path: raw bars available.
    if table_exists("china_bars_day"):
        d = read_sql("""
            SELECT bar_date, open, high, low, close, volume
            FROM china_bars_day
            WHERE symbol='TLT'
            ORDER BY bar_date
        """)
    elif table_exists("tlt_bars_day"):
        d = read_sql("""
            SELECT bar_date, open, high, low, close, volume
            FROM tlt_bars_day
            ORDER BY bar_date
        """)
    else:
        return pd.DataFrame()

    if d.empty:
        return d

    s = d["bar_date"].astype(str)
    mask = s.str.fullmatch(r"\d{8}", na=False)
    s = s.where(~mask, s.str.slice(0,4)+"-"+s.str.slice(4,6)+"-"+s.str.slice(6,8))
    d["bar_date"] = pd.to_datetime(s, errors="coerce")

    for c in ["open","high","low","close","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["bar_date","open","high","low","close"])

def aggregate_tlt(d, timeframe):
    if d.empty:
        return d
    if timeframe == "Daily":
        return d.copy()

    rule = "W-FRI" if timeframe == "Weekly" else "ME"
    x = d.set_index("bar_date").sort_index()

    try:
        out = pd.DataFrame({
            "open": x["open"].resample(rule).first(),
            "high": x["high"].resample(rule).max(),
            "low": x["low"].resample(rule).min(),
            "close": x["close"].resample(rule).last(),
            "volume": x["volume"].resample(rule).sum()
        }).dropna(subset=["open","high","low","close"])
    except ValueError:
        out = pd.DataFrame({
            "open": x["open"].resample("M").first(),
            "high": x["high"].resample("M").max(),
            "low": x["low"].resample("M").min(),
            "close": x["close"].resample("M").last(),
            "volume": x["volume"].resample("M").sum()
        }).dropna(subset=["open","high","low","close"])

    return out.reset_index()

def add_market_cap(df, u):
    if df.empty:
        return df

    base = df.copy()

    # Pivot reaction tables use "symbol"; some relationship tables use "ticker".
    # Normalize internally so either schema works.
    if "ticker" not in base.columns and "symbol" in base.columns:
        base["ticker"] = base["symbol"]

    if "ticker" not in base.columns:
        raise KeyError(
            f"Expected a symbol column named 'symbol' or 'ticker'. "
            f"Found columns: {list(base.columns)}"
        )

    meta = u[["symbol","company_name","market_cap_text","market_cap"]].rename(
        columns={"symbol":"ticker"}
    )

    base = base.drop(
        columns=[
            c for c in ["company_name","market_cap_text","market_cap"]
            if c in base.columns
        ],
        errors="ignore"
    )

    return base.merge(meta, on="ticker", how="left")

def tlt_candlestick_chart(tlt, pivots, timeframe):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=tlt["bar_date"],
        open=tlt["open"],
        high=tlt["high"],
        low=tlt["low"],
        close=tlt["close"],
        name="TLT",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ))

    if not pivots.empty:
        p = pivots.copy()
        p["pivot_date"] = pd.to_datetime(p["pivot_date"])

        bull = p[p["direction"]=="BULLISH"]
        bear = p[p["direction"]=="BEARISH"]

        if not bull.empty:
            y = []
            for _, r in bull.iterrows():
                row = tlt.iloc[(tlt["bar_date"]-r["pivot_date"]).abs().argsort()[:1]]
                y.append(float(row["low"].iloc[0]) if not row.empty else np.nan)

            fig.add_trace(go.Scatter(
                x=bull["pivot_date"], y=y,
                mode="markers",
                marker=dict(symbol="triangle-up", size=13, color="#00c853"),
                name="Bullish Pivot",
                text=bull["pattern"] + " / " + bull["strength"],
                hovertemplate="%{x}<br>%{text}<extra></extra>"
            ))

        if not bear.empty:
            y = []
            for _, r in bear.iterrows():
                row = tlt.iloc[(tlt["bar_date"]-r["pivot_date"]).abs().argsort()[:1]]
                y.append(float(row["high"].iloc[0]) if not row.empty else np.nan)

            fig.add_trace(go.Scatter(
                x=bear["pivot_date"], y=y,
                mode="markers",
                marker=dict(symbol="triangle-down", size=13, color="#ff1744"),
                name="Bearish Pivot",
                text=bear["pattern"] + " / " + bear["strength"],
                hovertemplate="%{x}<br>%{text}<extra></extra>"
            ))

    fig.update_layout(
        title=f"TLT {timeframe} Price with Reversal Pivots",
        height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
        margin=dict(l=20,r=20,t=60,b=20)
    )
    return fig

def tlt_volume_chart(tlt, timeframe):
    fig = go.Figure(go.Bar(
        x=tlt["bar_date"],
        y=tlt["volume"],
        name="Volume"
    ))
    fig.update_layout(
        title=f"TLT {timeframe} Volume",
        height=220,
        margin=dict(l=20,r=20,t=50,b=20)
    )
    return fig

def reaction_summary(rx, horizons, unit):
    rows = []
    for direction in ["BULLISH","BEARISH"]:
        q = rx[rx["direction"]==direction]
        if q.empty:
            continue

        for h in horizons:
            col=f"ret_{h}"
            vals=pd.to_numeric(q[col],errors="coerce").dropna()
            if vals.empty:
                continue

            if direction=="BULLISH":
                hit=(vals>0).mean()*100
            else:
                hit=(vals<0).mean()*100

            rows.append({
                "Direction": direction,
                "Horizon": f"{h} {unit}",
                "Events": q["pivot_date"].nunique(),
                "Average Return %": vals.mean(),
                "Median Return %": vals.median(),
                "Hit Rate %": hit
            })
    return pd.DataFrame(rows)

def average_path(paths):
    if paths.empty:
        return pd.DataFrame()
    return (
        paths.groupby(["direction","offset"],as_index=False)["normalized"]
        .mean()
    )

def pivot_level_universe_summary(rx, horizons, u):
    d = add_market_cap(rx,u)
    rows=[]
    for (pdt,direction,pattern,strength),q in d.groupby(
        ["pivot_date","direction","pattern","strength"], dropna=False
    ):
        row={
            "pivot_date":pdt,
            "direction":direction,
            "pattern":pattern,
            "strength":strength,
            "stocks":q["ticker"].nunique()
        }
        for h in horizons:
            col=f"ret_{h}"
            vals=pd.to_numeric(q[col],errors="coerce").dropna()
            row[f"avg_{h}"]=vals.mean() if len(vals) else np.nan
            row[f"median_{h}"]=vals.median() if len(vals) else np.nan
            row[f"positive_pct_{h}"]=(vals>0).mean()*100 if len(vals) else np.nan
            row[f"negative_pct_{h}"]=(vals<0).mean()*100 if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pivot_date",ascending=False)

u = load_universe()

st.title("TLT Pivot Analysis — Chinese U.S.-Listed Stocks")
st.caption(
    "TLT reversal pivots and subsequent Chinese-stock reactions. "
    "Daily, Weekly and Monthly analysis use the same reaction-explorer framework."
)

with st.sidebar:
    st.header("Analysis Controls")
    timeframe = st.radio("Timeframe", ["Daily","Weekly","Monthly"], horizontal=False)
    cfg_tf = TF[timeframe]

    if not u.empty:
        options = ["All Stocks"] + u["selector"].tolist()
    else:
        options = ["All Stocks"]

    selected_label = st.selectbox("Stock", options)
    selected_symbol = None if selected_label=="All Stocks" else selected_label.split(" — ",1)[0]

    direction_filter = st.multiselect(
        "Pivot Direction",
        ["BULLISH","BEARISH"],
        default=["BULLISH","BEARISH"]
    )

    strength_filter = st.multiselect(
        "Pivot Strength",
        ["STRONG","VERY_STRONG"],
        default=["STRONG","VERY_STRONG"]
    )

    confirmed_only = st.checkbox("Confirmed pivots only", value=True)

    st.divider()
    st.caption("Universe tables are ordered by market cap, highest first.")

ptab=cfg_tf["pivot_table"]
rtab=cfg_tf["reaction_table"]
path_table=cfg_tf["path_table"]
horizons=cfg_tf["horizons"]
unit=cfg_tf["unit"]

try:
    pivots=read_sql(f'SELECT * FROM "{ptab}" ORDER BY pivot_date DESC')
    rx=read_sql(f'SELECT * FROM "{rtab}"')
    paths=read_sql(f'SELECT * FROM "{path_table}"')
except Exception as e:
    st.error(
        f"Pivot tables for {timeframe} are not available. "
        f"Run pivot_{timeframe.lower()}.py first.\n\n{e}"
    )
    st.stop()

if pivots.empty:
    st.warning(f"No {timeframe.lower()} pivots were found.")
    st.stop()

pivots_f=pivots.copy()
if direction_filter:
    pivots_f=pivots_f[pivots_f["direction"].isin(direction_filter)]
if strength_filter:
    pivots_f=pivots_f[pivots_f["strength"].isin(strength_filter)]
if confirmed_only:
    pivots_f=pivots_f[pivots_f["confirmed"]==1]

valid_dates=set(pivots_f["pivot_date"].astype(str))
rx_f=rx[rx["pivot_date"].astype(str).isin(valid_dates)].copy()
paths_f=paths[paths["pivot_date"].astype(str).isin(valid_dates)].copy()

if selected_symbol:
    rx_f=rx_f[rx_f["symbol"]==selected_symbol]
    paths_f=paths_f[paths_f["symbol"]==selected_symbol]

latest = pivots_f.iloc[0] if not pivots_f.empty else pivots.iloc[0]

m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Timeframe",timeframe)
m2.metric("Filtered TLT Pivots",len(pivots_f))
m3.metric("Confirmed",int(pivots_f["confirmed"].sum()) if not pivots_f.empty else 0)
m4.metric("Latest Pivot",str(latest["pivot_date"]))
m5.metric("Latest Direction",str(latest["direction"]))

tabs=st.tabs([
    "Pivot Chart",
    "Pivot History",
    "Reaction Explorer",
    "Stock Reaction Explorer",
    "Backtest Summary",
    "Rules"
])

with tabs[0]:
    tlt_daily=load_tlt_daily()
    if tlt_daily.empty:
        st.warning("TLT price bars are not present in this database.")
    else:
        tlt_tf=aggregate_tlt(tlt_daily,timeframe)
        st.plotly_chart(
            tlt_candlestick_chart(tlt_tf,pivots_f,timeframe),
            use_container_width=True
        )
        st.plotly_chart(
            tlt_volume_chart(tlt_tf,timeframe),
            use_container_width=True
        )

    if not pivots_f.empty:
        latest= pivots_f.iloc[0]
        st.subheader("Latest Pivot")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Date",latest["pivot_date"])
        c2.metric("Direction",latest["direction"])
        c3.metric("Pattern",latest["pattern"])
        c4.metric("Strength",latest["strength"])

with tabs[1]:
    st.subheader(f"{timeframe} TLT Pivot History")
    hist_cols=[
        "pivot_date","direction","pattern","strength","score",
        "confirmed","confirmation_date","open","high","low","close"
    ]
    hist_cols=[c for c in hist_cols if c in pivots_f.columns]
    st.dataframe(
        pivots_f[hist_cols],
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        "Download Pivot History CSV",
        pivots_f.to_csv(index=False).encode("utf-8"),
        file_name=f"TLT_{timeframe.lower()}_pivot_history.csv",
        mime="text/csv"
    )

with tabs[2]:
    st.header("Universe Reaction Explorer")
    st.caption(
        "Shows how the Chinese-stock universe reacted after each selected TLT pivot."
    )

    if rx_f.empty:
        st.info("No reactions match the current filters.")
    else:
        universe_rx=rx_f if not selected_symbol else rx[rx["pivot_date"].astype(str).isin(valid_dates)].copy()
        pivot_sum=pivot_level_universe_summary(universe_rx,horizons,u)

        st.subheader("Pivot-by-Pivot Universe Reaction")
        display_cols=[
            "pivot_date","direction","pattern","strength","stocks"
        ]+[f"avg_{h}" for h in horizons]
        st.dataframe(
            pivot_sum[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Average Universe Reaction by Pivot Direction")
        uni_summary=reaction_summary(universe_rx,horizons,unit)
        st.dataframe(uni_summary,use_container_width=True,hide_index=True)

        if not uni_summary.empty:
            heat=uni_summary.pivot(
                index="Direction",
                columns="Horizon",
                values="Hit Rate %"
            )
            fig=px.imshow(
                heat,
                text_auto=".1f",
                aspect="auto",
                title="Direction-Aware Hit Rate Heatmap (%)"
            )
            st.plotly_chart(fig,use_container_width=True)

        latest_date=str(latest["pivot_date"])
        latest_rx=universe_rx[universe_rx["pivot_date"].astype(str)==latest_date].copy()
        if not latest_rx.empty:
            latest_rx=add_market_cap(latest_rx,u)
            latest_rx=latest_rx.sort_values(
                ["market_cap","ticker"],
                ascending=[False,True],
                na_position="last"
            )
            last_h=horizons[-1]
            cols=[
                "ticker","company_name","market_cap_text","direction",
                "pattern","strength",f"ret_{last_h}"
            ]
            cols=[c for c in cols if c in latest_rx.columns]
            st.subheader("Latest Pivot — Stock Reactions (Market Cap Ranked)")
            st.dataframe(
                latest_rx[cols],
                use_container_width=True,
                hide_index=True
            )

with tabs[3]:
    st.header("Stock Reaction Explorer")

    if not selected_symbol:
        st.info("Select a stock from the sidebar to open the Stock Reaction Explorer.")
    else:
        meta=u[u["symbol"]==selected_symbol]
        if not meta.empty:
            row=meta.iloc[0]
            c1,c2,c3=st.columns(3)
            c1.metric("Symbol",selected_symbol)
            c2.metric("Company",row["company_name"])
            c3.metric("Market Cap",row["market_cap_text"])

        stock_rx=rx_f.copy()
        stock_paths=paths_f.copy()

        if stock_rx.empty:
            st.info("No reaction history for this stock under the current filters.")
        else:
            st.subheader("Pivot Event History")
            cols=["pivot_date","direction","pattern","strength","confirmed"]+[f"ret_{h}" for h in horizons]
            cols=[c for c in cols if c in stock_rx.columns]
            st.dataframe(
                stock_rx.sort_values("pivot_date",ascending=False)[cols],
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Forward Return Summary")
            ss=reaction_summary(stock_rx,horizons,unit)
            st.dataframe(ss,use_container_width=True,hide_index=True)

            if not ss.empty:
                fig=px.bar(
                    ss,
                    x="Horizon",
                    y="Average Return %",
                    color="Direction",
                    barmode="group",
                    title=f"{selected_symbol} Average Return after TLT Pivots"
                )
                st.plotly_chart(fig,use_container_width=True)

                heat=ss.pivot(index="Direction",columns="Horizon",values="Hit Rate %")
                fig2=px.imshow(
                    heat,
                    text_auto=".1f",
                    aspect="auto",
                    title=f"{selected_symbol} Direction-Aware Hit Rate (%)"
                )
                st.plotly_chart(fig2,use_container_width=True)

            if not stock_paths.empty:
                avgp=average_path(stock_paths)
                fig3=px.line(
                    avgp,
                    x="offset",
                    y="normalized",
                    color="direction",
                    title=f"{selected_symbol} Average Normalized Path after TLT Pivots"
                )
                fig3.add_hline(y=100,line_dash="dash")
                st.plotly_chart(fig3,use_container_width=True)

            st.download_button(
                "Download Stock Event History CSV",
                stock_rx.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_symbol}_{timeframe.lower()}_tlt_pivot_reactions.csv",
                mime="text/csv"
            )

with tabs[4]:
    st.header("Backtest Summary")

    bt_rx=rx_f.copy()
    if bt_rx.empty:
        st.info("No events match the current filters.")
    else:
        label = selected_symbol if selected_symbol else "Chinese Stock Universe"
        st.caption(
            f"Historical reaction summary for {label}. "
            "This is an event study, not a trading recommendation."
        )

        bt=reaction_summary(bt_rx,horizons,unit)
        st.dataframe(bt,use_container_width=True,hide_index=True)

        if not bt.empty:
            final_h=f"ret_{horizons[-1]}"
            vals=pd.to_numeric(bt_rx[final_h],errors="coerce").dropna()

            c1,c2,c3,c4=st.columns(4)
            c1.metric("Event Rows",len(bt_rx))
            c2.metric(
                f"Average {horizons[-1]}-{unit} Return",
                f"{vals.mean():.2f}%" if len(vals) else "—"
            )
            c3.metric(
                "Best Event",
                f"{vals.max():.2f}%" if len(vals) else "—"
            )
            c4.metric(
                "Worst Event",
                f"{vals.min():.2f}%" if len(vals) else "—"
            )

            dist=bt_rx[["direction",final_h]].copy()
            dist[final_h]=pd.to_numeric(dist[final_h],errors="coerce")
            dist=dist.dropna()

            if not dist.empty:
                fig=px.histogram(
                    dist,
                    x=final_h,
                    color="direction",
                    barmode="overlay",
                    nbins=40,
                    title=f"Distribution of {horizons[-1]}-{unit} Forward Returns"
                )
                st.plotly_chart(fig,use_container_width=True)

with tabs[5]:
    st.header("Exact Pivot Rules")

    st.markdown("""
### Reversal patterns
The scanner looks for selective reversal candles around local extremes:

- **Hammer** — long lower wick near a local low.
- **Shooting Star** — long upper wick near a local high.
- **Outside Reversal** — current bar exceeds both sides of the prior range and reverses direction.
- **Piercing** — bullish reversal following a bearish bar.
- **Dark Cloud** — bearish reversal following a bullish bar.

### Strength grading
- **STRONG**
- **VERY_STRONG**

Strength increases when the reversal occurs at the appropriate local extreme.

### Confirmation
A pivot is confirmed by the next bar:
- Bullish: next close rises or next high exceeds the pivot high.
- Bearish: next close falls or next low breaks the pivot low.

### Forward-reaction windows
""")

    st.write(
        f"**{timeframe}:** " +
        ", ".join(f"{h} {unit}" for h in horizons)
    )

    st.markdown("""
### Reaction statistics
- Average forward return
- Median forward return
- Direction-aware hit rate
- Normalized post-pivot path
- Pivot-by-pivot event history
- Market-cap-ranked latest reactions

The dashboard uses the same TLT pivot date for every stock so the reaction study
measures how each Chinese U.S.-listed stock behaved after the same intermarket event.
""")
