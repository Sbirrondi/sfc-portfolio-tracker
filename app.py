"""
SFC Cattolica Investment Fund - Portfolio Tracker & Analyzer
=============================================================
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import io
import base64
from streamlit_lightweight_charts import renderLightweightCharts

from fund_manager import (
    load_positions, save_positions, load_transactions, add_transaction,
    delete_transaction, update_transaction, load_nav_history, load_fund_info, save_fund_info,
    get_isin_map, save_isin_map, get_overrides, save_overrides,
    enrich_positions, load_cash, save_cash, compute_cash_from_transactions,
    calculate_nav, snapshot_nav, update_fund_info,
    update_position_prices, compute_positions_from_transactions,
    recalculate_all, get_portfolio_summary,
)
from build_nav_history import fill_missing_nav_days
from analytics import (
    calculate_returns, cumulative_returns, total_return,
    annualized_return, annualized_volatility, sharpe_ratio,
    sortino_ratio, max_drawdown, drawdown_series, detect_frequency,
    calculate_alpha_beta, monthly_returns_table,
    performance_report, var_historical, cvar
)

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="SFC Investment Fund", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .main .block-container {
        padding-top: 0.5rem; max-width: 1600px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080810 0%, #0d0d1a 100%);
        border-right: 1px solid rgba(99,102,241,0.12);
    }

    /* ── Fund Banner ── */
    .fund-banner {
        background: linear-gradient(135deg, #0a0a14 0%, #12122a 50%, #0f1a2e 100%);
        padding: 1rem 1.5rem; border-radius: 12px; color: white;
        display: flex; align-items: center; gap: 1.2rem;
        border: 1px solid rgba(99,102,241,0.18);
        margin-bottom: 0.6rem; box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .fund-banner img { width: 52px; height: 52px; border-radius: 50%; background: white; padding: 3px; }
    .fund-banner h1 { margin:0; font-size:1.3rem; font-weight:700; letter-spacing:-0.5px; color:#e2e8f0; }
    .fund-banner p { margin:0.1rem 0 0; color:#64748b; font-size:0.75rem; letter-spacing:0.3px; }

    /* ── KPI Cards ── */
    .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.7rem; margin-bottom:0.8rem; }
    @media(max-width:768px){ .kpi-grid{grid-template-columns:repeat(2,1fr);} }
    .kpi-card {
        background: linear-gradient(135deg, #0d0d1a 0%, #13132a 100%);
        border: 1px solid rgba(99,102,241,0.10); border-radius:10px;
        padding: 0.9rem 1.1rem; transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: rgba(99,102,241,0.3); }
    .kpi-label { font-size:0.65rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.35rem; }
    .kpi-value { font-size:1.45rem; font-weight:700; color:#e2e8f0; line-height:1.2; }
    .kpi-delta { font-size:0.72rem; font-weight:500; margin-top:0.25rem; color:#64748b; }
    .kpi-delta .pos { color:#22c55e; font-weight:600; }
    .kpi-delta .neg { color:#ef4444; font-weight:600; }
    .accent-purple { border-left:3px solid #6366f1; }
    .accent-green { border-left:3px solid #22c55e; }
    .accent-blue { border-left:3px solid #3b82f6; }
    .accent-amber { border-left:3px solid #f59e0b; }

    /* ── Section Headers ── */
    .section-header {
        font-size:0.8rem; font-weight:600; color:#94a3b8; text-transform:uppercase;
        letter-spacing:0.8px; margin:1.2rem 0 0.5rem; padding-bottom:0.4rem;
        border-bottom:1px solid rgba(99,102,241,0.10);
    }

    /* ── Performance Table ── */
    .perf-table {
        width:100%; border-collapse:separate; border-spacing:0; font-size:0.8rem;
        border-radius:8px; overflow:hidden; border:1px solid rgba(99,102,241,0.10);
    }
    .perf-table thead th {
        background:#0d0d1a; color:#94a3b8; font-weight:600; font-size:0.65rem;
        text-transform:uppercase; letter-spacing:0.5px; padding:0.55rem 0.7rem;
        text-align:right; border-bottom:1px solid rgba(99,102,241,0.12);
    }
    .perf-table thead th:first-child { text-align:left; }
    .perf-table tbody td {
        padding:0.5rem 0.7rem; text-align:right; color:#cbd5e1;
        border-bottom:1px solid rgba(99,102,241,0.05);
    }
    .perf-table tbody td:first-child { text-align:left; font-weight:500; color:#e2e8f0; }
    .perf-table tbody tr:hover { background:rgba(99,102,241,0.04); }
    .perf-table .pos { color:#22c55e; font-weight:600; }
    .perf-table .neg { color:#ef4444; font-weight:600; }

    /* ── Risk / Stat Grid ── */
    .stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.45rem; }
    .stat-item {
        background:#0d0d1a; border:1px solid rgba(99,102,241,0.07);
        border-radius:8px; padding:0.6rem 0.8rem;
    }
    .stat-label { font-size:0.6rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; }
    .stat-value { font-size:1.05rem; font-weight:700; color:#e2e8f0; margin-top:0.15rem; }

    /* ── Movers List ── */
    .mover-item {
        display:flex; justify-content:space-between; align-items:center;
        padding:0.4rem 0; border-bottom:1px solid rgba(99,102,241,0.05); font-size:0.78rem;
    }
    .mover-item:last-child { border-bottom:none; }
    .mover-name { color:#cbd5e1; font-weight:500; max-width:68%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .mover-pnl { font-weight:700; }
    .mover-pnl.pos { color:#22c55e; }
    .mover-pnl.neg { color:#ef4444; }
    .mover-section { font-size:0.6rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin:0.5rem 0 0.3rem; }
    .mover-section:first-child { margin-top:0; }

    /* ── Streamlit Overrides ── */
    .stTabs [data-baseweb="tab-list"] {
        gap:0.2rem; background:rgba(13,13,26,0.6); padding:0.25rem;
        border-radius:10px; border:1px solid rgba(99,102,241,0.08);
    }
    .stTabs [data-baseweb="tab"] { border-radius:8px; padding:0.4rem 1rem; font-weight:500; font-size:0.8rem; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d0d1a 0%, #13132a 100%);
        border:1px solid rgba(99,102,241,0.08); border-radius:10px; padding:0.7rem 0.9rem;
    }
    [data-testid="stMetricLabel"] { font-size:0.68rem !important; text-transform:uppercase; letter-spacing:0.4px; }
    [data-testid="stExpander"] {
        border:1px solid rgba(99,102,241,0.08); border-radius:10px; background:rgba(13,13,26,0.3);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width:5px; }
    ::-webkit-scrollbar-track { background:#0a0a14; }
    ::-webkit-scrollbar-thumb { background:#2a2a4a; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────

# ── TradingView Lightweight Charts Helpers ────────────────────────────────

TV_DARK = {
    "layout": {
        "textColor": "#94a3b8",
        "background": {"type": "solid", "color": "rgba(0,0,0,0)"},
        "fontFamily": "'Inter', sans-serif",
        "fontSize": 11,
    },
    "grid": {
        "vertLines": {"color": "rgba(99,102,241,0.05)"},
        "horzLines": {"color": "rgba(99,102,241,0.05)"},
    },
    "crosshair": {"mode": 0},
    "timeScale": {"borderColor": "rgba(99,102,241,0.1)"},
    "rightPriceScale": {"borderColor": "rgba(99,102,241,0.1)"},
}


def _tv_opts(height=400, **extra):
    """Build chart options with dark theme + custom height."""
    opts = {**TV_DARK, "height": height}
    opts.update(extra)
    return opts


def _ts_data(dates, values):
    """Convert dates + values to TradingView [{time, value}] format."""
    out = []
    for d, v in zip(dates, values):
        if pd.notna(v):
            t = pd.Timestamp(d)
            out.append({"time": t.strftime("%Y-%m-%d"), "value": round(float(v), 4)})
    return out


def tv_line_chart(series_list, height=400, key=None):
    """Render one or more line/area series via TradingView Lightweight Charts.

    series_list: list of dicts with keys:
        dates, values, name, color, type ("Area"|"Line"|"Baseline"|"Histogram"),
        Optional: lineWidth, topColor, bottomColor
    """
    tv_series = []
    for s in series_list:
        stype = s.get("type", "Area")
        data = _ts_data(s["dates"], s["values"])
        if not data:
            continue
        opts = {"color": s.get("color", "#6366f1"), "lineWidth": s.get("lineWidth", 2)}
        if stype == "Area":
            opts["topColor"] = s.get("topColor", s.get("color", "#6366f1") + "18")
            opts["bottomColor"] = s.get("bottomColor", "rgba(0,0,0,0)")
            opts["lineColor"] = s.get("color", "#6366f1")
            opts["lineWidth"] = s.get("lineWidth", 2)
        if stype == "Histogram":
            opts["color"] = s.get("color", "#6366f1")
        if stype == "Baseline":
            opts["topLineColor"] = s.get("topColor", "#22c55e")
            opts["topFillColor1"] = s.get("topColor", "#22c55e") + "18"
            opts["topFillColor2"] = "rgba(0,0,0,0)"
            opts["bottomLineColor"] = s.get("bottomColor", "#ef4444")
            opts["bottomFillColor1"] = "rgba(0,0,0,0)"
            opts["bottomFillColor2"] = s.get("bottomColor", "#ef4444") + "18"
            opts["baseValue"] = {"type": "price", "price": s.get("baseValue", 0)}
        entry = {"type": stype, "data": data, "options": opts}
        if s.get("priceScale"):
            entry["priceScale"] = s["priceScale"]
        tv_series.append(entry)
    if tv_series:
        renderLightweightCharts([{"chart": _tv_opts(height), "series": tv_series}], key=key)


def fmt_num(n, decimals=2):
    """Format number with thousands separator (apostrophe) and decimals."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    if decimals == 0:
        s = f"{abs(n):,.0f}"
    else:
        s = f"{abs(n):,.{decimals}f}"
    # Replace comma with apostrophe for thousands
    s = s.replace(",", "'")
    prefix = "-" if n < 0 else ""
    return prefix + s


def fmt_eur_full(n):
    """Format as full EUR value with thousands separator."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    return "€" + fmt_num(n, 2)


def fmt_eur_short(n):
    """Format number as short EUR for KPIs."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    if abs(n) >= 1e6:
        return f"€{n/1e6:,.2f}M".replace(",", "'")
    elif abs(n) >= 1e3:
        return f"€{n/1e3:,.1f}K".replace(",", "'")
    return f"€{n:,.2f}".replace(",", "'")


def fmt_pct(n, decimals=2):
    """Format as percentage."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    return f"{n*100:+.{decimals}f}%" if n >= 0 else f"{n*100:.{decimals}f}%"


def get_logo_base64():
    logo_path = DATA_DIR / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def color_pnl(val):
    if isinstance(val, (int, float)):
        if val > 0: return "color: #22c55e"
        elif val < 0: return "color: #ef4444"
    return ""


def format_table_numbers(df, euro_cols=None, pct_cols=None, price_cols=None):
    """Format DataFrame columns for display with proper number formatting."""
    result = df.copy()
    if euro_cols:
        for col in euro_cols:
            if col in result.columns:
                result[col] = result[col].apply(lambda x: fmt_num(x, 2) if pd.notna(x) else "N/A")
    if pct_cols:
        for col in pct_cols:
            if col in result.columns:
                result[col] = result[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
    if price_cols:
        for col in price_cols:
            if col in result.columns:
                result[col] = result[col].apply(lambda x: fmt_num(x, 2) if pd.notna(x) else "N/A")
    return result


# ── Load Data ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_all_data():
    positions = load_positions()
    overrides = get_overrides()
    isin_map = get_isin_map()
    fund_info = load_fund_info()
    nav_history = load_nav_history()
    transactions = load_transactions()
    cash_data = load_cash()
    return positions, overrides, isin_map, fund_info, nav_history, transactions, cash_data


def _interpolate_nav_to_daily(nav_history: pd.DataFrame) -> pd.DataFrame:
    """Interpolate monthly NAV snapshots to daily resolution.
    Pure computation, no API calls - always works."""
    if nav_history.empty or len(nav_history) < 2:
        return pd.DataFrame()

    nav_df = nav_history.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")

    # Filter outliers
    initial_nav = nav_df["nav"].iloc[0]
    if initial_nav > 0:
        nav_df = nav_df[(nav_df["nav"] > initial_nav * 0.1) & (nav_df["nav"] < initial_nav * 5.0)]

    if len(nav_df) < 2:
        return pd.DataFrame()

    # Resample to daily and interpolate
    daily_idx = pd.date_range(nav_df.index[0], nav_df.index[-1], freq="D")
    daily = nav_df.reindex(daily_idx)
    daily["nav"] = daily["nav"].interpolate(method="linear")

    # Interpolate benchmark too if present
    if "benchmark" in daily.columns:
        daily["benchmark"] = pd.to_numeric(daily["benchmark"], errors="coerce").interpolate(method="linear")
    else:
        daily["benchmark"] = np.nan

    daily = daily.reset_index().rename(columns={"index": "date"})
    daily = daily.dropna(subset=["nav"])
    return daily


positions, overrides, isin_map, fund_info, nav_history, transactions, cash_data = load_all_data()
has_data = not positions.empty

if has_data:
    positions = enrich_positions(positions, overrides)
    # Recalculate pnl_pct if it's 0 or NaN
    mask = (positions["pnl_pct"] == 0) | positions["pnl_pct"].isna()
    positions.loc[mask, "pnl_pct"] = positions.loc[mask].apply(
        lambda r: r["pnl"] / r["invested_capital"] if r["invested_capital"] > 0 else 0, axis=1
    )
    total_value = positions["current_value"].sum()
    total_invested = positions["invested_capital"].sum()
    total_pnl = total_value - total_invested
    total_pnl_pct = (total_pnl / total_invested) if total_invested > 0 else 0

    liquidita = cash_data.get("balance", 0) if cash_data else 0
    nav_total = total_value + liquidita

    # Use NAV from history (includes realized PnL, dividends) as authoritative value
    if not nav_history.empty and "nav" in nav_history.columns:
        _nav_series = pd.to_numeric(nav_history["nav"], errors="coerce").dropna()
        if len(_nav_series) >= 1:
            nav_total = _nav_series.iloc[-1]


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(f"""<div style="text-align:center;margin:0.5rem 0 0.8rem;">
            <img src="data:image/png;base64,{logo_b64}" width="80"
                 style="border-radius:50%;background:white;padding:5px;box-shadow:0 2px 12px rgba(99,102,241,0.2);">
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;margin-bottom:0.3rem;">
        <span style="font-size:1.05rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.3px;">SFC Investment Fund</span><br>
        <span style="font-size:0.65rem;color:#64748b;letter-spacing:0.5px;text-transform:uppercase;">Starting Finance Club Cattolica</span>
    </div>""", unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigazione",
        ["🏠 Dashboard", "📋 Posizioni", "📈 Performance",
         "📊 Analytics Avanzate", "🏆 Contribuzione P&L",
         "🏛️ Analisi Fixed Income",
         "🎯 Ottimizzazione PTF",
         "🔬 X-Ray Esposizioni", "💹 Multipli & Fondamentali",
         "📝 Operazioni & Import", "⚙️ Gestione Info Strumenti"],
        label_visibility="collapsed",
    )

    st.divider()

    if has_data:
        # Compact fund summary
        _perf = (nav_total - fund_info.get("initial_nav", 10_000_000)) / fund_info.get("initial_nav", 10_000_000) if fund_info.get("initial_nav", 0) > 0 else 0
        _perf_color = "#22c55e" if _perf >= 0 else "#ef4444"
        st.markdown(f"""<div style="background:rgba(13,13,26,0.5);border:1px solid rgba(99,102,241,0.08);border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;">
            <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">NAV</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">{fmt_eur_short(nav_total)}</div>
            <div style="font-size:0.72rem;color:{_perf_color};font-weight:600;">{_perf*100:+.2f}% since inception</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="font-size:0.72rem;color:#94a3b8;line-height:1.7;padding:0 0.2rem;">
            <span style="color:#64748b;">Posizioni:</span> {len(positions)}<br>
            <span style="color:#64748b;">Cash:</span> {fmt_eur_short(liquidita)}<br>
            <span style="color:#64748b;">Inception:</span> {fund_info.get('inception_date', 'N/A')}
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="font-size:0.65rem;color:#475569;line-height:1.6;padding:0.3rem 0.2rem;margin-top:0.3rem;">
        Last update: {fund_info.get("last_updated", "—")}
    </div>""", unsafe_allow_html=True)

    # GitHub sync status
    try:
        from github_sync import get_sync_status
        sync = get_sync_status()
        if sync["enabled"]:
            st.markdown('<div style="font-size:0.6rem;color:#22c55e;padding:0 0.2rem;">● Sync attivo</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.6rem;color:#f59e0b;padding:0 0.2rem;">○ Sync non configurato</div>', unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    if st.button("🔄 Ricarica Dati", use_container_width=True):
        _status = st.empty()
        with st.spinner("Ricalcolando posizioni da transazioni e aggiornando prezzi..."):
            try:
                # Step 1: Recompute positions from transactions
                _status.info("Step 1/4 — Ricalcolo posizioni da transazioni...")
                fresh_pos = compute_positions_from_transactions()
                if not fresh_pos.empty:
                    # Step 1b: Preserve existing manual prices for bonds
                    existing = load_positions()
                    if not existing.empty:
                        old_prices = existing[["isin", "current_price", "fx_rate_current"]].copy()
                        old_prices = old_prices.rename(columns={
                            "current_price": "_old_price", "fx_rate_current": "_old_fx",
                        })
                        fresh_pos = fresh_pos.merge(old_prices, on="isin", how="left")
                        mask = (fresh_pos["current_price"] == 0) & (fresh_pos["_old_price"].fillna(0) > 0)
                        fresh_pos.loc[mask, "current_price"] = fresh_pos.loc[mask, "_old_price"]
                        mask_fx = fresh_pos["_old_fx"].fillna(0) > 0
                        fresh_pos.loc[mask_fx, "fx_rate_current"] = fresh_pos.loc[mask_fx, "_old_fx"]
                        fresh_pos.drop(columns=["_old_price", "_old_fx"], inplace=True)

                    # Step 2: Update live prices and FX rates
                    _status.info("Step 2/4 — Download prezzi live...")
                    updated = update_position_prices(fresh_pos, get_isin_map())
                    save_positions(updated)

                    # Step 3: Recompute cash and NAV snapshot for today
                    _status.info("Step 3/4 — Calcolo cash e NAV di oggi...")
                    cash = compute_cash_from_transactions()
                    save_cash({"balance": cash, "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")})
                    nav = calculate_nav(updated, cash)
                    update_fund_info(nav, len(updated))
                    snapshot_nav(nav)

                    # Step 4: Fill missing NAV history days (incremental)
                    _status.info("Step 4/4 — Ricostruzione giorni NAV mancanti...")
                    fill_missing_nav_days(progress_callback=lambda msg: _status.info(f"Step 4/4 — {msg}"))

            except Exception as e:
                st.warning(f"Errore aggiornamento: {e}")
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    import streamlit.components.v1 as components

    # ── Ticker Tape (TradingView) — live market context ───────────────
    ticker_tape = """
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
        {
          "symbols": [
            {"proName": "AMEX:SPY", "title": "S&P 500"},
            {"proName": "AMEX:QQQ", "title": "Nasdaq 100"},
            {"proName": "AMEX:EFA", "title": "EAFE"},
            {"proName": "FX:EURUSD", "title": "EUR/USD"},
            {"proName": "FX:EURGBP", "title": "EUR/GBP"},
            {"proName": "TVC:GOLD", "title": "Gold"},
            {"proName": "CBOE:TNX", "title": "US 10Y"},
            {"proName": "AMEX:TLT", "title": "US Treasury 20Y+"},
            {"proName": "AMEX:VT", "title": "World ETF"}
          ],
          "showSymbolLogo": true,
          "isTransparent": true,
          "displayMode": "adaptive",
          "colorTheme": "dark",
          "locale": "it"
        }
      </script>
    </div>"""
    components.html(ticker_tape, height=46)

    # ── Fund Banner ───────────────────────────────────────────────────
    logo_html = ""
    logo_b64 = get_logo_base64()
    if logo_b64:
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="SFC">'

    st.markdown(f"""
    <div class="fund-banner">
        {logo_html}
        <div>
            <h1>SFC Cattolica Investment Fund</h1>
            <p>Starting Finance Club Cattolica &middot; Simulated Investment Fund &middot; Since Oct 2023</p>
        </div>
    </div>""", unsafe_allow_html=True)

    if not has_data:
        st.info("Benvenuto! Vai su Operazioni & Import per caricare i dati.")
        st.stop()

    # ── Compute Dashboard Data ────────────────────────────────────────
    # Use nav_history as single source of truth for all performance calcs
    nav = nav_total
    if not nav_history.empty and "nav" in nav_history.columns:
        _nav_clean = pd.to_numeric(nav_history["nav"], errors="coerce").dropna()
        if len(_nav_clean) >= 2:
            initial_nav = _nav_clean.iloc[0]
        else:
            initial_nav = fund_info.get("initial_nav", 10_000_000)
    else:
        initial_nav = fund_info.get("initial_nav", 10_000_000)
    inception_perf = (nav - initial_nav) / initial_nav if initial_nav > 0 else total_pnl_pct

    # Compute benchmark performance from nav_history (live, not from stale JSON)
    bench_perf_val = 0
    if not nav_history.empty and "benchmark" in nav_history.columns:
        _bv = pd.to_numeric(nav_history["benchmark"], errors="coerce").dropna()
        if len(_bv) >= 2:
            bench_perf_val = (_bv.iloc[-1] / _bv.iloc[0] - 1) if _bv.iloc[0] > 0 else 0
    alpha = inception_perf - bench_perf_val

    unrealized_total = positions["unrealized_pnl"].sum() if "unrealized_pnl" in positions.columns else total_pnl
    realized_total = positions["realized_pnl"].sum() if "realized_pnl" in positions.columns else 0
    dividends_total = positions["dividends_received"].sum() if "dividends_received" in positions.columns else 0
    total_return_all = unrealized_total + realized_total + dividends_total
    cash_pct = liquidita / nav_total if nav_total > 0 else 0

    def _pc(v): return "pos" if v > 0 else ("neg" if v < 0 else "")
    def _pf(v): return f"{v*100:+.2f}%"

    # ── KPI Cards (custom HTML) ───────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card accent-purple">
            <div class="kpi-label">NAV Corrente</div>
            <div class="kpi-value">{fmt_eur_full(nav)}</div>
            <div class="kpi-delta"><span class="{_pc(inception_perf)}">{_pf(inception_perf)}</span> since inception</div>
        </div>
        <div class="kpi-card accent-green">
            <div class="kpi-label">Total Return</div>
            <div class="kpi-value">{fmt_eur_full(total_return_all)}</div>
            <div class="kpi-delta">Unreal. {fmt_eur_short(unrealized_total)} &middot; Real. {fmt_eur_short(realized_total + dividends_total)}</div>
        </div>
        <div class="kpi-card accent-blue">
            <div class="kpi-label">Alpha vs Benchmark</div>
            <div class="kpi-value"><span class="{_pc(alpha)}">{_pf(alpha)}</span></div>
            <div class="kpi-delta">Fund {_pf(inception_perf)} &middot; Bench {_pf(bench_perf_val)}</div>
        </div>
        <div class="kpi-card accent-amber">
            <div class="kpi-label">Cash &amp; Positions</div>
            <div class="kpi-value">{fmt_eur_full(liquidita)}</div>
            <div class="kpi-delta">{cash_pct:.1%} del NAV &middot; {len(positions)} posizioni attive</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── NAV vs Benchmark Chart ────────────────────────────────────────
    # Load daily NAV data (already daily from build_nav_history.py)
    nav_df = _interpolate_nav_to_daily(nav_history)

    if not nav_df.empty and len(nav_df) > 1:
        st.markdown('<div class="section-header">Andamento NAV vs Benchmark</div>', unsafe_allow_html=True)

        # Timeframe selector
        dash_period = st.selectbox(
            "Periodo", ["YTD", "1M", "3M", "6M", "1Y", "Dall'Inizio"],
            index=5, key="dash_period_sel",
        )
        _today = pd.Timestamp.today()
        if dash_period == "YTD":
            _dash_start = pd.Timestamp(_today.year, 1, 1)
        elif dash_period == "1M":
            _dash_start = _today - pd.DateOffset(months=1)
        elif dash_period == "3M":
            _dash_start = _today - pd.DateOffset(months=3)
        elif dash_period == "6M":
            _dash_start = _today - pd.DateOffset(months=6)
        elif dash_period == "1Y":
            _dash_start = _today - pd.DateOffset(years=1)
        else:
            _dash_start = nav_df["date"].iloc[0]

        nav_df_filtered = nav_df[nav_df["date"] >= _dash_start].copy()
        if len(nav_df_filtered) < 2:
            nav_df_filtered = nav_df.copy()

        nav_df_filtered["nav_index"] = nav_df_filtered["nav"] / nav_df_filtered["nav"].iloc[0] * 100
        if "benchmark" in nav_df_filtered.columns:
            bp = pd.to_numeric(nav_df_filtered["benchmark"], errors="coerce")
            fv = bp.dropna().iloc[0] if not bp.dropna().empty else 1
            nav_df_filtered["bench_index"] = bp / fv * 100

        _series = [{"dates": nav_df_filtered["date"], "values": nav_df_filtered["nav_index"],
                     "color": "#6366f1", "type": "Area", "lineWidth": 2}]
        if "bench_index" in nav_df_filtered.columns:
            _series.append({"dates": nav_df_filtered["date"], "values": nav_df_filtered["bench_index"],
                            "color": "#22c55e", "type": "Line", "lineWidth": 2})
        # Chart legend
        st.markdown("""<div style="display:flex;gap:1.5rem;justify-content:flex-end;margin-bottom:0.3rem;font-size:0.75rem;">
            <span><span style="display:inline-block;width:12px;height:3px;background:#6366f1;border-radius:2px;vertical-align:middle;margin-right:5px;"></span><span style="color:#94a3b8;">SFC Fund</span></span>
            <span><span style="display:inline-block;width:12px;height:3px;background:#22c55e;border-radius:2px;vertical-align:middle;margin-right:5px;"></span><span style="color:#94a3b8;">Benchmark (VNGA60)</span></span>
        </div>""", unsafe_allow_html=True)
        tv_line_chart(_series, height=370, key=f"dash_nav_{dash_period}")

        # Period performance summary
        _fund_p = nav_df_filtered["nav"].iloc[-1] / nav_df_filtered["nav"].iloc[0] - 1
        _cp = st.columns(3)
        _cp[0].metric(f"SFC Fund ({dash_period})", f"{_fund_p:+.2%}")
        if "benchmark" in nav_df_filtered.columns:
            _bclean = pd.to_numeric(nav_df_filtered["benchmark"], errors="coerce").dropna()
            if len(_bclean) >= 2:
                _bp = _bclean.iloc[-1] / _bclean.iloc[0] - 1
                _cp[1].metric(f"Benchmark ({dash_period})", f"{_bp:+.2%}")
                _cp[2].metric("Alpha", f"{_fund_p - _bp:+.2%}")

    # ── Two Columns: Perf Table + Allocation ──────────────────────────
    col_left, col_right = st.columns([1.15, 0.85])

    with col_left:
        st.markdown('<div class="section-header">Performance Overview</div>', unsafe_allow_html=True)

        if not nav_df.empty and len(nav_df) >= 1:
            latest_nav_v = nav_df["nav"].iloc[-1]
            latest_date = nav_df["date"].iloc[-1]
            bv = pd.to_numeric(nav_df.get("benchmark", pd.Series(dtype=float)), errors="coerce")

            rows_html = ""
            for label, start_dt in {
                "YTD": pd.Timestamp(latest_date.year, 1, 1),
                "1M": latest_date - pd.DateOffset(months=1),
                "3M": latest_date - pd.DateOffset(months=3),
                "6M": latest_date - pd.DateOffset(months=6),
                "1Y": latest_date - pd.DateOffset(years=1),
                "Since Inception": nav_df["date"].iloc[0],
            }.items():
                mask = nav_df["date"] >= start_dt
                subset = nav_df[mask]
                if len(subset) >= 1:
                    sn = subset["nav"].iloc[0]
                    fp = (latest_nav_v / sn - 1) if sn > 0 else 0
                    bs = bv[mask].dropna()
                    bp = ((bs.iloc[-1] / bs.iloc[0] - 1) if len(bs) >= 1 and bs.iloc[0] > 0 else None)
                    dp = (fp - bp) if bp is not None else None
                    b_str = f'<span class="{_pc(bp)}">{bp:+.2%}</span>' if bp is not None else '<span style="color:#475569">N/A</span>'
                    d_str = f'<span class="{_pc(dp)}">{dp:+.2%}</span>' if dp is not None else '<span style="color:#475569">N/A</span>'
                    rows_html += f'<tr><td>{label}</td><td><span class="{_pc(fp)}">{fp:+.2%}</span></td><td>{b_str}</td><td>{d_str}</td></tr>'

            if rows_html:
                st.markdown(f"""
                <table class="perf-table">
                    <thead><tr><th>Periodo</th><th>Fondo SFC</th><th>Benchmark</th><th>Alpha</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Asset Allocation</div>', unsafe_allow_html=True)
        macro = positions.groupby("macro_class")["current_value"].sum().reset_index()
        macro["pct"] = macro["current_value"] / nav_total * 100
        if liquidita > 0:
            liq_row = pd.DataFrame([{"macro_class": "Liquidità", "current_value": liquidita, "pct": liquidita / nav_total * 100}])
            macro = pd.concat([macro, liq_row], ignore_index=True)

        cmap = {"Equity": "#6366f1", "Fixed Income": "#22c55e", "Alternative": "#f59e0b", "Liquidità": "#64748b"}
        fig = px.pie(macro, values="pct", names="macro_class", hole=0.6,
                     color="macro_class", color_discrete_map=cmap)
        fig.update_layout(
            height=270, margin=dict(t=10, b=10, l=10, r=10),
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
            annotations=[dict(text=f"<b>{len(positions)}</b><br><span style='font-size:10px;color:#64748b'>posizioni</span>",
                              x=0.5, y=0.5, font_size=16, font_color="#94a3b8", showarrow=False)],
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    # ── Two Columns: P&L Breakdown + Risk | Top Movers ────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">P&L Breakdown &amp; Risk</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-item"><div class="stat-label">Non Realizzato</div>
                <div class="stat-value" style="color:{'#22c55e' if unrealized_total >= 0 else '#ef4444'}">{fmt_eur_short(unrealized_total)}</div></div>
            <div class="stat-item"><div class="stat-label">Realizzato + Div.</div>
                <div class="stat-value">{fmt_eur_short(realized_total + dividends_total)}</div></div>
            <div class="stat-item"><div class="stat-label">Investito Totale</div>
                <div class="stat-value">{fmt_eur_short(total_invested)}</div></div>
            <div class="stat-item"><div class="stat-label">Controvalore</div>
                <div class="stat-value">{fmt_eur_short(total_value)}</div></div>
        </div>""", unsafe_allow_html=True)

        # Risk metrics from NAV history
        if not nav_df.empty and len(nav_df) > 2:
            ns = pd.Series(nav_df["nav"].values, index=pd.DatetimeIndex(nav_df["date"]))
            ns = ns[ns > 0]
            if len(ns) > 2:
                nr = calculate_returns(ns)
                ppy, _ = detect_frequency(ns)
                vol = annualized_volatility(nr, ppy)
                sr = sharpe_ratio(nr, 0.03, ppy)
                mdd = max_drawdown(nr)
                st.markdown(f"""
                <div class="stat-grid" style="margin-top:0.45rem">
                    <div class="stat-item"><div class="stat-label">Volatilità Ann.</div>
                        <div class="stat-value">{vol*100:.1f}%</div></div>
                    <div class="stat-item"><div class="stat-label">Sharpe Ratio</div>
                        <div class="stat-value">{sr:.2f}</div></div>
                    <div class="stat-item"><div class="stat-label">Max Drawdown</div>
                        <div class="stat-value" style="color:#ef4444">{mdd*100:.1f}%</div></div>
                    <div class="stat-item"><div class="stat-label">Equity / Fixed Inc.</div>
                        <div class="stat-value">{positions[positions["macro_class"]=="Equity"]["current_value"].sum()/nav_total*100:.0f}% / {positions[positions["macro_class"]=="Fixed Income"]["current_value"].sum()/nav_total*100:.0f}%</div></div>
                </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Top &amp; Bottom Performers</div>', unsafe_allow_html=True)
        pnl_sorted = positions[positions["pnl_pct"] != 0].sort_values("pnl_pct", ascending=False)
        if not pnl_sorted.empty:
            top5 = pnl_sorted.head(5)
            btm5 = pnl_sorted.tail(5)
            mhtml = '<div class="mover-section">Best Performers</div>'
            for _, r in top5.iterrows():
                pv = r["pnl_pct"] * 100
                mhtml += f'<div class="mover-item"><span class="mover-name">{r["name"]}</span><span class="mover-pnl {_pc(r["pnl_pct"])}">{pv:+.1f}%</span></div>'
            mhtml += '<div class="mover-section">Worst Performers</div>'
            for _, r in btm5.iterrows():
                pv = r["pnl_pct"] * 100
                mhtml += f'<div class="mover-item"><span class="mover-name">{r["name"]}</span><span class="mover-pnl {_pc(r["pnl_pct"])}">{pv:+.1f}%</span></div>'
            st.markdown(mhtml, unsafe_allow_html=True)

    # ── Top 10 Holdings ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Top 10 Holdings</div>', unsafe_allow_html=True)
    top10 = positions.nlargest(10, "current_value")[
        ["name", "macro_class", "currency", "avg_cost", "current_price", "invested_capital", "current_value", "pnl", "pnl_pct"]
    ].copy()
    top10["weight"] = (top10["current_value"] / nav_total * 100).round(2)
    top10["pnl_pct_d"] = (top10["pnl_pct"] * 100).round(2)
    top10_display = top10[["name", "macro_class", "currency", "avg_cost", "current_price",
                           "invested_capital", "current_value", "pnl", "pnl_pct_d", "weight"]].copy()
    top10_display.columns = ["Nome", "Classe", "Valuta", "Costo", "Prezzo", "Investito", "Valore", "P&L", "P&L %", "Peso %"]
    top10_display = format_table_numbers(top10_display,
                                          euro_cols=["Investito", "Valore", "P&L"],
                                          price_cols=["Costo", "Prezzo"])
    st.dataframe(top10_display, use_container_width=True, hide_index=True, height=min(420, 10 * 38 + 50))

    # ── Market Context (TradingView widgets in tabs) ──────────────────
    st.markdown('<div class="section-header">Contesto di Mercato</div>', unsafe_allow_html=True)
    tab_bench, tab_overview, tab_cal = st.tabs(["Benchmark Live", "Market Overview", "Calendario Economico"])

    with tab_bench:
        tv_chart = """
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript"
            src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
            { "symbol": "AMEX:SPY", "interval": "D", "timezone": "Europe/Rome",
              "theme": "dark", "style": "1", "locale": "it_IT",
              "enable_publishing": false, "allow_symbol_change": true,
              "save_image": true, "hide_volume": false,
              "backgroundColor": "rgba(0,0,0,0)", "gridColor": "rgba(42,42,74,0.12)",
              "width": "100%", "height": "500",
              "studies": ["MASimple@tv-basicstudies"] }
          </script>
        </div>"""
        components.html(tv_chart, height=520)

    with tab_overview:
        tv_mkt = """
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript"
            src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
            { "colorTheme": "dark", "dateRange": "1M", "showChart": true,
              "locale": "it", "isTransparent": true, "showSymbolLogo": true,
              "showFloatingTooltip": true, "width": "100%", "height": "500",
              "plotLineColorGrowing": "rgba(99,102,241,0.8)",
              "plotLineColorFalling": "rgba(239,68,68,0.8)",
              "gridLineColor": "rgba(99,102,241,0.06)",
              "scaleFontColor": "rgba(148,163,184,1)",
              "belowLineFillColorGrowing": "rgba(99,102,241,0.05)",
              "belowLineFillColorFalling": "rgba(239,68,68,0.05)",
              "belowLineFillColorGrowingBottom": "rgba(0,0,0,0)",
              "belowLineFillColorFallingBottom": "rgba(0,0,0,0)",
              "symbolActiveColor": "rgba(99,102,241,0.12)",
              "tabs": [
                { "title": "Indici", "symbols": [
                    {"s": "AMEX:SPY", "d": "S&P 500"},
                    {"s": "AMEX:QQQ", "d": "Nasdaq 100"},
                    {"s": "AMEX:EFA", "d": "EAFE"},
                    {"s": "AMEX:VGK", "d": "Europe ETF"},
                    {"s": "AMEX:EWI", "d": "Italy ETF"},
                    {"s": "TVC:NI225", "d": "Nikkei 225"} ]},
                { "title": "Bond", "symbols": [
                    {"s": "CBOE:TNX", "d": "US 10Y Yield"},
                    {"s": "AMEX:TLT", "d": "US Treasury 20Y+"},
                    {"s": "AMEX:IEF", "d": "US Treasury 7-10Y"},
                    {"s": "AMEX:AGG", "d": "US Aggregate Bond"},
                    {"s": "AMEX:BND", "d": "Total Bond Market"} ]},
                { "title": "Forex", "symbols": [
                    {"s": "FX:EURUSD", "d": "EUR/USD"},
                    {"s": "FX:EURGBP", "d": "EUR/GBP"},
                    {"s": "FX:EURJPY", "d": "EUR/JPY"},
                    {"s": "FX:EURCHF", "d": "EUR/CHF"},
                    {"s": "FX:EURHKD", "d": "EUR/HKD"} ]},
                { "title": "Commodities", "symbols": [
                    {"s": "TVC:GOLD", "d": "Gold"},
                    {"s": "TVC:SILVER", "d": "Silver"},
                    {"s": "TVC:USOIL", "d": "WTI Crude"},
                    {"s": "TVC:UKOIL", "d": "Brent Crude"} ]}
              ] }
          </script>
        </div>"""
        components.html(tv_mkt, height=520)

    with tab_cal:
        tv_cal = """
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript"
            src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
            { "colorTheme": "dark", "isTransparent": true,
              "width": "100%", "height": "480",
              "locale": "it", "importanceFilter": "0,1",
              "countryFilter": "eu,us,gb,it" }
          </script>
        </div>"""
        components.html(tv_cal, height=500)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: POSIZIONI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Posizioni":
    st.markdown('<div class="section-header">Posizioni del Fondo</div>', unsafe_allow_html=True)

    if not has_data:
        st.info("Nessun dato disponibile.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("NAV Totale", fmt_eur_full(nav_total))
    c2.metric("Posizioni", len(positions))
    c3.metric("P&L Totale", fmt_eur_full(total_pnl))
    c4.metric("P&L %", f"{total_pnl_pct:.2%}")
    c5.metric("Liquidità", fmt_eur_full(liquidita))

    st.divider()

    def display_asset_class_table(df, title, emoji):
        """Display position table for one asset class - weights are relative to the class."""
        if df.empty:
            st.info(f"Nessuna posizione {title}.")
            return

        class_value = df["current_value"].sum()
        class_invested = df["invested_capital"].sum()
        class_pnl = class_value - class_invested
        class_pnl_pct = (class_pnl / class_invested * 100) if class_invested > 0 else 0
        class_weight = class_value / nav_total * 100

        ca, cb, cc, cd = st.columns(4)
        ca.metric(f"Valore {title}", fmt_eur_full(class_value))
        cb.metric("Peso su NAV", f"{class_weight:.1f}%")
        cc.metric("P&L Classe", fmt_eur_full(class_pnl))
        cd.metric("P&L %", f"{class_pnl_pct:+.2f}%")

        base_cols = ["isin", "name", "sector", "currency", "avg_cost", "current_price",
                     "invested_capital", "current_value", "pnl", "pnl_pct"]
        # Add FX columns if present
        has_fx_cols = "fx_effect" in df.columns and "price_effect" in df.columns
        if has_fx_cols:
            base_cols += ["price_effect", "fx_effect"]
        display = df[[c for c in base_cols if c in df.columns]].copy()
        # Weight on asset class (not on total ptf)
        display["weight_class"] = (display["current_value"] / class_value * 100).round(2)
        display["pnl_pct_d"] = (display["pnl_pct"] * 100).round(2)
        display = display.sort_values("current_value", ascending=False)

        show_cols = ["isin", "name", "sector", "currency", "avg_cost", "current_price",
                     "invested_capital", "current_value", "pnl", "pnl_pct_d"]
        col_names = ["ISIN", "Nome", "Settore", "Valuta", "Prezzo Carico", "Prezzo Attuale",
                      "Investito €", "Controvalore €", "P&L €", "P&L %"]
        if has_fx_cols:
            show_cols += ["price_effect", "fx_effect"]
            col_names += ["Eff. Prezzo €", "Eff. Cambio €"]
        show_cols.append("weight_class")
        col_names.append(f"Peso su {title} %")

        show = display[[c for c in show_cols if c in display.columns]].copy()
        show.columns = col_names
        fx_euro_cols = ["Investito €", "Controvalore €", "P&L €"]
        if has_fx_cols:
            fx_euro_cols += ["Eff. Prezzo €", "Eff. Cambio €"]
        show = format_table_numbers(show, euro_cols=fx_euro_cols,
                                     price_cols=["Prezzo Carico", "Prezzo Attuale"])
        st.dataframe(show, use_container_width=True, hide_index=True,
                     height=min(500, len(show) * 38 + 50))

    # Equity
    equity = positions[positions["macro_class"] == "Equity"]
    with st.expander(f"📈 **Equity** ({len(equity)} posizioni)", expanded=True):
        display_asset_class_table(equity, "Equity", "📈")

    # Fixed Income
    fi = positions[positions["macro_class"] == "Fixed Income"]
    with st.expander(f"🏛️ **Fixed Income** ({len(fi)} posizioni)", expanded=True):
        display_asset_class_table(fi, "Fixed Income", "🏛️")

    # Alternative
    alt = positions[positions["macro_class"] == "Alternative"]
    with st.expander(f"💎 **Alternative** ({len(alt)} posizioni)", expanded=True):
        display_asset_class_table(alt, "Alternative", "💎")

    # ── Full Portfolio Table (with weight on PTF) ─────────────────────────
    st.divider()
    st.markdown('<div class="section-header">Tutte le Posizioni (Peso su Portafoglio)</div>', unsafe_allow_html=True)

    all_base_cols = ["isin", "name", "macro_class", "sector", "currency", "avg_cost", "current_price",
                      "invested_capital", "current_value", "pnl", "pnl_pct"]
    has_fx = "fx_effect" in positions.columns and "price_effect" in positions.columns
    if has_fx:
        all_base_cols += ["price_effect", "fx_effect"]
    all_disp = positions[[c for c in all_base_cols if c in positions.columns]].copy()
    all_disp["weight_ptf"] = (all_disp["current_value"] / nav_total * 100).round(2)
    all_disp["pnl_pct_d"] = (all_disp["pnl_pct"] * 100).round(2)
    all_disp = all_disp.sort_values("current_value", ascending=False)

    all_show_cols = ["isin", "name", "macro_class", "sector", "currency", "avg_cost", "current_price",
                      "invested_capital", "current_value", "pnl", "pnl_pct_d"]
    all_col_names = ["ISIN", "Nome", "Classe", "Settore", "Valuta", "Prezzo Carico", "Prezzo Attuale",
                      "Investito €", "Controvalore €", "P&L €", "P&L %"]
    if has_fx:
        all_show_cols += ["price_effect", "fx_effect"]
        all_col_names += ["Eff. Prezzo €", "Eff. Cambio €"]
    all_show_cols.append("weight_ptf")
    all_col_names.append("Peso PTF %")

    show_all = all_disp[[c for c in all_show_cols if c in all_disp.columns]].copy()
    show_all.columns = all_col_names
    all_euro_cols = ["Investito €", "Controvalore €", "P&L €"]
    if has_fx:
        all_euro_cols += ["Eff. Prezzo €", "Eff. Cambio €"]
    show_all = format_table_numbers(show_all, euro_cols=all_euro_cols,
                                     price_cols=["Prezzo Carico", "Prezzo Attuale"])
    st.dataframe(show_all, use_container_width=True, hide_index=True,
                 height=min(800, len(show_all) * 38 + 50))

    csv_buf = io.StringIO()
    show_all.to_csv(csv_buf, index=False)
    st.download_button("📥 Esporta Posizioni CSV", csv_buf.getvalue(), "posizioni_fondo.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Performance":
    st.markdown('<div class="section-header">Performance Analysis</div>', unsafe_allow_html=True)

    if not has_data or nav_history.empty:
        st.info("Dati NAV insufficienti. Aggiorna i prezzi dalla pagina Operazioni & Import.")
        st.stop()

    nav_df = nav_history.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    nav_series = pd.Series(nav_df["nav"].values, index=pd.DatetimeIndex(nav_df["date"]))
    nav_series = nav_series[nav_series > 0]

    bench_series = None
    if "benchmark" in nav_df.columns:
        bench_vals = pd.to_numeric(nav_df["benchmark"], errors="coerce")
        bench_series = pd.Series(bench_vals.values, index=pd.DatetimeIndex(nav_df["date"]))
        bench_series = bench_series.dropna()

    if len(nav_series) < 2:
        st.warning("Servono almeno 2 punti NAV per l'analisi.")
        st.stop()

    periods_per_year, freq_label = detect_frequency(nav_series)
    st.caption(f"📊 Frequenza dati: **{freq_label}** ({len(nav_series)} osservazioni, {periods_per_year} periodi/anno)")

    # Cumulative chart (TradingView)
    nav_returns = calculate_returns(nav_series)
    nav_cum = cumulative_returns(nav_returns)

    _perf_series = [{"dates": nav_cum.index, "values": nav_cum.values * 100,
                     "color": "#6366f1", "type": "Area", "lineWidth": 2}]
    if bench_series is not None and len(bench_series) > 1:
        bench_returns = calculate_returns(bench_series)
        bench_cum = cumulative_returns(bench_returns)
        _perf_series.append({"dates": bench_cum.index, "values": bench_cum.values * 100,
                             "color": "#22c55e", "type": "Line", "lineWidth": 2})
    tv_line_chart(_perf_series, height=420, key="perf_cum")

    # Key Metrics
    st.markdown('<div class="section-header">Metriche Chiave</div>', unsafe_allow_html=True)
    report = performance_report(nav_series, bench_series)
    cols = st.columns(4)
    for i, (key, val) in enumerate(report.items()):
        with cols[i % 4]:
            st.metric(key, val)

    # Drawdown (TradingView)
    st.markdown('<div class="section-header">Drawdown</div>', unsafe_allow_html=True)
    dd = drawdown_series(nav_series) * 100
    tv_line_chart([{"dates": dd.index, "values": dd.values, "color": "#ef4444",
                    "type": "Area", "lineWidth": 2,
                    "topColor": "rgba(0,0,0,0)", "bottomColor": "rgba(239,68,68,0.25)"}],
                  height=230, key="perf_dd")

    # Monthly heatmap
    st.markdown('<div class="section-header">Performance Mensile</div>', unsafe_allow_html=True)
    monthly = monthly_returns_table(nav_series)
    if not monthly.empty:
        fig_heat = go.Figure(data=go.Heatmap(
            z=monthly.values * 100, x=monthly.columns.tolist(),
            y=[str(y) for y in monthly.index.tolist()],
            colorscale=[[0, "#ef4444"], [0.5, "#1a1a2e"], [1, "#22c55e"]], zmid=0,
            text=np.where(np.isnan(monthly.values), "", np.vectorize(lambda x: f"{x*100:.1f}%")(monthly.values)),
            texttemplate="%{text}", textfont={"size": 11}))
        fig_heat.update_layout(height=max(180, len(monthly) * 50 + 50), margin=dict(t=10, b=10, l=60, r=20),
                               template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── P&L per Posizione (Top and Bottom) ────────────────────────────────
    if has_data:
        st.markdown('<div class="section-header">P&L per Posizione (Migliori e Peggiori)</div>', unsafe_allow_html=True)
        pnl_df = positions[["name", "pnl", "pnl_pct", "macro_class"]].copy()
        pnl_df["pnl_pct_d"] = (pnl_df["pnl_pct"] * 100).round(2)

        # Filter out zero P&L entries
        pnl_df = pnl_df[pnl_df["pnl_pct_d"] != 0]

        if not pnl_df.empty:
            top = pnl_df.nlargest(10, "pnl_pct_d")
            bottom = pnl_df.nsmallest(5, "pnl_pct_d")
            combined = pd.concat([top, bottom]).drop_duplicates()
            combined = combined.sort_values("pnl_pct_d")

            colors = ["#ef4444" if x < 0 else "#22c55e" for x in combined["pnl_pct_d"]]

            fig_pnl = go.Figure(go.Bar(
                x=combined["pnl_pct_d"].values,
                y=combined["name"].values,
                orientation="h",
                marker_color=colors,
                text=[f"{x:+.1f}%" for x in combined["pnl_pct_d"]],
                textposition="outside",
            ))
            fig_pnl.update_layout(
                height=max(400, len(combined) * 32),
                margin=dict(t=10, b=30, l=220, r=60),
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="P&L %", yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("Nessun dato P&L disponibile.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS AVANZATE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics Avanzate":
    st.markdown('<div class="section-header">Analytics Avanzate</div>', unsafe_allow_html=True)
    st.caption("Powered by **QuantStats** · Metriche di rischio, rolling stats, drawdown analysis, Monte Carlo")

    if not has_data:
        st.info("Nessun dato disponibile.")
        st.stop()

    from analytics_plus import (
        build_daily_nav_series, advanced_risk_metrics, rolling_statistics,
        drawdown_details, generate_html_report, correlation_matrix,
        fetch_position_prices,
    )

    # ── Build or load daily NAV ──────────────────────────────────────────
    @st.cache_data(ttl=3600, show_spinner="Ricostruendo NAV giornaliero...")
    def get_daily_nav():
        return build_daily_nav_series(
            positions, isin_map,
            inception_date=fund_info.get("inception_date", "2023-10-01"),
            initial_nav=fund_info.get("initial_nav", 10_000_000),
            cash=liquidita,
        )

    daily_nav = get_daily_nav()

    if daily_nav.empty or len(daily_nav) < 5:
        st.warning("Dati giornalieri insufficienti. Verifica il mapping ISIN → Ticker.")
        st.stop()

    # Build return series
    nav_series = pd.Series(daily_nav["nav"].values, index=pd.DatetimeIndex(daily_nav["date"]))
    nav_returns = nav_series.pct_change().dropna()

    bench_returns = None
    if "benchmark" in daily_nav.columns:
        bench_vals = daily_nav["benchmark"].dropna()
        if len(bench_vals) > 5:
            bench_series_daily = pd.Series(bench_vals.values, index=pd.DatetimeIndex(daily_nav.loc[bench_vals.index, "date"]))
            bench_returns = bench_series_daily.pct_change().dropna()

    # ── Tab layout ────────────────────────────────────────────────────────
    tab_perf, tab_risk, tab_rolling, tab_dd, tab_report = st.tabs(
        ["📈 Performance Giornaliera", "📉 Metriche di Rischio", "🔄 Rolling Stats",
         "📊 Drawdown Analysis", "📄 Report HTML"])

    with tab_perf:
        st.markdown('<div class="section-header">NAV Giornaliero vs Benchmark (VNGA60)</div>', unsafe_allow_html=True)

        # Period selector
        period = st.selectbox("Periodo", ["YTD", "1M", "3M", "6M", "1Y", "Dall'Inizio"], index=0)
        today = pd.Timestamp.today()
        if period == "YTD":
            start_date = pd.Timestamp(today.year, 1, 1)
        elif period == "1M":
            start_date = today - pd.DateOffset(months=1)
        elif period == "3M":
            start_date = today - pd.DateOffset(months=3)
        elif period == "6M":
            start_date = today - pd.DateOffset(months=6)
        elif period == "1Y":
            start_date = today - pd.DateOffset(years=1)
        else:
            start_date = pd.Timestamp("2023-10-01")

        mask = daily_nav["date"] >= start_date
        filtered = daily_nav[mask].copy()

        if len(filtered) < 2:
            st.warning("Dati insufficienti per il periodo selezionato.")
        else:
            # Rebase to 100
            filtered["nav_base100"] = filtered["nav"] / filtered["nav"].iloc[0] * 100
            if "benchmark" in filtered.columns:
                bench_f = filtered["benchmark"].dropna()
                if not bench_f.empty:
                    first_bench = bench_f.iloc[0]
                    filtered["bench_base100"] = filtered["benchmark"] / first_bench * 100

            _an_series = [{"dates": filtered["date"], "values": filtered["nav_base100"],
                          "color": "#6366f1", "type": "Area", "lineWidth": 2}]
            if "bench_base100" in filtered.columns:
                _an_series.append({"dates": filtered["date"], "values": filtered["bench_base100"],
                                   "color": "#22c55e", "type": "Line", "lineWidth": 2})
            tv_line_chart(_an_series, height=420, key=f"analytics_nav_{period}")

            # Period performance
            fund_perf = (filtered["nav"].iloc[-1] / filtered["nav"].iloc[0] - 1)
            cols_perf = st.columns(3)
            cols_perf[0].metric(f"SFC Fund ({period})", f"{fund_perf:+.2%}")
            if "benchmark" in filtered.columns and not filtered["benchmark"].dropna().empty:
                b_clean = filtered["benchmark"].dropna()
                bench_perf_val = (b_clean.iloc[-1] / b_clean.iloc[0] - 1)
                cols_perf[1].metric(f"Benchmark ({period})", f"{bench_perf_val:+.2%}")
                cols_perf[2].metric("Scostamento", f"{fund_perf - bench_perf_val:+.2%}")

    with tab_risk:
        st.markdown('<div class="section-header">Metriche di Rischio Avanzate</div>', unsafe_allow_html=True)

        metrics = advanced_risk_metrics(nav_returns, bench_returns)
        if not metrics:
            st.warning("Dati insufficienti.")
        else:
            # Group metrics
            perf_keys = ["CAGR", "Sharpe", "Sortino", "Calmar", "Omega"]
            risk_keys = ["Volatilità Ann.", "Max Drawdown", "VaR 95%", "CVaR 95%", "Recovery Factor", "Ulcer Index"]
            dist_keys = ["Skew", "Kurtosis", "Tail Ratio"]
            trade_keys = ["Win Rate", "Best Day", "Worst Day", "Profit Factor", "Payoff Ratio", "Kelly Criterion"]
            bench_keys = ["Information Ratio", "Treynor Ratio", "Alpha", "Beta"]

            def show_metrics_group(title, keys):
                st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(keys), 5))
                for i, key in enumerate(keys):
                    val = metrics.get(key)
                    if val is not None:
                        if isinstance(val, float):
                            if "Rate" in key or "CAGR" in key or "Drawdown" in key or "VaR" in key or "CVaR" in key:
                                display = f"{val:.2%}"
                            elif abs(val) > 100:
                                display = f"{val:.0f}"
                            else:
                                display = f"{val:.3f}"
                        else:
                            display = str(val)
                        cols[i % len(cols)].metric(key, display)

            show_metrics_group("Performance", perf_keys)
            show_metrics_group("Rischio", risk_keys)
            show_metrics_group("Distribuzione", dist_keys)
            show_metrics_group("Win/Loss", trade_keys)
            if bench_returns is not None:
                show_metrics_group("vs Benchmark", bench_keys)

    with tab_rolling:
        st.markdown('<div class="section-header">Statistiche Rolling</div>', unsafe_allow_html=True)
        window = st.select_slider("Finestra (giorni)", options=[21, 42, 63, 126, 252], value=63)

        rolling = rolling_statistics(nav_returns, window=window)
        if rolling.empty:
            st.warning(f"Servono almeno {window} giorni di dati.")
        else:
            for col_name, color in [("Rolling Sharpe", "#6366f1"), ("Rolling Sortino", "#22c55e"), ("Rolling Volatility", "#f59e0b")]:
                if col_name in rolling.columns:
                    st.caption(col_name)
                    _bv = 0 if "Sharpe" in col_name or "Sortino" in col_name else None
                    if _bv is not None:
                        tv_line_chart([{"dates": rolling.index, "values": rolling[col_name],
                                        "color": color, "type": "Baseline", "baseValue": _bv,
                                        "topColor": "#22c55e", "bottomColor": "#ef4444"}],
                                      height=220, key=f"roll_{col_name}")
                    else:
                        tv_line_chart([{"dates": rolling.index, "values": rolling[col_name],
                                        "color": color, "type": "Area", "lineWidth": 2}],
                                      height=220, key=f"roll_{col_name}")

    with tab_dd:
        st.markdown('<div class="section-header">Dettaglio Drawdown</div>', unsafe_allow_html=True)

        # Drawdown series chart (TradingView)
        dd_series = nav_returns.copy()
        cum = (1 + dd_series).cumprod()
        running_max = cum.cummax()
        dd_pct = (cum / running_max - 1) * 100

        tv_line_chart([{"dates": dd_pct.index, "values": dd_pct.values, "color": "#ef4444",
                        "type": "Area", "lineWidth": 2,
                        "topColor": "rgba(0,0,0,0)", "bottomColor": "rgba(239,68,68,0.25)"}],
                      height=280, key="analytics_dd")

        # Drawdown details table
        dd_detail = drawdown_details(nav_returns)
        if not dd_detail.empty:
            st.markdown("**Top 10 Drawdown Periods**")
            st.dataframe(dd_detail, use_container_width=True, hide_index=True)
        else:
            st.info("Dettagli drawdown non disponibili.")

    with tab_report:
        st.markdown('<div class="section-header">Report HTML QuantStats</div>', unsafe_allow_html=True)
        st.markdown("Genera un report completo in formato HTML scaricabile, con tutte le metriche e grafici.")

        if st.button("📄 Genera Report HTML", use_container_width=True):
            with st.spinner("Generando report..."):
                report_path = generate_html_report(
                    nav_returns, bench_returns,
                    title="SFC Cattolica Investment Fund - Performance Report")
            if "Errore" not in str(report_path):
                st.success("Report generato!")
                with open(report_path, "r") as f:
                    html_content = f.read()
                st.download_button(
                    "📥 Scarica Report HTML",
                    html_content,
                    "sfc_performance_report.html",
                    "text/html")
            else:
                st.error(report_path)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CONTRIBUZIONE P&L
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏆 Contribuzione P&L":
    st.markdown('<div class="section-header">Analisi Contribuzione alla Performance</div>', unsafe_allow_html=True)

    if not has_data:
        st.info("Nessun dato disponibile.")
        st.stop()

    # ── Contribution by single stock ────────────────────────────────────
    st.markdown('<div class="section-header">Contributo per Singolo Titolo</div>', unsafe_allow_html=True)

    contrib_pos = positions[["name", "isin", "macro_class", "sector", "invested_capital",
                              "current_value", "pnl", "pnl_pct"]].copy()
    contrib_pos["weight_ptf"] = contrib_pos["current_value"] / nav_total
    # Contribution to portfolio P&L = weight × individual return
    total_invested = contrib_pos["invested_capital"].sum()
    contrib_pos["contrib_pnl_pct"] = contrib_pos["pnl"] / total_invested * 100  # contribution in pp

    # Top gainers & losers
    top_gain = contrib_pos.nlargest(10, "pnl")
    top_loss = contrib_pos.nsmallest(10, "pnl")

    col_g, col_l = st.columns(2)
    with col_g:
        st.markdown("**🟢 Top 10 Contributori Positivi**")
        fig_g = go.Figure(go.Bar(
            x=top_gain["pnl"].values,
            y=top_gain["name"].values,
            orientation="h", marker_color="#22c55e",
            text=[f"€{x:+,.0f}".replace(",", "'") for x in top_gain["pnl"]],
            textposition="outside"))
        fig_g.update_layout(
            height=max(350, len(top_gain) * 35),
            margin=dict(t=10, b=30, l=180, r=80),
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_l:
        st.markdown("**🔴 Top 10 Contributori Negativi**")
        fig_l = go.Figure(go.Bar(
            x=top_loss["pnl"].values,
            y=top_loss["name"].values,
            orientation="h", marker_color="#ef4444",
            text=[f"€{x:+,.0f}".replace(",", "'") for x in top_loss["pnl"]],
            textposition="outside"))
        fig_l.update_layout(
            height=max(350, len(top_loss) * 35),
            margin=dict(t=10, b=30, l=180, r=80),
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_l, use_container_width=True)

    # Full contribution table
    contrib_table = contrib_pos.sort_values("pnl", ascending=False).copy()
    contrib_table["pnl_pct_d"] = (contrib_table["pnl_pct"] * 100).round(2)
    contrib_table["contrib_pnl_pct"] = contrib_table["contrib_pnl_pct"].round(3)
    contrib_table["weight_ptf_d"] = (contrib_table["weight_ptf"] * 100).round(2)
    show_contrib = contrib_table[["name", "macro_class", "sector", "invested_capital",
                                    "current_value", "pnl", "pnl_pct_d", "contrib_pnl_pct", "weight_ptf_d"]].copy()
    show_contrib.columns = ["Nome", "Classe", "Settore", "Investito", "Controvalore",
                             "P&L €", "P&L %", "Contrib. PTF (pp)", "Peso PTF %"]
    show_contrib = format_table_numbers(show_contrib,
                                         euro_cols=["Investito", "Controvalore", "P&L €"])
    st.dataframe(show_contrib, use_container_width=True, hide_index=True, height=min(600, len(show_contrib) * 38 + 50))

    st.divider()

    # ── Contribution by Sector ──────────────────────────────────────────
    st.markdown('<div class="section-header">Contributo per Settore</div>', unsafe_allow_html=True)

    by_sector = contrib_pos.groupby("sector").agg(
        invested=("invested_capital", "sum"),
        value=("current_value", "sum"),
        pnl=("pnl", "sum"),
        positions=("name", "count")
    ).reset_index()
    by_sector["pnl_pct"] = ((by_sector["value"] / by_sector["invested"] - 1) * 100).round(2)
    by_sector["weight"] = (by_sector["value"] / nav_total * 100).round(2)
    by_sector = by_sector.sort_values("pnl", ascending=True)

    colors_s = ["#ef4444" if x < 0 else "#22c55e" for x in by_sector["pnl"]]
    fig_sec = go.Figure(go.Bar(
        x=by_sector["pnl"].values,
        y=by_sector["sector"].values,
        orientation="h", marker_color=colors_s,
        text=[f"€{x:+,.0f}".replace(",", "'") for x in by_sector["pnl"]],
        textposition="outside"))
    fig_sec.update_layout(
        height=max(400, len(by_sector) * 30),
        margin=dict(t=10, b=30, l=200, r=80),
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sec, use_container_width=True)

    sec_table = by_sector.sort_values("pnl", ascending=False)
    sec_show = sec_table[["sector", "positions", "invested", "value", "pnl", "pnl_pct", "weight"]].copy()
    sec_show.columns = ["Settore", "# Posizioni", "Investito", "Controvalore", "P&L €", "P&L %", "Peso PTF %"]
    sec_show = format_table_numbers(sec_show, euro_cols=["Investito", "Controvalore", "P&L €"])
    st.dataframe(sec_show, use_container_width=True, hide_index=True)

    st.divider()

    # ── Contribution by Macro Class ─────────────────────────────────────
    st.markdown('<div class="section-header">Contributo per Macro Classe</div>', unsafe_allow_html=True)

    by_macro = contrib_pos.groupby("macro_class").agg(
        invested=("invested_capital", "sum"),
        value=("current_value", "sum"),
        pnl=("pnl", "sum"),
        positions=("name", "count")
    ).reset_index()
    by_macro["pnl_pct"] = ((by_macro["value"] / by_macro["invested"] - 1) * 100).round(2)
    by_macro["weight"] = (by_macro["value"] / nav_total * 100).round(2)
    # Add liquidità row
    liq_row = pd.DataFrame([{"macro_class": "Liquidità", "invested": 0, "value": liquidita,
                              "pnl": 0, "positions": 0, "pnl_pct": 0,
                              "weight": round(liquidita / nav_total * 100, 2)}])
    by_macro = pd.concat([by_macro, liq_row], ignore_index=True)

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig_mp = px.pie(by_macro, values="value", names="macro_class", hole=0.5,
                         color="macro_class",
                         color_discrete_map={"Equity": "#6366f1", "Fixed Income": "#22c55e",
                                             "Alternative": "#f59e0b", "Liquidità": "#64748b"})
        fig_mp.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20),
                              template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              title="Allocazione per Valore")
        fig_mp.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fig_mp, use_container_width=True)

    with col_bar:
        by_macro_sorted = by_macro[by_macro["macro_class"] != "Liquidità"].sort_values("pnl")
        colors_m = ["#ef4444" if x < 0 else "#22c55e" for x in by_macro_sorted["pnl"]]
        fig_mb = go.Figure(go.Bar(
            x=by_macro_sorted["pnl"].values,
            y=by_macro_sorted["macro_class"].values,
            orientation="h", marker_color=colors_m,
            text=[f"€{x:+,.0f}".replace(",", "'") for x in by_macro_sorted["pnl"]],
            textposition="outside"))
        fig_mb.update_layout(
            height=350, margin=dict(t=20, b=30, l=120, r=80),
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            title="P&L per Macro Classe")
        st.plotly_chart(fig_mb, use_container_width=True)

    macro_show = by_macro[["macro_class", "positions", "invested", "value", "pnl", "pnl_pct", "weight"]].copy()
    macro_show.columns = ["Macro Classe", "# Posizioni", "Investito", "Controvalore", "P&L €", "P&L %", "Peso PTF %"]
    macro_show = format_table_numbers(macro_show, euro_cols=["Investito", "Controvalore", "P&L €"])
    st.dataframe(macro_show, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALISI FIXED INCOME
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏛️ Analisi Fixed Income":
    st.markdown('<div class="section-header">Analisi Portafoglio Fixed Income</div>', unsafe_allow_html=True)

    if not has_data:
        st.info("Nessun dato disponibile.")
        st.stop()

    fi_positions = positions[positions["macro_class"] == "Fixed Income"].copy()
    if fi_positions.empty:
        st.warning("Nessuna posizione Fixed Income nel portafoglio.")
        st.stop()

    fi_value = fi_positions["current_value"].sum()
    fi_invested = fi_positions["invested_capital"].sum()
    fi_pnl = fi_value - fi_invested
    fi_weight = fi_value / nav_total * 100

    # ── KPIs ──────────────────────────────────────────────────────────
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Valore FI", fmt_eur_full(fi_value))
    f2.metric("Peso su NAV", f"{fi_weight:.1f}%")
    f3.metric("Posizioni", str(len(fi_positions)))
    f4.metric("P&L", fmt_eur_full(fi_pnl))
    f5.metric("P&L %", f"{fi_pnl/fi_invested*100:+.2f}%" if fi_invested > 0 else "N/A")

    st.divider()

    # ── Manual bond data input ────────────────────────────────────────
    st.markdown('<div class="section-header">Dati Obbligazionari</div>', unsafe_allow_html=True)
    st.info("Inserisci cedola (%), scadenza e rating per ogni bond. "
            "Questi dati permettono di calcolare duration, YTM e analisi di sensitivity.")

    # Load bond metadata from overrides
    bond_data = []
    for _, row in fi_positions.iterrows():
        isin = row["isin"]
        ov = overrides.get(isin, {})
        bond_data.append({
            "ISIN": isin,
            "Nome": row["name"],
            "Cedola %": ov.get("coupon_rate", 0.0),
            "Scadenza": ov.get("maturity_date", ""),
            "Rating": ov.get("rating", "NR"),
            "Prezzo": row["current_price"],
            "Quantità": row["quantity"],
            "Controvalore €": row["current_value"],
        })
    bond_df = pd.DataFrame(bond_data)

    # Editable form for bond-specific data
    with st.expander("✏️ Modifica dati obbligazionari", expanded=False):
        with st.form("bond_data_form"):
            bond_isin = st.selectbox("Seleziona bond",
                                      fi_positions["isin"].tolist(),
                                      format_func=lambda x: f"{fi_positions[fi_positions['isin']==x]['name'].iloc[0]} ({x})")
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                coupon = st.number_input("Cedola annuale (%)", min_value=0.0, max_value=20.0, step=0.125,
                                          value=float(overrides.get(bond_isin, {}).get("coupon_rate", 0.0)))
            with bc2:
                maturity = st.text_input("Data scadenza (YYYY-MM-DD)",
                                          value=overrides.get(bond_isin, {}).get("maturity_date", ""))
            with bc3:
                rating = st.text_input("Rating (es. BBB+, A-)",
                                        value=overrides.get(bond_isin, {}).get("rating", "NR"))

            if st.form_submit_button("💾 Salva", use_container_width=True):
                current_ov = get_overrides()
                if bond_isin not in current_ov:
                    current_ov[bond_isin] = {}
                current_ov[bond_isin]["coupon_rate"] = coupon
                current_ov[bond_isin]["maturity_date"] = maturity
                current_ov[bond_isin]["rating"] = rating
                save_overrides(current_ov)
                st.success(f"✅ Dati aggiornati per {bond_isin}")
                st.cache_data.clear()
                st.rerun()

    # ── Bond analysis table ───────────────────────────────────────────
    st.markdown('<div class="section-header">Analisi Posizioni Fixed Income</div>', unsafe_allow_html=True)

    from datetime import datetime as dt
    today = dt.now()

    analysis_rows = []
    total_weighted_duration = 0
    total_weighted_ytm = 0
    total_weighted_coupon = 0

    for _, row in fi_positions.iterrows():
        isin = row["isin"]
        ov = overrides.get(isin, {})
        coupon_rate = float(ov.get("coupon_rate", 0))
        maturity_str = ov.get("maturity_date", "")
        rating = ov.get("rating", "NR")
        price = float(row["current_price"])
        qty = float(row["quantity"])
        value = float(row["current_value"])

        years_to_maturity = None
        ytm = None
        modified_duration = None
        annual_income = 0

        if maturity_str:
            try:
                mat_date = dt.strptime(maturity_str, "%Y-%m-%d")
                years_to_maturity = max((mat_date - today).days / 365.25, 0.01)
            except Exception:
                years_to_maturity = None

        if years_to_maturity and price > 0:
            face_value = 100  # Standard bond face value

            # Annual coupon income (EUR)
            annual_income = qty * face_value * coupon_rate / 100

            # Approximate YTM: (coupon + (face - price) / years) / ((face + price) / 2)
            coupon_payment = face_value * coupon_rate / 100
            ytm = (coupon_payment + (face_value - price) / years_to_maturity) / ((face_value + price) / 2)

            # Macaulay Duration (simplified for annual coupon bond)
            if ytm and ytm > -0.99:
                y = ytm
                n = years_to_maturity
                c = coupon_rate / 100
                if abs(y) > 1e-6:
                    mac_dur = ((1 + y) / y) - ((1 + y + n * (c - y)) / (c * ((1 + y)**n - 1) + y))
                    # Clamp to reasonable range
                    mac_dur = max(0, min(mac_dur, years_to_maturity))
                else:
                    mac_dur = years_to_maturity * 0.9  # Zero coupon approximation
                modified_duration = mac_dur / (1 + y)
            else:
                modified_duration = years_to_maturity * 0.9

            # Accumulate for weighted averages
            weight = value / fi_value if fi_value > 0 else 0
            if modified_duration:
                total_weighted_duration += modified_duration * weight
            if ytm:
                total_weighted_ytm += ytm * weight
            total_weighted_coupon += (coupon_rate / 100) * weight

        analysis_rows.append({
            "Nome": row["name"],
            "ISIN": isin,
            "Rating": rating,
            "Cedola %": f"{coupon_rate:.3f}" if coupon_rate else "N/A",
            "Scadenza": maturity_str or "N/A",
            "Anni a Scad.": f"{years_to_maturity:.1f}" if years_to_maturity else "N/A",
            "Prezzo": f"{price:.2f}" if price else "N/A",
            "YTM": f"{ytm:.2%}" if ytm else "N/A",
            "Mod. Duration": f"{modified_duration:.2f}" if modified_duration else "N/A",
            "Reddito Ann. €": fmt_eur_full(annual_income) if annual_income else "N/A",
            "Controvalore €": fmt_eur_full(value),
            "Peso FI %": f"{value/fi_value*100:.1f}" if fi_value > 0 else "N/A",
        })

    # Portfolio-level FI metrics
    st.markdown('<div class="section-header">Metriche Aggregate Fixed Income</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration Media (mod.)", f"{total_weighted_duration:.2f}" if total_weighted_duration else "N/A",
              help="Duration modificata media ponderata. Misura la sensibilità del portafoglio FI ai tassi.")
    m2.metric("YTM Medio", f"{total_weighted_ytm:.2%}" if total_weighted_ytm else "N/A",
              help="Yield to Maturity medio ponderato del portafoglio FI.")
    m3.metric("Cedola Media", f"{total_weighted_coupon:.2%}" if total_weighted_coupon else "N/A")
    total_annual_income = 0
    for _, row in fi_positions.iterrows():
        ov = overrides.get(row["isin"], {})
        c = float(ov.get("coupon_rate", 0))
        total_annual_income += float(row["quantity"]) * 100 * c / 100
    m4.metric("Reddito Annuo Stimato", fmt_eur_full(total_annual_income))

    # Detail table
    if analysis_rows:
        st.dataframe(pd.DataFrame(analysis_rows), use_container_width=True, hide_index=True,
                     height=min(500, len(analysis_rows) * 38 + 50))

    # ── Interest Rate Sensitivity ─────────────────────────────────────
    if total_weighted_duration > 0:
        st.divider()
        st.markdown('<div class="section-header">Sensitivity ai Tassi di Interesse</div>', unsafe_allow_html=True)
        st.caption("Impatto stimato sul valore del portafoglio FI per variazione dei tassi.")

        rate_changes = [-1.00, -0.75, -0.50, -0.25, -0.10, 0, 0.10, 0.25, 0.50, 0.75, 1.00]
        impacts = []
        for dr in rate_changes:
            # ΔValue ≈ -Duration × ΔRate × Value
            dv = -total_weighted_duration * (dr / 100) * fi_value
            new_val = fi_value + dv
            impacts.append({
                "Variazione Tassi (bp)": f"{dr*100:+.0f}",
                "Impatto €": dv,
                "Nuovo Valore FI": new_val,
                "Impatto %": dv / fi_value * 100 if fi_value else 0,
            })

        impact_df = pd.DataFrame(impacts)

        # Bar chart of impacts
        colors = ["#22c55e" if x >= 0 else "#ef4444" for x in impact_df["Impatto €"]]
        fig_sens = go.Figure(go.Bar(
            x=[f"{r*100:+.0f}bp" for r in rate_changes],
            y=impact_df["Impatto €"].values,
            marker_color=colors,
            text=[f"€{x:+,.0f}".replace(",", "'") for x in impact_df["Impatto €"]],
            textposition="outside"))
        fig_sens.update_layout(
            height=400, margin=dict(t=20, b=40, l=60, r=20),
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Variazione Tassi", yaxis_title="Impatto sul Valore FI (€)")
        st.plotly_chart(fig_sens, use_container_width=True)

        # Table
        show_impact = impact_df.copy()
        show_impact["Impatto €"] = show_impact["Impatto €"].apply(lambda x: f"€{x:+,.0f}".replace(",", "'"))
        show_impact["Nuovo Valore FI"] = show_impact["Nuovo Valore FI"].apply(lambda x: fmt_eur_full(x))
        show_impact["Impatto %"] = show_impact["Impatto %"].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(show_impact, use_container_width=True, hide_index=True)

    # ── Rating Distribution ───────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">Distribuzione per Rating</div>', unsafe_allow_html=True)
    rating_data = []
    for _, row in fi_positions.iterrows():
        ov = overrides.get(row["isin"], {})
        rating_data.append({
            "rating": ov.get("rating", "NR"),
            "value": row["current_value"]
        })
    rating_df = pd.DataFrame(rating_data)
    if not rating_df.empty:
        by_rating = rating_df.groupby("rating")["value"].sum().reset_index()
        by_rating["pct"] = (by_rating["value"] / fi_value * 100).round(1)
        by_rating = by_rating.sort_values("value", ascending=False)

        fig_rat = px.pie(by_rating, values="value", names="rating", hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_rat.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20),
                               template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               title="Distribuzione per Rating")
        fig_rat.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_rat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OTTIMIZZAZIONE PTF
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Ottimizzazione PTF":
    st.markdown('<div class="section-header">Ottimizzazione Portafoglio</div>', unsafe_allow_html=True)
    st.caption("Powered by **PyPortfolioOpt** · Efficient Frontier, Risk Parity, Risk Contribution")

    if not has_data:
        st.info("Nessun dato disponibile.")
        st.stop()

    from analytics_plus import (
        fetch_position_prices, optimize_max_sharpe, optimize_min_volatility,
        optimize_hrp, efficient_frontier_curve, risk_contribution, correlation_matrix,
    )

    # Only equity for optimization (bonds/crypto have different dynamics)
    equity_positions = positions[positions["macro_class"] == "Equity"].copy()
    if equity_positions.empty:
        st.warning("Nessuna posizione equity per l'ottimizzazione.")
        st.stop()

    st.info("**Nota:** L'ottimizzazione usa i prezzi storici degli ultimi 2 anni per le sole posizioni Equity. "
            "I risultati sono indicativi, non raccomandazioni operative.")

    tab_opt, tab_risk, tab_corr = st.tabs(
        ["🎯 Portafogli Ottimali", "⚡ Risk Contribution", "🔗 Correlazioni"])

    @st.cache_data(ttl=3600, show_spinner="Scaricando prezzi storici...")
    def get_equity_prices():
        return fetch_position_prices(equity_positions, isin_map, period="2y")

    prices = get_equity_prices()

    if prices.empty or len(prices) < 60:
        st.warning("Dati prezzi insufficienti per l'ottimizzazione (servono almeno 60 giorni).")
        st.stop()

    with tab_opt:
        st.markdown('<div class="section-header">Portafogli Ottimali</div>', unsafe_allow_html=True)

        col_ms, col_mv, col_hrp = st.columns(3)

        with col_ms:
            st.markdown("**📈 Max Sharpe**")
            result_ms = optimize_max_sharpe(prices)
            if "error" in result_ms:
                st.error(result_ms["error"])
            else:
                st.metric("Expected Return", f"{result_ms['expected_return']:.1%}")
                st.metric("Volatilità", f"{result_ms['volatility']:.1%}")
                st.metric("Sharpe Ratio", f"{result_ms['sharpe']:.2f}")

        with col_mv:
            st.markdown("**🛡️ Min Volatilità**")
            result_mv = optimize_min_volatility(prices)
            if "error" in result_mv:
                st.error(result_mv["error"])
            else:
                st.metric("Expected Return", f"{result_mv['expected_return']:.1%}")
                st.metric("Volatilità", f"{result_mv['volatility']:.1%}")
                st.metric("Sharpe Ratio", f"{result_mv['sharpe']:.2f}")

        with col_hrp:
            st.markdown("**🌳 Risk Parity (HRP)**")
            result_hrp = optimize_hrp(prices)
            if "error" in result_hrp:
                st.error(result_hrp["error"])
            else:
                st.metric("Expected Return", f"{result_hrp['expected_return']:.1%}")
                st.metric("Volatilità", f"{result_hrp['volatility']:.1%}")
                st.metric("Sharpe Ratio", f"{result_hrp['sharpe']:.2f}")

        # Show weights comparison
        st.divider()
        st.markdown('<div class="section-header">Confronto Pesi Ottimali vs Attuali</div>', unsafe_allow_html=True)

        # Current weights
        eq_total = equity_positions["current_value"].sum()
        current_w = {}
        for _, row in equity_positions.iterrows():
            name = row.get("name", "")
            if name in prices.columns:
                current_w[name] = row["current_value"] / eq_total

        # Build comparison table
        all_names = set(current_w.keys())
        for r in [result_ms, result_mv, result_hrp]:
            if "weights" in r:
                all_names.update(r["weights"].keys())

        comparison = []
        for name in sorted(all_names):
            comparison.append({
                "Strumento": name,
                "Attuale %": round(current_w.get(name, 0) * 100, 1),
                "Max Sharpe %": round(result_ms.get("weights", {}).get(name, 0) * 100, 1) if "weights" in result_ms else 0,
                "Min Vol %": round(result_mv.get("weights", {}).get(name, 0) * 100, 1) if "weights" in result_mv else 0,
                "HRP %": round(result_hrp.get("weights", {}).get(name, 0) * 100, 1) if "weights" in result_hrp else 0,
            })

        comp_df = pd.DataFrame(comparison)
        comp_df = comp_df[(comp_df.iloc[:, 1:] > 0).any(axis=1)]  # Remove all-zero rows
        comp_df = comp_df.sort_values("Attuale %", ascending=False)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Efficient frontier chart
        st.divider()
        st.markdown('<div class="section-header">Frontiera Efficiente</div>', unsafe_allow_html=True)
        frontier = efficient_frontier_curve(prices, n_points=30)
        if not frontier.empty:
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=frontier["volatility"] * 100, y=frontier["return"] * 100,
                mode="lines", name="Frontiera Efficiente",
                line=dict(color="#6366f1", width=3)))

            # Plot current portfolio
            if current_w:
                from pypfopt import expected_returns as er, risk_models as rm
                try:
                    mu = er.mean_historical_return(prices)
                    S = rm.CovarianceShrinkage(prices).ledoit_wolf()
                    w_arr = np.array([current_w.get(c, 0) for c in prices.columns])
                    w_arr = w_arr / w_arr.sum()
                    cur_ret = float(w_arr @ mu.values) * 100
                    cur_vol = float(np.sqrt(w_arr @ S.values @ w_arr)) * 100
                    fig_ef.add_trace(go.Scatter(
                        x=[cur_vol], y=[cur_ret], mode="markers",
                        marker=dict(size=14, color="#ef4444", symbol="star"),
                        name="PTF Attuale"))
                except Exception:
                    pass

            # Plot optimal portfolios
            for label, result, color in [
                ("Max Sharpe", result_ms, "#22c55e"),
                ("Min Vol", result_mv, "#f59e0b"),
                ("HRP", result_hrp, "#ff6b6b")]:
                if "volatility" in result:
                    fig_ef.add_trace(go.Scatter(
                        x=[result["volatility"] * 100], y=[result["expected_return"] * 100],
                        mode="markers", marker=dict(size=12, color=color, symbol="diamond"),
                        name=label))

            fig_ef.update_layout(
                height=500, margin=dict(t=20, b=40, l=50, r=20),
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Volatilità Annualizzata (%)", yaxis_title="Return Atteso Annualizzato (%)",
                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_ef, use_container_width=True)
        else:
            st.info("Impossibile generare la frontiera efficiente.")

    with tab_risk:
        st.markdown('<div class="section-header">Contributo al Rischio per Posizione</div>', unsafe_allow_html=True)
        st.markdown("Mostra quanto ogni posizione contribuisce al rischio totale del portafoglio.")

        rc_df = risk_contribution(equity_positions, isin_map, prices)
        if not rc_df.empty:
            # Bar chart: weight vs risk contribution
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Bar(
                x=rc_df["name"], y=rc_df["weight"],
                name="Peso %", marker_color="#6366f1"))
            fig_rc.add_trace(go.Bar(
                x=rc_df["name"], y=rc_df["pct_contribution"],
                name="Contributo Rischio %", marker_color="#ef4444"))
            fig_rc.update_layout(
                height=450, margin=dict(t=20, b=100, l=50, r=20),
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                barmode="group", xaxis_tickangle=-45,
                legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_rc, use_container_width=True)

            # Table
            rc_show = rc_df.copy()
            rc_show["weight"] = rc_show["weight"].round(2)
            rc_show["pct_contribution"] = rc_show["pct_contribution"].round(2)
            rc_show.columns = ["Strumento", "Peso %", "Risk Contribution", "% Rischio Totale"]
            st.dataframe(rc_show, use_container_width=True, hide_index=True)

            # Risk/Weight ratio
            rc_df["risk_weight_ratio"] = rc_df["pct_contribution"] / rc_df["weight"].replace(0, np.nan)
            outliers = rc_df[rc_df["risk_weight_ratio"] > 1.5].sort_values("risk_weight_ratio", ascending=False)
            if not outliers.empty:
                st.warning(f"⚠️ {len(outliers)} posizioni contribuiscono sproporzionatamente al rischio:")
                for _, row in outliers.iterrows():
                    st.caption(f"  **{row['name']}**: peso {row['weight']:.1f}% → rischio {row['pct_contribution']:.1f}% (ratio {row['risk_weight_ratio']:.1f}x)")
        else:
            st.info("Risk contribution non calcolabile con i dati disponibili.")

    with tab_corr:
        st.markdown('<div class="section-header">Matrice di Correlazione</div>', unsafe_allow_html=True)
        corr = correlation_matrix(prices)
        if not corr.empty:
            # Truncate long names
            short_names = [n[:20] if len(n) > 20 else n for n in corr.columns]

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=short_names, y=short_names,
                colorscale=[[0, "#ef4444"], [0.5, "#1a1a2e"], [1, "#22c55e"]],
                zmid=0, zmin=-1, zmax=1,
                text=np.vectorize(lambda x: f"{x:.2f}")(corr.values),
                texttemplate="%{text}", textfont={"size": 9}))
            fig_corr.update_layout(
                height=max(500, len(corr) * 25),
                margin=dict(t=10, b=10, l=10, r=10),
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)

            # Key insights
            upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            highest = upper_tri.stack().nlargest(3)
            lowest = upper_tri.stack().nsmallest(3)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Correlazioni più alte:**")
                for (n1, n2), val in highest.items():
                    st.caption(f"  {n1[:25]} ↔ {n2[:25]}: **{val:.2f}**")
            with c2:
                st.markdown("**Correlazioni più basse:**")
                for (n1, n2), val in lowest.items():
                    st.caption(f"  {n1[:25]} ↔ {n2[:25]}: **{val:.2f}**")
        else:
            st.info("Matrice di correlazione non disponibile.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: X-RAY ESPOSIZIONI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 X-Ray Esposizioni":
    st.markdown('<div class="section-header">X-Ray del Portafoglio</div>', unsafe_allow_html=True)

    if not has_data:
        st.info("Nessun dato disponibile.")
        st.stop()

    weights = positions["current_value"] / positions["current_value"].sum()
    hhi = (weights**2).sum()
    eff_pos = 1 / hhi if hhi > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posizioni Totali", len(positions))
    c2.metric("Posizioni Effettive", f"{eff_pos:.1f}")
    c3.metric("Top 5 Concentrazione", f"{weights.nlargest(5).sum():.1%}")
    c4.metric("HHI Index", f"{hhi:.4f}")
    st.divider()

    tab1, tab_ind, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🏢 Settore", "🔎 Industry", "🌍 Geografia", "💱 Valuta", "📊 Macro Classe", "🏭 Tipo Asset", "📈 Top Holdings"])

    def exposure_chart(df, group_col, label):
        if df.empty:
            st.info("Nessun dato.")
            return
        grouped = df.groupby(group_col).agg(
            value=("current_value", "sum"), count=("isin", "count"),
            names=("name", lambda x: ", ".join(x.head(5)))).reset_index()
        grouped["weight"] = (grouped["value"] / grouped["value"].sum() * 100).round(2)
        grouped["value_fmt"] = grouped["value"].apply(lambda x: fmt_num(x, 2))
        grouped = grouped.sort_values("weight", ascending=False)

        cl, cr = st.columns([1, 1])
        with cl:
            fig = px.pie(grouped, values="weight", names=group_col, hole=0.45,
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400, margin=dict(t=20, b=20), template="plotly_dark",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig.update_traces(textposition="outside", textinfo="percent+label", textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            show_g = grouped[[group_col, "weight", "value_fmt", "count", "names"]].copy()
            show_g.columns = [label, "Peso %", "Valore €", "# Strumenti", "Strumenti"]
            st.dataframe(show_g, use_container_width=True, hide_index=True)

    with tab1:
        if "sector" in positions.columns:
            exposure_chart(positions, "sector", "Settore")

    with tab_ind:
        if "industry" in positions.columns:
            exposure_chart(positions, "industry", "Industry")
        else:
            st.info("Nessun dato di industry disponibile. Configura le industry in overrides.json.")

    with tab2:
        if "country" in positions.columns:
            exposure_chart(positions, "country", "Paese")
            country_iso_map = {
                "United States": "USA", "Italy": "ITA", "France": "FRA", "Netherlands": "NLD",
                "United Kingdom": "GBR", "Germany": "DEU", "Australia": "AUS", "Austria": "AUT",
                "Romania": "ROU", "Brazil": "BRA", "China": "CHN", "Argentina": "ARG",
            }
            country_data = positions.groupby("country")["current_value"].sum().reset_index()
            country_data["weight"] = (country_data["current_value"] / country_data["current_value"].sum() * 100).round(2)
            country_data["iso3"] = country_data["country"].map(country_iso_map)
            mapped = country_data.dropna(subset=["iso3"])
            if not mapped.empty:
                fig_map = px.choropleth(mapped, locations="iso3", locationmode="ISO-3", color="weight",
                                         color_continuous_scale="Blues", hover_name="country", labels={"weight": "Peso %"})
                fig_map.update_layout(height=350, margin=dict(t=10, b=10, l=0, r=0), template="plotly_dark",
                                       geo=dict(showframe=False, bgcolor="rgba(0,0,0,0)"), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_map, use_container_width=True)

    with tab3:
        exposure_chart(positions, "currency", "Valuta")
    with tab4:
        exposure_chart(positions, "macro_class", "Macro Classe")
    with tab5:
        if "asset_sub_type" in positions.columns:
            exposure_chart(positions, "asset_sub_type", "Tipo Asset")

    with tab6:
        st.markdown('<div class="section-header">Top 20 Posizioni</div>', unsafe_allow_html=True)
        top20 = positions.nlargest(20, "current_value")[["name", "macro_class", "current_value"]].copy()
        top20["weight"] = (top20["current_value"] / nav_total * 100).round(2)
        # Sort by value for clean display
        top20 = top20.sort_values("current_value", ascending=True)

        fig_bar = go.Figure(go.Bar(
            x=top20["current_value"].values,
            y=top20["name"].values,
            orientation="h",
            marker_color=[{"Equity": "#6366f1", "Fixed Income": "#22c55e", "Alternative": "#f59e0b"}.get(mc, "#64748b")
                          for mc in top20["macro_class"]],
            text=[f"€{v:,.0f} ({w:.1f}%)".replace(",", "'") for v, w in zip(top20["current_value"], top20["weight"])],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=max(550, len(top20) * 30),
            margin=dict(t=10, b=30, l=250, r=120),
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Valore €",
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MULTIPLI & FONDAMENTALI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💹 Multipli & Fondamentali":
    st.markdown('<div class="section-header">Multipli & Analisi Fondamentale</div>', unsafe_allow_html=True)

    if not has_data:
        st.info("Nessun dato disponibile.")
        st.stop()

    st.info("**Nota:** I multipli vengono recuperati live da Yahoo Finance. "
            "Per strumenti senza mapping (obbligazioni), aggiungi i dati in **Gestione Info Strumenti**.")

    from data_fetcher import get_ticker_info

    @st.cache_data(ttl=600)
    def fetch_fundamentals(isins, _isin_map):
        results = []
        for isin in isins:
            ticker = _isin_map.get(isin)
            if ticker:
                info = get_ticker_info(ticker)
                info["isin"] = isin
                results.append(info)
        return pd.DataFrame(results)

    equity_pos = positions[positions["macro_class"] == "Equity"].copy()
    if equity_pos.empty:
        st.warning("Nessuna posizione equity.")
        st.stop()

    with st.spinner("Caricando dati fondamentali da Yahoo Finance..."):
        fundamentals = fetch_fundamentals(equity_pos["isin"].tolist(), isin_map)

    if not fundamentals.empty:
        merged = equity_pos.merge(fundamentals, on="isin", how="left", suffixes=("", "_yf"))
        # Normalize dividend_yield: yfinance sometimes returns it as % (e.g. 2.5) instead of decimal (0.025)
        if "dividend_yield" in merged.columns:
            merged["dividend_yield"] = merged["dividend_yield"].apply(
                lambda x: x / 100 if pd.notna(x) and x > 1 else x)
        total_eq_value = merged["current_value"].sum()
        merged["w"] = merged["current_value"] / total_eq_value

        st.markdown('<div class="section-header">Multipli Medi Ponderati (Equity)</div>', unsafe_allow_html=True)
        multiples_config = [("trailing_pe", "P/E Trailing"), ("forward_pe", "P/E Forward"),
                            ("price_to_book", "Price/Book"), ("ev_to_ebitda", "EV/EBITDA"),
                            ("dividend_yield", "Div Yield"), ("beta", "Beta")]
        mult_cols = st.columns(len(multiples_config))
        for i, (col_name, label) in enumerate(multiples_config):
            with mult_cols[i]:
                if col_name in merged.columns:
                    valid = merged[["w", col_name]].dropna()
                    valid = valid[(valid[col_name] > 0) & (valid[col_name] < 500)]
                    if not valid.empty:
                        w = valid["w"] / valid["w"].sum()
                        wavg = (w * valid[col_name]).sum()
                        if col_name == "dividend_yield":
                            st.metric(label, f"{wavg:.2%}")
                        elif col_name == "beta":
                            st.metric(label, f"{wavg:.2f}")
                        else:
                            st.metric(label, f"{wavg:.1f}x")
                    else:
                        st.metric(label, "N/A")
                else:
                    st.metric(label, "N/A")

        # Detail table
        st.markdown('<div class="section-header">Dettaglio per Posizione</div>', unsafe_allow_html=True)
        detail_cols = ["name", "trailing_pe", "forward_pe", "price_to_book", "ev_to_ebitda",
                       "profit_margin", "roe", "dividend_yield", "beta"]
        available_cols = [c for c in detail_cols if c in merged.columns]
        detail = merged[available_cols].copy()
        for col in ["profit_margin", "roe", "dividend_yield"]:
            if col in detail.columns:
                detail[col] = detail[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
        for col in ["trailing_pe", "forward_pe", "price_to_book", "ev_to_ebitda", "beta"]:
            if col in detail.columns:
                detail[col] = detail[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
        col_rename = {"name": "Nome", "trailing_pe": "P/E Trail.", "forward_pe": "P/E Fwd",
                      "price_to_book": "P/B", "ev_to_ebitda": "EV/EBITDA",
                      "profit_margin": "Margin", "roe": "ROE", "dividend_yield": "Div Yield", "beta": "Beta"}
        detail = detail.rename(columns={k: v for k, v in col_rename.items() if k in detail.columns})
        st.dataframe(detail, use_container_width=True, hide_index=True)

        # ── Contribution to P&L by Sector (replaces useless scatter) ──────
        st.markdown('<div class="section-header">Contributo al P&L per Settore</div>', unsafe_allow_html=True)
        contrib = positions[["name", "sector", "macro_class", "pnl", "current_value"]].copy()
        contrib = contrib[contrib["pnl"].notna() & (contrib["pnl"] != 0)]
        if not contrib.empty:
            by_class = contrib.groupby("sector").agg(
                total_pnl=("pnl", "sum"), total_value=("current_value", "sum"), count=("name", "count")
            ).reset_index()
            by_class = by_class.sort_values("total_pnl")
            colors = ["#ef4444" if x < 0 else "#22c55e" for x in by_class["total_pnl"]]
            fig_contrib = go.Figure(go.Bar(
                x=by_class["total_pnl"].values,
                y=by_class["sector"].values,
                orientation="h", marker_color=colors,
                text=[f"€{x:+,.0f}".replace(",", "'") for x in by_class["total_pnl"]],
                textposition="outside",
            ))
            fig_contrib.update_layout(
                height=max(400, len(by_class) * 28),
                margin=dict(t=10, b=30, l=220, r=80),
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="P&L €",
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
    else:
        st.warning("Nessun dato fondamentale disponibile. Verifica il mapping ISIN.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OPERAZIONI & IMPORT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📝 Operazioni & Import":
    st.markdown('<div class="section-header">Gestione Operazioni</div>', unsafe_allow_html=True)

    # Password gate for management pages
    _mgmt_pw = st.text_input("🔒 Password richiesta", type="password", key="pw_operazioni")
    if _mgmt_pw != "Maradona1":
        if _mgmt_pw:
            st.warning("Password errata")
        st.stop()

    # KPIs
    if has_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Posizioni", len(positions))
        c2.metric("Transazioni", len(transactions) if not transactions.empty else 0)
        c3.metric("Liquidità", fmt_eur_full(liquidita))
        c4.metric("NAV", fmt_eur_full(nav_total))
        st.divider()

    tab_new, tab_view, tab_edit, tab_prices, tab_manual = st.tabs(
        ["➕ Nuova Operazione", "📋 Registro Operazioni", "✏️ Modifica Eseguiti", "🔄 Aggiorna Prezzi", "✏️ Prezzi Manuali"])

    with tab_new:
        st.markdown("Registra una nuova operazione del fondo.")

        # Pre-fetch live FX rates for display
        from data_fetcher import get_fx_rate
        _fx_cache = {}
        def _get_live_fx(ccy):
            if ccy == "EUR":
                return 1.0
            if ccy not in _fx_cache:
                _fx_cache[ccy] = get_fx_rate(ccy, "EUR")
            return _fx_cache[ccy]

        with st.form("new_transaction", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                tx_type = st.selectbox("Tipo Operazione",
                                       ["BUY", "SELL", "DIVIDEND", "DEPOSIT", "WITHDRAWAL"])
                tx_date = st.date_input("Data", value=pd.Timestamp.today())
                tx_isin = st.text_input("ISIN", placeholder="es. US0378331005")
                tx_name = st.text_input("Nome Strumento", placeholder="es. Apple Inc.")
                tx_ticker = st.text_input("Ticker Yahoo Finance", placeholder="es. AAPL, ENEL.MI, PRY.MI",
                                           help="Per aggiornamento prezzi automatico. Lascia vuoto se non presente su Yahoo Finance.")

            with c2:
                tx_qty = st.number_input("Quantità / Importo", min_value=0.0, step=1.0,
                                          help="Per BUY/SELL: numero di titoli. Per DEPOSIT/WITHDRAWAL: importo €.")
                tx_price = st.number_input("Prezzo Unitario", min_value=0.0, step=0.01,
                                           help="Per DEPOSIT/WITHDRAWAL: lascia 1.0")
                tx_currency = st.selectbox("Valuta", ["EUR", "USD", "GBP", "CHF", "JPY", "AUD", "CAD", "NOK", "SEK", "DKK", "HKD", "SGD", "NZD", "BRL", "ZAR", "PLN", "CZK", "TRY"])
                tx_fx_auto = st.checkbox("Cambio FX automatico", value=True,
                                          help="Scarica il tasso di cambio corrente da Yahoo Finance")
                tx_fx = st.number_input("Cambio FX manuale (override)", min_value=0.0, value=0.0, step=0.001,
                                         help="Lascia 0 per usare il cambio automatico. Compila solo se vuoi forzare un tasso specifico.")

            c3, c4 = st.columns(2)
            with c3:
                mc_list = ["Equity", "Fixed Income", "Alternative", ""]
                tx_macro = st.selectbox("Macro Classe", mc_list, index=0)
                tx_sector = st.text_input("Settore", placeholder="es. Technology")
            with c4:
                at_list = ["Stock", "ETF", "Bond", "Crypto", "Commodity", "Fund"]
                tx_asset_sub = st.selectbox("Tipo Asset", at_list, index=0)
                tx_fees = st.number_input("Commissioni €", min_value=0.0, value=0.0, step=0.5)
                tx_notes = st.text_input("Note", placeholder="Opzionale")

            submitted = st.form_submit_button("✅ Registra Operazione", use_container_width=True)

            if submitted:
                if tx_type in ["BUY", "SELL"] and (not tx_isin or tx_qty <= 0 or tx_price <= 0):
                    st.error("Per BUY/SELL servono: ISIN, quantità > 0, prezzo > 0")
                elif tx_type in ["DEPOSIT", "WITHDRAWAL"] and tx_qty <= 0:
                    st.error("Inserisci un importo > 0")
                else:
                    # Resolve FX rate: manual override > auto-fetch > 1.0
                    if tx_fx > 0:
                        resolved_fx = tx_fx
                    elif tx_fx_auto and tx_currency != "EUR":
                        resolved_fx = _get_live_fx(tx_currency)
                    else:
                        resolved_fx = 1.0

                    add_transaction(
                        date_str=str(tx_date),
                        transaction_type=tx_type,
                        isin=tx_isin,
                        name=tx_name,
                        macro_class=tx_macro,
                        quantity=tx_qty,
                        price=tx_price if tx_price > 0 else 1.0,
                        currency=tx_currency,
                        fx_rate=resolved_fx,
                        fees=tx_fees,
                        notes=tx_notes,
                        sector=tx_sector,
                        asset_sub_type=tx_asset_sub,
                    )
                    # Auto-register ISIN → Ticker in isin_map
                    if tx_type in ["BUY", "SELL"] and tx_isin:
                        current_map = get_isin_map()
                        isin_key = tx_isin.strip()
                        if isin_key not in current_map or not current_map.get(isin_key):
                            ticker_val = tx_ticker.strip() if tx_ticker and tx_ticker.strip() else None
                            current_map[isin_key] = ticker_val
                            save_isin_map(current_map)
                    fx_msg = f" (FX: {resolved_fx:.4f} {tx_currency}/EUR)" if tx_currency != "EUR" else ""
                    st.success(f"✅ Operazione registrata: {tx_type} {tx_name or 'Cash'}{fx_msg}")
                    if tx_type == "BUY" and tx_isin and not tx_ticker.strip():
                        st.warning(f"⚠️ Non hai inserito il ticker per **{tx_isin}**. "
                                   f"I prezzi non si aggiorneranno automaticamente. "
                                   f"Puoi aggiungerlo dopo in **Gestione Info Strumenti → Mapping ISIN** "
                                   f"oppure aggiornare il prezzo manualmente nel tab **Prezzi Manuali**.")
                    st.cache_data.clear()
                    st.rerun()

        # Quick actions
        st.divider()
        st.markdown("**Azioni Rapide**")
        qc1, qc2 = st.columns(2)
        with qc1:
            if has_data:
                st.markdown("**Chiudi posizione** (vendi tutto)")
                close_isin = st.selectbox(
                    "Seleziona posizione da chiudere",
                    positions["isin"].tolist(),
                    format_func=lambda x: f"{positions[positions['isin']==x]['name'].iloc[0]} ({x})" if not positions[positions['isin']==x].empty else x,
                    key="close_pos_select"
                )
                close_price = st.number_input("Prezzo di vendita", min_value=0.0, step=0.01, key="close_price")
                if st.button("🔴 Chiudi Posizione", key="close_btn"):
                    if close_isin and close_price > 0:
                        row = positions[positions["isin"] == close_isin].iloc[0]
                        add_transaction(
                            date_str=str(pd.Timestamp.today().date()),
                            transaction_type="SELL",
                            isin=close_isin,
                            name=row["name"],
                            macro_class=row.get("macro_class", ""),
                            quantity=float(row["quantity"]),
                            price=close_price,
                            currency=row.get("currency", "EUR"),
                            sector=row.get("sector", ""),
                            asset_sub_type=row.get("asset_sub_type", "Stock"),
                        )
                        st.success(f"✅ Posizione chiusa: {row['name']}")
                        st.cache_data.clear()
                        st.rerun()

    with tab_view:
        if not transactions.empty:
            # Filter out migration rows for display by default
            show_migration = st.checkbox("Mostra operazioni iniziali (migrazione)", value=False)
            display_tx = transactions.copy()
            if not show_migration and "notes" in display_tx.columns:
                display_tx = display_tx[display_tx["notes"] != "migration_initial"]

            st.markdown(f"**{len(display_tx)} operazioni**")
            show_cols = ["date", "transaction_type", "name", "isin", "quantity", "price", "currency", "fees", "notes"]
            show_cols = [c for c in show_cols if c in display_tx.columns]
            st.dataframe(
                display_tx[show_cols].sort_values("date", ascending=False),
                use_container_width=True, hide_index=True, height=500)

            csv_buf = io.StringIO()
            display_tx.to_csv(csv_buf, index=False)
            st.download_button("📥 Esporta CSV", csv_buf.getvalue(), "transazioni_fondo.csv", "text/csv")
        else:
            st.info("Nessuna transazione registrata. Usa il tab **Nuova Operazione** per iniziare.")

    with tab_edit:
        st.markdown("Seleziona un'operazione eseguita per modificarne i dettagli.")
        if not transactions.empty:
            # Build display for selection
            edit_tx = transactions.copy()
            edit_tx["_idx"] = edit_tx.index
            edit_tx["_label"] = edit_tx.apply(
                lambda r: f"#{r['_idx']} | {str(r['date'])[:10]} | {r['transaction_type']} | {r.get('name', '')} | {r.get('isin', '')} | Qty: {r['quantity']} | Px: {r['price']}", axis=1
            )

            selected_label = st.selectbox(
                "Seleziona operazione da modificare",
                edit_tx["_label"].tolist()[::-1],  # Most recent first
                key="edit_tx_select"
            )

            if selected_label:
                sel_idx = int(selected_label.split("|")[0].replace("#", "").strip())
                row = transactions.loc[sel_idx]
                # Chiave dinamica basata sull'indice: forza il refresh dei widget
                k = f"_{sel_idx}"

                st.divider()
                st.markdown(f"**Modifica operazione #{sel_idx}** — *{row.get('name', '')}*")

                # Riepilogo attuale
                with st.expander("📋 Valori attuali", expanded=False):
                    sum_cols = st.columns(4)
                    sum_cols[0].markdown(f"**Data:** {str(row['date'])[:10]}")
                    sum_cols[1].markdown(f"**Tipo:** {row['transaction_type']}")
                    sum_cols[2].markdown(f"**Quantità:** {row['quantity']}")
                    sum_cols[3].markdown(f"**Prezzo:** {row['price']}")

                tx_types = ["BUY", "SELL", "DIVIDEND", "DEPOSIT", "WITHDRAWAL", "FEE"]
                currencies = ["EUR", "USD", "GBP", "CHF", "JPY", "AUD", "CAD", "NOK", "SEK", "DKK", "HKD", "SGD", "NZD", "BRL", "ZAR", "PLN", "CZK", "TRY"]

                with st.form(f"edit_transaction_form{k}", clear_on_submit=False):
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        edit_date = st.date_input("Data", value=pd.to_datetime(row["date"]).date(), key=f"edit_date{k}")
                        edit_type = st.selectbox(
                            "Tipo Operazione", tx_types,
                            index=tx_types.index(row["transaction_type"]) if row["transaction_type"] in tx_types else 0,
                            key=f"edit_type{k}"
                        )
                        edit_name = st.text_input("Nome Strumento", value=str(row.get("name", "")), key=f"edit_name{k}")
                        edit_isin = st.text_input("ISIN", value=str(row.get("isin", "")), key=f"edit_isin{k}")
                        edit_macro = st.text_input("Macro Classe", value=str(row.get("macro_class", "")), key=f"edit_macro{k}")
                        edit_sector = st.text_input("Settore", value=str(row.get("sector", "")), key=f"edit_sector{k}")

                    with ec2:
                        edit_qty = st.number_input("Quantità", value=float(row["quantity"]), step=1.0, key=f"edit_qty{k}")
                        edit_price = st.number_input("Prezzo", value=float(row["price"]), step=0.01, format="%.4f", key=f"edit_price{k}")
                        edit_currency = st.selectbox(
                            "Valuta", currencies,
                            index=currencies.index(row.get("currency", "EUR")) if row.get("currency", "EUR") in currencies else 0,
                            key=f"edit_currency{k}"
                        )
                        edit_fx = st.number_input("FX Rate", value=float(row.get("fx_rate", 1.0)), step=0.0001, format="%.4f", key=f"edit_fx{k}")
                        edit_fees = st.number_input("Commissioni", value=float(row.get("fees", 0.0)), step=0.01, key=f"edit_fees{k}")
                        edit_notes = st.text_input("Note", value=str(row.get("notes", "")), key=f"edit_notes{k}")

                    submit_col, delete_col = st.columns([3, 1])
                    with submit_col:
                        submitted = st.form_submit_button("💾 Salva Modifiche", use_container_width=True)
                    with delete_col:
                        deleted = st.form_submit_button("🗑️ Elimina", use_container_width=True)

                if submitted:
                    updates = {
                        "date": str(edit_date),
                        "transaction_type": edit_type,
                        "name": edit_name.strip(),
                        "isin": edit_isin.strip(),
                        "macro_class": edit_macro.strip(),
                        "sector": edit_sector.strip(),
                        "quantity": float(edit_qty),
                        "price": float(edit_price),
                        "currency": edit_currency,
                        "fx_rate": float(edit_fx),
                        "fees": float(edit_fees),
                        "notes": edit_notes.strip(),
                    }
                    update_transaction(sel_idx, updates)
                    st.success(f"✅ Operazione #{sel_idx} aggiornata!")
                    st.cache_data.clear()
                    st.rerun()

                if deleted:
                    delete_transaction(sel_idx)
                    st.success(f"🗑️ Operazione #{sel_idx} eliminata!")
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.info("Nessuna operazione da modificare.")

    with tab_prices:
        st.markdown("Aggiorna i prezzi di mercato e ricalcola NAV, P&L e pesi.")
        st.info("**Nota:** I prezzi vengono recuperati live da Yahoo Finance tramite il mapping ISIN → Ticker.")

        if st.button("🔄 Aggiorna Prezzi Live", use_container_width=True):
            with st.spinner("Recuperando prezzi da Yahoo Finance..."):
                fresh_positions = load_positions()
                if not fresh_positions.empty:
                    updated = update_position_prices(fresh_positions, get_isin_map())
                    save_positions(updated)

                    # Recalculate cash and NAV
                    cash = compute_cash_from_transactions()
                    nav = calculate_nav(updated, cash)
                    update_fund_info(nav, len(updated))

                    # Save NAV snapshot
                    snapshot_nav(nav)

                    st.success(f"✅ Prezzi aggiornati! NAV: {fmt_eur_full(nav)}")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("Nessuna posizione da aggiornare.")

        if has_data:
            st.divider()
            st.markdown("**Stato Prezzi Correnti**")
            price_status = positions[["name", "isin", "current_price", "avg_cost", "pnl_pct"]].copy()
            price_status["mapped"] = price_status["isin"].apply(lambda x: "✅" if isin_map.get(x) else "❌")
            price_status["ticker"] = price_status["isin"].apply(lambda x: isin_map.get(x, "—"))
            price_status["pnl_pct_d"] = (price_status["pnl_pct"] * 100).round(2)
            show_price = price_status[["name", "isin", "ticker", "mapped", "avg_cost", "current_price", "pnl_pct_d"]].copy()
            show_price.columns = ["Nome", "ISIN", "Ticker", "Mappato", "Prezzo Carico", "Prezzo Attuale", "P&L %"]
            show_price = format_table_numbers(show_price, price_cols=["Prezzo Carico", "Prezzo Attuale"])
            st.dataframe(show_price, use_container_width=True, hide_index=True)

    with tab_manual:
        st.markdown("Inserisci manualmente i prezzi per strumenti **non mappati** su Yahoo Finance "
                     "(es. obbligazioni, fondi, strumenti illiquidi).")

        if has_data:
            # Show only unmapped or all positions
            unmapped_pos = positions[positions["isin"].apply(lambda x: not isin_map.get(x))].copy()
            if unmapped_pos.empty:
                st.success("✅ Tutti gli strumenti sono mappati su Yahoo Finance.")
                st.caption("Se vuoi comunque aggiornare un prezzo manualmente, usa la tabella sotto.")
                target_positions = positions.copy()
            else:
                st.warning(f"⚠️ {len(unmapped_pos)} strumenti senza ticker Yahoo Finance:")
                for _, row in unmapped_pos.iterrows():
                    st.caption(f"  ❌ {row['name']} ({row['isin']}) — Prezzo attuale: {row['current_price']:.4f}")
                target_positions = unmapped_pos.copy()

            st.divider()
            st.markdown("**Aggiorna Prezzo Manuale**")

            with st.form("manual_price_form", clear_on_submit=True):
                mp_isin = st.selectbox(
                    "Seleziona strumento",
                    target_positions["isin"].tolist(),
                    format_func=lambda x: f"{positions[positions['isin']==x]['name'].iloc[0]} ({x})" if not positions[positions['isin']==x].empty else x)
                mp_price = st.number_input("Nuovo Prezzo", min_value=0.0, step=0.01,
                                            help="Inserisci il prezzo corrente dello strumento")

                if st.form_submit_button("💾 Aggiorna Prezzo", use_container_width=True):
                    if mp_isin and mp_price > 0:
                        fresh_pos = load_positions()
                        idx = fresh_pos[fresh_pos["isin"] == mp_isin].index
                        if not idx.empty:
                            old_price = fresh_pos.loc[idx[0], "current_price"]
                            fresh_pos.loc[idx, "current_price"] = mp_price
                            # Recalculate current_value and P&L
                            for i in idx:
                                qty = fresh_pos.loc[i, "quantity"]
                                fx = fresh_pos.loc[i, "fx_rate"] if "fx_rate" in fresh_pos.columns else 1.0
                                if pd.isna(fx) or fx == 0:
                                    fx = 1.0
                                fresh_pos.loc[i, "current_value"] = qty * mp_price * fx
                                avg_cost = fresh_pos.loc[i, "avg_cost"]
                                invested = fresh_pos.loc[i, "invested_capital"]
                                fresh_pos.loc[i, "pnl"] = fresh_pos.loc[i, "current_value"] - invested
                                if invested > 0:
                                    fresh_pos.loc[i, "pnl_pct"] = fresh_pos.loc[i, "pnl"] / invested
                            save_positions(fresh_pos)

                            # Recalculate NAV and mark manual update timestamp
                            cash = compute_cash_from_transactions()
                            nav = calculate_nav(fresh_pos, cash)
                            info = update_fund_info(nav, len(fresh_pos))
                            from datetime import datetime as _dt
                            info["last_manual_update"] = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                            save_fund_info(info)

                            name = positions[positions["isin"] == mp_isin]["name"].iloc[0]
                            st.success(f"✅ Prezzo aggiornato: **{name}** — {old_price:.4f} → {mp_price:.4f}")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error("Strumento non trovato nelle posizioni.")
                    else:
                        st.error("Inserisci un prezzo > 0")

            # Bulk manual price update
            st.divider()
            st.markdown("**Aggiornamento Multiplo (CSV)**")
            st.caption("Carica un CSV con colonne `isin` e `price` per aggiornare più prezzi contemporaneamente.")
            uploaded_csv = st.file_uploader("Carica CSV prezzi", type=["csv"], key="manual_csv")
            if uploaded_csv:
                try:
                    price_csv = pd.read_csv(uploaded_csv)
                    if "isin" in price_csv.columns and "price" in price_csv.columns:
                        fresh_pos = load_positions()
                        updated_count = 0
                        for _, row in price_csv.iterrows():
                            idx = fresh_pos[fresh_pos["isin"] == row["isin"]].index
                            if not idx.empty and row["price"] > 0:
                                fresh_pos.loc[idx, "current_price"] = row["price"]
                                for i in idx:
                                    qty = fresh_pos.loc[i, "quantity"]
                                    fx = fresh_pos.loc[i, "fx_rate"] if "fx_rate" in fresh_pos.columns else 1.0
                                    if pd.isna(fx) or fx == 0:
                                        fx = 1.0
                                    fresh_pos.loc[i, "current_value"] = qty * row["price"] * fx
                                    invested = fresh_pos.loc[i, "invested_capital"]
                                    fresh_pos.loc[i, "pnl"] = fresh_pos.loc[i, "current_value"] - invested
                                    if invested > 0:
                                        fresh_pos.loc[i, "pnl_pct"] = fresh_pos.loc[i, "pnl"] / invested
                                updated_count += 1
                        save_positions(fresh_pos)
                        cash = compute_cash_from_transactions()
                        nav = calculate_nav(fresh_pos, cash)
                        info = update_fund_info(nav, len(fresh_pos))
                        from datetime import datetime as _dt
                        info["last_manual_update"] = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_fund_info(info)
                        st.success(f"✅ Aggiornati {updated_count} prezzi. NAV: {fmt_eur_full(nav)}")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("Il CSV deve contenere le colonne `isin` e `price`.")
                except Exception as e:
                    st.error(f"Errore: {e}")
        else:
            st.info("Nessuna posizione. Aggiungi prima delle operazioni.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GESTIONE INFO STRUMENTI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Gestione Info Strumenti":
    st.markdown('<div class="section-header">Gestione Info Strumenti</div>', unsafe_allow_html=True)

    # Password gate for management pages
    _mgmt_pw = st.text_input("🔒 Password richiesta", type="password", key="pw_gestione_info")
    if _mgmt_pw != "Maradona1":
        if _mgmt_pw:
            st.warning("Password errata")
        st.stop()

    st.markdown("Modifica manualmente settore, paese e altre info per strumenti non coperti da Yahoo Finance.")

    tab_edit, tab_isin = st.tabs(["✏️ Modifica Info", "🔗 Mapping ISIN → Ticker"])

    with tab_edit:
        current_overrides = get_overrides()
        if has_data:
            selected_isin = st.selectbox(
                "Seleziona strumento",
                positions["isin"].tolist(),
                format_func=lambda x: f"{x} — {positions[positions['isin']==x]['name'].iloc[0] if not positions[positions['isin']==x].empty else x}")

            if selected_isin:
                existing = current_overrides.get(selected_isin, {})
                if existing and selected_isin != "_comment":
                    st.caption("Override correnti: " + ", ".join(f"{k}={v}" for k, v in existing.items()))

                with st.form(f"edit_{selected_isin}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        new_name = st.text_input("Nome", existing.get("name", ""))
                        new_sector = st.text_input("Settore", existing.get("sector", ""))
                        new_industry = st.text_input("Industry", existing.get("industry", ""))
                        new_country = st.text_input("Paese", existing.get("country", ""))
                    with c2:
                        at_list = ["Stock", "ETF", "Bond", "Crypto", "Commodity", "Fund"]
                        new_asset_type = st.selectbox("Tipo Asset", at_list,
                                                       index=at_list.index(existing.get("asset_type", "ETF")) if existing.get("asset_type") in at_list else 1)
                        mc_list = ["Equity", "Fixed Income", "Alternative"]
                        new_macro = st.selectbox("Macro Classe", mc_list,
                                                  index=mc_list.index(existing.get("macro_class", "Equity")) if existing.get("macro_class") in mc_list else 0)
                    if st.form_submit_button("💾 Salva", use_container_width=True):
                        update = {}
                        if new_name: update["name"] = new_name
                        if new_sector: update["sector"] = new_sector
                        if new_industry: update["industry"] = new_industry
                        if new_country: update["country"] = new_country
                        if new_asset_type: update["asset_type"] = new_asset_type
                        if new_macro: update["macro_class"] = new_macro
                        current_overrides[selected_isin] = update
                        save_overrides(current_overrides)
                        st.success(f"✅ Salvato per {selected_isin}")
                        st.cache_data.clear()
                        st.rerun()

    with tab_isin:
        st.markdown("**Mapping ISIN → Ticker Yahoo Finance**")
        st.info("Quando aggiungi una nuova operazione BUY, l'ISIN viene registrato automaticamente. "
                "Qui devi associare il **ticker Yahoo Finance** (es. AAPL, PRY.MI, ENEL.MI) "
                "per abilitare l'aggiornamento automatico dei prezzi. "
                "Per strumenti non presenti su Yahoo Finance (es. obbligazioni), "
                "usa il tab **Prezzi Manuali** in Operazioni & Import.")
        current_map = get_isin_map()
        unmapped = {k: v for k, v in current_map.items() if v is None and k != "_comment"}
        if unmapped:
            st.warning(f"⚠️ {len(unmapped)} strumenti senza mapping:")
            for isin in unmapped:
                name = ""
                if has_data:
                    match = positions[positions["isin"] == isin]
                    if not match.empty:
                        name = match.iloc[0].get("name", "")
                st.caption(f"  {isin} — {name}")

        with st.form("add_isin"):
            new_isin = st.text_input("ISIN")
            new_ticker = st.text_input("Ticker (es. AAPL, PRY.MI)")
            if st.form_submit_button("Aggiungi"):
                if new_isin and new_ticker:
                    current_map[new_isin.strip()] = new_ticker.strip()
                    save_isin_map(current_map)
                    st.success(f"✅ {new_isin} → {new_ticker}")
                    st.cache_data.clear()
                    st.rerun()

        map_df = pd.DataFrame([{"ISIN": k, "Ticker": v or "❌ Non mappato"} for k, v in current_map.items() if k != "_comment"])
        st.dataframe(map_df, use_container_width=True, hide_index=True, height=400)
