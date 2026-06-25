"""
Multi-benchmark comparison utilities for the SFC fund.

Defines a small universe of multi-asset benchmarks (the Vanguard LifeStrategy
ladder: 20/40/60/80 equity), downloads their price histories via the Yahoo
*bulk* endpoint (the same one used for live position prices — `.info` is
throttled on Streamlit Cloud datacenter IPs), aligns them to the fund's daily
NAV dates, and computes rebased series + period returns for a fund-vs-benchmark
comparison.

VNGA60 (V60A.DE) is the fund's official benchmark; VNGA40 is its mirror image
(40/60 vs 60/40). VNGA20/VNGA80 bracket the ladder so the fund can be compared
against more defensive and more aggressive multi-asset allocations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Benchmark KEY -> config. Order matters (used for legend / table ordering).
BENCHMARKS: dict[str, dict] = {
    "VNGA60": {
        "ticker": "V60A.DE", "label": "VNGA60 · 60/40", "equity": 60, "bond": 40,
        "color": "#22c55e", "primary": True,
        "desc": "Vanguard LifeStrategy 60% Equity — benchmark ufficiale del fondo",
    },
    "VNGA40": {
        "ticker": "V40A.DE", "label": "VNGA40 · 40/60", "equity": 40, "bond": 60,
        "color": "#f59e0b", "primary": False,
        "desc": "Vanguard LifeStrategy 40% Equity — il rovescio del nostro 60/40",
    },
    "VNGA20": {
        "ticker": "V20A.DE", "label": "VNGA20 · 20/80", "equity": 20, "bond": 80,
        "color": "#38bdf8", "primary": False,
        "desc": "Vanguard LifeStrategy 20% Equity — profilo difensivo",
    },
    "VNGA80": {
        "ticker": "V80A.DE", "label": "VNGA80 · 80/20", "equity": 80, "bond": 20,
        "color": "#ef4444", "primary": False,
        "desc": "Vanguard LifeStrategy 80% Equity — profilo aggressivo",
    },
}

PRIMARY_BENCHMARK = "VNGA60"
FUND_COLOR = "#6366f1"
FUND_LABEL = "Fondo SFC"

PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "Dall'Inizio"]


def benchmark_keys() -> list[str]:
    return list(BENCHMARKS.keys())


def benchmark_tickers() -> list[str]:
    return [cfg["ticker"] for cfg in BENCHMARKS.values()]


def _ticker_to_key() -> dict[str, str]:
    return {cfg["ticker"]: key for key, cfg in BENCHMARKS.items()}


def download_benchmark_prices(start, end=None) -> pd.DataFrame:
    """Daily close prices for all benchmarks via the Yahoo bulk endpoint.

    Returns a DataFrame indexed by (naive) date with one column per benchmark
    KEY (VNGA60, VNGA40, ...). Empty DataFrame on failure — callers degrade
    gracefully.
    """
    import yfinance as yf

    tickers = benchmark_tickers()
    start_buffer = pd.Timestamp(start) - pd.Timedelta(days=7)
    try:
        raw = yf.download(
            tickers, start=start_buffer, end=end,
            auto_adjust=False, progress=False,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"].copy()
    else:
        # Single-ticker fallback: yfinance drops the MultiIndex.
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        close = raw[[col]].copy()
        close.columns = [tickers[0]]

    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.rename(columns=_ticker_to_key())
    cols = [k for k in BENCHMARKS if k in close.columns]
    return close[cols].sort_index()


def align_to_dates(prices: pd.DataFrame, dates) -> pd.DataFrame:
    """Reindex benchmark prices onto the fund NAV dates (forward-filled)."""
    if prices is None or prices.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(dates)))
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize()
    p = prices.copy()
    p.index = pd.to_datetime(p.index).tz_localize(None).normalize()
    p = p[~p.index.duplicated(keep="last")].sort_index()
    union = p.index.union(idx)
    return p.reindex(union).ffill().reindex(idx)


def period_start_date(dates, period: str, inception) -> pd.Timestamp:
    """Anchor periods to the LAST available data point (not wall-clock today)
    so the comparison is consistent with the stored NAV series."""
    anchor = pd.Timestamp(pd.to_datetime(dates).max())
    if period == "YTD":
        return pd.Timestamp(anchor.year, 1, 1)
    if period == "1M":
        return anchor - pd.DateOffset(months=1)
    if period == "3M":
        return anchor - pd.DateOffset(months=3)
    if period == "6M":
        return anchor - pd.DateOffset(months=6)
    if period == "1Y":
        return anchor - pd.DateOffset(years=1)
    return pd.Timestamp(pd.to_datetime(inception))


def _first_on_or_after(series: pd.Series, start) -> float | None:
    s = series.dropna()
    s = s[s.index >= pd.Timestamp(start)]
    return float(s.iloc[0]) if not s.empty else None


def build_rebased_frame(
    nav_df: pd.DataFrame,
    bench_aligned: pd.DataFrame,
    period: str,
    inception,
    initial_nav: float,
    selected_keys: list[str],
) -> pd.DataFrame:
    """Return a tidy frame indexed by date with a `Fondo` column plus one column
    per selected benchmark key, each rebased to 100 at the period start.

    The fund uses `initial_nav` as base for the full-history view (to stay
    consistent with the headline since-inception figure); otherwise the first
    NAV in the window. Benchmarks rebase on their first value in the window.
    """
    nav = pd.Series(
        pd.to_numeric(nav_df["nav"], errors="coerce").values,
        index=pd.DatetimeIndex(pd.to_datetime(nav_df["date"])).tz_localize(None).normalize(),
    ).dropna()

    start = period_start_date(nav.index, period, inception)
    nav_win = nav[nav.index >= start]
    if len(nav_win) < 2:
        return pd.DataFrame()

    out = pd.DataFrame(index=nav_win.index)
    fund_base = float(initial_nav) if period == "Dall'Inizio" else float(nav_win.iloc[0])
    if fund_base <= 0:
        fund_base = float(nav_win.iloc[0])
    out[FUND_LABEL] = nav_win / fund_base * 100.0

    for key in selected_keys:
        if key not in bench_aligned.columns:
            continue
        b = bench_aligned[key].copy()
        b.index = pd.to_datetime(b.index).tz_localize(None).normalize()
        b_win = b[b.index >= start].dropna()
        base = _first_on_or_after(b, start)
        if base is None or base <= 0 or b_win.empty:
            continue
        out[key] = (b_win / base * 100.0).reindex(out.index).ffill()

    return out


def compute_return_table(
    nav_df: pd.DataFrame,
    bench_aligned: pd.DataFrame,
    inception,
    initial_nav: float,
    selected_keys: list[str],
) -> pd.DataFrame:
    """Return a table: rows = [Fondo, <benchmarks>], cols = PERIODS, values =
    total return % over each period."""
    nav = pd.Series(
        pd.to_numeric(nav_df["nav"], errors="coerce").values,
        index=pd.DatetimeIndex(pd.to_datetime(nav_df["date"])).tz_localize(None).normalize(),
    ).dropna()

    rows: dict[str, dict[str, float]] = {}
    fund_row: dict[str, float] = {}
    bench_rows: dict[str, dict[str, float]] = {k: {} for k in selected_keys}

    for period in PERIODS:
        start = period_start_date(nav.index, period, inception)
        nav_win = nav[nav.index >= start]
        if len(nav_win) >= 2:
            base = float(initial_nav) if period == "Dall'Inizio" else float(nav_win.iloc[0])
            if base <= 0:
                base = float(nav_win.iloc[0])
            fund_row[period] = (float(nav_win.iloc[-1]) / base - 1.0) * 100.0
        else:
            fund_row[period] = np.nan

        for key in selected_keys:
            if key not in bench_aligned.columns:
                bench_rows[key][period] = np.nan
                continue
            b = bench_aligned[key].dropna()
            b.index = pd.to_datetime(b.index).tz_localize(None).normalize()
            b_win = b[b.index >= start]
            if len(b_win) >= 2 and float(b_win.iloc[0]) > 0:
                bench_rows[key][period] = (float(b_win.iloc[-1]) / float(b_win.iloc[0]) - 1.0) * 100.0
            else:
                bench_rows[key][period] = np.nan

    rows[FUND_LABEL] = fund_row
    for key in selected_keys:
        rows[BENCHMARKS[key]["label"]] = bench_rows[key]

    table = pd.DataFrame(rows).T[PERIODS]
    return table


def benchmark_returns_series(bench_aligned: pd.DataFrame, key: str) -> pd.Series:
    """Daily simple returns for one benchmark, indexed by date."""
    if key not in bench_aligned.columns:
        return pd.Series(dtype=float)
    s = bench_aligned[key].dropna()
    return s.pct_change().dropna()


def macro_composition(equity_pct: float, bond_pct: float) -> pd.DataFrame:
    """Exact macro-asset composition for a LifeStrategy benchmark."""
    return pd.DataFrame({
        "macro_class": ["Equity", "Fixed Income"],
        "weight_pct": [float(equity_pct), float(bond_pct)],
    })
