"""
Benchmark underlying contribution utilities.

VNGA60 is held as a fund of ETF building blocks. This module converts those
building blocks into period performance and contribution in percentage points.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


EXCHANGE_SUFFIXES = {
    "LON": ".L",
    "LSE": ".L",
    "ETR": ".DE",
    "XETRA": ".DE",
    "BIT": ".MI",
    "MIL": ".MI",
}


def benchmark_symbol_to_yahoo(symbol: str) -> str:
    """Map a benchmark holding symbol such as ``LON: VHVG`` to Yahoo syntax."""
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    if ":" not in text:
        return text

    exchange, ticker = [part.strip() for part in text.split(":", 1)]
    if "." in ticker:
        return ticker
    return f"{ticker}{EXCHANGE_SUFFIXES.get(exchange, '')}"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _value_on_or_before(series: pd.Series | None, target_date):
    if series is None or len(series) == 0:
        return np.nan

    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan

    idx = pd.to_datetime(s.index, errors="coerce")
    normalized = pd.Series(pd.to_numeric(s.values, errors="coerce"), index=idx).dropna()
    normalized = normalized[normalized.index.notna()].sort_index()
    if normalized.empty:
        return np.nan

    if getattr(normalized.index, "tz", None) is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()

    target = pd.to_datetime(target_date).normalize()
    eligible = normalized[normalized.index <= target]
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[-1])


def _empty_result(benchmark_return_pct: float) -> dict:
    detail_cols = [
        "symbol", "ticker", "yahoo_ticker", "name", "weight_pct", "period_return_pct",
        "contribution_pp", "macro_class", "region", "data_status",
    ]
    summary_cols = [
        "benchmark_contribution_pp", "fund_contribution_pp", "active_contribution_pp",
        "benchmark_weight_pct",
    ]
    return {
        "detail": pd.DataFrame(columns=detail_cols),
        "macro_summary": pd.DataFrame(columns=["macro_class", *summary_cols]),
        "region_summary": pd.DataFrame(columns=["region", *summary_cols]),
        "reconstructed_return_pct": 0.0,
        "residual_pp": _safe_float(benchmark_return_pct),
    }


def _summarize_contribution(detail: pd.DataFrame, group_col: str, fund_contributions: pd.DataFrame | None) -> pd.DataFrame:
    if detail is None or detail.empty or group_col not in detail.columns:
        benchmark = pd.DataFrame(columns=[group_col, "benchmark_contribution_pp", "benchmark_weight_pct"])
    else:
        benchmark = detail.copy()
        benchmark[group_col] = benchmark[group_col].fillna("N/A").replace("", "N/A")
        benchmark = benchmark.groupby(group_col, dropna=False).agg(
            benchmark_contribution_pp=("contribution_pp", "sum"),
            benchmark_weight_pct=("weight_pct", "sum"),
        ).reset_index()

    if fund_contributions is None or fund_contributions.empty or group_col not in fund_contributions.columns:
        fund = pd.DataFrame(columns=[group_col, "fund_contribution_pp"])
    else:
        fund = fund_contributions.copy()
        fund[group_col] = fund[group_col].fillna("N/A").replace("", "N/A")
        fund["contribution_pp"] = pd.to_numeric(fund.get("contribution_pp", 0), errors="coerce").fillna(0.0)
        fund = fund.groupby(group_col, dropna=False)["contribution_pp"].sum().reset_index(name="fund_contribution_pp")

    result = pd.merge(benchmark, fund, on=group_col, how="outer")
    result[group_col] = result[group_col].fillna("N/A").replace("", "N/A")
    for col in ["benchmark_contribution_pp", "fund_contribution_pp", "benchmark_weight_pct"]:
        if col not in result.columns:
            result[col] = 0.0
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    result["active_contribution_pp"] = result["fund_contribution_pp"] - result["benchmark_contribution_pp"]
    for col in ["benchmark_contribution_pp", "fund_contribution_pp", "active_contribution_pp", "benchmark_weight_pct"]:
        result[col] = result[col].round(4)
    return result.sort_values("active_contribution_pp", ascending=False).reset_index(drop=True)


def compute_benchmark_underlying_contributions(
    holdings: pd.DataFrame,
    price_data: Mapping[str, pd.Series],
    start_date,
    end_date,
    benchmark_return_pct: float,
    fund_contributions: pd.DataFrame | None = None,
) -> dict:
    """Compute VNGA60 underlying return and contribution over a selected period."""
    if holdings is None or holdings.empty:
        return _empty_result(benchmark_return_pct)

    price_data = dict(price_data or {})
    rows = []
    for _, row in holdings.copy().iterrows():
        symbol = str(row.get("symbol", "") or "")
        clean_ticker = str(row.get("ticker", "") or "").strip().upper()
        yahoo_ticker = benchmark_symbol_to_yahoo(symbol) or clean_ticker
        series = price_data.get(yahoo_ticker)
        if series is None and clean_ticker:
            series = price_data.get(clean_ticker)

        start_value = _value_on_or_before(series, start_date)
        end_value = _value_on_or_before(series, end_date)
        has_data = pd.notna(start_value) and pd.notna(end_value) and start_value > 0
        period_return_pct = (end_value / start_value - 1) * 100 if has_data else np.nan
        weight_pct = _safe_float(row.get("weight_pct", 0.0))
        contribution_pp = weight_pct * period_return_pct / 100 if has_data else 0.0

        rows.append({
            "symbol": symbol,
            "ticker": clean_ticker,
            "yahoo_ticker": yahoo_ticker,
            "name": row.get("name", clean_ticker or symbol),
            "weight_pct": weight_pct,
            "period_return_pct": round(period_return_pct, 4) if has_data else np.nan,
            "contribution_pp": round(contribution_pp, 4),
            "macro_class": row.get("macro_class", "N/A") or "N/A",
            "region": row.get("region", "N/A") or "N/A",
            "data_status": "OK" if has_data else "N/A",
        })

    detail = pd.DataFrame(rows)
    detail = detail.sort_values("contribution_pp", ascending=False).reset_index(drop=True)
    reconstructed = float(detail["contribution_pp"].sum()) if not detail.empty else 0.0
    benchmark_return_pct = _safe_float(benchmark_return_pct)

    return {
        "detail": detail,
        "macro_summary": _summarize_contribution(detail, "macro_class", fund_contributions),
        "region_summary": _summarize_contribution(detail, "region", fund_contributions),
        "reconstructed_return_pct": round(reconstructed, 4),
        "residual_pp": round(benchmark_return_pct - reconstructed, 4),
    }
