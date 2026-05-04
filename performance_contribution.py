"""
Performance contribution utilities.

The fund is tracked in EUR. Transaction FX rates follow the existing tracker
convention: for non-EUR transactions, local currency amounts are divided by
``fx_rate`` to obtain EUR amounts.
"""

from __future__ import annotations

from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd


METADATA_COLUMNS = ["name", "macro_class", "sector", "currency", "asset_sub_type", "country"]


def _to_timestamp(value) -> pd.Timestamp:
    return pd.to_datetime(value).normalize()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_isin(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _transaction_value_eur(row: pd.Series, dividend: bool = False) -> float:
    qty = _safe_float(row.get("quantity", 0))
    price = _safe_float(row.get("price", 0))
    currency = str(row.get("currency", "EUR") or "EUR").upper()
    fx = _safe_float(row.get("fx_rate", 1.0), 1.0)

    amount = qty * price
    if dividend and price <= 0:
        amount = qty
    if currency != "EUR" and fx > 0 and abs(fx - 1.0) > 1e-12:
        amount = amount / fx
    return amount


def _metadata_lookup(transactions: pd.DataFrame, positions: pd.DataFrame) -> dict[str, dict]:
    lookup: dict[str, dict] = {}

    if transactions is not None and not transactions.empty and "isin" in transactions.columns:
        tx = transactions.copy()
        if "date" in tx.columns:
            tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
            tx = tx.sort_values("date")
        for _, row in tx.iterrows():
            isin = _clean_isin(row.get("isin"))
            if not isin:
                continue
            item = lookup.setdefault(isin, {})
            for col in METADATA_COLUMNS:
                val = row.get(col)
                if val is not None and not pd.isna(val) and str(val).strip():
                    item[col] = val

    if positions is not None and not positions.empty and "isin" in positions.columns:
        for _, row in positions.iterrows():
            isin = _clean_isin(row.get("isin"))
            if not isin:
                continue
            item = lookup.setdefault(isin, {})
            for col in METADATA_COLUMNS:
                val = row.get(col)
                if val is not None and not pd.isna(val) and str(val).strip():
                    item[col] = val

    return lookup


def compute_period_contributions(
    transactions: pd.DataFrame,
    positions: pd.DataFrame,
    start_date,
    end_date,
    nav_start: float,
    nav_end: float,
    start_values: Mapping[str, float] | None = None,
    end_values: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compute security-level contribution over a period.

    Contribution in EUR is:

        end market value - start market value - buys + sells + dividends

    Purchases include fees, sales are net of fees, and dividends are added as
    cash income. The contribution in percentage points is measured against the
    starting NAV of the selected period.
    """
    start = _to_timestamp(start_date)
    end = _to_timestamp(end_date)
    start_values = dict(start_values or {})
    end_values = dict(end_values or {})
    nav_start = _safe_float(nav_start)
    nav_end = _safe_float(nav_end)

    tx = transactions.copy() if transactions is not None else pd.DataFrame()
    if tx.empty:
        period_tx = pd.DataFrame()
    else:
        tx["date"] = pd.to_datetime(tx.get("date"), errors="coerce").dt.normalize()
        tx["isin"] = tx.get("isin", "").apply(_clean_isin)
        period_tx = tx[(tx["date"] > start) & (tx["date"] <= end) & (tx["isin"] != "")].copy()

    metadata = _metadata_lookup(tx, positions if positions is not None else pd.DataFrame())

    isins = set(start_values.keys()) | set(end_values.keys())
    if not period_tx.empty:
        isins |= set(period_tx["isin"].dropna().astype(str))

    rows = []
    for isin in sorted(isins):
        item_tx = period_tx[period_tx["isin"] == isin] if not period_tx.empty else pd.DataFrame()
        buys = 0.0
        sells = 0.0
        dividends = 0.0

        for _, row in item_tx.iterrows():
            tx_type = str(row.get("transaction_type", "")).upper()
            fees = _safe_float(row.get("fees", 0))
            if tx_type == "BUY":
                buys += _transaction_value_eur(row) + fees
            elif tx_type == "SELL":
                sells += _transaction_value_eur(row) - fees
            elif tx_type == "DIVIDEND":
                dividends += _transaction_value_eur(row, dividend=True)

        start_value = _safe_float(start_values.get(isin, 0.0))
        end_value = _safe_float(end_values.get(isin, 0.0))
        contribution = end_value - start_value - buys + sells + dividends
        base_capital = start_value + buys
        period_return = contribution / base_capital * 100 if base_capital > 0 else np.nan

        meta = metadata.get(isin, {})
        rows.append({
            "isin": isin,
            "name": meta.get("name", isin),
            "macro_class": meta.get("macro_class", "N/A"),
            "sector": meta.get("sector", "N/A"),
            "currency": meta.get("currency", "EUR"),
            "asset_sub_type": meta.get("asset_sub_type", ""),
            "country": meta.get("country", ""),
            "start_value": start_value,
            "end_value": end_value,
            "buys_eur": buys,
            "sells_eur": sells,
            "dividends_eur": dividends,
            "net_flows_eur": buys - sells - dividends,
            "contribution_eur": contribution,
            "contribution_pp": contribution / nav_start * 100 if nav_start > 0 else 0.0,
            "period_return_pct": period_return if pd.notna(period_return) else np.nan,
            "start_weight_pct": start_value / nav_start * 100 if nav_start > 0 else 0.0,
            "end_weight_pct": end_value / nav_end * 100 if nav_end > 0 else 0.0,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "isin", "name", "macro_class", "sector", "currency", "asset_sub_type", "country",
            "start_value", "end_value", "buys_eur", "sells_eur", "dividends_eur",
            "net_flows_eur", "contribution_eur", "contribution_pp", "period_return_pct",
            "start_weight_pct", "end_weight_pct",
        ])

    result = pd.DataFrame(rows)
    return result.sort_values("contribution_eur", ascending=False).reset_index(drop=True)


def summarize_contributions(contributions: pd.DataFrame, group_col: str, nav_end: float) -> pd.DataFrame:
    """Aggregate period contribution by sector, macro class, currency, etc."""
    if contributions is None or contributions.empty or group_col not in contributions.columns:
        return pd.DataFrame()

    contributions = contributions.copy()
    defaults = {
        "isin": "",
        "start_value": 0.0,
        "end_value": 0.0,
        "buys_eur": 0.0,
        "sells_eur": 0.0,
        "dividends_eur": 0.0,
        "contribution_eur": 0.0,
        "contribution_pp": 0.0,
    }
    for col, default in defaults.items():
        if col not in contributions.columns:
            contributions[col] = default

    nav_end = _safe_float(nav_end)
    grouped = contributions.groupby(group_col, dropna=False).agg(
        positions=("isin", "nunique"),
        start_value=("start_value", "sum"),
        end_value=("end_value", "sum"),
        buys_eur=("buys_eur", "sum"),
        sells_eur=("sells_eur", "sum"),
        dividends_eur=("dividends_eur", "sum"),
        contribution_eur=("contribution_eur", "sum"),
        contribution_pp=("contribution_pp", "sum"),
    ).reset_index()
    grouped["end_weight_pct"] = np.where(
        nav_end > 0,
        grouped["end_value"] / nav_end * 100,
        0.0,
    )
    grouped["period_return_pct"] = np.where(
        (grouped["start_value"] + grouped["buys_eur"]) > 0,
        grouped["contribution_eur"] / (grouped["start_value"] + grouped["buys_eur"]) * 100,
        np.nan,
    )

    numeric_cols = [
        "start_value", "end_value", "buys_eur", "sells_eur", "dividends_eur",
        "contribution_eur", "contribution_pp", "end_weight_pct", "period_return_pct",
    ]
    for col in numeric_cols:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round(4)

    return grouped.sort_values("contribution_eur", ascending=False).reset_index(drop=True)


def period_bounds(nav_history: pd.DataFrame, period: str, end_date: date | str | None = None):
    """Return start/end NAV rows for a standard contribution period."""
    if nav_history is None or nav_history.empty:
        return None

    nav = nav_history.copy()
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = nav.dropna(subset=["date", "nav"]).sort_values("date")
    if nav.empty:
        return None

    end_ts = _to_timestamp(end_date) if end_date is not None else nav["date"].max()
    nav = nav[nav["date"] <= end_ts]
    if nav.empty:
        return None

    end_row = nav.iloc[-1]
    end_ts = end_row["date"]
    period_key = str(period).upper()
    if period_key == "1M":
        target = end_ts - pd.DateOffset(months=1)
    elif period_key == "3M":
        target = end_ts - pd.DateOffset(months=3)
    elif period_key == "6M":
        target = end_ts - pd.DateOffset(months=6)
    elif period_key == "1Y":
        target = end_ts - pd.DateOffset(years=1)
    elif period_key == "YTD":
        target = pd.Timestamp(year=end_ts.year, month=1, day=1)
    else:
        target = nav["date"].min()

    start_candidates = nav[nav["date"] <= target]
    start_row = start_candidates.iloc[-1] if not start_candidates.empty else nav.iloc[0]
    if start_row["date"] >= end_row["date"] and len(nav) > 1:
        start_row = nav.iloc[-2]

    return {
        "start_date": start_row["date"],
        "end_date": end_row["date"],
        "nav_start": float(start_row["nav"]),
        "nav_end": float(end_row["nav"]),
    }
