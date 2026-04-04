"""
Build daily NAV history from transaction history + historical market prices.
=============================================================================
Tracks positions and cash day-by-day from inception, fetches historical prices
from yfinance for mapped tickers, and applies FX rates for non-EUR positions.
Bonds without tickers use their last known price (invested value / qty).

Usage:
    python build_nav_history.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import yfinance as yf
import json
from bisect import bisect_right
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent / "data"

# Module-level constants for ticker/currency overrides
CRYPTO_PROXY = {
    "GB00BJYDH287.SG": {"proxy": "BTC-EUR", "scale": 0.00023982},
    "GB00BLD4ZM24.SG": {"proxy": "ETH-EUR", "scale": 0.03031801},
}
# Tickers whose yfinance currency differs from transaction currency
TICKER_CURRENCY = {
    "BTEC.L": "USD",
    "JD.L": "GBX",
    "EVO.ST": "SEK",
    "RTO1.F": "EUR",
    "CTEC.AS": "EUR",
}


def load_transactions():
    df = pd.read_csv(DATA_DIR / "fund_transactions.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_isin_map():
    with open(DATA_DIR / "isin_map.json") as f:
        return json.load(f)


def load_fund_info():
    with open(DATA_DIR / "fund_info.json") as f:
        return json.load(f)


def load_manual_bond_prices():
    """Load current manual prices for bonds from fund_positions.csv."""
    pos_file = DATA_DIR / "fund_positions.csv"
    if not pos_file.exists():
        return {}
    df = pd.read_csv(pos_file)
    bond_prices = {}
    for _, row in df.iterrows():
        isin = row.get("isin", "")
        price = float(row.get("current_price", 0) or 0)
        currency = row.get("currency", "EUR") or "EUR"
        if price > 0:
            bond_prices[isin] = {"price": price, "currency": currency}
    return bond_prices


def build_daily_positions_and_cash(transactions: pd.DataFrame):
    """
    For each transaction date, compute the running state of:
    - positions: dict of ISIN -> {qty, total_cost_eur, total_cost_local, currency}
    - cash: float
    Returns list of (date, positions_snapshot, cash) sorted by date.
    """
    positions = {}  # isin -> {qty, cost_eur, cost_local, currency}
    cash = 0.0
    events = []  # (date, positions_copy, cash)

    for _, row in transactions.iterrows():
        tx_date = row["date"]
        tx_type = row.get("transaction_type", "")
        isin = row.get("isin", "")
        qty = float(row.get("quantity", 0) or 0)
        price = float(row.get("price", 0) or 0)
        fees = float(row.get("fees", 0) or 0)
        currency = row.get("currency", "EUR") or "EUR"
        fx = float(row.get("fx_rate", 1.0) or 1.0)

        if tx_type == "DEPOSIT":
            cash += qty
        elif tx_type == "WITHDRAWAL":
            cash -= qty
        elif tx_type == "BUY" and isin:
            cost_local = qty * price
            cost_eur = cost_local
            if currency != "EUR" and fx > 0 and fx != 1.0:
                cost_eur = cost_local / fx
            cash -= (cost_eur + fees)

            if isin not in positions:
                positions[isin] = {"qty": 0, "cost_eur": 0, "cost_local": 0, "currency": currency}
            positions[isin]["qty"] += qty
            positions[isin]["cost_eur"] += cost_eur
            positions[isin]["cost_local"] += cost_local
            positions[isin]["currency"] = currency

        elif tx_type == "SELL" and isin:
            if isin in positions and positions[isin]["qty"] > 0:
                p = positions[isin]
                sell_qty = min(qty, p["qty"])
                avg_cost_eur = p["cost_eur"] / p["qty"] if p["qty"] > 0 else 0
                avg_cost_local = p["cost_local"] / p["qty"] if p["qty"] > 0 else 0

                proceeds_local = sell_qty * price
                proceeds_eur = proceeds_local
                if currency != "EUR" and fx > 0 and fx != 1.0:
                    proceeds_eur = proceeds_local / fx
                cash += (proceeds_eur - fees)

                p["cost_eur"] -= sell_qty * avg_cost_eur
                p["cost_local"] -= sell_qty * avg_cost_local
                p["qty"] -= sell_qty

                if p["qty"] < 1e-8:
                    del positions[isin]

        elif tx_type == "DIVIDEND":
            div = qty * price if price > 0 else qty
            if currency != "EUR" and fx > 0 and fx != 1.0:
                div = div / fx
            cash += div

        # Record state after this transaction
        pos_snapshot = {k: dict(v) for k, v in positions.items()}
        events.append((tx_date, pos_snapshot, cash))

    return events


def _extract_close(data, tickers):
    """Extract Close prices from yfinance download result."""
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0).unique()
        col_name = "Close" if "Close" in level0 else ("Price" if "Price" in level0 else None)
        if not col_name:
            return pd.DataFrame()
        close = data[col_name].copy()
    else:
        if "Close" in data.columns:
            close = data[["Close"]].copy()
            close.columns = [tickers[0]]
        else:
            return pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close


def download_prices(tickers: list, start_date: str):
    """Download historical close prices in batches to avoid data loss."""
    print(f"Downloading prices for {len(tickers)} tickers from {start_date}...")

    end_str = (pd.Timestamp.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    all_close = pd.DataFrame()
    batch_size = 10

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, start=start_date, end=end_str, auto_adjust=True, progress=True, threads=True)
            close = _extract_close(data, batch)
            if not close.empty:
                close = close.ffill()
                if all_close.empty:
                    all_close = close
                else:
                    all_close = all_close.join(close, how="outer")
        except Exception as e:
            print(f"Batch download failed for {batch}: {e}")

    if not all_close.empty:
        # Remove single-day spikes (>5x median in rolling window)
        for col in all_close.columns:
            series = all_close[col]
            median = series.rolling(10, min_periods=3, center=True).median()
            spike = (series > median * 5) | (series < median / 5)
            all_close.loc[spike, col] = np.nan
        all_close = all_close.ffill()

    return all_close


def download_fx_rates(currencies: set, start_date: str):
    """Download historical FX rates for currency pairs vs EUR."""
    fx_tickers = [f"{ccy}EUR=X" for ccy in currencies]
    if not fx_tickers:
        return {}

    print(f"Downloading FX rates for {currencies}...")
    end_str = (pd.Timestamp.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        data = yf.download(fx_tickers, start=start_date, end=end_str, auto_adjust=True, progress=True)
    except Exception:
        return {}

    if data.empty:
        return {}

    fx_data = {}
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0).unique()
        col_name = "Close" if "Close" in level0 else ("Price" if "Price" in level0 else None)
        if col_name:
            close = data[col_name]
            if isinstance(close, pd.Series):
                pair = fx_tickers[0]
                fx_data[pair] = close.ffill()
            else:
                for pair in fx_tickers:
                    if pair in close.columns:
                        fx_data[pair] = close[pair].ffill()
    elif "Close" in data.columns:
        pair = fx_tickers[0]
        fx_data[pair] = data["Close"].ffill()

    return fx_data


def build_daily_nav():
    """Main function: build daily NAV history from transactions."""
    transactions = load_transactions()
    isin_map = load_isin_map()
    fund_info = load_fund_info()

    if transactions.empty:
        print("No transactions found.")
        return

    inception = pd.to_datetime(fund_info.get("inception_date", "2023-10-01"))
    benchmark_ticker = fund_info.get("benchmark_ticker", "V60A.DE")

    # Build position events from transactions
    events = build_daily_positions_and_cash(transactions)
    if not events:
        print("No events to process.")
        return

    # Collect all ISINs and their currencies
    all_isins = set()
    all_currencies = set()
    for _, pos_snap, _ in events:
        for isin, info in pos_snap.items():
            all_isins.add(isin)
            if info["currency"] != "EUR":
                all_currencies.add(info["currency"])

    # Map ISINs to tickers (using module-level CRYPTO_PROXY and TICKER_CURRENCY)
    mapped_tickers = {}
    unmapped_isins = set()
    crypto_scale = {}
    for isin in all_isins:
        ticker = isin_map.get(isin)
        if ticker:
            cp = CRYPTO_PROXY.get(ticker)
            if cp:
                mapped_tickers[isin] = cp["proxy"]
                crypto_scale[cp["proxy"]] = cp["scale"]
            else:
                mapped_tickers[isin] = ticker
        else:
            unmapped_isins.add(isin)

    # Ensure all needed currencies are downloaded
    for t, ccy in TICKER_CURRENCY.items():
        actual = "GBP" if ccy == "GBX" else ccy
        if actual != "EUR":
            all_currencies.add(actual)

    print(f"Mapped tickers: {len(mapped_tickers)}, Unmapped (bonds etc): {len(unmapped_isins)}")

    # Download all historical prices
    all_tickers = list(set(mapped_tickers.values()))
    if benchmark_ticker:
        all_tickers.append(benchmark_ticker)

    start_str = str(inception.date())
    prices = download_prices(all_tickers, start_str)
    fx_data = download_fx_rates(all_currencies, start_str)

    if prices.empty:
        print("No price data available.")
        return

    # Load manual bond prices for unmapped instruments
    manual_prices = load_manual_bond_prices()

    # Build daily date range
    end_date = pd.Timestamp.today()
    all_dates = pd.date_range(inception, end_date, freq="B")  # Business days

    # Reindex prices and FX data to cover ALL business days (holidays included)
    # so that holidays get ffilled from the previous trading day
    prices = prices.reindex(all_dates).ffill()
    for pair in fx_data:
        fx_data[pair] = fx_data[pair].reindex(all_dates).ffill()

    # For each date, find the applicable position state using bisect
    event_dates = [e[0] for e in events]

    nav_records = []

    for date in all_dates:
        # Binary search for latest event on or before this date
        idx = bisect_right(event_dates, date) - 1
        if idx < 0:
            continue  # Before first transaction
        _, positions_snap, cash = events[idx]

        # Calculate portfolio value
        port_value = 0.0
        for isin, info in positions_snap.items():
            qty = info["qty"]
            currency = info["currency"]
            cost_eur = info["cost_eur"]
            cost_local = info["cost_local"]

            if qty <= 0:
                continue

            # Helper: get FX rate for this currency on this date
            def get_fx_rate(ccy):
                if ccy == "EUR":
                    return 1.0
                fx_pair = f"{ccy}EUR=X"
                if fx_pair in fx_data and date in fx_data[fx_pair].index:
                    fx_val = fx_data[fx_pair].loc[date]
                    if pd.notna(fx_val) and fx_val > 0:
                        return float(fx_val)
                return None

            # Try to get market price from downloaded data
            ticker = mapped_tickers.get(isin)
            price_found = False
            if ticker and ticker in prices.columns:
                if date in prices.index:
                    p = prices.loc[date, ticker]
                    if pd.notna(p) and p > 0:
                        if ticker in crypto_scale:
                            value = qty * float(p) * crypto_scale[ticker]
                        else:
                            # Use actual yfinance currency if known
                            price_ccy = TICKER_CURRENCY.get(ticker, currency)
                            px = float(p)
                            if price_ccy == "GBX":
                                px /= 100.0
                                price_ccy = "GBP"
                            fx_rate = get_fx_rate(price_ccy)
                            if fx_rate is None:
                                fx_rate = 1.0
                            value = qty * px * fx_rate
                        port_value += value
                        price_found = True

            if not price_found:
                # For unmapped instruments (bonds), use manual price if available
                # Interpolate linearly from cost_local to current manual price
                if isin in manual_prices:
                    mp = manual_prices[isin]
                    current_price = mp["price"]
                    bond_ccy = mp["currency"]
                    avg_cost_local = cost_local / qty if qty > 0 else 0

                    # Linear interpolation between avg_cost and current_price
                    # based on time fraction from first purchase to today
                    first_buy_date = None
                    for ev_d, ev_p, _ in events:
                        if isin in ev_p and ev_p[isin]["qty"] > 0:
                            first_buy_date = ev_d
                            break
                    if first_buy_date and end_date > first_buy_date:
                        frac = (date - first_buy_date).days / (end_date - first_buy_date).days
                        frac = max(0.0, min(1.0, frac))
                        interp_price = avg_cost_local + (current_price - avg_cost_local) * frac
                    else:
                        interp_price = current_price

                    # Convert to EUR
                    fx_rate = get_fx_rate(bond_ccy)
                    if fx_rate is None:
                        fx_rate = cost_eur / cost_local if cost_local > 0 else 1.0
                    value = qty * interp_price * fx_rate
                    port_value += value
                else:
                    # No price at all — use invested capital
                    port_value += cost_eur

        total_nav = port_value + cash

        # Benchmark value
        bench_val = np.nan
        if benchmark_ticker and benchmark_ticker in prices.columns:
            if date in prices.index:
                bv = prices.loc[date, benchmark_ticker]
                if pd.notna(bv):
                    bench_val = float(bv)

        nav_records.append({
            "date": date,
            "nav": round(total_nav, 2),
            "benchmark": round(bench_val, 4) if pd.notna(bench_val) else np.nan,
        })

    result = pd.DataFrame(nav_records)
    print(f"\nGenerated {len(result)} daily NAV points")
    print(f"Date range: {result['date'].iloc[0]} to {result['date'].iloc[-1]}")
    print(f"NAV range: {result['nav'].min():,.0f} to {result['nav'].max():,.0f}")
    print(f"Latest NAV: {result['nav'].iloc[-1]:,.2f}")

    # Save
    output_path = DATA_DIR / "fund_nav_history.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return result


def fill_missing_nav_days(progress_callback=None):
    """Incrementally fill only missing business days in nav_history.csv.

    Reads existing history, finds gaps, downloads prices only for
    the missing date range, computes NAV for those days, and merges back.
    Returns the updated DataFrame or None if nothing to do.
    """
    nav_file = DATA_DIR / "fund_nav_history.csv"
    if not nav_file.exists():
        if progress_callback:
            progress_callback("Nessuna storia NAV trovata, esegui prima un rebuild completo.")
        return None

    # Load existing history
    existing = pd.read_csv(nav_file, parse_dates=["date"])
    existing = existing.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    existing_dates = set(existing["date"].dt.normalize())

    # Determine expected business days range
    transactions = load_transactions()
    if transactions.empty:
        return None
    fund_info = load_fund_info()
    inception = pd.to_datetime(fund_info.get("inception_date", "2023-10-01"))
    today = pd.Timestamp.today().normalize()
    all_bdays = pd.date_range(inception, today, freq="B")

    # Find missing days
    missing_dates = sorted([d for d in all_bdays if d not in existing_dates])
    if not missing_dates:
        if progress_callback:
            progress_callback("Storia NAV completa, nessun giorno mancante.")
        return existing

    if progress_callback:
        progress_callback(f"Trovati {len(missing_dates)} giorni mancanti da ricostruire...")

    # Find the date range we need prices for (min to max of missing dates)
    range_start = missing_dates[0]
    range_end = missing_dates[-1]

    # Build position events from transactions
    events = build_daily_positions_and_cash(transactions)
    if not events:
        return None
    event_dates = [e[0] for e in events]

    # Collect ISINs and currencies needed for the missing period
    all_isins = set()
    all_currencies = set()
    for _, pos_snap, _ in events:
        for isin, info in pos_snap.items():
            all_isins.add(isin)
            if info["currency"] != "EUR":
                all_currencies.add(info["currency"])

    isin_map = load_isin_map()
    # Uses module-level CRYPTO_PROXY and TICKER_CURRENCY
    mapped_tickers = {}
    unmapped_isins = set()
    crypto_scale = {}
    for isin in all_isins:
        ticker = isin_map.get(isin)
        if ticker:
            cp = CRYPTO_PROXY.get(ticker)
            if cp:
                mapped_tickers[isin] = cp["proxy"]
                crypto_scale[cp["proxy"]] = cp["scale"]
            else:
                mapped_tickers[isin] = ticker
        else:
            unmapped_isins.add(isin)

    for t, ccy in TICKER_CURRENCY.items():
        actual = "GBP" if ccy == "GBX" else ccy
        if actual != "EUR":
            all_currencies.add(actual)

    # Download prices only for the missing range (with 5-day buffer for ffill)
    benchmark_ticker = fund_info.get("benchmark_ticker", "V60A.DE")
    all_tickers = list(set(mapped_tickers.values()))
    if benchmark_ticker:
        all_tickers.append(benchmark_ticker)

    dl_start = (range_start - timedelta(days=5)).strftime("%Y-%m-%d")
    dl_end = (range_end + timedelta(days=2)).strftime("%Y-%m-%d")

    if progress_callback:
        progress_callback(f"Download prezzi dal {range_start.date()} al {range_end.date()}...")

    prices = download_prices(all_tickers, dl_start)
    fx_data = download_fx_rates(all_currencies, dl_start) if all_currencies else {}
    manual_prices = load_manual_bond_prices()
    end_date = today

    # Reindex prices and FX to cover missing dates (holidays get ffilled)
    missing_idx = pd.DatetimeIndex(missing_dates)
    if not prices.empty:
        prices = prices.reindex(prices.index.union(missing_idx)).ffill()
    for pair in fx_data:
        fx_data[pair] = fx_data[pair].reindex(fx_data[pair].index.union(missing_idx)).ffill()

    # Calculate NAV only for missing dates
    new_records = []
    for date in missing_dates:
        idx = bisect_right(event_dates, date) - 1
        if idx < 0:
            continue
        _, positions_snap, cash = events[idx]

        port_value = 0.0
        for isin, info in positions_snap.items():
            qty = info["qty"]
            currency = info["currency"]
            cost_eur = info["cost_eur"]
            cost_local = info["cost_local"]
            if qty <= 0:
                continue

            def get_fx_rate(ccy):
                if ccy == "EUR":
                    return 1.0
                fx_pair = f"{ccy}EUR=X"
                if fx_pair in fx_data and date in fx_data[fx_pair].index:
                    fx_val = fx_data[fx_pair].loc[date]
                    if pd.notna(fx_val) and fx_val > 0:
                        return float(fx_val)
                return None

            ticker = mapped_tickers.get(isin)
            price_found = False
            if ticker and not prices.empty and ticker in prices.columns:
                if date in prices.index:
                    p = prices.loc[date, ticker]
                    if pd.notna(p) and p > 0:
                        if ticker in crypto_scale:
                            port_value += qty * float(p) * crypto_scale[ticker]
                        else:
                            price_ccy = TICKER_CURRENCY.get(ticker, currency)
                            px = float(p)
                            if price_ccy == "GBX":
                                px /= 100.0
                                price_ccy = "GBP"
                            fx_rate = get_fx_rate(price_ccy) or 1.0
                            port_value += qty * px * fx_rate
                        price_found = True

            if not price_found:
                if isin in manual_prices:
                    mp = manual_prices[isin]
                    current_price = mp["price"]
                    bond_ccy = mp["currency"]
                    avg_cost_local = cost_local / qty if qty > 0 else 0
                    first_buy_date = None
                    for ev_d, ev_p, _ in events:
                        if isin in ev_p and ev_p[isin]["qty"] > 0:
                            first_buy_date = ev_d
                            break
                    if first_buy_date and end_date > first_buy_date:
                        frac = (date - first_buy_date).days / (end_date - first_buy_date).days
                        frac = max(0.0, min(1.0, frac))
                        interp_price = avg_cost_local + (current_price - avg_cost_local) * frac
                    else:
                        interp_price = current_price
                    fx_rate = get_fx_rate(bond_ccy)
                    if fx_rate is None:
                        fx_rate = cost_eur / cost_local if cost_local > 0 else 1.0
                    port_value += qty * interp_price * fx_rate
                else:
                    port_value += cost_eur

        total_nav = port_value + cash
        bench_val = np.nan
        if benchmark_ticker and not prices.empty and benchmark_ticker in prices.columns:
            if date in prices.index:
                bv = prices.loc[date, benchmark_ticker]
                if pd.notna(bv):
                    bench_val = float(bv)

        new_records.append({
            "date": date,
            "nav": round(total_nav, 2),
            "benchmark": round(bench_val, 4) if pd.notna(bench_val) else np.nan,
        })

    if not new_records:
        return existing

    new_df = pd.DataFrame(new_records)
    # Merge: existing + new, drop duplicates keeping existing
    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    # Forward-fill benchmark NaNs
    merged["benchmark"] = pd.to_numeric(merged["benchmark"], errors="coerce")
    merged["benchmark"] = merged["benchmark"].ffill()

    merged.to_csv(nav_file, index=False)
    if progress_callback:
        progress_callback(f"Aggiunti {len(new_records)} giorni alla storia NAV.")

    return merged


if __name__ == "__main__":
    build_daily_nav()
