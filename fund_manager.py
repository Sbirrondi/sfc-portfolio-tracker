"""
Fund Manager - Motore centrale del portafoglio SFC.
Gestisce: transazioni, posizioni, calcolo NAV, storico, pesi, P&L.
Questo modulo è il "single source of truth" del fondo.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import json
import time

DATA_DIR = Path(__file__).parent / "data"

# ── File paths ───────────────────────────────────────────────────────────────
TRANSACTIONS_FILE = DATA_DIR / "fund_transactions.csv"
POSITIONS_FILE = DATA_DIR / "fund_positions.csv"
NAV_HISTORY_FILE = DATA_DIR / "fund_nav_history.csv"
FUND_INFO_FILE = DATA_DIR / "fund_info.json"
ISIN_MAP_FILE = DATA_DIR / "isin_map.json"
OVERRIDES_FILE = DATA_DIR / "overrides.json"
CASH_FILE = DATA_DIR / "fund_cash.json"

TRANSACTION_TYPES = ["BUY", "SELL", "DIVIDEND", "DEPOSIT", "WITHDRAWAL"]


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _sync_to_github(filename: str, message: str = None):
    """Sync a data file to GitHub (non-blocking, fails silently)."""
    try:
        from github_sync import sync_data_file, is_enabled
        if is_enabled():
            sync_data_file(filename, message)
    except Exception:
        pass  # Never break the app for sync issues


# ══════════════════════════════════════════════════════════════════════════════
# TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_transactions() -> pd.DataFrame:
    """Carica tutte le transazioni dal CSV."""
    ensure_data_dir()
    if TRANSACTIONS_FILE.exists():
        df = pd.read_csv(TRANSACTIONS_FILE)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame(columns=[
        "date", "transaction_type", "isin", "name", "macro_class",
        "quantity", "price", "currency", "fx_rate", "fees", "notes"
    ])


def save_transactions(df: pd.DataFrame):
    """Salva le transazioni su CSV."""
    ensure_data_dir()
    df.to_csv(TRANSACTIONS_FILE, index=False)
    _sync_to_github("fund_transactions.csv", "Update transactions")


def add_transaction(
    date_str: str,
    transaction_type: str,
    isin: str,
    name: str,
    macro_class: str,
    quantity: float,
    price: float,
    currency: str = "EUR",
    fx_rate: float = 1.0,
    fees: float = 0.0,
    notes: str = "",
    sector: str = "",
    asset_sub_type: str = "Stock",
) -> pd.DataFrame:
    """
    Aggiunge una transazione e aggiorna posizioni/NAV.
    Per BUY/SELL: isin, quantity, price obbligatori.
    Per DEPOSIT/WITHDRAWAL: quantity = importo in EUR.
    Per DIVIDEND: quantity = importo ricevuto, isin = strumento.
    """
    df = load_transactions()

    new_row = pd.DataFrame([{
        "date": pd.to_datetime(date_str),
        "transaction_type": transaction_type.upper(),
        "isin": isin.strip() if isin else "",
        "name": name.strip() if name else "",
        "macro_class": macro_class.strip() if macro_class else "",
        "quantity": float(quantity),
        "price": float(price),
        "currency": currency.strip(),
        "fx_rate": float(fx_rate),
        "fees": float(fees),
        "notes": notes,
        "sector": sector,
        "asset_sub_type": asset_sub_type,
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    save_transactions(df)

    # Update positions and cash after every transaction
    recalculate_all()

    return df


def delete_transaction(index: int) -> pd.DataFrame:
    """Elimina una transazione per indice e ricalcola tutto."""
    df = load_transactions()
    if 0 <= index < len(df):
        df = df.drop(index).reset_index(drop=True)
        save_transactions(df)
        recalculate_all()
    return df


def update_transaction(index: int, updates: dict) -> pd.DataFrame:
    """
    Modifica una transazione esistente per indice.
    updates: dict con le colonne da aggiornare (es. {"price": 105.5, "quantity": 200}).
    Dopo la modifica ricalcola posizioni, cash e NAV.
    """
    df = load_transactions()
    if 0 <= index < len(df):
        for col, val in updates.items():
            if col in df.columns:
                if col == "date":
                    df.at[index, col] = pd.to_datetime(val)
                else:
                    df.at[index, col] = val
        df = df.sort_values("date").reset_index(drop=True)
        save_transactions(df)
        recalculate_all()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# POSITIONS - Calcolate dalle transazioni
# ══════════════════════════════════════════════════════════════════════════════

def compute_positions_from_transactions() -> pd.DataFrame:
    """
    Calcola le posizioni correnti dal registro transazioni.
    Usa il metodo del costo medio ponderato (weighted average cost).
    Traccia anche: dividendi ricevuti, P&L realizzato, total return.
    """
    df = load_transactions()
    if df.empty:
        return pd.DataFrame()

    # Get all ISINs from trades AND dividends
    relevant = df[df["transaction_type"].isin(["BUY", "SELL", "DIVIDEND"])].copy()
    if relevant.empty:
        return pd.DataFrame()

    all_isins = relevant["isin"].dropna().unique()

    positions = []
    for isin in all_isins:
        if not isin:
            continue
        isin_tx = df[df["isin"] == isin].sort_values("date")

        qty = 0.0
        total_cost_eur = 0.0
        total_cost_local = 0.0
        realized_pnl = 0.0
        dividends_received = 0.0
        last_name = ""
        last_macro = ""
        last_currency = "EUR"
        last_sector = ""
        last_asset_sub = "Stock"

        for _, row in isin_tx.iterrows():
            tx_type = row.get("transaction_type", "")
            last_name = row.get("name", "") or last_name
            last_macro = row.get("macro_class", "") or last_macro
            last_currency = row.get("currency", "EUR") or last_currency
            last_sector = row.get("sector", "") or last_sector
            last_asset_sub = row.get("asset_sub_type", "Stock") or last_asset_sub
            fx = row.get("fx_rate", 1.0) or 1.0

            if tx_type == "BUY":
                cost_local = row["quantity"] * row["price"]
                cost_in_eur = cost_local
                if last_currency != "EUR" and fx > 0:
                    cost_in_eur = cost_local / fx
                total_cost_local += cost_local
                total_cost_eur += cost_in_eur + (row.get("fees", 0) or 0)
                qty += row["quantity"]

            elif tx_type == "SELL":
                if qty > 0:
                    avg_cost_per_unit = total_cost_eur / qty
                    avg_cost_local_per_unit = total_cost_local / qty if total_cost_local > 0 else avg_cost_per_unit
                    sell_qty = min(row["quantity"], qty)

                    # Calculate realized P&L on this sale
                    sell_proceeds_local = sell_qty * row["price"]
                    sell_proceeds_eur = sell_proceeds_local
                    if last_currency != "EUR" and fx > 0:
                        sell_proceeds_eur = sell_proceeds_local / fx
                    sell_proceeds_eur -= (row.get("fees", 0) or 0)
                    realized_pnl += sell_proceeds_eur - (sell_qty * avg_cost_per_unit)

                    total_cost_eur -= sell_qty * avg_cost_per_unit
                    total_cost_local -= sell_qty * avg_cost_local_per_unit
                    qty -= sell_qty

            elif tx_type == "DIVIDEND":
                div_amount = row["quantity"] * (row["price"] if row["price"] > 0 else 1.0)
                if last_currency != "EUR" and fx > 0 and fx != 1.0:
                    div_amount = div_amount / fx
                dividends_received += div_amount

        if abs(qty) > 1e-8 and qty > 0:
            avg_cost = total_cost_eur / qty
            if last_currency != "EUR" and total_cost_local > 0:
                avg_cost_local = total_cost_local / qty
                avg_fx = total_cost_eur / total_cost_local
            else:
                avg_cost_local = avg_cost
                avg_fx = 1.0

            positions.append({
                "isin": isin,
                "name": last_name,
                "macro_class": last_macro,
                "sector": last_sector,
                "currency": last_currency,
                "quantity": round(qty, 6),
                "avg_cost": round(avg_cost, 4),
                "avg_cost_local": round(avg_cost_local, 4),
                "avg_fx": round(avg_fx, 6),
                "invested_capital": round(total_cost_eur, 2),
                "current_value": 0,
                "current_price": 0,
                "fx_rate_current": avg_fx,
                "pnl": 0,
                "pnl_pct": 0,
                "unrealized_pnl": 0,
                "realized_pnl": round(realized_pnl, 2),
                "dividends_received": round(dividends_received, 2),
                "total_return": 0,  # unrealized + realized + dividends
                "price_effect": 0,
                "fx_effect": 0,
                "weight_on_ptf": 0,
                "asset_sub_type": last_asset_sub,
            })

    return pd.DataFrame(positions)


def load_positions() -> pd.DataFrame:
    """Carica le posizioni dal file CSV."""
    if POSITIONS_FILE.exists():
        return pd.read_csv(POSITIONS_FILE)
    return pd.DataFrame()


def save_positions(df: pd.DataFrame):
    """Salva le posizioni su file CSV."""
    ensure_data_dir()
    df.to_csv(POSITIONS_FILE, index=False)
    _sync_to_github("fund_positions.csv", "Update positions")


def update_position_prices(positions: pd.DataFrame, isin_map: dict) -> pd.DataFrame:
    """
    Aggiorna i prezzi correnti delle posizioni usando yfinance.
    Calcola P&L totale e decompone in effetto prezzo + effetto cambio.
    Il fondo è denominato in EUR.
    """
    if positions.empty:
        return positions

    from data_fetcher import get_ticker_info, get_fx_rate

    df = positions.copy()

    # Ensure numeric columns
    num_cols = ["quantity", "avg_cost", "invested_capital", "current_value",
                "current_price", "pnl", "pnl_pct"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Ensure FX/decomposition columns exist
    for col in ["avg_cost_local", "avg_fx", "fx_rate_current", "price_effect", "fx_effect"]:
        if col not in df.columns:
            df[col] = 1.0 if "fx" in col else 0.0

    df["avg_cost_local"] = pd.to_numeric(df["avg_cost_local"], errors="coerce").fillna(0)
    df["avg_fx"] = pd.to_numeric(df["avg_fx"], errors="coerce").fillna(1.0)
    df["avg_fx"] = df["avg_fx"].replace(0.0, 1.0)

    # Fetch FX rates once per currency (avoid repeated API calls)
    currencies_needed = df[df["currency"] != "EUR"]["currency"].unique()
    fx_rates = {}
    for ccy in currencies_needed:
        try:
            fx_rates[ccy] = get_fx_rate(ccy, "EUR")
        except Exception:
            fx_rates[ccy] = 1.0

    for idx, row in df.iterrows():
        isin = row.get("isin", "")
        ticker = isin_map.get(isin)
        currency = row.get("currency", "EUR")
        qty = float(row["quantity"])

        live_price = None
        yf_currency = currency  # Will be updated from yfinance if available
        if ticker:
            try:
                info = get_ticker_info(ticker)
                lp = info.get("current_price", 0)
                if lp and lp > 0:
                    live_price = lp
                    df.at[idx, "current_price"] = round(live_price, 4)
                # Use the actual trading currency from yfinance for FX conversion
                # This ensures correct conversion even if transaction currency was wrong
                reported_ccy = info.get("currency", "")
                if reported_ccy and reported_ccy != "":
                    yf_currency = reported_ccy
            except Exception:
                pass

        # Get current FX rate based on actual trading currency
        if yf_currency != "EUR":
            if yf_currency not in fx_rates:
                try:
                    fx_rates[yf_currency] = get_fx_rate(yf_currency, "EUR")
                except Exception:
                    fx_rates[yf_currency] = 1.0
            fx_now = fx_rates.get(yf_currency, 1.0)
        else:
            fx_now = 1.0
        df.at[idx, "fx_rate_current"] = round(fx_now, 6)

        # Use existing price if no live price fetched
        if live_price is None:
            live_price = float(row.get("current_price", 0) or 0)

        if live_price > 0:
            # Current value in EUR
            value_eur = qty * live_price * fx_now
            df.at[idx, "current_value"] = round(value_eur, 2)

            invested = float(row["invested_capital"])
            unrealized = value_eur - invested
            df.at[idx, "pnl"] = round(unrealized, 2)
            df.at[idx, "unrealized_pnl"] = round(unrealized, 2)
            df.at[idx, "pnl_pct"] = round(unrealized / invested, 6) if invested > 0 else 0

            # Total return = unrealized + realized + dividends
            realized = float(row.get("realized_pnl", 0) or 0)
            dividends = float(row.get("dividends_received", 0) or 0)
            total_ret = unrealized + realized + dividends
            df.at[idx, "total_return"] = round(total_ret, 2)

            # Decompose unrealized P&L for non-EUR positions:
            # unrealized_pnl = price_effect + fx_effect
            # price_effect = qty * (live_price - avg_cost_local) * avg_fx_at_purchase
            # fx_effect    = qty * live_price * (fx_now - avg_fx_at_purchase)
            # NOTE: decomposition only valid when yfinance currency matches transaction currency
            effective_ccy = yf_currency if yf_currency != currency else currency
            currencies_match = (yf_currency == currency) or (yf_currency == "EUR" and currency == "EUR")
            if effective_ccy != "EUR" and currencies_match:
                avg_fx_purchase = float(row.get("avg_fx", 1.0) or 1.0)
                avg_local = float(row.get("avg_cost_local", 0) or 0)
                if avg_fx_purchase > 0 and avg_fx_purchase != 1.0 and avg_local > 0:
                    p_effect = qty * (live_price - avg_local) * avg_fx_purchase
                    f_effect = qty * live_price * (fx_now - avg_fx_purchase)
                    df.at[idx, "price_effect"] = round(p_effect, 2)
                    df.at[idx, "fx_effect"] = round(f_effect, 2)
                else:
                    df.at[idx, "price_effect"] = round(unrealized, 2)
                    df.at[idx, "fx_effect"] = 0.0
            else:
                # Cannot decompose when yfinance and transaction currencies differ
                df.at[idx, "price_effect"] = round(unrealized, 2)
                df.at[idx, "fx_effect"] = 0.0

    # Calculate weights
    total_value = df["current_value"].sum()
    if total_value > 0:
        df["weight_on_ptf"] = (df["current_value"] / total_value).round(6)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CASH MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_cash() -> dict:
    """Carica il saldo cassa."""
    if CASH_FILE.exists():
        with open(CASH_FILE) as f:
            return json.load(f)
    return {"balance": 0, "last_updated": None}


def save_cash(cash_data: dict):
    """Salva il saldo cassa."""
    ensure_data_dir()
    with open(CASH_FILE, "w") as f:
        json.dump(cash_data, f, indent=2, default=str)
    _sync_to_github("fund_cash.json", "Update cash")


def compute_cash_from_transactions() -> float:
    """
    Calcola il saldo cassa corrente dalle transazioni.
    DEPOSIT aggiunge, WITHDRAWAL toglie.
    BUY toglie (quantity * price + fees), SELL aggiunge (quantity * price - fees).
    DIVIDEND aggiunge (quantity * price).
    """
    df = load_transactions()
    if df.empty:
        return 0.0

    cash = 0.0
    for _, row in df.iterrows():
        tx_type = row.get("transaction_type", "")
        qty = float(row.get("quantity", 0) or 0)
        price = float(row.get("price", 0) or 0)
        fees = float(row.get("fees", 0) or 0)
        fx = float(row.get("fx_rate", 1.0) or 1.0)

        if tx_type == "DEPOSIT":
            cash += qty  # Direct EUR amount
        elif tx_type == "WITHDRAWAL":
            cash -= qty
        elif tx_type == "BUY":
            cost = qty * price
            currency = row.get("currency", "EUR")
            if currency != "EUR" and fx != 1.0:
                cost = cost / fx
            cash -= (cost + fees)
        elif tx_type == "SELL":
            proceeds = qty * price
            currency = row.get("currency", "EUR")
            if currency != "EUR" and fx != 1.0:
                proceeds = proceeds / fx
            cash += (proceeds - fees)
        elif tx_type == "DIVIDEND":
            cash += qty * price if price > 0 else qty

    return round(cash, 2)


# ══════════════════════════════════════════════════════════════════════════════
# NAV CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_nav(positions: pd.DataFrame, cash: float) -> float:
    """Calcola il NAV corrente = somma posizioni + liquidità."""
    pos_value = positions["current_value"].sum() if not positions.empty else 0
    return round(pos_value + cash, 2)


def snapshot_nav(nav: float, benchmark_value: float = None):
    """Salva un punto NAV nello storico."""
    ensure_data_dir()

    # Sanity check: NAV should be within reasonable range (50% to 500% of initial)
    fund_info = load_fund_info()
    initial_nav = fund_info.get("initial_nav", 10_000_000)
    if initial_nav > 0:
        ratio = nav / initial_nav
        if ratio > 5.0 or ratio < 0.1 or nav <= 0:
            return None  # Reject obviously wrong NAV values

    today = datetime.now().strftime("%Y-%m-%d")

    # Load existing history
    if NAV_HISTORY_FILE.exists():
        history = pd.read_csv(NAV_HISTORY_FILE)
        history["date"] = pd.to_datetime(history["date"])
    else:
        history = pd.DataFrame(columns=["date", "nav", "benchmark"])

    # Check if today already has an entry → update it
    today_mask = history["date"].dt.strftime("%Y-%m-%d") == today
    if today_mask.any():
        history.loc[today_mask, "nav"] = nav
        if benchmark_value is not None:
            history.loc[today_mask, "benchmark"] = benchmark_value
    else:
        new_row = pd.DataFrame([{
            "date": pd.to_datetime(today),
            "nav": nav,
            "benchmark": benchmark_value
        }])
        history = pd.concat([history, new_row], ignore_index=True)

    history = history.sort_values("date").reset_index(drop=True)
    history.to_csv(NAV_HISTORY_FILE, index=False)
    _sync_to_github("fund_nav_history.csv", "Update NAV history")
    return history


def load_nav_history() -> pd.DataFrame:
    """Carica lo storico NAV."""
    if NAV_HISTORY_FILE.exists():
        df = pd.read_csv(NAV_HISTORY_FILE, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame(columns=["date", "nav", "benchmark"])


def rebuild_daily_nav() -> pd.DataFrame:
    """
    Ricostruisce uno storico NAV giornaliero interpolando i punti mensili
    e scaricando i valori giornalieri del benchmark (VNGA60.MI) da yfinance.
    I punti NAV mensili vengono interpolati linearmente per avere un valore al giorno.
    Il benchmark viene preso dai prezzi di chiusura effettivi.
    """
    from data_fetcher import get_benchmark_prices

    history = load_nav_history()
    if history.empty or len(history) < 2:
        return history

    history = history.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    history["date"] = pd.to_datetime(history["date"])
    history["nav"] = pd.to_numeric(history["nav"], errors="coerce")

    # Remove obviously wrong entries
    initial_nav = history["nav"].iloc[0]
    if initial_nav > 0:
        history = history[(history["nav"] > initial_nav * 0.1) & (history["nav"] < initial_nav * 5.0)]
        history = history.reset_index(drop=True)

    if len(history) < 2:
        return history

    # Create daily date range
    start = history["date"].iloc[0]
    end = history["date"].iloc[-1]
    daily_dates = pd.date_range(start=start, end=end, freq="B")  # Business days

    # Interpolate NAV to daily
    history_indexed = history.set_index("date")["nav"]
    daily_nav = history_indexed.reindex(daily_dates).interpolate(method="linear")
    daily_nav = daily_nav.ffill().bfill()

    # Fetch benchmark daily prices
    benchmark_daily = None
    try:
        bench_prices = get_benchmark_prices("VNGA60", start=start.strftime("%Y-%m-%d"))
        if bench_prices is not None and len(bench_prices) > 0:
            benchmark_daily = bench_prices.reindex(daily_dates).ffill().bfill()
    except Exception:
        pass

    # Build result DataFrame
    result = pd.DataFrame({
        "date": daily_dates,
        "nav": daily_nav.values,
    })

    if benchmark_daily is not None and len(benchmark_daily) > 0:
        result["benchmark"] = benchmark_daily.values
    else:
        # Interpolate existing benchmark values
        if "benchmark" in history.columns:
            bench_indexed = pd.to_numeric(
                history.set_index("date")["benchmark"], errors="coerce"
            )
            daily_bench = bench_indexed.reindex(daily_dates).interpolate(method="linear")
            result["benchmark"] = daily_bench.ffill().bfill().values

    # Save
    result.to_csv(NAV_HISTORY_FILE, index=False)
    _sync_to_github("fund_nav_history.csv", "Rebuild daily NAV history")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FUND INFO
# ══════════════════════════════════════════════════════════════════════════════

def load_fund_info() -> dict:
    """Carica le info del fondo."""
    if FUND_INFO_FILE.exists():
        with open(FUND_INFO_FILE) as f:
            return json.load(f)
    return {
        "fund_name": "SFC Cattolica Investment Fund",
        "inception_date": "2023-10-01",
        "initial_nav": 10_000_000,
        "currency": "EUR",
        "benchmark": "S&P 500 Equal Weight",
    }


def save_fund_info(info: dict):
    """Salva le info del fondo."""
    ensure_data_dir()
    with open(FUND_INFO_FILE, "w") as f:
        json.dump(info, f, indent=2, default=str)
    _sync_to_github("fund_info.json", "Update fund info")


def update_fund_info(nav: float, positions_count: int):
    """Aggiorna automaticamente fund_info con NAV corrente e performance."""
    info = load_fund_info()
    initial_nav = info.get("initial_nav", 10_000_000)
    info["current_nav"] = nav
    info["positions_count"] = positions_count
    if initial_nav > 0:
        info["performance_since_inception"] = (nav - initial_nav) / initial_nav
    info["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_fund_info(info)
    return info


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG FILES
# ══════════════════════════════════════════════════════════════════════════════

def get_isin_map() -> dict:
    """Carica il mapping ISIN → ticker."""
    if ISIN_MAP_FILE.exists():
        with open(ISIN_MAP_FILE) as f:
            return json.load(f)
    return {}


def save_isin_map(isin_map: dict):
    """Salva il mapping ISIN → ticker."""
    ensure_data_dir()
    with open(ISIN_MAP_FILE, "w") as f:
        json.dump(isin_map, f, indent=4)
    _sync_to_github("isin_map.json", "Update ISIN mapping")


def get_overrides() -> dict:
    """Carica le override manuali."""
    if OVERRIDES_FILE.exists():
        with open(OVERRIDES_FILE) as f:
            return json.load(f)
    return {}


def save_overrides(overrides: dict):
    """Salva le override manuali."""
    ensure_data_dir()
    with open(OVERRIDES_FILE, "w") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)
    _sync_to_github("overrides.json", "Update overrides")


def enrich_positions(pos_df: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """Arricchisce le posizioni con dati dalle override (settore, paese, ecc.)."""
    if pos_df.empty:
        return pos_df

    df = pos_df.copy()

    # Ensure key columns exist
    for col in ["current_value", "invested_capital", "pnl", "pnl_pct",
                 "weight_on_ptf", "quantity", "avg_cost", "current_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "sector" not in df.columns:
        df["sector"] = "N/A"
    if "country" not in df.columns:
        df["country"] = "N/A"

    for idx, row in df.iterrows():
        isin = row.get("isin", "")
        if isin in overrides and isin != "_comment":
            o = overrides[isin]
            for key, val in o.items():
                if key not in df.columns:
                    df[key] = "N/A"
                df.at[idx, key] = val

    return df


# ══════════════════════════════════════════════════════════════════════════════
# RECALCULATE ALL - Chiamata dopo ogni transazione
# ══════════════════════════════════════════════════════════════════════════════

def recalculate_all():
    """
    Ricalcola tutto dopo una modifica alle transazioni:
    1. Posizioni dalle transazioni
    2. Cassa dalle transazioni
    3. Salva posizioni aggiornate
    4. Aggiorna fund_info
    """
    # Compute positions from transactions
    positions = compute_positions_from_transactions()

    # Compute cash
    cash = compute_cash_from_transactions()
    save_cash({"balance": cash, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")})

    # Save positions (prices will be updated separately via live fetch)
    if not positions.empty:
        # Merge with existing positions to preserve current_price
        # (current_value/pnl will be recalculated from current_price + new position data)
        existing = load_positions()
        if not existing.empty:
            # Only preserve current_price and fx_rate_current from old positions
            preserve_cols = ["current_price", "fx_rate_current"]
            existing_prices = existing[["isin"] + [c for c in preserve_cols if c in existing.columns]].copy()
            # Rename to avoid conflicts
            existing_prices = existing_prices.rename(columns={
                "current_price": "_old_current_price",
                "fx_rate_current": "_old_fx_rate_current",
            })
            positions = positions.merge(existing_prices, on="isin", how="left")

            # Use old current_price if we don't have a new one
            if "_old_current_price" in positions.columns:
                old_price = pd.to_numeric(positions["_old_current_price"], errors="coerce").fillna(0)
                positions["current_price"] = old_price
                positions.drop(columns=["_old_current_price"], inplace=True)
            if "_old_fx_rate_current" in positions.columns:
                old_fx = pd.to_numeric(positions["_old_fx_rate_current"], errors="coerce").fillna(1.0)
                old_fx = old_fx.replace(0.0, 1.0)
                positions["fx_rate_current"] = old_fx
                positions.drop(columns=["_old_fx_rate_current"], inplace=True)

            # Recalculate current_value and P&L from current_price + new position data
            for idx, row in positions.iterrows():
                cp = float(row.get("current_price", 0) or 0)
                qty = float(row.get("quantity", 0) or 0)
                ccy = row.get("currency", "EUR")
                fx_now = float(row.get("fx_rate_current", 1.0) or 1.0)

                if cp > 0 and qty > 0:
                    value_eur = qty * cp * fx_now
                    invested = float(row.get("invested_capital", 0) or 0)
                    unrealized = value_eur - invested
                    positions.at[idx, "current_value"] = round(value_eur, 2)
                    positions.at[idx, "pnl"] = round(unrealized, 2)
                    positions.at[idx, "unrealized_pnl"] = round(unrealized, 2)
                    positions.at[idx, "pnl_pct"] = round(unrealized / invested, 6) if invested > 0 else 0

        save_positions(positions)

    # Update fund info
    nav = calculate_nav(positions, cash)
    update_fund_info(nav, len(positions))


# ══════════════════════════════════════════════════════════════════════════════
# MIGRATION - Import posizioni esistenti come transazioni iniziali
# ══════════════════════════════════════════════════════════════════════════════

def migrate_positions_to_transactions():
    """
    Migra le posizioni esistenti (dal CSV) in transazioni BUY iniziali.
    Usa la data di inception del fondo come data della transazione.
    Idempotente: non duplica se la migrazione è già avvenuta.
    """
    positions = load_positions()
    if positions.empty:
        return False

    existing_tx = load_transactions()

    # Check if migration already happened (look for migration marker)
    if not existing_tx.empty and "notes" in existing_tx.columns and "migration_initial" in existing_tx["notes"].values:
        return False  # Already migrated

    fund_info = load_fund_info()
    inception_date = fund_info.get("inception_date", "2023-10-01")

    new_transactions = []

    # First, add the initial deposit
    initial_nav = fund_info.get("initial_nav", 10_000_000)
    new_transactions.append({
        "date": pd.to_datetime(inception_date),
        "transaction_type": "DEPOSIT",
        "isin": "",
        "name": "Conferimento iniziale",
        "macro_class": "",
        "quantity": initial_nav,
        "price": 1.0,
        "currency": "EUR",
        "fx_rate": 1.0,
        "fees": 0,
        "notes": "migration_initial",
        "sector": "",
        "asset_sub_type": "",
    })

    # Then, add BUY for each position
    for _, row in positions.iterrows():
        qty = float(row.get("quantity", 0) or 0)
        avg_cost = float(row.get("avg_cost", 0) or 0)
        if qty <= 0 or avg_cost <= 0:
            continue

        new_transactions.append({
            "date": pd.to_datetime(inception_date),
            "transaction_type": "BUY",
            "isin": str(row.get("isin", "")),
            "name": str(row.get("name", "")),
            "macro_class": str(row.get("macro_class", "")),
            "quantity": qty,
            "price": avg_cost,
            "currency": str(row.get("currency", "EUR")),
            "fx_rate": 1.0,  # Already in position currency
            "fees": 0,
            "notes": "migration_initial",
            "sector": str(row.get("sector", row.get("classification", ""))),
            "asset_sub_type": str(row.get("asset_sub_type", "Stock")),
        })

    if new_transactions:
        new_df = pd.DataFrame(new_transactions)
        # Append to existing transactions (preserve any real transactions)
        if not existing_tx.empty and "notes" in existing_tx.columns:
            real_tx = existing_tx[existing_tx["notes"] != "migration_initial"]
        else:
            real_tx = existing_tx if not existing_tx.empty else pd.DataFrame()
        combined = pd.concat([new_df, real_tx], ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)
        save_transactions(combined)

    return True


def get_portfolio_summary() -> dict:
    """Restituisce un riepilogo completo del portafoglio."""
    positions = load_positions()
    overrides = get_overrides()
    isin_map = get_isin_map()
    fund_info = load_fund_info()
    cash_data = load_cash()
    nav_history = load_nav_history()

    if not positions.empty:
        positions = enrich_positions(positions, overrides)

    cash = cash_data.get("balance", 0)
    total_value = positions["current_value"].sum() if not positions.empty else 0
    nav = total_value + cash

    return {
        "positions": positions,
        "fund_info": fund_info,
        "isin_map": isin_map,
        "overrides": overrides,
        "nav_history": nav_history,
        "cash": cash,
        "total_value": total_value,
        "nav": nav,
    }
