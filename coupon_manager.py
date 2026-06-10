"""
Registrazione automatica delle cedole obbligazionarie.

Le cedole configurate in data/bond_coupons.json vengono registrate come
transazioni DIVIDEND (importo in EUR, convenzione del registro) la prima
volta che si aggiornano i prezzi dopo la data di stacco. Il processo è
idempotente: una cedola già presente a registro per lo stesso ISIN entro
7 giorni dalla data attesa non viene duplicata, quindi le cedole inserite
a mano non creano doppioni.

Per aggiungere un nuovo bond basta una voce in bond_coupons.json:
  frequency "monthly"  -> payment_day (giorno del mese) + amount in EUR/valuta
  frequency "semiannual"/"annual" -> payment_dates ["MM-DD", ...]
Gli importi in valuta vengono convertiti in EUR al cambio della data di
stacco (storico yfinance), con fallback sul cambio corrente.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import fund_manager as fm

COUPONS_FILE = fm.DATA_DIR / "bond_coupons.json"

# Tolleranza per riconoscere una cedola già registrata (anche a mano)
_DEDUP_DAYS = 7


def load_coupon_config() -> list:
    """Carica la configurazione cedole. Ritorna lista di voci abilitate."""
    if not COUPONS_FILE.exists():
        return []
    with open(COUPONS_FILE) as f:
        cfg = json.load(f)
    return [e for e in cfg.get("coupons", []) if e.get("enabled", True)]


def _due_dates(entry: dict, today: pd.Timestamp) -> list:
    """Genera le date di stacco dovute tra first_payment e min(today, last_payment)."""
    first = pd.Timestamp(entry["first_payment"])
    last = pd.Timestamp(entry.get("last_payment", "2099-12-31"))
    end = min(today, last)
    if end < first:
        return []

    dates = []
    freq = entry.get("frequency", "monthly")
    if freq == "monthly":
        day = int(entry["payment_day"])
        cur = first
        while cur <= end:
            dates.append(cur)
            # mese successivo, stesso giorno
            y, m = (cur.year + 1, 1) if cur.month == 12 else (cur.year, cur.month + 1)
            cur = pd.Timestamp(year=y, month=m, day=day)
    else:  # semiannual / annual
        for year in range(first.year, end.year + 1):
            for mmdd in entry.get("payment_dates", []):
                mm, dd = mmdd.split("-")
                d = pd.Timestamp(year=year, month=int(mm), day=int(dd))
                if first <= d <= end:
                    dates.append(d)
    return sorted(dates)


def _fx_to_eur(currency: str, on_date: pd.Timestamp) -> float:
    """Cambio valuta->EUR alla data di stacco (storico), con fallback sul live.

    Ritorna 0.0 se non determinabile (la cedola NON viene registrata).
    """
    if currency == "EUR":
        return 1.0
    try:
        import yfinance as yf
        h = yf.Ticker(f"{currency}EUR=X").history(
            start=(on_date - timedelta(days=7)).strftime("%Y-%m-%d"),
            end=(on_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )["Close"].dropna()
        if len(h):
            return float(h.iloc[-1])
    except Exception:
        pass
    try:
        from data_fetcher import get_fx_rate
        v = get_fx_rate(currency, "EUR")
        if v and v > 0 and abs(v - 1.0) > 1e-6:
            return float(v)
    except Exception:
        pass
    return 0.0


def process_due_coupons(today: str = None, recalculate: bool = False) -> list:
    """Registra le cedole scadute e non ancora a registro.

    Ritorna la lista delle cedole aggiunte:
    [{"date", "isin", "name", "amount_eur"}, ...]
    Con recalculate=True ricalcola anche posizioni/cash/NAV (non serve nel
    flusso di aggiornamento prezzi dell'app, che ricalcola subito dopo).
    """
    config = load_coupon_config()
    if not config:
        return []

    today_ts = pd.Timestamp(today) if today else pd.Timestamp.now().normalize()

    tx = fm.load_transactions()
    positions = fm.load_positions()
    held = set(positions[positions.get("quantity", 0) > 0]["isin"]) if not positions.empty else set()

    booked = []
    new_rows = []
    for entry in config:
        isin = entry["isin"]
        # Se il bond non è più in portafoglio non registriamo nuove cedole
        if held and isin not in held:
            continue

        existing = tx[(tx["transaction_type"] == "DIVIDEND") & (tx["isin"] == isin)]
        existing_dates = pd.to_datetime(existing["date"]).tolist() if not existing.empty else []

        for due in _due_dates(entry, today_ts):
            if any(abs((due - d).days) <= _DEDUP_DAYS for d in existing_dates):
                continue  # già registrata (anche manualmente)

            currency = entry.get("currency", "EUR")
            amount = float(entry["amount"])
            fx = _fx_to_eur(currency, due)
            if fx <= 0:
                continue  # cambio non determinabile: meglio non registrare
            amount_eur = round(amount * fx, 2)

            note = f"Cedola automatica ({entry.get('description', entry['name'])})"
            if currency != "EUR":
                note += f" - {amount:,.2f} {currency} @ {fx:.4f}"

            new_rows.append({
                "date": due.strftime("%Y-%m-%d"),
                "transaction_type": "DIVIDEND",
                "isin": isin,
                "name": entry["name"],
                "macro_class": "Fixed Income",
                "quantity": amount_eur,
                "price": 1.0,
                "currency": "EUR",
                "fx_rate": 1.0,
                "fees": 0.0,
                "notes": note,
                "sector": entry.get("sector", ""),
                "asset_sub_type": "Bond",
            })
            existing_dates.append(due)
            booked.append({"date": due.strftime("%Y-%m-%d"), "isin": isin,
                           "name": entry["name"], "amount_eur": amount_eur})

    if new_rows:
        tx = pd.concat([tx, pd.DataFrame(new_rows)], ignore_index=True)
        tx["date"] = pd.to_datetime(tx["date"]).dt.strftime("%Y-%m-%d")
        tx = tx.sort_values("date", kind="stable").reset_index(drop=True)
        fm.save_transactions(tx)
        if recalculate:
            fm.recalculate_all()

    return booked
