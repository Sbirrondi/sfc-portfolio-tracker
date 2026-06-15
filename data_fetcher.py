"""
Data Fetcher - Recupera dati di mercato da yfinance.
Gestisce: prezzi storici, info fondamentali, benchmark, FX rates.
Cache in memoria per evitare chiamate ripetute.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
import time

# In-memory cache
_info_cache = {}
_price_cache = {}


# ── Ticker Info ──────────────────────────────────────────────────────────────

def get_ticker_info(ticker: str) -> dict:
    """
    Recupera le info fondamentali di un ticker da yfinance.
    Restituisce un dict con campi normalizzati.
    """
    if ticker in _info_cache:
        return _info_cache[ticker]

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        normalized = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "currency": info.get("currency", ""),
            "exchange": info.get("exchange", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "asset_type": _classify_asset(info, ticker),

            # Valuation multiples
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),

            # Profitability
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),

            # Dividends & yield
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "payout_ratio": info.get("payoutRatio"),

            # Risk
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "50d_avg": info.get("fiftyDayAverage"),
            "200d_avg": info.get("twoHundredDayAverage"),

            # Price
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),

            # Additional
            "description": info.get("longBusinessSummary", ""),
            "employees": info.get("fullTimeEmployees"),
        }

        _info_cache[ticker] = normalized
        return normalized

    except Exception as e:
        fallback = {
            "ticker": ticker, "name": ticker, "sector": "N/A",
            "industry": "N/A", "country": "N/A", "currency": "",
            "exchange": "N/A", "market_cap": 0, "asset_type": "Unknown",
            "trailing_pe": None, "forward_pe": None, "price_to_book": None,
            "ev_to_ebitda": None, "ev_to_revenue": None, "peg_ratio": None,
            "price_to_sales": None, "profit_margin": None, "operating_margin": None,
            "roe": None, "roa": None, "revenue_growth": None, "earnings_growth": None,
            "dividend_yield": None, "dividend_rate": None, "payout_ratio": None,
            "beta": None, "52w_high": None, "52w_low": None,
            "50d_avg": None, "200d_avg": None, "current_price": 0,
            "description": "", "employees": None,
        }
        _info_cache[ticker] = fallback
        return fallback


def _classify_asset(info: dict, ticker: str) -> str:
    """Classifica il tipo di asset in base alle info yfinance."""
    qtype = info.get("quoteType", "").upper()
    if qtype == "ETF":
        return "ETF"
    elif qtype == "MUTUALFUND":
        return "Fund"
    elif qtype == "CRYPTOCURRENCY":
        return "Crypto"
    elif qtype == "FUTURE" or qtype == "COMMODITY":
        return "Commodity"
    elif qtype == "EQUITY":
        return "Stock"
    # Fallback heuristics
    ticker_upper = ticker.upper()
    if ticker_upper.endswith("-USD") or ticker_upper.startswith("BTC") or ticker_upper.startswith("ETH"):
        return "Crypto"
    return "Stock"


# ── Historical Prices ────────────────────────────────────────────────────────

def get_historical_prices(
    tickers: list,
    start: str = None,
    end: str = None,
    period: str = "5y"
) -> dict:
    """
    Scarica prezzi storici (Close adjusted) per una lista di ticker.
    Restituisce: {ticker: pd.Series}
    """
    cache_key = tuple(sorted(tickers)) + (start, end, period)
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    result = {}

    try:
        if start:
            data = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=True
            )
        else:
            data = yf.download(
                tickers, period=period,
                auto_adjust=True, progress=False, threads=True
            )

        if data.empty:
            return result

        if len(tickers) == 1:
            # Single ticker: data has simple columns
            if "Close" in data.columns:
                series = data["Close"].dropna()
                result[tickers[0]] = series
        else:
            # Multiple tickers: multi-level columns
            if "Close" in data.columns.get_level_values(0):
                close_data = data["Close"]
                for ticker in tickers:
                    if ticker in close_data.columns:
                        series = close_data[ticker].dropna()
                        if not series.empty:
                            result[ticker] = series

    except Exception as e:
        # Fallback: fetch one by one
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                if start:
                    hist = t.history(start=start, end=end, auto_adjust=True)
                else:
                    hist = t.history(period=period, auto_adjust=True)
                if not hist.empty and "Close" in hist.columns:
                    result[ticker] = hist["Close"].dropna()
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")

    _price_cache[cache_key] = result
    return result


def get_current_prices(tickers: list) -> dict:
    """Recupera i prezzi correnti per una lista di ticker."""
    prices = {}
    for ticker in tickers:
        info = get_ticker_info(ticker)
        price = info.get("current_price", 0)
        if price and price > 0:
            prices[ticker] = price
        else:
            # Fallback: last close from history
            try:
                hist = get_historical_prices([ticker], period="5d")
                if ticker in hist and not hist[ticker].empty:
                    prices[ticker] = hist[ticker].iloc[-1]
            except Exception as e:
                logger.warning(f"Failed to get current price for {ticker}: {e}")
                prices[ticker] = 0
    return prices


def get_current_prices_bulk(tickers: list) -> dict:
    """
    Ultimo prezzo di chiusura per molti ticker con UNA chiamata bulk a
    yf.download (in batch da 10), invece di una chiamata .info per ticker.

    Perché: l'endpoint quoteSummary usato da .info viene bloccato/throttato
    dagli IP datacenter (es. Streamlit Cloud) e torna vuoto — questo congelava
    silenziosamente i prezzi. L'endpoint chart/download usato qui è molto più
    tollerante (è lo stesso che tiene aggiornato lo storico NAV).

    Restituisce {ticker: prezzo} solo per i ticker con un prezzo valido.
    """
    tickers = [t for t in dict.fromkeys(tickers) if t]  # dedup, drop empty
    if not tickers:
        return {}

    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    out = {}
    batch_size = 10

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch, start=start, end=end,
                auto_adjust=True, progress=False, threads=True
            )
            if data is None or data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                lvl0 = data.columns.get_level_values(0)
                col = "Close" if "Close" in lvl0 else ("Price" if "Price" in lvl0 else None)
                if not col:
                    continue
                close = data[col]
                for tk in batch:
                    if tk in close.columns:
                        s = close[tk].dropna()
                        if not s.empty and float(s.iloc[-1]) > 0:
                            out[tk] = float(s.iloc[-1])
            else:
                # Single ticker: simple columns
                if "Close" in data.columns:
                    s = data["Close"].dropna()
                    if not s.empty and float(s.iloc[-1]) > 0:
                        out[batch[0]] = float(s.iloc[-1])
        except Exception as e:
            logger.warning(f"bulk price batch failed for {batch}: {e}")

    return out


# ── Benchmark Data ───────────────────────────────────────────────────────────

BENCHMARKS = {
    "VNGA60": "VNGA60.MI",           # Vanguard LifeStrategy 60% Equity UCITS ETF (EUR, Milan)
    "S&P 500 Equal Weight": "RSP",   # Invesco S&P 500 Equal Weight ETF
    "S&P 500": "^GSPC",
    "MSCI World": "URTH",
    "MSCI EM": "EEM",
    "NASDAQ 100": "^NDX",
    "Euro Stoxx 50": "^STOXX50E",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "DAX": "^GDAXI",
    "FTSE MIB": "FTSEMIB.MI",
    "BTC": "BTC-USD",
    "Gold": "GC=F",
    "US 10Y Bond": "^TNX",
}


def get_benchmark_prices(benchmark_key: str, start: str = None, period: str = "5y") -> pd.Series:
    """Scarica i prezzi storici di un benchmark."""
    ticker = BENCHMARKS.get(benchmark_key, benchmark_key)
    prices = get_historical_prices([ticker], start=start, period=period)
    return prices.get(ticker, pd.Series(dtype=float))


# ── FX Rates ─────────────────────────────────────────────────────────────────

def get_fx_rate(from_currency: str, to_currency: str = "EUR") -> float:
    """Recupera il tasso di cambio corrente. Prova coppia diretta, poi inversa."""
    if from_currency == to_currency:
        return 1.0

    # Try direct pair first (e.g. HKDEUR=X)
    try:
        pair = f"{from_currency}{to_currency}=X"
        t = yf.Ticker(pair)
        hist = t.history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].dropna().iloc[-1])
            if rate > 0:
                return rate
    except Exception as e:
        logger.warning(f"FX direct {from_currency}{to_currency}: {e}")

    # Try reverse pair and invert (e.g. EURHKD=X → 1/rate)
    try:
        pair_rev = f"{to_currency}{from_currency}=X"
        t2 = yf.Ticker(pair_rev)
        hist2 = t2.history(period="5d")
        if not hist2.empty:
            rate_rev = float(hist2["Close"].dropna().iloc[-1])
            if rate_rev > 0:
                return 1.0 / rate_rev
    except Exception as e:
        logger.warning(f"FX reverse {to_currency}{from_currency}: {e}")

    logger.warning(f"Failed to get FX rate {from_currency}/{to_currency}, returning 1.0")
    return 1.0


def get_fx_history(from_currency: str, to_currency: str = "EUR", period: str = "5y") -> pd.Series:
    """Scarica la serie storica di un tasso di cambio."""
    if from_currency == to_currency:
        return pd.Series(dtype=float)
    pair = f"{from_currency}{to_currency}=X"
    prices = get_historical_prices([pair], period=period)
    return prices.get(pair, pd.Series(dtype=float))


# ── Batch Info Fetcher ───────────────────────────────────────────────────────

def get_all_tickers_info(tickers: list) -> pd.DataFrame:
    """Recupera le info per tutti i ticker del portafoglio. Restituisce un DataFrame."""
    infos = []
    for ticker in tickers:
        info = get_ticker_info(ticker)
        infos.append(info)
        time.sleep(0.2)  # Rate limiting
    return pd.DataFrame(infos)


# ── Bond Price Scraping ──────────────────────────────────────────────────────

# Map ISIN -> Borsa Italiana URL path for bonds
_BOND_BORSA_IT_MAP = {
    "IT0005433690": "mot/btp/scheda",
    "IT0005436693": "mot/btp/scheda",
    "AT0000A2HLC4": "mot/euro-obbligazioni/scheda",
    "DE0001141836": "mot/euro-obbligazioni/scheda",
    "XS2262211076": "mot/euro-obbligazioni/scheda",
    "IT0005640666": "mot/btp/scheda",
}

# Map ISIN -> Euronext market code for bonds available via Euronext ajax API
# NB: da giugno 2026 l'endpoint Euronext risponde con payload cifrato ("ct": ...),
# quindi non è più utilizzabile: preferire _BOND_TV_MAP.
_BOND_EURONEXT_MAP = {}

# Map ISIN -> TradingView exchange:ticker for bonds via TradingView scanner API
_BOND_TV_MAP = {
    "XS2943818059": "SWB:XS2943818059",    # Iliad 5.375% 2030
    "US91282CMN82": "OTCB:91282CMN8",       # US Treasury 4.25% Feb 2028
    "GB00BSQNRC93": "DUS:GBBSQNRC9",       # UK Bond 4.375% 24/28
    "US912810PV44": "FWB:US912810PV4",      # US TIPS 1.75% Jan 2028
    "AU000XCLWAX7": "EUROTLX:AU000XCLWAX7",  # Australia 2.75% Nov 2029 (EuroTLX)
}

_bond_price_cache = {}


def _fetch_tradingview_prices(isins: list) -> dict:
    """Fetch bond prices from TradingView scanner API. Returns {isin: price}."""
    tv_tickers = {isin: _BOND_TV_MAP[isin] for isin in isins if isin in _BOND_TV_MAP}
    if not tv_tickers:
        return {}

    try:
        import requests as req_lib
        url = "https://scanner.tradingview.com/bond/scan"
        payload = {
            "symbols": {
                "tickers": list(tv_tickers.values()),
                "query": {"types": []}
            },
            "columns": ["close", "name", "currency"]
        }
        resp = req_lib.post(url, json=payload,
                           headers={"Content-Type": "application/json"},
                           timeout=10)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        results = {}
        # Build reverse map: tv_ticker -> isin
        rev_map = {v: k for k, v in tv_tickers.items()}
        for item in data.get("data", []):
            ticker = item.get("s", "")
            vals = item.get("d", [])
            if vals and len(vals) > 0:
                price = vals[0]
                if isinstance(price, (int, float)) and 5 < price < 200:
                    isin = rev_map.get(ticker)
                    if isin:
                        results[isin] = price
        return results
    except Exception:
        return {}


def _fetch_euronext_price(isin: str) -> float:
    """Fetch bond price from Euronext ajax API."""
    market = _BOND_EURONEXT_MAP.get(isin)
    if not market:
        return 0.0
    url = f"https://live.euronext.com/en/ajax/getDetailedQuote/{isin}-{market}"
    try:
        import requests as req_lib
        import re
        from bs4 import BeautifulSoup

        resp = req_lib.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code != 200 or "unknown" in resp.text.lower():
            return 0.0

        soup = BeautifulSoup(resp.text, "html.parser")
        full_text = soup.get_text()

        # Look for "% XX.XX" pattern (Euronext bond price format)
        for m in re.finditer(r'%\s*(\d{1,3}[.,]\d{1,5})', full_text):
            price_str = m.group(1).replace(",", ".")
            price = float(price_str)
            if 5 < price < 200:
                return price

        # Also try plain number patterns
        for m in re.finditer(r'(\d{1,3}),(\d{2,5})', full_text):
            price = float(f"{m.group(1)}.{m.group(2)}")
            if 10 < price < 150:
                return price
    except Exception:
        pass
    return 0.0


def get_bond_price_from_borsa_italiana(isin: str) -> float:
    """Fetch bond price: tries Borsa Italiana first, then Euronext API."""
    if isin in _bond_price_cache:
        return _bond_price_cache[isin]

    # Try Euronext API (for non-Italian bonds)
    if isin in _BOND_EURONEXT_MAP:
        price = _fetch_euronext_price(isin)
        if price > 0:
            _bond_price_cache[isin] = price
            return price

    # Try TradingView API
    if isin in _BOND_TV_MAP:
        tv_prices = _fetch_tradingview_prices([isin])
        if isin in tv_prices:
            _bond_price_cache[isin] = tv_prices[isin]
            return tv_prices[isin]

    url_path = _BOND_BORSA_IT_MAP.get(isin)
    if not url_path:
        return 0.0

    url = f"https://www.borsaitaliana.it/borsa/obbligazioni/{url_path}/{isin}.html"
    try:
        import requests as req_lib
        import re
        from bs4 import BeautifulSoup

        resp = req_lib.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code != 200:
            return 0.0

        soup = BeautifulSoup(resp.text, "html.parser")

        # Method 1: Look for price in <strong> tags (main price display)
        for tag in soup.find_all("strong"):
            text = tag.get_text(strip=True)
            m = re.match(r'^(\d{1,3}),(\d{2,5})$', text)
            if m:
                price = float(f"{m.group(1)}.{m.group(2)}")
                if 5 < price < 200:
                    _bond_price_cache[isin] = price
                    return price

        # Method 2: Search full page text for price patterns
        full_text = soup.get_text()
        for m in re.finditer(r'(\d{1,3}),(\d{2,5})', full_text):
            price = float(f"{m.group(1)}.{m.group(2)}")
            if 10 < price < 150:
                _bond_price_cache[isin] = price
                return price

    except Exception:
        pass

    return 0.0


def get_all_bond_prices() -> dict:
    """Fetch prices for all mapped bonds. Returns {isin: price}."""
    results = {}

    # Batch fetch TradingView bonds in one API call (efficient)
    tv_isins = list(_BOND_TV_MAP.keys())
    if tv_isins:
        tv_prices = _fetch_tradingview_prices(tv_isins)
        for isin, price in tv_prices.items():
            _bond_price_cache[isin] = price
            results[isin] = price

    # Fetch Borsa Italiana and Euronext bonds individually
    for isin in list(_BOND_BORSA_IT_MAP.keys()) + list(_BOND_EURONEXT_MAP.keys()):
        price = get_bond_price_from_borsa_italiana(isin)
        if price > 0:
            results[isin] = price

    return results


def clear_cache():
    """Svuota la cache."""
    global _info_cache, _price_cache, _bond_price_cache
    _info_cache = {}
    _price_cache = {}
    _bond_price_cache = {}
