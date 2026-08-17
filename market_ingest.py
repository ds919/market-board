#!/usr/bin/env python3
"""
market_ingest.py -- nightly market-board data engine.

Pipeline:
  1. Backfill (first run) or nightly update of daily OHLCV into SQLite.
     Primary feed: yfinance (one batched call). Fallback: Stooq per-ticker.
  2. Compute per-ticker technicals, universe breadth, regime, index panel,
     and theme scores (with 1-day deltas persisted for tomorrow).
  3. Emit dashboard_data.json in the exact shape the Market Board reads.

Run:
  python market_ingest.py             # backfill if DB empty, else nightly update
  python market_ingest.py --backfill  # force a full-history pull
  python market_ingest.py --selftest  # synthetic data, NO network -- validates logic

Deps: pip install yfinance pandas numpy   (sqlite3 is stdlib)
"""
import argparse, os, json, sqlite3, sys
import numpy as np
import pandas as pd

DB_PATH   = "market.db"
JSON_OUT  = "dashboard_data.json"
REPLAY_CSV = "macro_replay.csv"
HIST_CSV   = "macro_history.csv"   # committed to the repo; shared by local + Action
UNIVERSE  = "UNIVERSE V1"
BACKFILL_PERIOD = os.environ.get("BACKFILL_PERIOD", "7y")
# 7y reaches back through the 2020 COVID crash -- the only genuine bear market
# available for testing the regime engine. Scores are only valid ~200 trading
# days after the pull start (the 200DMA warmup), so 7y gives clean scores from
# roughly 2020 onward. Override with e.g. BACKFILL_PERIOD=max or 2y.
NIGHTLY_PERIOD  = "10d"   # small incremental pull

INDEX_ETFS = [("SPY", "SPX"), ("QQQ", "NDX"), ("IWM", "R2K"),
              ("RSP", "Equal Weight SPX"), ("DIA", "DJIA")]
INDEX_SYMS = [s for s, _ in INDEX_ETFS]

SECTOR_ETFS = [("XLK", "Technology"), ("XLF", "Financials"), ("XLE", "Energy"),
               ("XLV", "Health Care"), ("XLI", "Industrials"), ("XLY", "Cons Disc"),
               ("XLP", "Cons Staples"), ("XLU", "Utilities"), ("XLB", "Materials"),
               ("XLRE", "Real Estate"), ("XLC", "Comm Svcs"), ("SMH", "Semiconductors"),
               ("IBB", "Biotech"), ("XBI", "Biotech SMID"), ("ITA", "Aero & Defense"),
               ("KWEB", "China Internet"), ("URA", "Uranium"), ("TAN", "Solar")]
SECTOR_SYMS = [s for s, _ in SECTOR_ETFS]

MACRO_ETFS = [("GLD", "Gold"), ("SLV", "Silver"), ("USO", "Crude Oil"),
              ("UUP", "US Dollar"), ("TLT", "20Y Treasuries"), ("HYG", "High Yield"),
              ("BITO", "Bitcoin"), ("UNG", "Nat Gas"), ("DBC", "Commodities")]
MACRO_SYMS = [s for s, _ in MACRO_ETFS]

# Macro Risk Engine inputs (Sec.2 of spec). Most already exist in the index/
# sector/macro lists; these are the genuinely new ones. Excluded from breadth.
MACRO_STRUCT_EXTRA = [("BTC-USD", "Bitcoin"), ("IEI", "3-7Y Treasuries"),
                      ("CPER", "Copper"),
                      # volatility term structure: front to back, plus vol-of-vol
                      ("^VIX9D", "VIX 9-Day"), ("^VIX", "VIX 30-Day"),
                      ("^VIX3M", "VIX 3-Month"), ("^VIX6M", "VIX 6-Month"),
                      ("^VVIX", "Vol of VIX")]
MACRO_STRUCT_SYMS = [s for s, _ in MACRO_STRUCT_EXTRA]
NON_MEMBERS = set(INDEX_SYMS) | set(SECTOR_SYMS) | set(MACRO_SYMS) | set(MACRO_STRUCT_SYMS)

# ---- Theme map: ~250-name universe. Edit freely -- a ticker may appear in more
# ---- than one theme (that's realistic; it just contributes to both scores).
THEMES = {
    "Semiconductors": ["NVDA", "AMD", "AVGO", "TSM", "MU", "INTC", "QCOM", "ASML", "AMAT", "LRCX", "KLAC", "ADI", "TXN", "NXPI", "MRVL", "ON", "MCHP", "SWKS", "QRVO", "TER", "ENTG", "ARM", "SNPS", "CDNS", "ALAB"],
    "Software Infrastructure": ["MSFT", "ORCL", "NOW", "SNOW", "DDOG", "NET", "MDB", "GTLB", "FROG", "ESTC", "TEAM", "WDAY", "DT", "PATH", "TWLO", "HUBS", "VEEV", "CRM", "ADBE", "INTU"],
    "Cybersecurity": ["CRWD", "PANW", "ZS", "S", "OKTA", "FTNT", "CHKP", "RPD", "TENB", "QLYS", "VRNS"],
    "Robotics & Automation": ["ISRG", "ROK", "ZBRA", "SYM", "OMCL", "CGNX", "EMR", "HON", "PTC", "ABBNY", "NDSN", "SERV"],
    "AI Power & Datacenter": ["VST", "CEG", "NRG", "TLN", "GEV", "PWR", "ETN", "VRT", "SMR", "OKLO", "BWXT", "NNE", "LEU", "CEG"],
    "Datacenter REITs": ["EQIX", "DLR", "AMT", "CCI", "IRM"],
    "Optics & Photonics": ["COHR", "LITE", "FN", "AAOI", "POET", "CIEN", "IPGP", "NVMI"],
    "Quantum": ["IONQ", "RGTI", "QBTS", "QUBT", "ARQQ", "INFQ", "LAES"],
    "Space & Defense Tech": ["RDW", "RKLB", "ASTS", "LUNR", "PL", "LMT", "NOC", "RTX", "GD", "LHX", "AVAV", "KTOS", "LDOS", "BAH"],
    "Biotech": ["VRTX", "REGN", "AMGN", "GILD", "BIIB", "MRNA", "ALNY", "BMRN", "INCY", "NBIX", "SRPT", "IONS", "EXEL", "UTHR"],
    "AI Healthcare & Diagnostics": ["TEM", "CAI", "VCYT", "WGS", "NEO", "PSNL", "RXRX", "SDGR", "ABSI", "RLAY", "CERT", "GH"],
    "Medical Devices": ["BSX", "MDT", "SYK", "ABT", "EW", "DXCM", "PODD", "BFLY"],
    "Financials": ["JPM", "BAC", "GS", "MS", "WFC", "C", "SCHW", "BLK", "BX", "KKR", "APO", "AXP", "COF", "USB"],
    "Fintech & Payments": ["V", "MA", "PYPL", "FIS", "GPN", "AFRM", "TOST", "SOFI", "HOOD", "NU"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "VLO", "HAL", "DVN", "FANG", "WMB", "KMI"],
    "Defensives": ["PG", "KO", "PEP", "WMT", "COST", "MCD", "JNJ", "CL", "KMB", "GIS", "MDLZ", "MO"],
    "Consumer Momentum": ["AMZN", "NFLX", "SBUX", "NKE", "LULU", "CMG", "ABNB", "BKNG", "DKNG", "RH", "DECK", "ONON", "CAVA"],
    "EV & Autonomy": ["TSLA", "RIVN", "LCID", "GM", "F", "NIO", "XPEV", "LI"],
    "China Tech": ["BABA", "JD", "PDD", "BIDU", "TCEHY", "NTES", "TME", "BEKE"],
    "Industrials": ["CAT", "DE", "GE", "MMM", "UNP", "UPS", "PH", "CMI", "ITW", "ETN"],
    "Materials & Uranium": ["FCX", "NEM", "LIN", "APD", "CCJ", "UEC", "DNN", "NXE"],
    "Crypto-linked": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT"],
    "Mega-cap Platforms": ["AAPL", "GOOGL", "META", "MSFT", "AMZN", "NVDA"],
}

# ---- Universe source -------------------------------------------------------
# THEMES above is the built-in fallback. If SHEET_CSV_URL is set (env var or the
# literal below), the universe is loaded from a published Google Sheet CSV with
# columns: ticker, theme. Edit the sheet -> next ingest picks it up. If the sheet
# is unreachable or malformed, we fall back to the built-in THEMES so the board
# never breaks.
SHEET_CSV_URL = os.environ.get("SHEET_CSV_URL", "").strip()

def load_universe_from_sheet(url):
    import urllib.request, csv, io
    req = urllib.request.Request(url, headers={"User-Agent": "market-ingest"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read().decode("utf-8", "ignore")
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        raise ValueError("sheet empty")
    # detect header
    hdr = [c.strip().lower() for c in rows[0]]
    ti = hdr.index("ticker") if "ticker" in hdr else 0
    hi = hdr.index("theme") if "theme" in hdr else 1
    start_row = 1 if ("ticker" in hdr or "theme" in hdr) else 0
    themes = {}
    for row in rows[start_row:]:
        if len(row) <= max(ti, hi):
            continue
        tk = row[ti].strip().upper()
        th = row[hi].strip()
        if not tk or not th:
            continue
        themes.setdefault(th, [])
        if tk not in themes[th]:
            themes[th].append(tk)
    total = len({t for v in themes.values() for t in v})
    if total < 10:
        raise ValueError(f"sheet only yielded {total} tickers -- refusing, using fallback")
    return themes

_THEMES_CACHE = {"v": None}

def active_themes():
    """THEMES to use this run: sheet if configured & healthy, else built-in fallback.
    CACHED for the process: this is called from universe() which is called per
    replay date -- without the cache a 1,650-day replay issued 1,650 HTTP
    requests to the published sheet."""
    if _THEMES_CACHE["v"] is not None:
        return _THEMES_CACHE["v"]
    if SHEET_CSV_URL:
        try:
            th = load_universe_from_sheet(SHEET_CSV_URL)
            print(f"[universe] loaded {sum(len(v) for v in th.values())} entries from sheet")
            _THEMES_CACHE["v"] = th
            return th
        except Exception as e:
            print(f"[universe] sheet load failed ({e}); using built-in THEMES")
    _THEMES_CACHE["v"] = THEMES
    return THEMES


def universe():
    return sorted(set([s for m in active_themes().values() for s in m] + INDEX_SYMS + SECTOR_SYMS + MACRO_SYMS + MACRO_STRUCT_SYMS))

# ------------------------------------------------------------------ storage
def init_db(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS prices(
        ticker TEXT, d TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY(ticker, d))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS theme_scores(
        d TEXT, theme TEXT, score REAL, PRIMARY KEY(d, theme))""")
    conn.commit()

def upsert(conn, ticker, df):
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    rows = [(ticker, ts.strftime("%Y-%m-%d"), float(r.open), float(r.high),
             float(r.low), float(r.close), float(r.volume))
            for ts, r in df.iterrows()]
    conn.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)", rows)

def load_prices(conn, ticker):
    df = pd.read_sql("SELECT d, open, high, low, close, volume FROM prices "
                     "WHERE ticker=? ORDER BY d", conn, params=(ticker,))
    if df.empty:
        return df
    df["d"] = pd.to_datetime(df["d"])
    return df.set_index("d")

# ------------------------------------------------------------------ feeds
def fetch_yf(tickers, period):
    import yfinance as yf, time
    raw = None
    for attempt, wait in enumerate([0, 2, 5, 10]):        # 3 retries: 2s/5s/10s
        if wait:
            time.sleep(wait)
        try:
            raw = yf.download(tickers, period=period, auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
            if raw is not None and not raw.empty:
                break
        except Exception as e:
            print(f"[ingest] yfinance attempt {attempt + 1} failed: {e}")
    if raw is None or raw.empty:
        print("[ingest] yfinance unavailable after retries; relying on Stooq fallback")
        return {}
    out = {}
    for t in tickers:
        try:
            df = raw[t] if len(tickers) > 1 else raw
            df = df.dropna()
            if not df.empty:
                out[t] = df
        except Exception:
            pass
    return out

def fetch_stooq(ticker):
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        df = pd.read_csv(url)
        if "Date" not in df.columns or df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None

def ingest(conn, period):
    tickers = universe()
    frames = fetch_yf(tickers, period)
    missing = [t for t in tickers if t not in frames]
    for t in missing:                       # Stooq fallback for anything Yahoo dropped
        df = fetch_stooq(t)
        if df is not None and not df.empty:
            frames[t] = df
    for t, df in frames.items():
        upsert(conn, t, df)
    conn.commit()
    if missing:
        still = [t for t in missing if t not in frames]
        print(f"[ingest] yfinance missed {len(missing)}, Stooq recovered "
              f"{len(missing) - len(still)}" + (f", still missing: {still}" if still else ""))
    return set(frames)


def fetch_earnings(tickers):
    """Best-effort next-earnings date per ticker via yfinance. Network; skipped in selftest."""
    import yfinance as yf
    import datetime as _dt
    out = {}
    today = _dt.date.today()
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            ed = None
            if isinstance(cal, dict):
                v = cal.get("Earnings Date")
                if isinstance(v, (list, tuple)) and v:
                    ed = v[0]
                elif v:
                    ed = v
            if ed is not None:
                d = ed.date() if hasattr(ed, "date") else ed
                if isinstance(d, _dt.date) and d >= today:
                    out[t] = d.strftime("%Y-%m-%d")
        except Exception:
            pass
    return out

# ------------------------------------------------------------------ metrics
def _v(x):                                  # valid (non-NaN) number?
    return x is not None and x == x

def ticker_metrics(df):
    if df is None or len(df) < 21:
        return None
    c, h, l = df["close"], df["high"], df["low"]
    close = float(c.iloc[-1])
    sma20  = c.rolling(20).mean().iloc[-1]
    sma50  = c.rolling(50).mean().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    ret = lambda n: (close / c.iloc[-1 - n] - 1) * 100 if len(c) > n else np.nan
    v = df["volume"]
    v20 = v.rolling(20).mean().iloc[-1]
    rvol = float(v.iloc[-1] / v20) if _v(v20) and v20 else np.nan
    return {
        "close": close,
        "above20":  bool(close > sma20)  if _v(sma20)  else None,
        "above50":  bool(close > sma50)  if _v(sma50)  else None,
        "above200": bool(close > sma200) if _v(sma200) else None,
        "ret1": ret(1), "ret5": ret(5), "ret21": ret(21), "ret63": ret(63),
        "rvol": rvol,
        "dist50": (close / sma50 - 1) * 100 if _v(sma50) else np.nan,
        "atr_ext": (close - sma50) / atr if _v(atr) and atr and _v(sma50) else np.nan,
        "new20high": bool(close >= c.rolling(20).max().iloc[-1]),
        "new20low":  bool(close <= c.rolling(20).min().iloc[-1]),
        "new52high": bool(close >= c.rolling(252).max().iloc[-1]) if len(c) >= 252 else False,
        "up3": bool(_v(ret(1)) and ret(1) >= 3),
        "down3": bool(_v(ret(1)) and ret(1) <= -3),
    }

def breadth(metrics):
    mem = [m for t, m in metrics.items() if t not in NON_MEMBERS and m]
    n = len(mem)
    def pct(k):
        vals = [m[k] for m in mem if m[k] is not None]
        return 100 * sum(vals) / len(vals) if vals else 0.0
    return {
        "n": n, "pct20": pct("above20"), "pct50": pct("above50"), "pct200": pct("above200"),
        "nh20": sum(m["new20high"] for m in mem), "nl20": sum(m["new20low"] for m in mem),
        "nh52": sum(m["new52high"] for m in mem),
        "up3": sum(m["up3"] for m in mem), "down3": sum(m["down3"] for m in mem),
    }

def regime_label(b):
    if b["pct50"] < 40 and b["nl20"] > b["nh20"]:
        return "RISK-OFF"
    if b["pct50"] > 60 and b["nh20"] > b["nl20"]:
        return "RISK-ON"
    return "NEUTRAL"

def theme_score(members, metrics, spy_ret21):
    ms = [metrics[t] for t in members if t in metrics and metrics[t]]
    if not ms:
        return None
    breadth50 = 100 * np.mean([1 if m["above50"] else 0 for m in ms])
    rs = 100 * np.mean([1 if (_v(m["ret21"]) and m["ret21"] > spy_ret21) else 0 for m in ms])
    avg21 = np.nanmean([m["ret21"] for m in ms])
    mom = float(np.clip(50 + 2.5 * (avg21 if _v(avg21) else 0), 0, 100))
    return int(round(float(np.clip(0.4 * breadth50 + 0.3 * rs + 0.3 * mom, 0, 100))))

def status(score, delta):
    if score >= 75:      return "DOMINANT"
    if score >= 58:      return "STRONG"
    if delta <= -8:      return "DETERIORATING"
    if score < 25 and delta < 0: return "FADING"
    if score < 40:       return "WEAK"
    return "NEUTRAL"

# ------------------------------------------------------------------ assemble

# ------------------------------------------------------------------ macro risk engine
def _close_series(conn, ticker, as_of=None):
    df = load_prices(conn, ticker)
    if df is None or df.empty:
        return None
    c = df["close"]
    return c if as_of is None else c[c.index <= pd.Timestamp(as_of)]

def _basket_series(conn, tickers, as_of=None):
    """Equal-weight, base-100 normalised composite of several tickers. Used for
    basket-vs-basket ratios where no single ETF represents the concept."""
    cols = []
    for t in tickers:
        c = _close_series(conn, t, as_of)
        if c is not None and len(c) > 5:
            cols.append(c / c.iloc[0] * 100.0)
    if not cols:
        return None
    df = pd.concat(cols, axis=1, join="inner").dropna()
    return None if df.empty else df.mean(axis=1)

def _ratio_signal(conn, num, den, inverted=False, as_of=None):
    """5/21 EMA crossover on a date-aligned price ratio. Returns dict or None.
    Date alignment matters: BTC-USD trades 7d/wk vs 5 for equities, so we inner-
    join on common dates before computing the ratio and its EMAs."""
    a = _basket_series(conn, num, as_of) if isinstance(num, (list, tuple)) else _close_series(conn, num, as_of)
    b = _basket_series(conn, den, as_of) if isinstance(den, (list, tuple)) else _close_series(conn, den, as_of)
    if a is None or b is None:
        return None
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < 30 or (j.iloc[:, 1] == 0).any():
        return None
    r = j.iloc[:, 0] / j.iloc[:, 1]
    e5, e21 = r.ewm(span=5, adjust=False).mean(), r.ewm(span=21, adjust=False).mean()
    up = bool(e5.iloc[-1] > e21.iloc[-1])
    score = (1 if up else -1) * (-1 if inverted else 1)
    sign = (e5 > e21)
    flips = sign != sign.shift()
    last_flip = flips[flips].index[-1] if flips.any() else sign.index[0]
    age = int((sign.index >= last_flip).sum()) - 1

    # CONVICTION WEIGHT: a ratio barely above its 21-EMA should not count the same
    # as one decisively trending. Separation is normalized against its own recent
    # distribution (so each pair is judged on its own scale), then damped for very
    # fresh crosses which flip back often.
    sep = (e5 - e21).abs() / e21.abs().replace(0, np.nan)
    sep_now = float(sep.iloc[-1]) if _v(sep.iloc[-1]) else 0.0
    sep_ref = float(sep.tail(252).quantile(0.75)) if len(sep.dropna()) > 30 else sep_now
    mag_w = min(1.0, sep_now / sep_ref) if sep_ref and sep_ref > 0 else 0.5
    age_w = min(1.0, (age + 1) / 5.0)          # full weight once the cross is 4+ days old
    weight = max(0.25, round(mag_w * age_w, 3))
    _nm = "+".join(num) if isinstance(num, (list, tuple)) else num
    _dn = "+".join(den) if isinstance(den, (list, tuple)) else den
    return {"pair": f"{_nm}/{_dn}", "value": round(float(r.iloc[-1]), 3),
            "ema5": round(float(e5.iloc[-1]), 3), "ema21": round(float(e21.iloc[-1]), 3),
            "direction": "up" if up else "down", "score": score, "age_days": age,
            "weight": weight, "weighted_score": round(score * weight, 2)}

def _atr14(df):
    hc = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    return hc.rolling(14).mean()

def _agreement(macro_status, breadth_regime):
    """Do the two independent engines agree? Divergence is itself information."""
    if not breadth_regime:
        return None
    m = "on" if "Risk-On" in macro_status else "off" if "Risk-Off" in macro_status else "neutral"
    br = str(breadth_regime).upper()
    b = "on" if "ON" in br else "off" if "OFF" in br else "neutral"
    return m == b


def _load_hist_csv():
    """Regime history from the committed CSV. The GitHub Action runs against a
    CACHED market.db that does not contain a locally-run --replay, so the DB
    alone is not a reliable history store. The CSV travels with the repo, so
    local runs and the Action always see the same history."""
    try:
        if os.path.exists(HIST_CSV):
            df = pd.read_csv(HIST_CSV)
            out = {}
            for _, r in df.iterrows():
                spx = r["spx"] if "spx" in df.columns and pd.notna(r.get("spx")) else None
                out[str(r["d"])] = (float(r["raw"]), float(r["smooth"]), str(r["status"]),
                                    float(spx) if spx is not None else None)
            return out
    except Exception as e:
        print(f"[hist] could not read {HIST_CSV}: {e}")
    return {}

def _save_hist_csv(rows):
    try:
        pd.DataFrame([{"d": d, "raw": v[0], "smooth": v[1], "status": v[2],
                       "spx": (v[3] if len(v) > 3 else None)}
                      for d, v in sorted(rows.items())]).to_csv(HIST_CSV, index=False)
    except Exception as e:
        print(f"[hist] could not write {HIST_CSV}: {e}")


_LEAD_CACHE = {"v": None}

def _theme_leadership(conn, as_of=None):
    """Share of themes whose equal-weight basket is BEATING SPY over 21 sessions.

    Rationale (from a practitioner framing): a regime engine can say conditions
    improved while the assets you would actually buy have not moved. Requiring the
    leadership to confirm before allowing the top band is a different question
    from 'is the composite high' -- it asks whether anything is actually working.

    Cached per process: the basket series are built once from the full price
    history and then sliced by date, so a 1,650-day replay does not rebuild 23
    baskets per day (the same mistake that made breadth history unusably slow).
    """
    if _LEAD_CACHE["v"] is None:
        series = {}
        for th, syms in active_themes().items():
            b = _basket_series(conn, syms)          # full history, no as_of
            if b is not None and len(b) > 30:
                series[th] = b
        spy = _close_series(conn, "SPY")
        _LEAD_CACHE["v"] = (series, spy)
    series, spy = _LEAD_CACHE["v"]
    if not series or spy is None:
        return None
    cut = pd.Timestamp(as_of) if as_of is not None else None
    sp = spy if cut is None else spy[spy.index <= cut]
    if len(sp) < 22:
        return None
    spy_ret = float(sp.iloc[-1] / sp.iloc[-22] - 1)
    beat = tot = 0
    for th, b in series.items():
        bb = b if cut is None else b[b.index <= cut]
        if len(bb) < 22:
            continue
        tot += 1
        if float(bb.iloc[-1] / bb.iloc[-22] - 1) > spy_ret:
            beat += 1
    return round(100.0 * beat / tot, 1) if tot else None


_SIG_CACHE = {"v": None}

def _combo_signals(conn, tickers, as_of=None):
    """Reproduce homily_combo.pine's graded dot logic in Python so the Screen tab
    can show where the DASHBOARD and the CHART agree.

    Mirrors the live indicator settings: Fast MACD 7/16/6 on the chart timeframe
    (the config that replaced the lower-TF intrabar approach), with the same 0-3
    conviction score -- regime, cross magnitude vs ATR, position vs the 20-EMA
    basis.

    HONEST CONTEXT: these dots backtested at PF 1.26 and return-per-drawdown 0.62
    versus buy-and-hold's 28. Agreement between this and the screen means two of
    your tools point the same way; it does not mean two validated edges align.

    KNOWN COST: the logic now exists in two places (Pine and here) and can drift.
    If the indicator's lengths change, change them here too.
    """
    if _SIG_CACHE["v"] is not None and as_of is None:
        return _SIG_CACHE["v"]
    out = {}
    for t in tickers:
        df = load_prices(conn, t)
        if df is None or len(df) < 220:
            continue
        if as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of)]
            if len(df) < 220:
                continue
        c = df["close"]
        fm = c.ewm(span=7, adjust=False).mean() - c.ewm(span=16, adjust=False).mean()
        fs = fm.rolling(6).mean()
        if len(fm) < 3 or pd.isna(fs.iloc[-1]) or pd.isna(fs.iloc[-2]):
            continue
        up_now, up_prev = fm.iloc[-1] > fs.iloc[-1], fm.iloc[-2] > fs.iloc[-2]
        atr = _atr14(df).iloc[-1]
        ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
        ema21 = c.ewm(span=21, adjust=False).mean().iloc[-1]
        e200 = c.ewm(span=200, adjust=False).mean()
        px = float(c.iloc[-1])

        buy_regime = px > e200.iloc[-1] and e200.iloc[-1] >= e200.iloc[-11]
        sell_regime = px < ema21 or e200.iloc[-1] < e200.iloc[-11]
        cross_mag = abs(float(fm.iloc[-1])) >= 0.60 * float(atr) if _v(atr) else False
        buy_pos  = px <= ema20 + 0.5 * float(atr) if _v(atr) else False
        sell_pos = px >= ema20 + 1.5 * float(atr) if _v(atr) else False

        buy_score  = int(buy_regime) + int(cross_mag) + int(buy_pos)
        sell_score = int(sell_regime) + int(cross_mag) + int(sell_pos)

        # bars since the most recent cross, so a stale signal is visible as stale
        sign = (fm > fs)
        flips = sign != sign.shift()
        age = 0
        idx = list(sign.index)
        for i in range(len(idx) - 1, 0, -1):
            if bool(flips.iloc[i]):
                age = len(idx) - 1 - i
                break
        out[t] = {"state": "up" if up_now else "down",
                  "fresh_buy":  bool(up_now and not up_prev),
                  "fresh_sell": bool((not up_now) and up_prev),
                  "buy_conv": buy_score, "sell_conv": sell_score,
                  "cross_age": age}
    if as_of is None:
        _SIG_CACHE["v"] = out
    return out

def build_macro_structure(conn, breadth_pct200, breadth_pct50=None, breadth_regime=None, as_of=None):
    import datetime as _dt
    RATIOS = [("RSP", "SPY", False, "Market Breadth"),
              ("XLY", "XLP", False, "Consumer Demand"),
              ("HYG", "IEI", False, "Credit Spreads"),
              ("IWM", "SPY", False, "Small Cap Appetite"),
              ("BTC-USD", "GLD", False, "Digital Risk vs Safety"),
              ("CPER", "GLD", False, "Industrial Demand"),
              ("UUP", "SPY", True,  "US Dollar vs Equities")]
    # TESTED AND REMOVED: a cyclical-vs-defensive basket ratio
    #   (["XLK","XLY","XLI","XLF"] / ["XLP","XLU","XLV"])
    # Rationale was sound -- rotation into defensives as regime confirmation --
    # and it DID improve entries (Aug 2024 8/07->8/06, Nov 2025 11/20->11/18,
    # deeper scores at turns). But it crowds: it reads the same "are cyclicals
    # leading" dimension as XLY/XLP and RSP/SPY, so it amplified rather than
    # added. Regime changes 84->95, Full Risk-On days 95->179, the 2020 and late
    # 2022 recoveries fragmented, and Spring 2025 LOST its April re-entry -- the
    # April bottom would have been scored Neutral instead of Risk-Off. Not worth
    # two days of earlier warning. _basket_series() is retained for future use.
    ratios = []
    total = 0
    for num, den, inv, label in RATIOS:
        sig = _ratio_signal(conn, num, den, inv, as_of)
        if sig:
            sig["label"] = label
            sig["inverted"] = inv
            ratios.append(sig)
            total += sig["weighted_score"]

    # ---- VOLATILITY TERM STRUCTURE ----------------------------------------
    # Read the curve as a SHAPE (9D -> 30D -> 3M -> 6M) rather than one pair.
    # In calm markets the curve is upward-sloping (contango): near-dated vol is
    # cheaper than far-dated. Stress inverts it from the FRONT first, so counting
    # how many segments are inverted grades severity instead of just flagging it.
    # VVIX (vol-of-vol) falling is a separate calming signal.
    def _last(t):
        c = _close_series(conn, t, as_of)
        return round(float(c.iloc[-1]), 3) if c is not None and len(c) else None

    v9, vix_close, vix3_close, v6m = _last("^VIX9D"), _last("^VIX"), _last("^VIX3M"), _last("^VIX6M")
    vvix_s = _close_series(conn, "^VVIX", as_of)
    vvix_close = round(float(vvix_s.iloc[-1]), 3) if vvix_s is not None and len(vvix_s) else None
    vvix_10d = (round(float(vvix_s.iloc[-11]), 3)
                if vvix_s is not None and len(vvix_s) > 11 else None)

    # CURVE SPREAD, and critically its DIRECTION. The 7y replay showed the vol
    # component pinned at -2 for entire recoveries: COVID stayed backwardated
    # 2/24 through 4/23 while price rallied 28% off the 3/23 bottom. Realized vol
    # stays elevated after a low, so the curve's LEVEL is a lagging indicator on
    # the way out. Its CHANGE is not -- the spread starts normalizing right at
    # the bottom, which is exactly when the level still says "stress".
    spread_now = None
    spread_5d = None
    try:
        _v9s = _close_series(conn, "^VIX9D", as_of)
        _v3s = _close_series(conn, "^VIX3M", as_of)
        if _v9s is not None and _v3s is not None:
            _j = pd.concat([_v9s, _v3s], axis=1, join="inner").dropna()
            if len(_j) > 6:
                _sp = _j.iloc[:, 0] - _j.iloc[:, 1]      # 9D minus 3M; >0 = inverted
                spread_now = round(float(_sp.iloc[-1]), 3)
                spread_5d = round(float(_sp.iloc[-6]), 3)
    except Exception:
        pass
    # improving = still inverted but the spread is narrowing meaningfully
    curve_improving = (spread_now is not None and spread_5d is not None
                       and spread_now > 0 and spread_now < spread_5d - 1.0)

    # each adjacent pair: True when in contango (healthy)
    segs = []
    for a, b, lbl in ((v9, vix_close, "9D<30D"), (vix_close, vix3_close, "30D<3M"),
                      (vix3_close, v6m, "3M<6M")):
        segs.append({"pair": lbl, "ok": (a is not None and b is not None and a < b),
                     "known": a is not None and b is not None})
    known = [x for x in segs if x["known"]]
    inverted = sum(1 for x in known if not x["ok"])
    backwardated = inverted > 0

    if not known:
        curve = "unknown"
        vix_score = 0
    else:
        # front-end inversion is the earliest and most reliable stress tell
        front_inv = (not segs[0]["ok"]) and segs[0]["known"]
        if inverted == 0:
            curve = "contango"
        elif inverted >= 2:
            curve = "backwardated"
        else:
            curve = "front-inverted" if front_inv else "flat"

        if curve == "backwardated":
            vix_score = -2          # multiple segments inverted = severe stress
        elif curve in ("front-inverted", "flat"):
            vix_score = -1
        else:
            # curve healthy -> fall back to the absolute level.
            # CALM VOL IS NOT SCORED BULLISH. The --vixtest hypothesis test came
            # back 4/4: forward SPX returns after STRESS readings beat those after
            # CALM readings in every window, monotonically, by +1.9pp at 21d and
            # +5.2pp at 63d out-of-sample ("volatility paradox" -- suppressed vol
            # invites leverage, which precedes poor returns). Calm scores 0, NOT
            # -1: full inversion is a larger claim than 4/4 on ~78 effective
            # observations supports, and this is a CONCURRENT classifier.
            vix_score = -1 if (vix_close is not None and vix_close > 20.0) else 0
            # vol-of-vol falling only offsets an elevated-level penalty; it never
            # adds a positive.
            if (vix_score < 0 and vvix_close is not None and vvix_10d is not None
                    and vvix_close < vvix_10d):
                vix_score = 0

        # DECAY THE PENALTY WHEN THE CURVE IS REPAIRING. Applied AFTER the chain
        # above so every branch has assigned vix_score first. Still inverted, but
        # the 9D-3M spread has narrowed >1pt over 5 sessions -- stress receding.
        # Halved rather than cleared: the curve is still inverted, so a full
        # reprieve would overstate the improvement.
        if vix_score < 0 and curve_improving:
            vix_score = vix_score / 2.0

    total += vix_score

    # ---- ABSOLUTE components (the fix for rotation masquerading as risk appetite) ----
    # Four of the seven ratios are equity-vs-equity and are scale-invariant to the
    # market's direction: RSP/SPY can rise during a selloff. These two inputs anchor
    # the score to whether the market is actually going UP.
    spy_s = _close_series(conn, "SPY", as_of)
    trend_score = 0
    spy_vs_200 = None
    if spy_s is not None and len(spy_s) > 200:
        sma200 = spy_s.rolling(200).mean().iloc[-1]
        spy_vs_200 = round(float((spy_s.iloc[-1] / sma200 - 1) * 100), 2)
        trend_score = 1 if spy_s.iloc[-1] > sma200 else -1
    total += trend_score

    breadth_score = 0
    if breadth_pct200 is not None:
        breadth_score = 1 if breadth_pct200 > 50 else -1
    total += breadth_score

    total = round(total, 2)
    # CALIBRATED BANDS. The naive mapping (+1 = "Moderate Risk-On") was far too
    # generous once conviction weighting shrank typical ratio contributions: a
    # score of +1 out of 10 is noise, not risk appetite. The absolute components
    # alone contribute +2 whenever SPY is above its 200-SMA with >50% breadth --
    # i.e. any non-bear tape -- so the risk-on threshold must sit ABOVE that floor
    # to mean anything. Requiring +3.5 forces the intermarket engine to actually
    # agree before the label turns bullish.
    # BANDS. +7 for Full Risk-On was empirically unreachable (0 of 300 replayed
    # days); the observed max was +6.70. Lowered to +5.0, which fires ~10% of the
    # time. Moderate stays at +3.5 -- it sits just above the +2 floor the absolute
    # components contribute in any non-bear tape, which is an independent reason
    # rather than a curve-fit. Full Risk-Off stays at -3.0: the percentile fit
    # suggested -1.7, but that was derived from a 14-month BULL sample where deep
    # negatives are rare, and adopting it would flag full risk-off on ordinary
    # pullbacks. -3.0 fired on 5% of days and caught the genuine stress episodes.
    B_FULL_ON, B_MOD_ON, B_FULL_OFF = 5.0, 3.5, -3.0

    # SMOOTHED REGIME. The raw score flips across a band boundary on decimal noise,
    # which made the daily label alternate green/yellow day to day (visible as
    # striping when overlaid on SPX). The LABEL is driven by a short EMA of the
    # score instead; the raw score stays visible for the daily read. Only PRIOR
    # dates are used, so this is lookahead-free during replay.
    # ASYMMETRIC SMOOTHING. A single span forces one trade-off: fast enough to
    # catch deterioration early, or slow enough to stop the label striping. We do
    # not need symmetry -- for a drawdown-first mandate the correct bias is FAST
    # DOWN / SLOW UP: react quickly when risk is rising, re-engage reluctantly.
    # Symmetric span=5 delayed the Feb-2025 risk-off warning by about a week;
    # span=3 on the way down recovers most of that, while span=8 on the way up
    # keeps the regime blocks readable and deliberately slows re-entry.
    # SPAN_UP was 8. The 7y replay showed the failure is EXITS, not entries:
    # COVID exited 4/27 after price had already rallied 28% off the 3/23 low;
    # Spring 2025 exited 5/05 after +13%; Aug 2024 spent its entire risk-off
    # window in a rally. Cause: components are structurally slow to RECOVER
    # (backwardation persists after a bottom, breadth-above-200DMA cannot repair
    # quickly), and span-8 up-smoothing plus a 10-session dwell floor put two
    # brakes on the same wheel. Env-overridable so the effect can be measured.
    SPAN_DOWN = int(os.environ.get("SPAN_DOWN", "3"))
    SPAN_UP   = int(os.environ.get("SPAN_UP", "8"))
    SMOOTH_SPAN = SPAN_DOWN  # reported; the active span depends on direction
    raw_score = total
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS macro_scores (d TEXT PRIMARY KEY, score REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS macro_smooth (d TEXT PRIMARY KEY, s REAL)")
        cutoff = (pd.Timestamp(as_of).date().isoformat() if as_of else _session_date(conn))
        prev_sm = conn.execute(
            "SELECT s FROM macro_smooth WHERE d < ? ORDER BY d DESC LIMIT 1", (cutoff,)).fetchone()
        if prev_sm is None:
            smooth = float(total)
        else:
            prev = float(prev_sm[0])
            span = SPAN_DOWN if float(total) < prev else SPAN_UP
            alpha = 2.0 / (span + 1.0)
            smooth = prev + alpha * (float(total) - prev)
        conn.execute("INSERT OR REPLACE INTO macro_smooth VALUES (?,?)", (cutoff, float(smooth)))
        conn.commit()
    except Exception:
        smooth = float(total)
    total = round(smooth, 2)

    # HYSTERESIS: the score must clear a threshold by `hyst` to ENTER a regime,
    # but only fall back through it to EXIT. Without this the label flips on
    # decimal noise around a boundary (35 changes in 300 days).
    # PER-BAND HYSTERESIS. The score spends a lot of time near +5.0, so a single
    # narrow buffer produced repeated Full<->Moderate flips (July 2025 flipped 4x
    # in 3 weeks). A wider buffer at the top band kills that churn; the risk-off
    # boundary keeps a narrow buffer so defensive signals stay responsive.
    HYST = {"Full Risk-On": 0.8, "Moderate Risk-On": 0.4, "Full Risk-Off": 0.3}
    hyst = HYST["Moderate Risk-On"]   # reported default
    prev_lab = None
    prev_run = 0            # consecutive sessions the previous label has held
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS macro_hist "
                     "(d TEXT PRIMARY KEY, raw REAL, smooth REAL, status TEXT)")
        _cut = (pd.Timestamp(as_of).date().isoformat() if as_of else _session_date(conn))
        _hrows = conn.execute(
            "SELECT status FROM macro_hist WHERE d < ? ORDER BY d DESC LIMIT 60", (_cut,)).fetchall()
        if _hrows:
            prev_lab = _hrows[0][0]
            for (_st,) in _hrows:
                if _st == prev_lab:
                    prev_run += 1
                else:
                    break
        else:
            _r = conn.execute("SELECT v FROM macro_state WHERE k='label'").fetchone()
            prev_lab = _r[0] if _r else None
    except Exception:
        pass

    def _band(x, entering_from=None):
        if x >= B_FULL_ON:    return "Full Risk-On"
        if x >= B_MOD_ON:     return "Moderate Risk-On"
        if x > B_FULL_OFF:    return "Neutral / Choppy"
        return "Full Risk-Off"

    # MINIMUM DWELL. 2022 exposed the real failure mode: in a grinding decline the
    # score oscillates around -3.0 and the label flipped in/out of Full Risk-Off
    # ~14 times in a year (three times in eight days in Feb 2022). Widening the
    # hysteresis buffer cannot fix that -- the score genuinely crossed back and
    # forth by more than the buffer. A dwell floor does: once Full Risk-Off is
    # entered it holds for MIN_DWELL_OFF sessions regardless of score. This costs
    # nothing in 2020 or 2025, where every episode ran far longer than the floor.
    # NEUTRAL SLOPE. 53% of sessions land in Neutral, which is honest but not
    # actionable. Rather than narrow the band (which would overclaim conviction),
    # split it by the score's own direction over ~5 sessions. No predictive claim
    # is made -- it only reports whether conditions are getting better or worse.
    slope = None
    try:
        _cut2 = (pd.Timestamp(as_of).date().isoformat() if as_of else _session_date(conn))
        _prev5 = conn.execute(
            "SELECT smooth FROM macro_hist WHERE d < ? ORDER BY d DESC LIMIT 5", (_cut2,)).fetchall()
        if len(_prev5) >= 3:
            slope = round(float(total) - float(_prev5[-1][0]), 2)
    except Exception:
        pass

    MIN_DWELL_OFF = 10
    # ---- LEADERSHIP GATE ----------------------------------------------------
    # Full Risk-On additionally requires that a real share of themes are actually
    # OUTPERFORMING SPY. Without this the top band can fire off the composite
    # alone while nothing is leading -- "the score says go but the assets you
    # would buy have not moved". Gate applies ONLY to the top band; Moderate and
    # Risk-Off are untouched, so this cannot make the engine slower to defend.
    LEAD_MIN = float(os.environ.get("LEAD_MIN", "40"))
    useLeadGate = os.environ.get("LEAD_GATE", "1") != "0"
    leadership = None
    try:
        leadership = _theme_leadership(conn, as_of)
    except Exception:
        pass

    raw_status = _band(total)
    if (useLeadGate and raw_status == "Full Risk-On"
            and leadership is not None and leadership < LEAD_MIN):
        raw_status = "Moderate Risk-On"
    status = raw_status
    dwell_held = False
    if str(prev_lab) == "Full Risk-Off" and raw_status != "Full Risk-Off" and prev_run < MIN_DWELL_OFF:
        status = "Full Risk-Off"
        dwell_held = True

    # ---- PRICE RECOVERY OVERRIDE (LATCHED) ---------------------------------
    # Exits from Risk-Off are structurally late: the composite stays deep-negative
    # through recoveries because credit/breadth/vol repair AFTER price does. That
    # is what a V-bottom is, so no reweighting of those inputs fixes it. This adds
    # price -- a different information type that recovers first.
    #
    # LATCHED, and that matters. A stateless version oscillated 8 times in two
    # weeks during COVID: it released to Neutral, then next bar prev_lab was no
    # longer "Full Risk-Off" so the override stopped applying, the still-negative
    # composite forced Risk-Off again, and the cycle repeated. The latch persists
    # the release until price actually breaks back down.
    #
    # Releases a defensive stance only; it never initiates risk-on.
    PRICE_RECOVERY_PCT = float(os.environ.get("PRICE_RECOVERY_PCT", "8.0"))
    RECOVERY_LOOKBACK  = int(os.environ.get("RECOVERY_LOOKBACK", "20"))
    price_override = False
    recovery_pct = None
    _ov = None
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS macro_override (k TEXT PRIMARY KEY, v TEXT)")
        _r = conn.execute("SELECT v FROM macro_override WHERE k='latch'").fetchone()
        _ov = json.loads(_r[0]) if _r else None
    except Exception:
        pass

    try:
        _sp = _close_series(conn, "SPY", as_of)
        if _sp is not None and len(_sp) > RECOVERY_LOOKBACK:
            _win = _sp.iloc[-RECOVERY_LOOKBACK:]
            _low = float(_win.min())
            _now = float(_sp.iloc[-1])
            _bars_since_low = len(_win) - 1 - int(_win.values.argmin())
            if _low > 0:
                recovery_pct = round((_now / _low - 1) * 100, 2)

            # ---- latch already active: hold the release, or break it ----
            if _ov and _ov.get("active"):
                _trig_low = float(_ov.get("low", _low))
                if _now < _trig_low:
                    _ov = None                      # price undercut the low -> latch broken
                elif raw_status == "Full Risk-Off":
                    status = "Neutral / Choppy"     # keep holding the release
                    price_override = True
                    dwell_held = False
                else:
                    _ov = None                      # composite recovered on its own

            # ---- not latched: can we release? ----
            elif (str(prev_lab) == "Full Risk-Off" and status == "Full Risk-Off"
                  and recovery_pct is not None and recovery_pct >= PRICE_RECOVERY_PCT
                  and _bars_since_low >= 3 and vix_score > -2):
                status = "Neutral / Choppy"
                price_override = True
                dwell_held = False
                _ov = {"active": True, "low": _low}

        conn.execute("DELETE FROM macro_override WHERE k='latch'")
        if _ov and _ov.get("active"):
            conn.execute("INSERT OR REPLACE INTO macro_override VALUES ('latch', ?)",
                         (json.dumps(_ov),))
        conn.commit()
    except Exception:
        pass
    prev_lab = ("Neutral / Choppy" if str(prev_lab).startswith("Neutral") else prev_lab)
    if (not dwell_held) and prev_lab and raw_status != prev_lab:
        h_on   = HYST["Full Risk-On"]
        h_mod  = HYST["Moderate Risk-On"]
        h_off  = HYST["Full Risk-Off"]
        # entering a regime requires clearing its threshold by that band's buffer
        if raw_status == "Full Risk-On"       and total < B_FULL_ON + h_on:   status = prev_lab
        elif raw_status == "Moderate Risk-On":
            # dropping OUT of Full Risk-On also needs the wide buffer, so the
            # top band does not oscillate on decimal moves
            if prev_lab == "Full Risk-On" and total > B_FULL_ON - h_on:
                status = prev_lab
            elif total < B_MOD_ON + h_mod:
                status = prev_lab
        elif raw_status == "Full Risk-Off"    and total > B_FULL_OFF - h_off: status = prev_lab
        elif raw_status == "Neutral / Choppy":
            if prev_lab in ("Full Risk-On", "Moderate Risk-On") and total > B_MOD_ON - h_mod:
                status = prev_lab
            elif prev_lab == "Full Risk-Off" and total < B_FULL_OFF + h_off:
                status = prev_lab

    # Slope is reported as a SEPARATE readout, never folded into the label.
    # Baking it in produced ~230 regime changes (from 96) because Improving and
    # Deteriorating flip against each other every few sessions. As an adjacent
    # indicator it cannot churn the regime series at all, and it still answers
    # "are conditions getting better or worse inside Neutral?".
    neutral_dir = None
    if slope is not None:
        if slope >= 0.35:
            neutral_dir = "improving"
        elif slope <= -0.35:
            neutral_dir = "deteriorating"
        else:
            neutral_dir = "flat"

    # expose the split so the UI can show what is actually driving the score
    ratio_component = round(total - vix_score - trend_score - breadth_score, 2)

    # ---- asset scorecards (Sec.4) ----
    def _checks(ticker, checks):
        passed = [{"label": lbl, "pass": bool(ok)} for lbl, ok in checks]
        n = sum(1 for c in passed if c["pass"])
        return {"ticker": ticker, "score": f"{n}/3", "n": n, "checks": passed}

    def _sma(sr, n): return sr.rolling(n).mean()
    def _ema(sr, n): return sr.ewm(span=n, adjust=False).mean()
    def _ret(sr, n): return (sr.iloc[-1] / sr.iloc[-1 - n] - 1) * 100 if len(sr) > n else None

    spy = _close_series(conn, "SPY", as_of); qqq = _close_series(conn, "QQQ", as_of)
    iwm = _close_series(conn, "IWM", as_of); btc = _close_series(conn, "BTC-USD", as_of)
    spy_r21 = _ret(spy, 21) if spy is not None else None
    r_rsp = next((r for r in ratios if r["pair"] == "RSP/SPY"), None)
    r_hyg = next((r for r in ratios if r["pair"] == "HYG/IEI"), None)
    r_cop = next((r for r in ratios if r["pair"] == "CPER/GLD"), None)

    engines = []
    if spy is not None and len(spy) > 200:
        engines.append(_checks("SPX", [
            ("Close > 200d SMA", spy.iloc[-1] > _sma(spy, 200).iloc[-1]),
            (">50% universe > 200d SMA", (breadth_pct200 or 0) > 50),
            ("RSP/SPY 5>21 EMA", bool(r_rsp and r_rsp["direction"] == "up"))]))
    if qqq is not None and len(qqq) > 60:
        qdf = load_prices(conn, "QQQ")
        qdf = qdf if as_of is None else qdf[qdf.index <= pd.Timestamp(as_of)]
        qatr = _atr14(qdf).iloc[-1]
        q21r = _ret(qqq, 21)
        engines.append(_checks("QQQ", [
            ("Close > 21d EMA", qqq.iloc[-1] > _ema(qqq, 21).iloc[-1]),
            ("21d return > SPY", q21r is not None and spy_r21 is not None and q21r > spy_r21),
            ("Not stretched (<=50SMA+3ATR)", qqq.iloc[-1] <= _sma(qqq, 50).iloc[-1] + 3 * qatr)]))
    if iwm is not None and len(iwm) > 60:
        i21r = _ret(iwm, 21)
        engines.append(_checks("IWM", [
            ("Close > 50d SMA", iwm.iloc[-1] > _sma(iwm, 50).iloc[-1]),
            ("HYG/IEI 5>21 EMA", bool(r_hyg and r_hyg["direction"] == "up")),
            ("21d return > SPY", i21r is not None and spy_r21 is not None and i21r > spy_r21)]))
    if btc is not None and len(btc) > 60:
        bdf = load_prices(conn, "BTC-USD")
        bdf = bdf if as_of is None else bdf[bdf.index <= pd.Timestamp(as_of)]
        batr = _atr14(bdf).iloc[-1]
        engines.append(_checks("BTC", [
            ("Close > 21d EMA", btc.iloc[-1] > _ema(btc, 21).iloc[-1]),
            ("Vol contained (ATR/px < 5%)", (batr / btc.iloc[-1]) < 0.05 if btc.iloc[-1] else False),
            ("CPER/GLD 5>21 EMA", bool(r_cop and r_cop["direction"] == "up"))]))

    # ---- relative performance spreads vs SPY (Sec.5; NOTE: momentum, not valuation) ----
    relval = []
    if spy is not None:
        spy21, spy63 = _ret(spy, 21), _ret(spy, 63)
        for t in ["RSP", "QQQ", "IWM", "BTC-USD"]:
            sr = _close_series(conn, t, as_of)
            if sr is None:
                continue
            a21, a63 = _ret(sr, 21), _ret(sr, 63)
            relval.append({"ticker": t,
                "spread_1m": round(a21 - spy21, 3) if a21 is not None and spy21 is not None else None,
                "spread_3m": round(a63 - spy63, 3) if a63 is not None and spy63 is not None else None})

    # ---- freshness ----
    conn.execute("CREATE TABLE IF NOT EXISTS macro_state (k TEXT PRIMARY KEY, v TEXT)")
    conn.execute("INSERT OR REPLACE INTO macro_state VALUES ('label', ?)", (status,))
    conn.execute("CREATE TABLE IF NOT EXISTS macro_hist "
                 "(d TEXT PRIMARY KEY, raw REAL, smooth REAL, status TEXT)")
    _hd = (pd.Timestamp(as_of).date().isoformat() if as_of else _session_date(conn))
    conn.execute("INSERT OR REPLACE INTO macro_hist VALUES (?,?,?,?)",
                 (_hd, float(raw_score), float(total), status))
    conn.execute("CREATE TABLE IF NOT EXISTS macro_scores (d TEXT PRIMARY KEY, score REAL)")
    today_iso = (pd.Timestamp(as_of).date().isoformat() if as_of else _session_date(conn))
    conn.execute("INSERT OR REPLACE INTO macro_scores VALUES (?,?)", (today_iso, float(total)))
    conn.commit()
    # regime history joined to SPX, for the overlay chart on the Macro tab
    regime_history = []
    try:
        merged = _load_hist_csv()
        spx_px = _close_series(conn, "SPY", as_of)
        for d_, rw, sm, st in conn.execute(
                "SELECT d, raw, smooth, status FROM macro_hist").fetchall():
            ts = pd.Timestamp(str(d_))
            px = float(spx_px.loc[ts]) if spx_px is not None and ts in spx_px.index else None
            if px is None and str(d_) in merged and len(merged[str(d_)]) > 3:
                px = merged[str(d_)][3]        # keep the price the CSV already carries
            merged[str(d_)] = (float(rw), float(sm), str(st), px)
        _save_hist_csv(merged)
        # CRITICAL: the chart reads SPX from the CSV, not the DB. The GitHub Action
        # runs against a CACHED database that may hold far less price history than a
        # local 7y backfill -- if the price came from the DB, every date older than
        # the cache would lose its SPX, get filtered out, and the chart would
        # silently shrink. Anything the front end needs must live in the repo.
        # unconfirmed exits, read from the per-date detail blob so the chart can
        # mark them historically (the runtime flag only ever described "today")
        unconf = set()
        try:
            for d_, j in conn.execute("SELECT d, j FROM macro_detail").fetchall():
                try:
                    if json.loads(j).get("exit_unconfirmed"):
                        unconf.add(str(d_))
                except Exception:
                    pass
        except Exception:
            pass

        hrows = [(d, v[0], v[1], v[2], (v[3] if len(v) > 3 else None))
                 for d, v in sorted(merged.items())][-1600:]

        # ---- PRICE / SCORE DIVERGENCE (descriptive flag, NOT a signal) --------
        # --divtest over 2020-2026: price DOWN >=2% over 10 sessions while the
        # score held flat or rose beat the base-rate forward return at every
        # horizon in both windows (21d: 3.61 vs 1.10 IS, 5.36 vs 1.67 OOS). The
        # mirror case -- price up, score unconfirmed -- showed essentially
        # nothing.
        #
        # THE SAMPLE IS FAR TOO SMALL TO TRADE: n=55 in-sample, n=20 out, and at
        # a 21-day horizon the effective independent count out-of-sample is about
        # ONE. This is emitted so observations ACCUMULATE going forward instead of
        # a conclusion being drawn from twenty. Treat it as descriptive, like the
        # unconfirmed-exit marker.
        DIV_WIN, DIV_PX, DIV_SC = 10, 2.0, 0.0
        for i, (d_, rw, sm, st, px) in enumerate(hrows):
            row = {"d": d_, "score": round(float(sm), 2),
                   "raw": round(float(rw), 2), "status": st,
                   "spx": round(float(px), 2) if px is not None else None}
            if d_ in unconf:
                row["unconfirmed_exit"] = True
            if i >= DIV_WIN:
                p0 = hrows[i - DIV_WIN][4]
                s0 = hrows[i - DIV_WIN][2]
                if p0 and px:
                    px_chg = (float(px) / float(p0) - 1) * 100.0
                    sc_chg = float(sm) - float(s0)
                    if px_chg <= -DIV_PX and sc_chg >= DIV_SC:
                        row["div_bull"] = True      # price fell, conditions did not
                    elif px_chg >= DIV_PX and sc_chg <= -DIV_SC:
                        row["div_bear"] = True      # price rose, conditions did not
            regime_history.append(row)
    except Exception as e:
        print(f"[macro] regime_history unavailable: {e}")

    # ---- historical series for charts elsewhere on the board ----
    # Breadth is DERIVED from the price matrix (not stored), so it is available
    # for the full 7y without any new storage. Theme scores come from the
    # theme_scores table, which has been accumulating since the board went live.
    # These series are only consumed by the live JSON. Computing them per replay
    # date meant rebuilding the whole breadth matrix ~1,650 times -- skip entirely
    # when as_of is set.
    breadth_history = []
    try:
        if as_of is not None:
            raise StopIteration
        _names = [t for t in universe() if t not in NON_MEMBERS]
        _pct = breadth_matrix(conn, _names)
        if _pct is not None:
            _pct = _pct.dropna()
            if as_of is not None:
                _pct = _pct[_pct.index <= pd.Timestamp(as_of)]
            for d_, v in list(_pct.items())[-1600:]:
                breadth_history.append({"d": d_.date().isoformat(), "pct200": round(float(v), 1)})
    except StopIteration:
        pass
    except Exception as e:
        print(f"[hist] breadth series unavailable: {e}")

    theme_history = {}
    try:
        if as_of is not None:
            raise StopIteration
        trows = conn.execute(
            "SELECT theme, d, score FROM theme_scores ORDER BY d").fetchall()
        for th, d_, sc in trows:
            theme_history.setdefault(th, []).append({"d": str(d_), "s": round(float(sc), 1)})
        theme_history = {k: v[-260:] for k, v in theme_history.items()}
    except StopIteration:
        pass
    except Exception as e:
        print(f"[hist] theme series unavailable: {e}")

    hist_rows = conn.execute("SELECT d, score FROM macro_scores ORDER BY d DESC LIMIT 60").fetchall()
    score_history = [{"d": d, "score": sc} for d, sc in reversed(hist_rows)]
    score_delta = round(float(total) - float(score_history[-2]["score"]), 2) if len(score_history) > 1 else 0

    last_d = conn.execute("SELECT max(d) FROM prices WHERE ticker='SPY'").fetchone()[0]
    stale = True
    if last_d:
        try:
            age = (_dt.date.today() - _dt.date.fromisoformat(str(last_d)[:10])).days
            stale = age > 4
        except Exception:
            pass

    # Exit-quality flag: leaving Full Risk-Off while the intermarket component is
    # still falling means the all-clear came from trend/breadth/vol, not from
    # cross-asset confirmation. Seen on 2025-03-21 and 2025-11-25.
    exit_unconfirmed = False
    try:
        prow = conn.execute(
            "SELECT status, raw FROM macro_hist WHERE d < ? ORDER BY d DESC LIMIT 1",
            ((pd.Timestamp(as_of).date().isoformat() if as_of else _session_date(conn)),)
        ).fetchone()
        if prow and prow[0] == "Full Risk-Off" and status != "Full Risk-Off":
            prev_im = conn.execute(
                "SELECT intermarket FROM macro_components WHERE d < ? ORDER BY d DESC LIMIT 1",
                ((pd.Timestamp(as_of).date().isoformat() if as_of else _session_date(conn)),)
            ).fetchone()
            if prev_im and ratio_component < float(prev_im[0]):
                exit_unconfirmed = True
    except Exception:
        pass
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS macro_components (d TEXT PRIMARY KEY, intermarket REAL)")
        conn.execute("INSERT OR REPLACE INTO macro_components VALUES (?,?)",
                     ((pd.Timestamp(as_of).date().isoformat() if as_of else _session_date(conn)),
                      float(ratio_component)))
        conn.commit()
    except Exception:
        pass

    _dstamp = (pd.Timestamp(as_of).date().isoformat() if as_of else _session_date(conn))
    _spx_now = None
    try:
        _sp = _close_series(conn, "SPY", as_of)
        _spx_now = round(float(_sp.iloc[-1]), 2) if _sp is not None and len(_sp) else None
    except Exception:
        pass
    _store_detail(conn, _dstamp, {
        "score": total, "raw": raw_score, "status": status, "spx": _spx_now,
        "vix": vix_score, "trend": trend_score, "breadth": breadth_score,
        "intermarket": ratio_component, "curve": curve,
        "exit_unconfirmed": bool(exit_unconfirmed),
        "vix9d": v9, "vix30d": vix_close, "vix3m": vix3_close, "vix6m": v6m, "vvix": vvix_close,
        "breadth_pct200": breadth_pct200, "spy_vs_200": spy_vs_200,
        "ratios": [{"pair": r["pair"], "value": r["value"], "direction": r["direction"],
                    "weight": r.get("weight"), "weighted_score": r.get("weighted_score")}
                   for r in ratios]})

    # ---- DEPLOYMENT GUIDE ----------------------------------------------------
    # The board described conditions but never connected them to a position size,
    # so a reading changed what you knew and not what you did. This maps regime to
    # a suggested exposure band.
    #
    # THESE NUMBERS ARE A POLICY YOU SET, NOT A MODEL OUTPUT. Nothing in the
    # testing supports a specific percentage -- the IC analysis found no forward
    # predictive power in the components. What the history DOES support is the
    # ordering: risk-off periods sat on real drawdowns, so being smaller then is
    # consistent with a drawdown-first mandate. Override via env to match your own
    # risk tolerance; the defaults are deliberately conservative.
    DEPLOY_MAP = {
        "Full Risk-On":     float(os.environ.get("DEPLOY_FULL_ON",  "100")),
        "Moderate Risk-On": float(os.environ.get("DEPLOY_MOD_ON",   "75")),
        "Neutral / Choppy": float(os.environ.get("DEPLOY_NEUTRAL",  "50")),
        "Full Risk-Off":    float(os.environ.get("DEPLOY_OFF",      "25")),
    }
    deploy_pct = DEPLOY_MAP.get(status, 50.0)
    deploy_notes = []
    if dwell_held:
        deploy_notes.append("held defensive by the dwell floor")
    if price_override:
        deploy_notes.append("risk-off released on price recovery, not on conditions")
    if exit_unconfirmed:
        deploy_notes.append("exit not confirmed by cross-asset signals")
    if leadership is not None and leadership < LEAD_MIN:
        deploy_notes.append(f"leadership thin ({leadership}% of themes beating SPY)")
    if ratio_component is not None and ratio_component < 0 and total > 0:
        deploy_notes.append("score carried by trend/breadth; cross-asset negative")

    return {
        "updated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system_health": {"data_status": "stale" if stale else "ok", "last_close": last_d},
        "global_regime": {"status": status, "score": total, "raw_score": round(raw_score, 2),
                          "smooth_span": {"down": SPAN_DOWN, "up": SPAN_UP}, "delta_1d": score_delta,
                          "min": -11, "max": 10,
                          "breadth_engine_regime": breadth_regime,
                          "exit_unconfirmed": exit_unconfirmed,
                          "deploy_pct": deploy_pct, "deploy_notes": deploy_notes,
                          "components": {"intermarket": ratio_component, "vix": vix_score,
                                         "trend": trend_score, "breadth": breadth_score},
                          "bands": {"full_on": B_FULL_ON, "moderate_on": B_MOD_ON, "full_off": B_FULL_OFF, "hysteresis": HYST},
                          "raw_status": raw_status, "dwell_held": dwell_held,
                          "leadership_pct": leadership, "lead_min": LEAD_MIN,
                          "price_override": price_override, "recovery_pct": recovery_pct,
                          "slope_5d": slope, "neutral_dir": neutral_dir,
                          "min_dwell_off": MIN_DWELL_OFF, "prev_run_days": prev_run,
                          "agrees": _agreement(status, breadth_regime)},
        "absolute_trend": {"spy_vs_200sma_pct": spy_vs_200, "trend_score": trend_score,
                           "breadth_pct200": round(breadth_pct200, 1) if breadth_pct200 is not None else None,
                           "breadth_pct50": round(breadth_pct50, 1) if breadth_pct50 is not None else None,
                           "breadth_score": breadth_score},
        "score_history": score_history,
        "regime_history": regime_history,
        "breadth_history": breadth_history,
        "theme_history": theme_history,
        "intermarket_ratios": ratios,
        "volatility": {"vix9d": v9, "vix_close": vix_close, "vix3m_close": vix3_close,
                       "vix6m": v6m, "vvix": vvix_close, "vvix_10d_ago": vvix_10d,
                       "curve": curve, "segments": segs, "inverted_segments": inverted,
                       "spread_9d_3m": spread_now, "spread_5d_ago": spread_5d,
                       "curve_improving": bool(curve_improving),
                       "backwardated": bool(backwardated), "score": vix_score},
        "asset_engines": engines,
        "relative_valuation": relval,
    }


def breadth_matrix(conn, tickers):
    """Precompute an above-200DMA boolean matrix ONCE (dates x tickers) so a
    multi-year replay doesn't re-query SQLite per ticker per date. Turns
    ~127k queries into one pass."""
    closes = {}
    for t in tickers:
        c = _close_series(conn, t)
        if c is not None and len(c) >= 200:
            closes[t] = c
    if not closes:
        return None
    px = pd.DataFrame(closes).sort_index()
    sma200 = px.rolling(200).mean()
    above = (px > sma200) & sma200.notna()
    valid = sma200.notna()
    pct = (above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)) * 100
    return pct

def historical_breadth(conn, as_of, tickers, pct_series=None):
    """% of universe above its 200-DMA as of a past date. Uses only data through
    that date -- no lookahead. Pass pct_series from breadth_matrix() for speed."""
    if pct_series is not None:
        sub = pct_series[pct_series.index <= pd.Timestamp(as_of)]
        return float(sub.iloc[-1]) if len(sub) and _v(sub.iloc[-1]) else None
    above = total = 0
    for t in tickers:
        c = _close_series(conn, t, as_of)
        if c is None or len(c) < 200:
            continue
        sma = c.rolling(200).mean().iloc[-1]
        if not _v(sma):
            continue
        total += 1
        if c.iloc[-1] > sma:
            above += 1
    return (100.0 * above / total) if total else None

def repair_history(conn):
    """Backfill missing columns in macro_history.csv from the DB WITHOUT a full
    recompute. Use when a change only adds a stored field (e.g. the spx column)
    rather than altering the scoring itself -- seconds instead of minutes."""
    merged = _load_hist_csv()
    if not merged:
        print("[repair] no history csv -- run --replay first"); return
    spx_px = _close_series(conn, "SPY")
    filled = 0
    for d_, v in list(merged.items()):
        px = v[3] if len(v) > 3 else None
        if px is None and spx_px is not None:
            ts = pd.Timestamp(d_)
            if ts in spx_px.index:
                merged[d_] = (v[0], v[1], v[2], float(spx_px.loc[ts]))
                filled += 1
    _save_hist_csv(merged)
    print(f"[repair] filled {filled} missing prices across {len(merged)} rows -> {HIST_CSV}")

def replay(conn, start, force=False):
    """Recompute the macro score for every trading day from `start` to the last
    stored close, writing results into macro_scores. Every input is truncated to
    each date -- no lookahead."""
    import datetime as _dt
    spy = _close_series(conn, "SPY")
    if spy is None:
        print("[replay] no SPY data"); return
    dates = [d for d in spy.index if d >= pd.Timestamp(start)]
    if not dates:
        print(f"[replay] no trading days on/after {start}"); return

    # INCREMENTAL BY DEFAULT. Recomputing 1,650 days takes minutes and is only
    # necessary when the SCORING changes. Routine catch-up needs just the missing
    # dates. Use --force after any methodology change.
    if not force:
        try:
            have = {r[0] for r in conn.execute("SELECT d FROM macro_hist").fetchall()}
        except Exception:
            have = set()
        todo = [d for d in dates if d.date().isoformat() not in have]
        if len(todo) < len(dates):
            print(f"[replay] incremental: {len(todo)} of {len(dates)} days need computing "
                  f"({len(dates)-len(todo)} already stored). Use --force to recompute all.")
        dates = todo
        if not dates:
            print("[replay] nothing to compute -- history is current. "
                  "Use --force if the scoring changed.")
            return
    names = [t for t in universe() if t not in NON_MEMBERS]
    conn.execute("CREATE TABLE IF NOT EXISTS macro_smooth (d TEXT PRIMARY KEY, s REAL)")
    conn.execute("DELETE FROM macro_smooth WHERE d >= ?", (str(pd.Timestamp(start).date()),))
    conn.execute("CREATE TABLE IF NOT EXISTS macro_override (k TEXT PRIMARY KEY, v TEXT)")
    conn.execute("DELETE FROM macro_override")
    conn.commit()
    print("[replay] precomputing breadth matrix...")
    pct_series = breadth_matrix(conn, names)
    conn.execute("CREATE TABLE IF NOT EXISTS macro_scores (d TEXT PRIMARY KEY, score REAL)")
    print(f"[replay] {len(dates)} trading days from {dates[0].date()} to {dates[-1].date()}")
    verbose = len(dates) <= 60
    if verbose:
        print(f"{'date':<12}{'score':>7}  {'status':<20}{'intermkt':>9}{'vix':>5}{'trend':>6}{'brdth':>6}")
        print("-" * 70)
    rows = []
    statuses = []
    for d in dates:
        b200 = historical_breadth(conn, d, names, pct_series)
        try:
            ms = build_macro_structure(conn, b200, None, None, as_of=d)
        except Exception as e:
            print(f"{str(d.date()):<12}  build failed: {e}")
            continue
        g = ms["global_regime"]; c = g["components"]
        rows.append((d.date().isoformat(), g["score"]))
        conn.execute("INSERT OR REPLACE INTO macro_scores VALUES (?,?)",
                     (d.date().isoformat(), float(g["raw_score"])))
        conn.commit()
        statuses.append((d.date().isoformat(), g["status"], g["score"], c["intermarket"]))
        if verbose:
            print(f"{str(d.date()):<12}{g['score']:>7.2f}  {g['status']:<20}"
                  f"{c['intermarket']:>9.2f}{c['vix']:>5}{c['trend']:>6}{c['breadth']:>6}")
        elif len(rows) % 50 == 0:
            print(f"  ...{len(rows)}/{len(dates)} days")
    conn.commit()

    # CSV export so the SPX-vs-regime overlay can be rebuilt in one step
    try:
        spy_px = _close_series(conn, "SPY")
        out = pd.DataFrame(statuses, columns=["date", "status", "score", "intermarket"])
        out["spx"] = [float(spy_px.loc[pd.Timestamp(d)]) if pd.Timestamp(d) in spy_px.index else None
                      for d in out["date"]]
        out["risk_on"] = (out["status"].str.contains("Risk-On")).astype(int)
        out["neutral"] = (out["status"] == "Neutral / Choppy").astype(int)
        out["risk_off"] = (out["status"] == "Full Risk-Off").astype(int)
        out.to_csv(REPLAY_CSV, index=False)
        print(f"[replay] overlay data -> {REPLAY_CSV}  (columns: date, spx, status, score, "
              f"risk_on/neutral/risk_off flags)")
    except Exception as e:
        print(f"[replay] csv export skipped: {e}")
    print(f"\n[replay] wrote {len(rows)} scores to macro_scores")
    if len(rows) > 1:
        first, last = rows[0][1], rows[-1][1]
        print(f"[replay] {rows[0][0]} {first:+.2f}  ->  {rows[-1][0]} {last:+.2f}   (change {last-first:+.2f})")
    if statuses:
        vals = [r[2] for r in statuses]
        print(f"[replay] score range {min(vals):+.2f} to {max(vals):+.2f}   mean {sum(vals)/len(vals):+.2f}")
        from collections import Counter
        cnt = Counter(r[1] for r in statuses)
        print("[replay] days in each regime:")
        for k in ["Full Risk-On", "Moderate Risk-On", "Neutral / Choppy", "Full Risk-Off"]:
            n = cnt.get(k, 0)
            print(f"           {k:<20}{n:>5}  ({100*n/len(statuses):.0f}%)")
        # regime transitions, so you can eyeball whether it led or lagged turns
        print("[replay] regime changes:")
        prev = None
        for d_iso, st, sc_, im in statuses:
            if st != prev:
                if prev is not None:
                    print(f"           {d_iso}  {prev}  ->  {st}   (score {sc_:+.2f}, intermkt {im:+.2f})")
                prev = st


def calibrate(conn, exclude_warmup=True):
    """Recommend regime bands from the REPLAYED score history instead of judgment.
    Requires --replay to have been run first. Excludes the warmup period where the
    200-day SMA was unavailable (trend/breadth == 0), since those scores are
    ratio-only and not comparable."""
    rows = conn.execute("SELECT d, score FROM macro_scores ORDER BY d").fetchall()
    if len(rows) < 60:
        print(f"[calibrate] only {len(rows)} scores stored -- run --replay first"); return
    ser = pd.Series({pd.Timestamp(d): float(sc) for d, sc in rows}).sort_index()

    # warmup detection: the absolute components need 200 bars of SPY history
    spy = _close_series(conn, "SPY")
    if exclude_warmup and spy is not None and len(spy) > 200:
        valid_from = spy.index[199]
        dropped = int((ser.index < valid_from).sum())
        if dropped:
            print(f"[calibrate] excluding {dropped} warmup days before {valid_from.date()} "
                  f"(200-day SMA unavailable -> ratio-only scores)")
            ser = ser[ser.index >= valid_from]

    n = len(ser)
    print(f"\n[calibrate] {n} usable days  {ser.index[0].date()} -> {ser.index[-1].date()}")
    print(f"  min {ser.min():+.2f}   max {ser.max():+.2f}   mean {ser.mean():+.2f}   sd {ser.std():.2f}")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("  percentiles: " + "  ".join(f"p{q}={ser.quantile(q/100):+.2f}" for q in qs))

    # symmetric, percentile-anchored bands: extremes = tails, moderate = shoulders
    full_on   = round(float(ser.quantile(0.90)), 1)
    mod_on    = round(float(ser.quantile(0.65)), 1)
    full_off  = round(float(ser.quantile(0.10)), 1)
    print(f"\n  RECOMMENDED BANDS (p90 / p65 / p10):")
    print(f"    Full Risk-On     >= {full_on:+.1f}")
    print(f"    Moderate Risk-On >= {mod_on:+.1f}")
    print(f"    Full Risk-Off    <= {full_off:+.1f}")

    def dist(fon, mon, foff):
        c = {"Full Risk-On": int((ser >= fon).sum()),
             "Moderate Risk-On": int(((ser >= mon) & (ser < fon)).sum()),
             "Neutral / Choppy": int(((ser > foff) & (ser < mon)).sum()),
             "Full Risk-Off": int((ser <= foff).sum())}
        return c
    cur = dist(7.0, 3.5, -3.0)
    new = dist(full_on, mod_on, full_off)
    print(f"\n  {'regime':<20}{'current':>16}{'recommended':>16}")
    for k in ["Full Risk-On", "Moderate Risk-On", "Neutral / Choppy", "Full Risk-Off"]:
        print(f"    {k:<18}{cur[k]:>6} ({100*cur[k]/n:>3.0f}%){new[k]:>10} ({100*new[k]/n:>3.0f}%)")

    # label stability: how often would the regime flip?
    def flips(fon, mon, foff):
        lab = pd.cut(ser, [-99, foff, mon, fon, 99], labels=["off","neutral","mod","full"])
        return int((lab != lab.shift()).sum() - 1)
    print(f"\n  raw band crossings (NO smoothing/hysteresis/dwell applied):")
    print(f"    current bands {flips(7.0,3.5,-3.0)}   recommended {flips(full_on,mod_on,full_off)}")
    print("    NOTE: these are NOT comparable to the regime-change count from --replay,")
    print("    which applies asymmetric smoothing, per-band hysteresis and the dwell")
    print("    floor. Use the band levels below; ignore these crossing counts.")


def _store_detail(conn, d_iso, payload):
    """Full per-date component detail, so the composite score is auditable and
    exportable. One JSON blob per date keeps the schema stable as inputs change."""
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS macro_detail (d TEXT PRIMARY KEY, j TEXT)")
        conn.execute("INSERT OR REPLACE INTO macro_detail VALUES (?,?)", (d_iso, json.dumps(payload)))
    except Exception as e:
        print(f"[detail] store failed: {e}")

def export_excel(conn, out="macro_components.xlsx"):
    """Excel workbook of every day's inputs and the resulting composite score."""
    try:
        import openpyxl  # noqa
    except ImportError:
        print("[export] pip install openpyxl"); return
    try:
        rows = conn.execute("SELECT d, j FROM macro_detail ORDER BY d").fetchall()
    except Exception:
        print("[export] no detail table yet -- run --replay (or a normal run) first"); return
    if not rows:
        print("[export] no detail stored -- run --replay first"); return
    recs = []
    for d_, j in rows:
        try:
            o = json.loads(j)
        except Exception:
            continue
        rec = {"date": d_, "composite_score": o.get("score"), "raw_score": o.get("raw"),
               "status": o.get("status"), "spx": o.get("spx"),
               "vix_score": o.get("vix"), "trend_score": o.get("trend"),
               "breadth_score": o.get("breadth"), "intermarket_total": o.get("intermarket"),
               "curve": o.get("curve"), "vix9d": o.get("vix9d"), "vix30d": o.get("vix30d"),
               "vix3m": o.get("vix3m"), "vix6m": o.get("vix6m"), "vvix": o.get("vvix"),
               "breadth_pct200": o.get("breadth_pct200"), "spy_vs_200sma_pct": o.get("spy_vs_200")}
        for r in o.get("ratios", []):
            key = r["pair"].replace("/", "_").replace("-", "")
            rec[f"{key}_value"] = r.get("value")
            rec[f"{key}_dir"] = r.get("direction")
            rec[f"{key}_conv"] = r.get("weight")
            rec[f"{key}_score"] = r.get("weighted_score")
        recs.append(rec)
    df = pd.DataFrame(recs)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Daily Components", index=False)
        cols = ["date", "composite_score", "status", "spx", "intermarket_total",
                "vix_score", "trend_score", "breadth_score"]
        df[[c for c in cols if c in df]].to_excel(xl, sheet_name="Summary", index=False)
    print(f"[export] {len(df)} days x {len(df.columns)} fields -> {out}")


def information_coefficient(conn, split="2024-01-01"):
    """Which components actually carry forward-looking information?

    Measures each component's Spearman rank correlation (the "information
    coefficient") against FORWARD SPX returns at 5/21/63 days -- not concurrent
    returns, which only tell you a component moves with the market and is
    therefore confirmation rather than signal.

    Reported IN-SAMPLE and OUT-OF-SAMPLE separately. This is the whole point: a
    component that scores well in-sample and collapses out-of-sample is noise
    that happened to fit, and the split is what exposes it.

    TWO CAVEATS THAT LIMIT HOW HARD THESE NUMBERS CAN BE PUSHED:
      1. Overlapping windows. Consecutive 21-day forward returns share 20 days,
         so the effective independent sample is ~n/21, not n. With ~1650 days
         that is roughly 79 independent observations -- enough to RANK components
         loosely, nowhere near enough to justify precise fitted weights.
      2. For macro/breadth signals an IC of 0.05-0.15 is considered good. Do not
         expect large numbers, and expect several components to be statistically
         indistinguishable from one another.

    Use the output for COARSE TIERING (2x / 1x / 0.5x), never fitted coefficients.
    """
    try:
        rows = conn.execute("SELECT d, j FROM macro_detail ORDER BY d").fetchall()
    except Exception:
        print("[ic] no detail table -- run --replay first"); return
    if len(rows) < 300:
        print(f"[ic] only {len(rows)} days stored -- run --replay 2020-01-01 first"); return

    recs = {}
    for d_, j in rows:
        try:
            o = json.loads(j)
        except Exception:
            continue
        rec = {"score": o.get("score"), "raw": o.get("raw"),
               "intermarket": o.get("intermarket"), "vix": o.get("vix"),
               "trend": o.get("trend"), "breadth": o.get("breadth"),
               "spx": o.get("spx")}
        for r in o.get("ratios", []):
            rec[r["pair"]] = r.get("weighted_score")
        recs[pd.Timestamp(d_)] = rec
    df = pd.DataFrame.from_dict(recs, orient="index").sort_index()

    spx = _close_series(conn, "SPY")
    if spx is None:
        print("[ic] no SPY data"); return
    px = spx.reindex(df.index).ffill()

    horizons = [5, 21, 63]
    for h in horizons:
        df[f"fwd{h}"] = (px.shift(-h) / px - 1) * 100

    comps = [c for c in df.columns
             if c not in ("spx",) and not c.startswith("fwd")]
    cut = pd.Timestamp(split)
    ins, oos = df[df.index < cut], df[df.index >= cut]

    def ic_of(frame, comp, h):
        sub = frame[[comp, f"fwd{h}"]].dropna()
        if len(sub) < 60 or sub[comp].nunique() < 3:
            return None
        return float(sub[comp].corr(sub[f"fwd{h}"], method="spearman"))

    print(f"\n[ic] Spearman IC vs FORWARD SPX returns")
    print(f"     in-sample  {ins.index.min().date()} -> {ins.index.max().date()}  ({len(ins)} days)")
    print(f"     out-sample {oos.index.min().date()} -> {oos.index.max().date()}  ({len(oos)} days)")
    print(f"     effective independent obs at 21d: ~{len(df)//21}")
    print(f"     NOISE BANDS (|IC| below these is indistinguishable from zero):")
    for h in horizons:
        eff = max(len(df) // h, 2)
        se = 1.0 / (eff ** 0.5)
        print(f"       {h:>3}d: effective n ~{eff:<5} SE ~{se:.3f}   2-SE band +/-{2*se:.3f}")
    print(f"     {len(comps)} components x {len(horizons)} horizons x 2 windows = "
          f"{len(comps)*len(horizons)*2} tests -- expect several to exceed the band by chance.\n")
    hdr = f"{'component':<24}" + "".join(f"{'IS '+str(h)+'d':>9}{'OOS '+str(h)+'d':>10}" for h in horizons) + "   verdict"
    print(hdr); print("-" * len(hdr))

    results = []
    for c in comps:
        cells, holds = [], []
        for h in horizons:
            a, b = ic_of(ins, c, h), ic_of(oos, c, h)
            cells.append((a, b))
            if a is not None and b is not None:
                # "holds" = same sign in both windows and non-trivial magnitude
                holds.append(abs(a) >= 0.05 and abs(b) >= 0.03 and (a > 0) == (b > 0))
        n_hold = sum(holds)
        verdict = "STRONG" if n_hold >= 2 else "weak" if n_hold == 1 else "noise"
        line = f"{c:<24}"
        for a, b in cells:
            line += f"{('%.3f' % a) if a is not None else '  -':>9}{('%.3f' % b) if b is not None else '  -':>10}"
        print(line + f"   {verdict}")
        best = max((abs(a) for a, _ in cells if a is not None), default=0)
        results.append((c, n_hold, best))

    print("\n[ic] SUGGESTED COARSE TIERS (not fitted weights):")
    for c, n_hold, _ in sorted(results, key=lambda r: -r[1]):
        if c in ("score", "raw"):
            continue     # composites, not inputs
        tier = "2.0x" if n_hold >= 2 else "1.0x" if n_hold == 1 else "0.5x  (candidate to drop)"
        print(f"     {c:<24} {tier}")
    print("\n     'score'/'raw' are the composite itself -- shown as a benchmark:")
    print("     if a single component beats the composite out-of-sample, the")
    print("     weighting is diluting real signal with noise.")


def vix_sign_test(conn, split="2024-01-01"):
    """ONE pre-specified hypothesis, not a fishing expedition.

    The IC analysis showed the vix component negative in every window at every
    horizon -- sign-consistent across six cells, which is harder to obtain by
    chance than any single large IC value. There is also a documented mechanism:
    the "volatility paradox" (suppressed volatility encourages leverage, which
    precedes poor forward returns). The engine currently scores CALM vol as
    risk-on POSITIVE. If the sign is genuinely inverted, that is an error rather
    than a weighting question.

    This compares forward SPX returns conditioned on the vol component's sign,
    in-sample and out-of-sample, and reports whether the pattern holds in BOTH.
    """
    try:
        rows = conn.execute("SELECT d, j FROM macro_detail ORDER BY d").fetchall()
    except Exception:
        print("[vixtest] no detail -- run --replay first"); return
    recs = {}
    for d_, j in rows:
        try:
            o = json.loads(j)
        except Exception:
            continue
        recs[pd.Timestamp(d_)] = {"vix": o.get("vix"), "curve": o.get("curve")}
    df = pd.DataFrame.from_dict(recs, orient="index").sort_index()
    spx = _close_series(conn, "SPY")
    if spx is None or df.empty:
        print("[vixtest] insufficient data"); return
    px = spx.reindex(df.index).ffill()
    for h in (21, 63):
        df[f"fwd{h}"] = (px.shift(-h) / px - 1) * 100

    cut = pd.Timestamp(split)
    print(f"\n[vixtest] mean forward SPX return (%) by vol-component sign")
    print(f"          split at {split}   (n shown per cell)\n")
    hdr = f"{'vol component':<16}{'IS 21d':>12}{'OOS 21d':>12}{'IS 63d':>12}{'OOS 63d':>12}"
    print(hdr); print("-" * len(hdr))
    buckets = [("positive (calm)", lambda v: v > 0),
               ("zero", lambda v: v == 0),
               ("negative (stress)", lambda v: v < 0)]
    table = {}
    for label, fn in buckets:
        line = f"{label:<16}"
        for h in (21, 63):
            for frame, _lbl in ((df[df.index < cut], "IS"), (df[df.index >= cut], "OOS")):
                sub = frame[frame["vix"].apply(lambda v: fn(v) if pd.notna(v) else False)]
                col = sub[f"fwd{h}"].dropna()
                table[(label, h, _lbl)] = float(col.mean()) if len(col) else None
        for h in (21, 63):
            for _lbl in ("IS", "OOS"):
                v = table.get((label, h, _lbl))
                line += f"{('%+.2f' % v) if v is not None else '   -':>12}"
        print(line)

    print("\n[vixtest] verdict:")
    holds = 0
    for h in (21, 63):
        for _lbl in ("IS", "OOS"):
            calm = table.get(("positive (calm)", h, _lbl))
            stress = table.get(("negative (stress)", h, _lbl))
            if calm is not None and stress is not None and stress > calm:
                holds += 1
    print(f"          stress > calm forward returns in {holds}/4 windows")
    if holds >= 3:
        print("          -> supports INVERTING the vol sign (calm vol is not bullish).")
    elif holds <= 1:
        print("          -> does NOT support inverting. Current sign stands.")
    else:
        print("          -> inconclusive (2/4). Do not change the sign on this.")
    print("          NOTE: forward-return differences of <1-2% over 21d are inside")
    print("          the noise band given ~78 effective independent observations.")


def divergence_test(conn, split="2024-01-01", lookback=10):
    """Does SPX moving WITHOUT the regime score confirming carry information?

    Premise: when price rallies hard but the composite stays flat or falls, the
    move is unconfirmed by underlying conditions and may be likelier to fail.
    Mirror case: price falls while the score improves may mark washouts.

    Method: over a rolling `lookback` window, measure SPX % change and score
    change, classify into four quadrants, measure FORWARD SPX returns from each.
    In-sample / out-of-sample split -- a pattern in only one window is noise.

    LIMITS (printed with the output):
      - Overlapping windows: effective independent n is ~days/horizon.
      - 4 quadrants x 3 horizons x 2 windows = 24 cells; some will look good by
        chance alone.
      - Compare against the BASE RATE, not zero. SPX rose through most of the
        sample, so "price rose afterward" is the default outcome.
    """
    try:
        rows = conn.execute("SELECT d, j FROM macro_detail ORDER BY d").fetchall()
    except Exception:
        print("[div] no detail -- run --replay first"); return
    recs = {}
    for d_, j in rows:
        try:
            o = json.loads(j)
        except Exception:
            continue
        recs[pd.Timestamp(d_)] = {"score": o.get("score"), "spx": o.get("spx")}
    df = pd.DataFrame.from_dict(recs, orient="index").sort_index().dropna()
    if len(df) < 300:
        print(f"[div] only {len(df)} usable days -- run --replay 2020-01-01 --force"); return

    df["spx_chg"]   = (df["spx"] / df["spx"].shift(lookback) - 1) * 100
    df["score_chg"] = df["score"] - df["score"].shift(lookback)
    for h in (5, 21, 63):
        df[f"fwd{h}"] = (df["spx"].shift(-h) / df["spx"] - 1) * 100
    df = df.dropna(subset=["spx_chg", "score_chg"])

    px_up = df["spx_chg"] >= 2.0
    px_dn = df["spx_chg"] <= -2.0
    sc_up = df["score_chg"] >= 1.0
    sc_dn = df["score_chg"] <= -1.0

    quads = {
        "price UP + score UP (confirmed)":      px_up & sc_up,
        "price UP + score flat/dn (UNCONF)":    px_up & ~sc_up,
        "price DN + score DN (confirmed)":      px_dn & sc_dn,
        "price DN + score flat/up (washout?)":  px_dn & ~sc_dn,
    }

    cut = pd.Timestamp(split)
    print(f"\n[div] price vs score divergence, {lookback}-session windows")
    print(f"      thresholds: |SPX| >= 2.0%, |score change| >= 1.0")
    print(f"      IS: {df.index.min().date()} -> {(cut - pd.Timedelta(days=1)).date()}"
          f"   OOS: {cut.date()} -> {df.index.max().date()}\n")

    ins_all, oos_all = df[df.index < cut], df[df.index >= cut]
    print(f"      {'BASE RATE (all days)':<38}{len(ins_all):>7}{len(oos_all):>8}", end="")
    for h in (5, 21, 63):
        print(f"{ins_all[f'fwd{h}'].mean():>9.2f}{oos_all[f'fwd{h}'].mean():>10.2f}", end="")
    print()
    hdr = (f"      {'quadrant':<38}{'n(IS)':>7}{'n(OOS)':>8}"
           + "".join(f"{('IS'+str(h)):>9}{('OOS'+str(h)):>10}" for h in (5, 21, 63)))
    print("      " + "-" * (len(hdr) - 6))
    print(hdr)
    for name, mask in quads.items():
        ins, oos = df[mask & (df.index < cut)], df[mask & (df.index >= cut)]
        line = f"      {name:<38}{len(ins):>7}{len(oos):>8}"
        for h in (5, 21, 63):
            a = ins[f"fwd{h}"].mean() if len(ins) > 10 else None
            b = oos[f"fwd{h}"].mean() if len(oos) > 10 else None
            line += (f"{a:>9.2f}{b:>10.2f}" if a is not None and b is not None
                     else f"{'-':>9}{'-':>10}")
        print(line)
    print("\n      Read against the BASE RATE row, not zero. A quadrant means something")
    print("      only if it differs from base AND the sign holds in BOTH windows.")
    print("      Effective independent observations at 21d is roughly n/21.")


def band_forward_test(conn, split="2024-01-01"):
    """Do LOW scores actually precede BETTER forward returns?

    Motivated by an eyeball observation that the score dipping into the risk-off
    band tends to be followed by SPX rising. If true at scale it would mean the
    chart's colour semantics (red = bad) are misleading as a forward read, even
    though they are correct as a description of CURRENT conditions.

    Reports mean forward SPX return by score band, in-sample and out-of-sample,
    against the all-days base rate.

    WATCH THE CONFOUND: SPX rose over most of this sample, and low scores cluster
    inside drawdowns. "Buy when the score is low" is therefore close to "buy the
    dip in a rising market" -- which works until it doesn't. A positive result
    here is NOT evidence the colours should flip; it is evidence that drawdowns
    in this sample resolved upward.
    """
    try:
        rows = conn.execute("SELECT d, j FROM macro_detail ORDER BY d").fetchall()
    except Exception:
        print("[band] no detail -- run --replay first"); return
    recs = {}
    for d_, j in rows:
        try:
            o = json.loads(j)
        except Exception:
            continue
        if o.get("score") is not None and o.get("spx") is not None:
            recs[pd.Timestamp(d_)] = {"score": o["score"], "spx": o["spx"]}
    df = pd.DataFrame.from_dict(recs, orient="index").sort_index()
    if len(df) < 300:
        print(f"[band] only {len(df)} usable days"); return
    for h in (5, 21, 63):
        df[f"fwd{h}"] = (df["spx"].shift(-h) / df["spx"] - 1) * 100

    bands = [("risk-off      (<= -3)",      df["score"] <= -3),
             ("weak          (-3 to 0)",    (df["score"] > -3) & (df["score"] <= 0)),
             ("mild          (0 to 3.5)",   (df["score"] > 0) & (df["score"] < 3.5)),
             ("risk-on       (3.5 to 5)",   (df["score"] >= 3.5) & (df["score"] < 5)),
             ("full risk-on  (>= 5)",       df["score"] >= 5)]

    cut = pd.Timestamp(split)
    ins, oos = df[df.index < cut], df[df.index >= cut]
    print(f"\n[band] mean forward SPX return (%) by score band")
    print(f"       IS {df.index.min().date()} -> {(cut-pd.Timedelta(days=1)).date()}"
          f" | OOS {cut.date()} -> {df.index.max().date()}\n")
    hdr = (f"       {'band':<26}{'n(IS)':>7}{'n(OOS)':>8}"
           + "".join(f"{('IS'+str(h)):>9}{('OOS'+str(h)):>10}" for h in (5,21,63)))
    print(f"       {'BASE RATE (all days)':<26}{len(ins):>7}{len(oos):>8}"
          + "".join(f"{ins[f'fwd{h}'].mean():>9.2f}{oos[f'fwd{h}'].mean():>10.2f}" for h in (5,21,63)))
    print("       " + "-"*(len(hdr)-7)); print(hdr)
    for name, mask in bands:
        a, b = df[mask & (df.index < cut)], df[mask & (df.index >= cut)]
        line = f"       {name:<26}{len(a):>7}{len(b):>8}"
        for h in (5,21,63):
            x = a[f"fwd{h}"].mean() if len(a) > 10 else None
            y = b[f"fwd{h}"].mean() if len(b) > 10 else None
            line += (f"{x:>9.2f}{y:>10.2f}") if x is not None and y is not None else f"{'-':>9}{'-':>10}"
        print(line)
    print("\n       Compare each row to BASE RATE, not zero, and require the sign to")
    print("       hold in BOTH windows. Effective independent obs at 21d is ~n/21.")

    # ---- SECOND TABLE: by CHANGE in the score, not its level -----------------
    # The engine currently labels off the LEVEL. The level is structurally slow to
    # recover (credit/breadth/vol repair after price does), which is why exits ran
    # late. The CHANGE may carry the information earlier. This measures whether it
    # actually does, before any trigger is rebuilt around it.
    for win in (5, 10, 21):
        df[f"chg{win}"] = df["score"] - df["score"].shift(win)
    print()
    for win in (10,):
        c = f"chg{win}"
        cbands = [(f"falling hard  (<= -3)",  df[c] <= -3),
                  (f"falling       (-3 to -1)", (df[c] > -3) & (df[c] <= -1)),
                  (f"flat          (-1 to 1)",  (df[c] > -1) & (df[c] < 1)),
                  (f"rising        (1 to 3)",   (df[c] >= 1) & (df[c] < 3)),
                  (f"rising hard   (>= 3)",     df[c] >= 3)]
        print(f"[band] mean forward SPX return (%) by {win}-SESSION CHANGE in score\n")
        hdr2 = (f"       {'change band':<26}{'n(IS)':>7}{'n(OOS)':>8}"
                + "".join(f"{('IS'+str(h)):>9}{('OOS'+str(h)):>10}" for h in (5,21,63)))
        print(f"       {'BASE RATE (all days)':<26}{len(ins):>7}{len(oos):>8}"
              + "".join(f"{ins[f'fwd{h}'].mean():>9.2f}{oos[f'fwd{h}'].mean():>10.2f}" for h in (5,21,63)))
        print("       " + "-"*(len(hdr2)-7)); print(hdr2)
        for name, mask in cbands:
            a, b = df[mask & (df.index < cut)], df[mask & (df.index >= cut)]
            line = f"       {name:<26}{len(a):>7}{len(b):>8}"
            for h in (5,21,63):
                x = a[f"fwd{h}"].mean() if len(a) > 10 else None
                y = b[f"fwd{h}"].mean() if len(b) > 10 else None
                line += (f"{x:>9.2f}{y:>10.2f}") if x is not None and y is not None else f"{'-':>9}{'-':>10}"
            print(line)
    print("\n       READ THE TWO TABLES TOGETHER. If the CHANGE bands separate forward")
    print("       returns more cleanly than the LEVEL bands -- bigger spread, signs")
    print("       holding in both windows -- then triggering on rate-of-change is")
    print("       better supported than triggering on level. If they look similar,")
    print("       the level is not the problem and switching would add complexity")
    print("       for nothing.")

def dispatch_webhook(prev_status, new_status, payload):
    """POST a regime-shift alert to WEBHOOK_URL if set. Never crashes the run."""
    url = os.environ.get("WEBHOOK_URL", "").strip()
    if not url or prev_status == new_status:
        return
    try:
        import urllib.request
        body = json.dumps({"event": "macro_regime_shift", "from": prev_status,
                           "to": new_status, "payload": payload}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print(f"[webhook] regime shift {prev_status} -> {new_status} dispatched")
    except Exception as e:
        print(f"[webhook] dispatch failed (non-fatal): {e}")


def _session_date(conn):
    """The board's as-of date = last EQUITY close, not max() across all tickers.
    BTC-USD trades 24/7, so after ~20:00 ET (00:00 UTC) it already has a bar for
    the next calendar day while equities are still on the prior session. Taking a
    global max() let crypto drag the board date a day ahead."""
    r = conn.execute("SELECT max(d) FROM prices WHERE ticker='SPY'").fetchone()
    if r and r[0]:
        return str(r[0])[:10]
    r = conn.execute("SELECT max(d) FROM prices").fetchone()
    return str(r[0])[:10] if r and r[0] else _dt.date.today().isoformat()

def compute_and_emit(conn):
    today = _session_date(conn)
    if not today:
        raise SystemExit("no price data in DB -- run ingest first")

    metrics = {}
    for t in universe():
        metrics[t] = ticker_metrics(load_prices(conn, t))

    b = breadth(metrics)
    spy = metrics.get("SPY") or {}
    spy_ret21 = spy.get("ret21", 0.0) if _v(spy.get("ret21", np.nan)) else 0.0

    _TH = active_themes()
    # theme scores + persist + 1-day delta
    prev = dict(conn.execute(
        "SELECT theme, score FROM theme_scores WHERE d=(SELECT max(d) FROM theme_scores WHERE d<?)",
        (today,)).fetchall())
    scores = {th: s for th, syms in _TH.items()
              if (s := theme_score(syms, metrics, spy_ret21)) is not None}
    for th, s in scores.items():
        conn.execute("INSERT OR REPLACE INTO theme_scores VALUES (?,?,?)", (today, th, s))
    conn.commit()
    delta = {th: round(scores[th] - prev[th], 1) if th in prev else 0.0 for th in scores}

    by_score = sorted(scores, key=lambda t: -scores[t])
    rank = {t: i + 1 for i, t in enumerate(by_score)}
    def row(t):
        return {"r": rank[t], "name": t,
                "cnt": len([x for x in THEMES[t] if x in metrics and metrics[x]]),
                "score": scores[t], "delta": delta[t], "st": status(scores[t], delta[t])}
    dominant = [row(t) for t in by_score[:8]]
    emerging = [row(t) for t in sorted(scores, key=lambda t: -abs(delta[t]))[:8]]

    dom_names = [t for t in by_score if status(scores[t], delta[t]) == "DOMINANT"][:3]
    fading    = [t for t in sorted(scores, key=lambda t: delta[t])
                 if status(scores[t], delta[t]) in ("DETERIORATING", "FADING")][:3]
    emerg     = [t for t in by_score
                 if delta[t] >= 5 and status(scores[t], delta[t]) != "DOMINANT"][:3]

    idx = []
    for sym, desc in INDEX_ETFS:
        m = metrics.get(sym)
        if not m:
            continue
        idx.append({"tk": sym, "desc": desc,
                    "d1": round(m["ret1"], 2), "d5": round(m["ret5"], 2),
                    "d50": round(m["dist50"], 1), "atr": round(m["atr_ext"], 1)})

    # ---- tab data: ETF grid, RVOL, momentum, extension ----
    t2theme = {}
    for th, syms in _TH.items():
        for t in syms:
            t2theme.setdefault(t, th)

    def rnd(x, n=2):
        return round(float(x), n) if _v(x) else None

    names = [(t, m) for t, m in metrics.items() if m and t not in NON_MEMBERS]

    macro = []
    for sym, desc in MACRO_ETFS:
        m = metrics.get(sym)
        if m:
            macro.append({"tk": sym, "desc": desc, "d1": rnd(m["ret1"]), "d5": rnd(m["ret5"]),
                          "d21": rnd(m["ret21"]), "d50": rnd(m["dist50"], 1)})

    etfs = []
    for sym, desc in SECTOR_ETFS:
        m = metrics.get(sym)
        if m:
            etfs.append({"tk": sym, "desc": desc, "d1": rnd(m["ret1"]), "d5": rnd(m["ret5"]),
                         "d21": rnd(m["ret21"]), "d50": rnd(m["dist50"], 1), "atr": rnd(m["atr_ext"], 1)})

    def row(t, m, extra):
        base = {"tk": t, "theme": t2theme.get(t, "—"), "d1": rnd(m["ret1"])}
        base.update(extra)
        return base

    rv = sorted([x for x in names if _v(x[1]["rvol"])], key=lambda x: -x[1]["rvol"])[:18]
    rvol_rows = [row(t, m, {"rvol": rnd(m["rvol"]), "d5": rnd(m["ret5"])}) for t, m in rv]

    mo = sorted([x for x in names if _v(x[1]["ret21"])], key=lambda x: -x[1]["ret21"])
    mrow = lambda t, m: row(t, m, {"r21": rnd(m["ret21"], 1), "r63": rnd(m["ret63"], 1)})
    momentum = {"leaders":  [mrow(t, m) for t, m in mo[:18]],
                "laggards": [mrow(t, m) for t, m in mo[-18:][::-1]]}

    ex = sorted([x for x in names if _v(x[1]["atr_ext"])], key=lambda x: -x[1]["atr_ext"])
    erow = lambda t, m: row(t, m, {"atr": rnd(m["atr_ext"], 1), "d50": rnd(m["dist50"], 1)})
    extension = {"high": [erow(t, m) for t, m in ex[:18]],
                 "low":  [erow(t, m) for t, m in ex[-18:][::-1]]}

    universe_map = [{"theme": th, "tickers": sorted(syms)} for th, syms in _TH.items()]

    verify_map = {}
    for t, m in metrics.items():
        if not m or t in NON_MEMBERS:
            continue
        verify_map[t] = {
            "close": rnd(m["close"]), "d1": rnd(m["ret1"]), "d5": rnd(m["ret5"]),
            "d21": rnd(m["ret21"]), "d63": rnd(m["ret63"]),
            "dist50": rnd(m["dist50"], 1), "atr": rnd(m["atr_ext"], 1),
            "rvol": rnd(m["rvol"]), "a50": m["above50"], "a200": m["above200"],
            "nh20": m["new20high"], "nl20": m["new20low"],
        }

    # ---- SCREEN: a JOIN across metrics already computed, not new signal -------
    # The IC analysis found no forward-predictive power in these components, so
    # this is explicitly a SHORTLIST TO RESEARCH, not a buy list. Its value is
    # saving the cross-referencing you would otherwise do by hand across the
    # Momentum / Extension / RVOL / Themes / Events tabs.
    _earn_soon = set()
    try:
        _today = pd.Timestamp(today)
        for _t, _d in globals().get("_EARNINGS", {}).items():
            if 0 <= (pd.Timestamp(_d) - _today).days <= 7:
                _earn_soon.add(_t)
    except Exception:
        pass

    _theme_rank = {r["name"]: r["score"] for r in dominant}
    for r in emerging:
        _theme_rank.setdefault(r["name"], r["score"])

    try:
        _sig = _combo_signals(conn, [t for t, _ in names])
    except Exception as e:
        print(f"[screen] combo signals unavailable: {e}")
        _sig = {}

    screen_long, screen_avoid = [], []
    for t, m in names:
        if not _v(m.get("ret21")) or not _v(m.get("atr_ext")):
            continue
        th = t2theme.get(t, "—")
        th_score = _theme_rank.get(th)
        rs = m["ret21"] - (spy_ret21 or 0.0)          # relative strength vs SPY
        ext = m["atr_ext"]
        rec = {"tk": t, "theme": th,
               "theme_score": th_score,
               "rs21": rnd(rs, 1), "d1": rnd(m["ret1"]),
               "atr_ext": rnd(ext, 1), "dist50": rnd(m["dist50"], 1),
               "rvol": rnd(m["rvol"]),
               "above200": bool(m["above200"]),
               "earnings_7d": t in _earn_soon}
        _sg = _sig.get(t)
        if _sg:
            rec.update({"sig_state": _sg["state"], "sig_buy_conv": _sg["buy_conv"],
                        "sig_sell_conv": _sg["sell_conv"], "sig_age": _sg["cross_age"],
                        "sig_fresh_buy": _sg["fresh_buy"], "sig_fresh_sell": _sg["fresh_sell"]})
        # LONG shortlist: outperforming, in an uptrend, and NOT already stretched
        # (extension is a caution, not a virtue -- +4 ATR is a bad entry even on
        # a strong name).
        if rs > 0 and m["above200"] and ext < 3.0:
            rec["rank"] = round(rs - max(0.0, ext - 1.0) * 2.0
                                + (th_score - 50) / 10.0 if th_score is not None else rs, 2)
            screen_long.append(rec)
        # AVOID/TRIM: lagging and below the 200-DMA, or very stretched
        elif (rs < 0 and not m["above200"]) or ext > 4.0:
            rec["rank"] = round(rs - max(0.0, ext - 1.0) * 2.0, 2)
            screen_avoid.append(rec)
    for r in screen_long:
        r["confluence"] = bool(r.get("sig_state") == "up" and (r.get("sig_buy_conv", 0) >= 2))
    for r in screen_avoid:
        r["confluence"] = bool(r.get("sig_state") == "down" and (r.get("sig_sell_conv", 0) >= 2))
    # confluent names first, then by rank
    screen_long.sort(key=lambda r: (-int(r.get("confluence", False)), -r["rank"]))
    screen_avoid.sort(key=lambda r: (-int(r.get("confluence", False)), r["rank"]))
    screen = {"long": screen_long[:20], "avoid": screen_avoid[:20],
              "regime": None}     # filled after the macro block below

    earn_map = globals().get("_EARNINGS", {})
    t2th = t2theme
    earnings = []
    for t, d8 in sorted(earn_map.items(), key=lambda kv: kv[1]):
        earnings.append({"tk": t, "theme": t2th.get(t, "—"), "date": d8})

    # ---- macro structure (Sec.3-6): graceful -- on failure, preserve previous ----
    try:
        macro_structure = build_macro_structure(conn, b.get("pct200"), b.get("pct50"), regime_label(b))
        conn.execute("CREATE TABLE IF NOT EXISTS macro_state (k TEXT PRIMARY KEY, v TEXT)")
        prev = conn.execute("SELECT v FROM macro_state WHERE k='regime'").fetchone()
        pend = conn.execute("SELECT v FROM macro_state WHERE k='pending'").fetchone()
        confirmed = prev[0] if prev else None
        pending = pend[0] if pend else None
        new_status = macro_structure["global_regime"]["status"]
        # a regime shift must hold for TWO consecutive runs before it is confirmed
        # and webhooked -- kills boundary flapping (e.g. score oscillating 0<->1).
        if confirmed is None or new_status == confirmed:
            conn.execute("DELETE FROM macro_state WHERE k='pending'")
            if confirmed is None:
                conn.execute("INSERT OR REPLACE INTO macro_state VALUES ('regime', ?)", (new_status,))
        elif new_status == pending:
            dispatch_webhook(confirmed, new_status, macro_structure["global_regime"])
            conn.execute("INSERT OR REPLACE INTO macro_state VALUES ('regime', ?)", (new_status,))
            conn.execute("DELETE FROM macro_state WHERE k='pending'")
        else:
            conn.execute("INSERT OR REPLACE INTO macro_state VALUES ('pending', ?)", (new_status,))
        conn.commit()
        macro_structure["global_regime"]["confirmed_status"] = new_status if (confirmed is None or new_status == confirmed or new_status == pending) else confirmed
    except Exception as e:
        print(f"[macro] build failed ({e}); preserving previous payload as stale")
        macro_structure = None
        try:
            with open(JSON_OUT) as f:
                old = json.load(f).get("macro_structure")
            if old:
                old["system_health"] = {"data_status": "stale", "last_close": old.get("system_health", {}).get("last_close")}
                macro_structure = old
        except Exception:
            pass

    out = {
        "date": today, "universe": UNIVERSE, "tickers": b["n"],
        "regime": {
            "label": regime_label(b),
            "breadth": f"{b['pct50']:.0f}% > 50DMA, {b['pct200']:.0f}% > 200DMA",
            "dominant": dom_names or ["\u2014"],
            "emerging": emerg,
            "emergingNote": "no themes improving" if not emerg else "",
            "fading": fading or ["\u2014"],
            "newHighs": b["nh20"], "newLows": b["nl20"], "highs52w": b["nh52"],
            "up3": b["up3"], "down3": b["down3"],
        },
        "indices": idx,
        "breadth": [
            ["% > 20DMA", f"{b['pct20']:.1f}%"], ["% > 50DMA", f"{b['pct50']:.1f}%"],
            ["% > 200DMA", f"{b['pct200']:.1f}%"], ["Total names", str(b["n"])],
            ["New 20D highs", str(b["nh20"])], ["New 52W highs", str(b["nh52"])],
            ["Up 3%+", str(b["up3"])], ["Down 3%+", str(b["down3"])],
        ],
        "dominant": dominant, "emerging": emerging,
        "etfs": etfs, "macro": macro, "rvol": rvol_rows, "momentum": momentum,
        "screen": screen,
        "universe_map": universe_map, "verify_map": verify_map,
        "macro_structure": macro_structure,
        "extension": extension, "earnings": earnings,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=2)
    return out

# ------------------------------------------------------------------ self-test
def make_synthetic(ticker, n=320):
    rng = np.random.default_rng(abs(hash(ticker)) % (2 ** 32))
    drift = rng.normal(0.0004, 0.0009)          # per-ticker trend, so statuses vary
    rets = rng.normal(drift, 0.02, n)
    close = 100 * np.exp(np.cumsum(rets))
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low  = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol  = rng.integers(1_000_000, 10_000_000, n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    m = len(idx)   # bdate_range can return n-1 when today is a weekend; align to it
    return pd.DataFrame({"open": openp[:m], "high": high[:m], "low": low[:m],
                         "close": close[:m], "volume": vol[:m]}, index=idx)

def selftest():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for t in universe():
        upsert(conn, t, make_synthetic(t))
    conn.commit()
    out = compute_and_emit(conn)
    print("SELFTEST OK -- emitted", JSON_OUT)
    print(f"  regime={out['regime']['label']}  tickers={out['tickers']}  "
          f"indices={len(out['indices'])}  themes={len(out['dominant'])}")
    print("  sample dominant rows:")
    for r in out["dominant"][:4]:
        print(f"    {r['r']:>2} {r['name']:<24} score={r['score']:>3} "
              f"d={r['delta']:>5} {r['st']}")
    keys = {"date", "universe", "tickers", "regime", "indices", "breadth", "dominant", "emerging"}
    assert keys <= set(out), "missing top-level keys"
    assert len(out["indices"]) == len(INDEX_ETFS), "index panel incomplete"
    print("  JSON contract check: PASS")

# ------------------------------------------------------------------ main
def verify(tickers):
    """Audit mode: print the full calculation chain per ticker so each value can
    be checked against Yahoo/your broker by hand. Read-only; no network, no write."""
    conn = sqlite3.connect(DB_PATH)
    asof = conn.execute("SELECT max(d) FROM prices").fetchone()[0]
    print(f"\n=== VERIFY  (data as-of {asof}) ===")
    print("Compare 'close' and the returns against Yahoo for that same date.\n")
    for t in tickers:
        t = t.upper()
        df = load_prices(conn, t)
        if df is None or len(df) < 21:
            print(f"{t:<7} no / insufficient data in market.db\n")
            continue
        m = ticker_metrics(df)
        c = df["close"]
        last5 = list(zip([d.strftime("%Y-%m-%d") for d in df.index[-5:]],
                         [round(float(x), 2) for x in c.iloc[-5:]]))
        sma50  = float(c.rolling(50).mean().iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float("nan")
        def show(x, suf="", nd=2):
            return "n/a" if not _v(x) else f"{x:+.{nd}f}{suf}" if suf == "%" else f"{x:.{nd}f}{suf}"
        print(f"{t:<7} close {m['close']:.2f}   ({len(c)} bars in DB)")
        print(f"        last 5 closes: " + ", ".join(f"{d}={v}" for d, v in last5))
        print(f"        1D {show(m['ret1'],'%')}   5D {show(m['ret5'],'%')}   "
              f"21D {show(m['ret21'],'%')}   63D {show(m['ret63'],'%')}")
        print(f"        50DMA {sma50:.2f}  -> dist {show(m['dist50'],'%',1)}   "
              f"(above50={m['above50']})")
        a200 = f"{sma200:.2f}" if _v(sma200) else "n/a (<200 bars)"
        print(f"        200DMA {a200}   above200={m['above200']}")
        print(f"        ATR ext {show(m['atr_ext'],'',1)}   RVOL {show(m['rvol'],'x',2)}   "
              f"new20high={m['new20high']}  new20low={m['new20low']}\n")
    conn.close()

def _rewrite_themes(new_themes):
    """Rewrite the THEMES = {...} block in this source file, preserving the rest."""
    import re, io
    src = open(__file__, encoding="utf-8").read()
    # locate 'THEMES = {' ... matching closing brace at column 0 ('}\n')
    start = src.index("THEMES = {")
    depth = 0
    i = start + len("THEMES = ")
    end = None
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end is None:
        raise RuntimeError("could not locate end of THEMES block")
    # pretty-print the new dict
    lines = ["THEMES = {"]
    for th in new_themes:
        toks = new_themes[th]
        inner = ", ".join(f'"{t}"' for t in toks)
        lines.append(f'    "{th}": [{inner}],')
    lines.append("}")
    block = "\n".join(lines)
    open(__file__, "w", encoding="utf-8").write(src[:start] + block + src[end:])

def edit_universe(add=None, remove=None, theme=None):
    themes = {k: list(v) for k, v in THEMES.items()}
    if add:
        add = add.upper()
        if not theme:
            print("--add requires --theme \"Theme Name\""); return
        if theme not in themes:
            print(f"unknown theme '{theme}'. existing: {', '.join(themes)}"); return
        if add in themes[theme]:
            print(f"{add} already in {theme}")
        else:
            themes[theme].append(add)
            print(f"added {add} -> {theme}")
    if remove:
        remove = remove.upper()
        hit = [th for th, ts in themes.items() if remove in ts]
        if not hit:
            print(f"{remove} not found in any theme")
        else:
            for th in hit:
                themes[th] = [t for t in themes[th] if t != remove]
            print(f"removed {remove} from: {', '.join(hit)}")
    _rewrite_themes(themes)
    total = len({t for ts in themes.values() for t in ts})
    print(f"universe now {total} named tickers. Run --backfill to pull history for new names.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="force full-history pull")
    ap.add_argument("--selftest", action="store_true", help="synthetic data, no network")
    ap.add_argument("--verify", nargs="+", metavar="TICKER",
                    help="audit calc chain for given tickers vs the DB (read-only)")
    ap.add_argument("--bandtest", action="store_true",
                    help="mean forward SPX return by score band")
    ap.add_argument("--divtest", action="store_true",
                    help="test whether price/score divergence predicts forward SPX")
    ap.add_argument("--div-lookback", type=int, default=10, metavar="N",
                    help="window for --divtest (default 10 sessions)")
    ap.add_argument("--vixtest", action="store_true",
                    help="test whether the vol component's sign should be inverted")
    ap.add_argument("--ic", action="store_true",
                    help="information-coefficient analysis: which components predict forward SPX")
    ap.add_argument("--ic-split", metavar="YYYY-MM-DD", default="2024-01-01",
                    help="in-sample / out-of-sample boundary for --ic")
    ap.add_argument("--export", action="store_true",
                    help="write macro_components.xlsx from stored per-date detail")
    ap.add_argument("--calibrate", action="store_true",
                    help="recommend regime bands from replayed score history")
    ap.add_argument("--replay", metavar="YYYY-MM-DD",
                    help="compute macro scores from this date forward (incremental; no lookahead)")
    ap.add_argument("--force", action="store_true",
                    help="with --replay: recompute every date, not just missing ones")
    ap.add_argument("--repair-history", action="store_true",
                    help="backfill missing columns in macro_history.csv without recomputing")
    ap.add_argument("--add", metavar="TICKER", help="add a ticker (needs --theme)")
    ap.add_argument("--remove", metavar="TICKER", help="remove a ticker from all themes")
    ap.add_argument("--theme", metavar="NAME", help="theme name for --add")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.verify:
        verify(args.verify)
        return

    if args.bandtest:
        conn = sqlite3.connect(DB_PATH)
        band_forward_test(conn, args.ic_split)
        conn.close()
        return

    if args.divtest:
        conn = sqlite3.connect(DB_PATH)
        divergence_test(conn, args.ic_split, args.div_lookback)
        conn.close()
        return

    if args.vixtest:
        conn = sqlite3.connect(DB_PATH)
        vix_sign_test(conn, args.ic_split)
        conn.close()
        return

    if args.ic:
        conn = sqlite3.connect(DB_PATH)
        information_coefficient(conn, args.ic_split)
        conn.close()
        return

    if args.export:
        conn = sqlite3.connect(DB_PATH)
        export_excel(conn)
        conn.close()
        return

    if args.calibrate:
        conn = sqlite3.connect(DB_PATH)
        calibrate(conn)
        conn.close()
        return

    if args.repair_history:
        conn = sqlite3.connect(DB_PATH)
        repair_history(conn)
        conn.close()
        return

    if args.replay:
        conn = sqlite3.connect(DB_PATH)
        replay(conn, args.replay, force=args.force)
        conn.close()
        return

    if args.add or args.remove:
        edit_universe(add=args.add, remove=args.remove, theme=args.theme)
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    have = conn.execute("SELECT count(*) FROM prices").fetchone()[0]
    period = BACKFILL_PERIOD if (args.backfill or have == 0) else NIGHTLY_PERIOD
    print(f"[run] {'backfill' if period == BACKFILL_PERIOD else 'nightly'} pull ({period})")
    ingest(conn, period)
    try:
        named = [t for t in universe() if t not in NON_MEMBERS]
        globals()["_EARNINGS"] = fetch_earnings(named)
        print(f"[run] earnings dates found for {len(globals()['_EARNINGS'])} names")
    except Exception as e:
        globals()["_EARNINGS"] = {}
        print(f"[run] earnings fetch skipped: {e}")
    out = compute_and_emit(conn)
    print(f"[run] {out['date']}  regime={out['regime']['label']}  "
          f"tickers={out['tickers']}  ->  {JSON_OUT}")

if __name__ == "__main__":
    main()
