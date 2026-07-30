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

def active_themes():
    """THEMES to use this run: sheet if configured & healthy, else built-in fallback."""
    if SHEET_CSV_URL:
        try:
            th = load_universe_from_sheet(SHEET_CSV_URL)
            print(f"[universe] loaded {sum(len(v) for v in th.values())} entries from sheet")
            return th
        except Exception as e:
            print(f"[universe] sheet load failed ({e}); using built-in THEMES")
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

def _ratio_signal(conn, num, den, inverted=False, as_of=None):
    """5/21 EMA crossover on a date-aligned price ratio. Returns dict or None.
    Date alignment matters: BTC-USD trades 7d/wk vs 5 for equities, so we inner-
    join on common dates before computing the ratio and its EMAs."""
    a, b = _close_series(conn, num, as_of), _close_series(conn, den, as_of)
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
    return {"pair": f"{num}/{den}", "value": round(float(r.iloc[-1]), 3),
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
            return {str(r["d"]): (float(r["raw"]), float(r["smooth"]), str(r["status"]))
                    for _, r in df.iterrows()}
    except Exception as e:
        print(f"[hist] could not read {HIST_CSV}: {e}")
    return {}

def _save_hist_csv(rows):
    try:
        pd.DataFrame([{"d": d, "raw": v[0], "smooth": v[1], "status": v[2]}
                      for d, v in sorted(rows.items())]).to_csv(HIST_CSV, index=False)
    except Exception as e:
        print(f"[hist] could not write {HIST_CSV}: {e}")

def build_macro_structure(conn, breadth_pct200, breadth_pct50=None, breadth_regime=None, as_of=None):
    import datetime as _dt
    RATIOS = [("RSP", "SPY", False, "Market Breadth"),
              ("XLY", "XLP", False, "Consumer Demand"),
              ("HYG", "IEI", False, "Credit Spreads"),
              ("IWM", "SPY", False, "Small Cap Appetite"),
              ("BTC-USD", "GLD", False, "Digital Risk vs Safety"),
              ("CPER", "GLD", False, "Industrial Demand"),
              ("UUP", "SPY", True,  "US Dollar vs Equities")]
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
            # curve healthy -> fall back to the absolute level
            # CALM VOL IS NOT SCORED BULLISH. The --vixtest hypothesis test came
            # back 4/4: forward SPX returns after STRESS readings beat those after
            # CALM readings in every window, monotonically (stress > zero > calm),
            # by +1.9pp at 21d and +5.2pp at 63d out-of-sample. Mechanism is the
            # "volatility paradox" -- suppressed vol invites leverage, which
            # precedes poor returns.
            #
            # Calm is therefore scored 0 rather than +1. NOT inverted to -1: full
            # inversion is a larger claim than 4/4 on ~78 effective independent
            # observations supports, and the engine is a CONCURRENT classifier --
            # "vol is calm now" remains a true statement about present conditions.
            vix_score = -1 if (vix_close is not None and vix_close > 20.0) else 0
            # vol-of-vol falling no longer adds a positive; it only offsets an
            # elevated-level penalty back toward neutral.
            if (vix_score < 0 and vvix_close is not None and vvix_10d is not None
                    and vvix_close < vvix_10d):
                vix_score = 0
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
    SPAN_DOWN, SPAN_UP = 3, 8
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
    raw_status = _band(total)
    status = raw_status
    dwell_held = False
    if str(prev_lab) == "Full Risk-Off" and raw_status != "Full Risk-Off" and prev_run < MIN_DWELL_OFF:
        status = "Full Risk-Off"
        dwell_held = True
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
        for d_, rw, sm, st in conn.execute(
                "SELECT d, raw, smooth, status FROM macro_hist").fetchall():
            merged[str(d_)] = (float(rw), float(sm), str(st))   # DB wins on overlap
        _save_hist_csv(merged)
        hrows = [(d, v[0], v[1], v[2]) for d, v in sorted(merged.items())][-1600:]
        spx_px = _close_series(conn, "SPY", as_of)
        for d_, rw, sm, st in hrows:
            ts = pd.Timestamp(d_)
            px = float(spx_px.loc[ts]) if spx_px is not None and ts in spx_px.index else None
            regime_history.append({"d": d_, "score": round(float(sm), 2),
                                   "raw": round(float(rw), 2), "status": st,
                                   "spx": round(px, 2) if px is not None else None})
    except Exception as e:
        print(f"[macro] regime_history unavailable: {e}")

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
        "vix9d": v9, "vix30d": vix_close, "vix3m": vix3_close, "vix6m": v6m, "vvix": vvix_close,
        "breadth_pct200": breadth_pct200, "spy_vs_200": spy_vs_200,
        "ratios": [{"pair": r["pair"], "value": r["value"], "direction": r["direction"],
                    "weight": r.get("weight"), "weighted_score": r.get("weighted_score")}
                   for r in ratios]})

    return {
        "updated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system_health": {"data_status": "stale" if stale else "ok", "last_close": last_d},
        "global_regime": {"status": status, "score": total, "raw_score": round(raw_score, 2),
                          "smooth_span": {"down": SPAN_DOWN, "up": SPAN_UP}, "delta_1d": score_delta,
                          "min": -11, "max": 10,
                          "breadth_engine_regime": breadth_regime,
                          "exit_unconfirmed": exit_unconfirmed,
                          "components": {"intermarket": ratio_component, "vix": vix_score,
                                         "trend": trend_score, "breadth": breadth_score},
                          "bands": {"full_on": B_FULL_ON, "moderate_on": B_MOD_ON, "full_off": B_FULL_OFF, "hysteresis": HYST},
                          "raw_status": raw_status, "dwell_held": dwell_held,
                          "slope_5d": slope, "neutral_dir": neutral_dir,
                          "min_dwell_off": MIN_DWELL_OFF, "prev_run_days": prev_run,
                          "agrees": _agreement(status, breadth_regime)},
        "absolute_trend": {"spy_vs_200sma_pct": spy_vs_200, "trend_score": trend_score,
                           "breadth_pct200": round(breadth_pct200, 1) if breadth_pct200 is not None else None,
                           "breadth_pct50": round(breadth_pct50, 1) if breadth_pct50 is not None else None,
                           "breadth_score": breadth_score},
        "score_history": score_history,
        "regime_history": regime_history,
        "intermarket_ratios": ratios,
        "volatility": {"vix9d": v9, "vix_close": vix_close, "vix3m_close": vix3_close,
                       "vix6m": v6m, "vvix": vvix_close, "vvix_10d_ago": vvix_10d,
                       "curve": curve, "segments": segs, "inverted_segments": inverted,
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

def replay(conn, start):
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
    names = [t for t in universe() if t not in NON_MEMBERS]
    conn.execute("CREATE TABLE IF NOT EXISTS macro_smooth (d TEXT PRIMARY KEY, s REAL)")
    conn.execute("DELETE FROM macro_smooth WHERE d >= ?", (str(pd.Timestamp(start).date()),))
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
                    help="recompute macro scores from this date forward (no lookahead)")
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

    if args.replay:
        conn = sqlite3.connect(DB_PATH)
        replay(conn, args.replay)
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
