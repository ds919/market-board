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
UNIVERSE  = "UNIVERSE V1"
BACKFILL_PERIOD = "2y"    # enough history for 200DMA + 52-week highs
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
NON_MEMBERS = set(INDEX_SYMS) | set(SECTOR_SYMS) | set(MACRO_SYMS)

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
    return sorted(set([s for m in active_themes().values() for s in m] + INDEX_SYMS + SECTOR_SYMS + MACRO_SYMS))

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
    import yfinance as yf
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
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
def compute_and_emit(conn):
    today = conn.execute("SELECT max(d) FROM prices").fetchone()[0]
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
