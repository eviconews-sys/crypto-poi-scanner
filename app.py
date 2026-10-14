
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st

BASE = "https://api.binance.com"
TIMEFRAMES = ["15m", "1h", "4h"]
QUOTE = "USDT"

st.set_page_config(page_title="Crypto POI Scanner", page_icon="🎯", layout="wide")

st.title("🎯 Crypto POI Scanner")
st.caption("Binance Spot • Public data • Educational research tool")

st.info(
    "POIs in this version are recent swing highs/lows. "
    "These are starter rules, not personalized trading signals. "
    "CAB is not included until its rules are defined."
)

@st.cache_data(ttl=60, show_spinner=False)
def get_markets(limit):
    r = requests.get(BASE + "/api/v3/exchangeInfo", timeout=15)
    r.raise_for_status()
    info = r.json()

    allowed = {
        s["symbol"] for s in info["symbols"]
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == QUOTE
        and s.get("isSpotTradingAllowed", False)
        and not any(x in s["baseAsset"] for x in
                    ["UP", "DOWN", "BULL", "BEAR"])
    }

    r = requests.get(BASE + "/api/v3/ticker/24hr", timeout=15)
    r.raise_for_status()
    tickers = r.json()
    rows = [
        x for x in tickers
        if x["symbol"] in allowed
        and float(x.get("quoteVolume", 0)) > 0
    ]
    rows.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return rows[:limit]


def get_candles(symbol, tf):
    r = requests.get(
        BASE + "/api/v3/klines",
        params={"symbol": symbol, "interval": tf, "limit": 100},
        timeout=12,
    )
    r.raise_for_status()
    data = r.json()
    cols = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # Exclude the candle that may still be forming.
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    return df.reset_index(drop=True)


def analyze(symbol, tf, ticker):
    df = get_candles(symbol, tf)
    if len(df) < 30:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["close"])
    ranges = df["high"] - df["low"]
    atr = float(ranges.tail(14).mean())
    if atr <= 0 or price <= 0:
        return None

    # Simple three-candle swing points; use recent completed candles.
    swing_highs, swing_lows = [], []
    for i in range(max(2, len(df) - 60), len(df) - 2):
        if df["high"].iloc[i] >= max(df["high"].iloc[i-2:i+3]):
            swing_highs.append(float(df["high"].iloc[i]))
        if df["low"].iloc[i] <= min(df["low"].iloc[i-2:i+3]):
            swing_lows.append(float(df["low"].iloc[i]))

    tolerance = max(atr * 0.25, price * 0.0015)
    levels = []
    for label, values in [("Swing high", swing_highs),
                          ("Swing low", swing_lows)]:
        if values:
            level = min(values, key=lambda x: abs(x - price))
            distance = abs(price - level)
            if distance <= max(atr * 2, price * 0.015):
                levels.append((label, level, distance))

    poi = min(levels, key=lambda x: x[2]) if levels else None
    touched = bool(poi and poi[2] <= tolerance)

    o, h, l, c = [float(last[k]) for k in
                  ["open", "high", "low", "close"]]
    po, ph, pl, pc = [float(prev[k]) for k in
                      ["open", "high", "low", "close"]]

    bull_engulf = (
        pc < po and c > o and o <= pc and c >= po
    )
    bear_engulf = (
        pc > po and c < o and o >= pc and c <= po
    )

    # Two-bar reversal starter definition:
    # prior candle direction reverses, with close beyond prior midpoint.
    bull_two = pc < po and c > o and c > (po + pc) / 2
    bear_two = pc > po and c < o and c < (po + pc) / 2

    avg_range = float(ranges.iloc[-21:-1].mean())
    avg_vol = float(df["volume"].iloc[-21:-1].mean())
    small_bar = avg_range > 0 and (h - l) <= avg_range * 0.7
    low_vol = avg_vol > 0 and float(last["volume"]) <= avg_vol * 0.7

    # Heuristic candidates only; these are not standalone confirmations.
    no_supply = c < o and small_bar and low_vol and c >= l + (h-l)*0.5
    no_demand = c > o and small_bar and low_vol and c <= l + (h-l)*0.5

    patterns = []
    if bull_engulf:
        patterns.append("Bullish engulfing")
    if bear_engulf:
        patterns.append("Bearish engulfing")
    if bull_two:
        patterns.append("Bullish 2-bar reversal candidate")
    if bear_two:
        patterns.append("Bearish 2-bar reversal candidate")
    if no_supply:
        patterns.append("No-supply candidate")
    if no_demand:
        patterns.append("No-demand candidate")

    return {
        "Symbol": symbol,
        "Timeframe": tf,
        "Price": price,
        "POI": poi[0] if poi else "—",
        "POI level": poi[1] if poi else None,
        "Distance %": round(poi[2] / price * 100, 3) if poi else None,
        "POI touch": "TOUCH" if touched else (
            "Near" if poi and poi[2] <= atr else "Watch"
        ),
        "Patterns": ", ".join(patterns) if patterns else "—",
        "24h quote volume": float(ticker.get("quoteVolume", 0)),
        "Candle time (UTC)": pd.to_datetime(
            int(df["time"].iloc[-1]), unit="ms", utc=True
        ).strftime("%Y-%m-%d %H:%M"),
    }


with st.sidebar:
    st.header("Scanner settings")
    market_limit = st.slider(
        "Number of highest-volume markets", 10, 50, 25, step=5
    )
    selected_tfs = st.multiselect(
        "Timeframes", TIMEFRAMES, default=TIMEFRAMES
    )
    refresh = st.checkbox("Auto-refresh every 60 seconds", value=False)
    st.caption("Refresh fetches new data. It does not run in the background.")

if refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="60">',
        unsafe_allow_html=True,
    )

if not selected_tfs:
    st.warning("Choose at least one timeframe.")
    st.stop()

if st.button("🔄 Scan market now", type="primary") or "results" not in st.session_state:
    results = []
    errors = 0
    try:
        with st.spinner("Loading markets and scanning completed candles..."):
            markets = get_markets(market_limit)
            lookup = {x["symbol"]: x for x in markets}
            jobs = [
                (sym, tf, lookup[sym])
                for sym in lookup for tf in selected_tfs
            ]
            progress = st.progress(0)
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [
                    pool.submit(analyze, sym, tf, tick)
                    for sym, tf, tick in jobs
                ]
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        row = future.result()
                        if row:
                            results.append(row)
                    except Exception:
                        errors += 1
                    progress.progress(i / len(futures))
            st.session_state["results"] = results
            st.session_state["errors"] = errors
            st.session_state["scan_time"] = time.strftime(
                "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
            )
    except Exception as e:
        st.error("Could not load Binance data. Please try again later.")
        st.caption(str(e))

results = st.session_state.get("results", [])
if results:
    df = pd.DataFrame(results)
    st.caption(
        f"Last scan: {st.session_state.get('scan_time', '—')} · "
        f"{len(df)} market/timeframe checks · "
        f"{st.session_state.get('errors', 0)} failed requests"
    )

    touches = df[df["POI touch"] == "TOUCH"].copy()
    touches = touches.sort_values("Distance %", na_position="last")
    st.subheader("🎯 POI touches")
    st.caption("Markets whose last completed candle closed near a recent swing level.")
    if touches.empty:
        st.write("No POI touches found in this scan.")
    else:
        st.dataframe(touches, use_container_width=True, hide_index=True)

    near = df[df["POI touch"] == "Near"].copy()
    near = near.sort_values("Distance %", na_position="last")
    with st.expander("Markets approaching POIs"):
        st.dataframe(near, use_container_width=True, hide_index=True)

    setups = df[df["Patterns"] != "—"].copy()
    st.subheader("🕯️ Candle pattern candidates")
    st.caption(
        "Pattern matches are candidates, not trade instructions. "
        "Confirmation rules and market context still matter."
    )
    if setups.empty:
        st.write("No pattern candidates found in this scan.")
    else:
        st.dataframe(setups, use_container_width=True, hide_index=True)

    st.download_button(
        "Download scan as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="crypto_poi_scan.csv",
        mime="text/csv",
    )
else:
    st.write("Press **Scan market now** to start.")

st.divider()
st.caption(
    "Educational tool only. Not financial advice. "
    "POI swing rules and candle patterns are simplified and may produce false positives."
)
