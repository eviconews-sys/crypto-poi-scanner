import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# MEXC public USDT perpetual futures API
BASE_URL = "https://contract.mexc.com"
TIMEOUT = 12
USER_AGENT = "MEXC-MultiTF-POI-Scanner/2.0"

POI_TFS = {
    "4H": ("Hour4", 4 * 60 * 60),
    "1H": ("Min60", 60 * 60),
    "15M": ("Min15", 15 * 60),
    "5M": ("Min5", 5 * 60),
}
ENTRY_TFS = {
    "1M": ("Min1", 60),
    "3M": ("Min3", 3 * 60),
    "5M": ("Min5", 5 * 60),
    "15M": ("Min15", 15 * 60),
    "1H": ("Min60", 60 * 60),
    "4H": ("Hour4", 4 * 60 * 60),
}
PREFERRED_ENTRY = {"4H": ["1H"], "1H": ["15M"], "15M": ["5M", "3M"], "5M": ["3M", "1M"]}
POI_PRIORITY = {
    "Fibonacci golden zone": 10,
    "Liquidity pool": 9,
    "Structure support/resistance": 8,
    "Order block + FVG": 7,
    "Order block": 6,
    "Fair value gap": 6,
    "Breaker-like level": 5,
    "Breakout retest": 4,
    "Chart pattern": 3,
    "Trendline break": 2,
    "Climactic action bar": 1,
    "Supply/demand zone": 1,
}
TRIGGER_PRIORITY = {
    "Climactic action bar": 6,
    "Two-bar reversal": 5,
    "Engulfing candle": 4,
    "Effort vs result": 3,
    "No-supply/no-demand test": 2,
    "Shakeout/upthrust": 1,
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def api_get(path, params=None):
    response = SESSION.get(BASE_URL + path, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        if payload.get("success") is False:
            raise RuntimeError(str(payload.get("message", "MEXC API error")))
        return payload.get("data", payload)
    return payload


@st.cache_data(ttl=20, show_spinner=False)
def get_contracts():
    data = api_get("/api/v1/contract/detail")
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response from MEXC contract-detail endpoint.")
    contracts = []
    for x in data:
        symbol = str(x.get("symbol", ""))
        quote = str(x.get("quoteCoin", ""))
        settle = str(x.get("settleCoin", ""))
        state = x.get("state", 0)
        # MEXC contract state 0 is generally enabled; tolerate missing state.
        enabled = state in (0, "0", None, "")
        if symbol.endswith("_USDT") and quote == "USDT" and settle == "USDT" and enabled:
            contracts.append(symbol)
    return sorted(set(contracts))


@st.cache_data(ttl=10, show_spinner=False)
def get_tickers():
    data = api_get("/api/v1/contract/ticker")
    if isinstance(data, dict):
        data = [data]
    result = {}
    for item in data if isinstance(data, list) else []:
        symbol = item.get("symbol")
        try:
            price = float(item.get("lastPrice"))
            volume = float(item.get("amount24", item.get("volume24", 0)) or 0)
            bid = float(item.get("bid1", 0) or 0)
            ask = float(item.get("ask1", 0) or 0)
            result[symbol] = {
                "price": price, "volume": volume, "bid": bid, "ask": ask,
                "spread_pct": ((ask - bid) / price * 100) if price and ask >= bid > 0 else None,
            }
        except (TypeError, ValueError):
            continue
    return result


@st.cache_data(ttl=15, show_spinner=False)
def get_candles(symbol, interval, seconds, limit=120):
    now = int(time.time())
    start = now - seconds * (limit + 4)
    # MEXC futures does not consistently expose a native 3-minute interval.
    # Build 3-minute candles from 1-minute candles instead.
    if interval == "Min3":
        base_seconds = 60
        base_start = now - base_seconds * (limit * 3 + 6)
        data = api_get(
            f"/api/v1/contract/kline/{symbol}",
            {"interval": "Min1", "start": base_start, "end": now},
        )
        if not isinstance(data, dict) or not data.get("time"):
            return pd.DataFrame()
        n = min(len(data.get("time", [])), len(data.get("open", [])),
                len(data.get("high", [])), len(data.get("low", [])),
                len(data.get("close", [])), len(data.get("vol", [])))
        base = pd.DataFrame({
            "time": data["time"][:n], "open": data["open"][:n],
            "high": data["high"][:n], "low": data["low"][:n],
            "close": data["close"][:n], "volume": data["vol"][:n],
        })
        for col in ["time", "open", "high", "low", "close", "volume"]:
            base[col] = pd.to_numeric(base[col], errors="coerce")
        base = base.dropna().sort_values("time").drop_duplicates("time")
        base = base[base["time"] + 60 <= now]
        if base.empty:
            return pd.DataFrame()
        base["bucket"] = (base["time"] // 180) * 180
        grouped = base.groupby("bucket", as_index=False).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum"),
        )
        counts = base.groupby("bucket").size()
        grouped = grouped[grouped["bucket"].map(counts) == 3]
        grouped = grouped.rename(columns={"bucket": "time"})
        grouped = grouped[grouped["time"] + 180 <= now]
        return grouped.tail(limit).reset_index(drop=True)

    data = api_get(
        f"/api/v1/contract/kline/{symbol}",
        {"interval": interval, "start": start, "end": now},
    )
    if not isinstance(data, dict) or not data.get("time"):
        return pd.DataFrame()
    n = min(len(data.get("time", [])), len(data.get("open", [])),
            len(data.get("high", [])), len(data.get("low", [])),
            len(data.get("close", [])), len(data.get("vol", [])))
    if n == 0:
        return pd.DataFrame()
    df = pd.DataFrame({
        "time": data["time"][:n], "open": data["open"][:n],
        "high": data["high"][:n], "low": data["low"][:n],
        "close": data["close"][:n], "volume": data["vol"][:n],
    })
    for col in ["time", "open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates("time")
    # Only completed candles are used for patterns and structural calculations.
    df = df[df["time"] + seconds <= now]
    return df.tail(limit).reset_index(drop=True)


def candle_time(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def trend_info(df):
    if len(df) < 30:
        return "Unclear", 0
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    last = float(close.iloc[-1])
    if last > ema20 > ema50:
        return "Uptrend", 2
    if last < ema20 < ema50:
        return "Downtrend", 2
    if last > ema50:
        return "Mild uptrend", 1
    if last < ema50:
        return "Mild downtrend", 1
    return "Sideways / unclear", 0


def pivots(df, lookback=70):
    d = df.tail(lookback).reset_index(drop=True)
    highs, lows = [], []
    for i in range(2, len(d) - 2):
        h = float(d.loc[i, "high"])
        l = float(d.loc[i, "low"])
        if h >= float(d.loc[i-2:i+2, "high"].max()):
            highs.append((i, h, int(d.loc[i, "time"])))
        if l <= float(d.loc[i-2:i+2, "low"].min()):
            lows.append((i, l, int(d.loc[i, "time"])))
    return d, highs, lows


def make_pois(df):
    if len(df) < 35:
        return []
    d, highs, lows = pivots(df)
    ranges = (d["high"] - d["low"]).tail(20)
    atr = float(ranges.mean())
    if atr <= 0:
        return []
    price = float(d["close"].iloc[-1])
    trend, _ = trend_info(df)
    pois = []

    def add(name, side, low, high, ts, fresh=True, note=""):
        low, high = sorted((float(low), float(high)))
        if high <= low or low <= 0:
            return
        # Discard zones price has already closed through decisively.
        if side == "LONG" and price < low - atr * 0.15:
            return
        if side == "SHORT" and price > high + atr * 0.15:
            return
        # Only keep current-price zones and nearby approaching zones.
        distance = max(low - price, 0, price - high)
        if distance > atr * 0.65:
            return
        inside = low <= price <= high
        # Recent touch count: distinguish tested from consumed.
        after = d[d["time"] > ts]
        touches = int(((after["low"] <= high) & (after["high"] >= low)).sum())
        # Repeated contacts are lower quality; do not automatically delete first touch.
        if touches >= 4:
            return
        poi_rank = POI_PRIORITY.get(name, 1)
        trend_match = (side == "LONG" and "uptrend" in trend.lower()) or (
            side == "SHORT" and "downtrend" in trend.lower()
        )
        score = poi_rank * 5 + (12 if trend_match else 0) - touches * 4
        if inside:
            score += 10
        else:
            score -= min(8, int(distance / atr * 8))
        pois.append({
            "type": name, "side": side, "low": low, "high": high,
            "time": int(ts), "inside": inside, "distance": distance,
            "atr": atr, "touches": touches, "score": score,
            "note": note, "trend": trend,
        })

    # Fibonacci golden zone from latest meaningful swing high/low.
    if highs and lows:
        latest_high = highs[-1]
        latest_low = lows[-1]
        lo_price, hi_price = latest_low[1], latest_high[1]
        if latest_low[0] < latest_high[0] and hi_price > lo_price:
            diff = hi_price - lo_price
            # Up-swing retracement: long golden zone.
            z1, z2 = hi_price - diff * 0.618, hi_price - diff * 0.5
            add("Fibonacci golden zone", "LONG", z1, z2, latest_high[2],
                note="0.50–0.618 retracement of latest swing")
        elif latest_high[0] < latest_low[0] and hi_price > lo_price:
            diff = hi_price - lo_price
            z1, z2 = lo_price + diff * 0.5, lo_price + diff * 0.618
            add("Fibonacci golden zone", "SHORT", z1, z2, latest_low[2],
                note="0.50–0.618 retracement of latest swing")

    # Liquidity and structure at confirmed swing levels.
    for idx, level, ts in highs[-8:]:
        add("Liquidity pool", "SHORT", level - atr * .12, level + atr * .12, ts,
            note="Recent swing-high liquidity")
        add("Structure support/resistance", "SHORT", level - atr * .15, level + atr * .15, ts)
    for idx, level, ts in lows[-8:]:
        add("Liquidity pool", "LONG", level - atr * .12, level + atr * .12, ts,
            note="Recent swing-low liquidity")
        add("Structure support/resistance", "LONG", level - atr * .15, level + atr * .15, ts)

    # Order blocks: last opposing candle before a local break of structure.
    for i in range(max(5, len(d) - 35), len(d) - 1):
        prior_high = float(d.loc[max(0, i-10):i-1, "high"].max())
        prior_low = float(d.loc[max(0, i-10):i-1, "low"].min())
        o, c = float(d.loc[i, "open"]), float(d.loc[i, "close"])
        nxt_close = float(d.loc[i+1, "close"])
        if c < o and nxt_close > prior_high:
            add("Order block", "LONG", float(d.loc[i, "low"]), float(d.loc[i, "high"]),
                int(d.loc[i, "time"]), note="Last bearish candle before upside break")
        if c > o and nxt_close < prior_low:
            add("Order block", "SHORT", float(d.loc[i, "low"]), float(d.loc[i, "high"]),
                int(d.loc[i, "time"]), note="Last bullish candle before downside break")

    # Three-candle FVG; combine with nearby OB where zones overlap/touch.
    for i in range(max(2, len(d) - 40), len(d)):
        if i < 2:
            continue
        left_high = float(d.loc[i-2, "high"])
        left_low = float(d.loc[i-2, "low"])
        cur_high = float(d.loc[i, "high"])
        cur_low = float(d.loc[i, "low"])
        ts = int(d.loc[i, "time"])
        if cur_low > left_high:
            add("Fair value gap", "LONG", left_high, cur_low, ts, note="Bullish 3-candle gap")
        if cur_high < left_low:
            add("Fair value gap", "SHORT", cur_high, left_low, ts, note="Bearish 3-candle gap")

    # Range breakout/retest, trendline break proxy, and supply/demand zones.
    recent = d.tail(20)
    prior = d.iloc[-21:-1]
    if len(prior):
        range_high, range_low = float(prior["high"].max()), float(prior["low"].min())
        last = d.iloc[-1]
        if float(last["close"]) > range_high:
            add("Breakout retest", "LONG", range_high - atr*.15, range_high + atr*.15,
                int(last["time"]), note="Close broke recent range high")
        if float(last["close"]) < range_low:
            add("Breakout retest", "SHORT", range_low - atr*.15, range_low + atr*.15,
                int(last["time"]), note="Close broke recent range low")

    # CAB / climactic bar heuristic: unusually wide candle and high volume.
    vol_avg = float(d["volume"].tail(20).mean())
    for i in range(max(2, len(d)-8), len(d)-1):
        row = d.iloc[i]
        bar_range = float(row["high"] - row["low"])
        if bar_range > atr * 1.7 and vol_avg > 0 and float(row["volume"]) > vol_avg * 1.5:
            side = "LONG" if float(row["close"]) > (float(row["high"] + row["low"]) / 2) else "SHORT"
            add("Climactic action bar", side, float(row["low"]), float(row["high"]),
                int(row["time"]), note="Wide-range, high-volume CAB heuristic")

    # Supply/demand fallback around the most recent wide candles.
    for i in range(max(1, len(d)-12), len(d)-1):
        row = d.iloc[i]
        bar_range = float(row["high"] - row["low"])
        if bar_range >= atr * 1.4:
            side = "LONG" if float(row["close"]) > float(row["open"]) else "SHORT"
            add("Supply/demand zone", side, float(row["low"]), float(row["high"]),
                int(row["time"]), note="Wide-range candle zone")

    # Mark nearby OB/FVG combinations as highest-confidence confluence label.
    for p in pois:
        if p["type"] not in ("Order block", "Fair value gap"):
            continue
        for q in pois:
            if q is p or q["side"] != p["side"]:
                continue
            if q["type"] in ("Order block", "Fair value gap"):
                if max(p["low"], q["low"]) <= min(p["high"], q["high"]) + atr * .12:
                    p["score"] += 10
                    p["note"] = (p["note"] + "; OB/FVG confluence").strip("; ")

    # Deduplicate same-type zones with nearly identical price bounds.
    pois.sort(key=lambda x: x["score"], reverse=True)
    unique = []
    for p in pois:
        duplicate = any(
            q["side"] == p["side"] and q["type"] == p["type"]
            and abs(q["low"] - p["low"]) <= atr * .08
            and abs(q["high"] - p["high"]) <= atr * .08
            for q in unique
        )
        if not duplicate:
            unique.append(p)
    return unique


def detect_triggers(df):
    out = []
    if len(df) < 8:
        return out
    a, b = df.iloc[-2], df.iloc[-1]
    o1, c1 = float(a["open"]), float(a["close"])
    o2, c2 = float(b["open"]), float(b["close"])
    h1, l1 = float(a["high"]), float(a["low"])
    h2, l2 = float(b["high"]), float(b["low"])
    body2 = abs(c2-o2)
    rng2 = max(h2-l2, 1e-12)
    lower_wick = min(o2, c2)-l2
    upper_wick = h2-max(o2, c2)
    prior = df.iloc[-7:-2]
    avg_range = float((prior["high"]-prior["low"]).mean())
    avg_vol = float(prior["volume"].mean())
    vol2 = float(b["volume"])

    # Highest priority: CAB heuristic, high volume and wide candle.
    if avg_range > 0 and avg_vol > 0 and rng2 > avg_range*1.7 and vol2 > avg_vol*1.5:
        out.append(("Climactic action bar", "LONG" if c2 > (h2+l2)/2 else "SHORT"))

    if c1 < o1 and c2 > o2 and c2 > (o1+c1)/2:
        out.append(("Two-bar reversal", "LONG"))
    if c1 > o1 and c2 < o2 and c2 < (o1+c1)/2:
        out.append(("Two-bar reversal", "SHORT"))

    if c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1:
        out.append(("Engulfing candle", "LONG"))
    if c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1:
        out.append(("Engulfing candle", "SHORT"))

    if avg_vol > 0 and avg_range > 0 and vol2 > avg_vol*1.5 and rng2 < avg_range*.8:
        if lower_wick > upper_wick and c2 > o2:
            out.append(("Effort vs result", "LONG"))
        elif upper_wick > lower_wick and c2 < o2:
            out.append(("Effort vs result", "SHORT"))

    if avg_vol > 0 and avg_range > 0 and rng2 < avg_range*.75 and vol2 < avg_vol*.75:
        if c2 > o2 and c2 >= l2+rng2*.65:
            out.append(("No-supply/no-demand test", "LONG"))
        if c2 < o2 and c2 <= l2+rng2*.35:
            out.append(("No-supply/no-demand test", "SHORT"))

    if l2 < l1 and c2 > l1 and lower_wick > body2 and c2 > o2:
        out.append(("Shakeout/upthrust", "LONG"))
    if h2 > h1 and c2 < h1 and upper_wick > body2 and c2 < o2:
        out.append(("Shakeout/upthrust", "SHORT"))

    # Deduplicate; preserve priority order.
    result, seen = [], set()
    for name, side in out:
        if (name, side) not in seen:
            result.append((name, side))
            seen.add((name, side))
    return result


def entry_timeframes(poi_tf, same_tf=True):
    preferred = PREFERRED_ENTRY[poi_tf]
    values = preferred.copy()
    if same_tf and poi_tf not in values:
        values.append(poi_tf)
    # Keep all trigger choices available as fallback, but score preferred higher.
    for tf in ENTRY_TFS:
        if tf not in values:
            values.append(tf)
    return values


def scan_contract(symbol, ticker, enabled_poi_tfs, same_tf, trend_filter, max_spread):
    price = ticker["price"]
    spread = ticker.get("spread_pct")
    if spread is not None and spread > max_spread:
        return [], None
    rows = []
    trend_by_tf = {}
    for poi_tf in enabled_poi_tfs:
        interval, seconds = POI_TFS[poi_tf]
        df = get_candles(symbol, interval, seconds, limit=120)
        if df.empty or len(df) < 35:
            continue
        trend, trend_strength = trend_info(df)
        trend_by_tf[poi_tf] = trend
        pois = make_pois(df)
        for poi in pois:
            # Re-evaluate zone distance using live ticker price rather than candle close.
            low, high = poi["low"], poi["high"]
            atr = poi["atr"]
            distance = max(low-price, 0, price-high)
            if distance > atr * 0.65:
                continue
            inside = low <= price <= high
            entry_candidates = entry_timeframes(poi_tf, same_tf)
            best_trigger = None
            best_entry_tf = None
            best_entry_time = None
            best_trigger_score = 0
            for entry_tf in entry_candidates:
                entry_interval, entry_seconds = ENTRY_TFS[entry_tf]
                # Avoid duplicate request when same candle interval is already available.
                edf = df if entry_interval == interval else get_candles(
                    symbol, entry_interval, entry_seconds, limit=100
                )
                if edf.empty:
                    continue
                triggers = detect_triggers(edf)
                matching = [(name, side) for name, side in triggers if side == poi["side"]]
                if not matching:
                    continue
                name, side = sorted(
                    matching, key=lambda x: TRIGGER_PRIORITY.get(x[0], 0), reverse=True
                )[0]
                alignment_bonus = 18 if entry_tf in PREFERRED_ENTRY[poi_tf] else (
                    8 if entry_tf == poi_tf else 0
                )
                trigger_score = TRIGGER_PRIORITY.get(name, 0)*4 + alignment_bonus
                if trigger_score > best_trigger_score:
                    best_trigger_score = trigger_score
                    best_trigger = name
                    best_entry_tf = entry_tf
                    best_entry_time = int(edf.iloc[-1]["time"])

            trend_text = trend
            aligned = (poi["side"] == "LONG" and "uptrend" in trend.lower()) or (
                poi["side"] == "SHORT" and "downtrend" in trend.lower()
            )
            if trend_filter and not aligned:
                continue
            status = "INSIDE ZONE" if inside else "APPROACHING"
            if poi["touches"] >= 2:
                freshness = f"Tested ({poi['touches']} touches)"
            elif poi["touches"] == 1:
                freshness = "Touched once"
            else:
                freshness = "Fresh / untested"
            alignment = "Preferred alignment" if best_entry_tf in PREFERRED_ENTRY.get(poi_tf, []) else (
                "Same timeframe" if best_entry_tf == poi_tf else "Other timeframe"
            )
            score = poi["score"] + (20 if aligned else -6) + best_trigger_score
            if inside:
                score += 10
            if spread is not None and spread < 0.08:
                score += 5
            if ticker["volume"] > 0:
                score += min(10, int(ticker["volume"] / 1_000_000))
            rows.append({
                "Score": score,
                "Symbol": symbol,
                "Live price": price,
                "Side": poi["side"],
                "POI TF": poi_tf,
                "POI type": poi["type"],
                "Zone low": round(low, 10),
                "Zone high": round(high, 10),
                "Price status": status,
                "Distance %": round(distance / price * 100, 3) if price else None,
                "Trend": trend_text,
                "Trend aligned": "Yes" if aligned else "No",
                "Freshness": freshness,
                "Entry TF": best_entry_tf or "Waiting",
                "Entry trigger": best_trigger or "No trigger yet",
                "Entry alignment": alignment if best_entry_tf else "Waiting",
                "POI priority": poi["type"],
                "Trigger priority": best_trigger or "None",
                "POI candle UTC": candle_time(poi["time"]),
                "Entry candle UTC": candle_time(best_entry_time) if best_entry_time else "Not confirmed",
                "Notes": poi["note"],
                "Spread %": round(spread, 4) if spread is not None else None,
            })
    return rows, trend_by_tf


st.set_page_config(page_title="MEXC Live POI Scanner", page_icon="📈", layout="wide")
st.title("MEXC Futures — Live POI & Entry Scanner")
st.caption("Rule-based research scanner only. It does not place orders or guarantee trades.")

with st.sidebar:
    st.header("Scan settings")
    selected_tfs = st.multiselect(
        "POI timeframes to scan",
        list(POI_TFS.keys()),
        default=["4H", "1H", "15M", "5M"],
    )
    same_tf = st.checkbox("Allow entry trigger on same timeframe", value=True)
    trend_filter = st.checkbox("Only show POIs aligned with their timeframe trend", value=False)
    max_spread = st.slider("Maximum allowed spread (%)", 0.02, 1.0, 0.25, 0.01)
    workers = st.slider("Parallel requests (higher = faster, more rate-limit risk)", 1, 8, 4)
    max_contracts = st.number_input(
        "Contract limit (0 = all eligible)", min_value=0, max_value=3000, value=0, step=100,
        help="All contracts may take a long time and may be rate-limited. Start with 100–300 to test."
    )
    st.caption("Preferred alignment: 4H→1H; 1H→15M; 15M→5M/3M; 5M→3M/1M.")

st.markdown(
    "**POI priority:** Fibonacci golden zone → liquidity → structure → OB/FVG → breaker-like "
    "levels → breakout → chart-pattern proxy → trendline-break proxy → CAB → supply/demand.  \n"
    "**Entry priority:** CAB → two-bar reversal → engulfing → effort/result → no-supply/no-demand → shakeout/upthrust."
)

if st.button("Scan all selected MEXC markets", type="primary", use_container_width=True):
    if not selected_tfs:
        st.warning("Select at least one POI timeframe.")
    else:
        try:
            with st.spinner("Loading MEXC contract list and live tickers..."):
                contracts = get_contracts()
                tickers = get_tickers()
            eligible = [
                s for s in contracts
                if s in tickers and tickers[s]["price"] > 0 and tickers[s]["volume"] > 0
            ]
            # Liquidity-first ordering by 24h reported amount/turnover.
            eligible.sort(key=lambda s: tickers[s]["volume"], reverse=True)
            total_eligible = len(eligible)
            if int(max_contracts) > 0:
                eligible = eligible[:int(max_contracts)]
            st.info(
                f"MEXC eligible USDT perpetuals: {total_eligible}. "
                f"Contracts selected for this scan: {len(eligible)}. "
                "The live ticker snapshot is used for inside/approaching status."
            )
            all_rows = []
            errors = 0
            progress = st.progress(0)
            status = st.empty()

            # Each worker scans one contract; individual requests are cached and timed out.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        scan_contract, symbol, tickers[symbol], selected_tfs,
                        same_tf, trend_filter, max_spread
                    ): symbol for symbol in eligible
                }
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    symbol = futures[future]
                    status.write(f"Scanning {symbol} ({completed}/{len(eligible)})…")
                    try:
                        rows, _ = future.result()
                        all_rows.extend(rows)
                    except Exception:
                        errors += 1
                    progress.progress(completed / max(1, len(eligible)))

            progress.empty()
            status.empty()
            all_rows.sort(key=lambda r: r["Score"], reverse=True)
            st.session_state["results"] = all_rows
            st.session_state["scan_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            st.session_state["scan_errors"] = errors
            st.session_state["scanned_count"] = len(eligible)
            st.session_state["eligible_count"] = total_eligible
            st.session_state["selected_tfs"] = selected_tfs
        except Exception as exc:
            st.error(f"MEXC API request failed: {exc}")
            st.info("Try again later, reduce parallel requests, or check whether MEXC futures API is reachable from your host.")

if "results" in st.session_state:
    st.caption(
        f"Last scan: {st.session_state['scan_utc']} · "
        f"Contracts scanned: {st.session_state['scanned_count']}/"
        f"{st.session_state['eligible_count']} · "
        f"Contract scan errors: {st.session_state['scan_errors']}"
    )
    results = st.session_state["results"]
    if not results:
        st.warning(
            "No valid nearby POIs found under the current filters. This can be normal. "
            "Try disabling the trend-only filter, increasing the spread limit, or scanning fewer timeframes."
        )
    else:
        df = pd.DataFrame(results)
        st.subheader("Ranked live POIs")
        st.write(f"POI candidates: **{len(df)}** · Inside zones: **{int((df['Price status']=='INSIDE ZONE').sum())}** · Approaching: **{int((df['Price status']=='APPROACHING').sum())}")
        trigger_df = df[df["Entry trigger"] != "No trigger yet"]
        st.markdown("### Entry-trigger confirmed at a POI")
        if trigger_df.empty:
            st.info("No matching completed-candle trigger was detected at the selected POIs.")
        else:
            st.dataframe(trigger_df, use_container_width=True, hide_index=True)
        st.markdown("### All valid POIs near live price")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download all POIs (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="mexc_live_poi_scanner.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Important limitations: zone and trigger labels are mechanical approximations, not discretionary "
    "Wyckoff/CAB analysis. Fibonacci swing selection, liquidity, order blocks, breakers, trendlines, "
    "and chart patterns need further validation/backtesting. A score is a ranking aid, not a probability "
    "of profit. Check funding, spread, leverage, liquidation risk, and stop placement independently."
)
