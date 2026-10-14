
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# MEXC public perpetual-futures API
BASE_URL = "https://contract.mexc.com"
TIMEOUT = 12

st.set_page_config(
    page_title="MEXC Futures POI Scanner",
    page_icon="📊",
    layout="wide",
)

POI_INTERVALS = {
    "15 minutes": ("Min15", 900),
    "30 minutes": ("Min30", 1800),
    "1 hour": ("Min60", 3600),
    "4 hours": ("Hour4", 14400),
    "1 day": ("Day1", 86400),
}

ENTRY_INTERVALS = {
    "1 minute": ("Min1", 60),
    "5 minutes": ("Min5", 300),
    "15 minutes": ("Min15", 900),
    "30 minutes": ("Min30", 1800),
    "1 hour": ("Min60", 3600),
    "4 hours": ("Hour4", 14400),
}


def api_get(path, params=None):
    response = requests.get(
        BASE_URL + path,
        params=params,
        timeout=TIMEOUT,
        headers={"User-Agent": "MEXC-POI-Scanner/1.0"},
    )
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(payload.get("message", "MEXC API error"))

    return payload.get("data", payload)


@st.cache_data(ttl=30, show_spinner=False)
def get_contracts():
    """Get eligible USDT-settled perpetual contracts."""
    data = api_get("/api/v1/contract/detail")
    if not isinstance(data, list):
        raise RuntimeError("Unexpected contract-list response from MEXC.")

    contracts = []
    for item in data:
        symbol = str(item.get("symbol", ""))
        quote = str(item.get("quoteCoin", ""))
        settle = str(item.get("settleCoin", ""))

        if (
            symbol.endswith("_USDT")
            and quote == "USDT"
            and settle == "USDT"
        ):
            contracts.append(symbol)

    return sorted(set(contracts))


def get_tickers():
    """Get the current public ticker snapshot."""
    data = api_get("/api/v1/contract/ticker")
    if isinstance(data, dict):
        data = [data]

    result = {}
    for item in data if isinstance(data, list) else []:
        symbol = item.get("symbol")
        price = item.get("lastPrice")
        if symbol and price is not None:
            try:
                result[symbol] = {
                    "price": float(price),
                    "volume": float(item.get("amount24", 0) or 0),
                    "timestamp": item.get("timestamp"),
                }
            except (TypeError, ValueError):
                continue
    return result


def get_candles(symbol, interval, seconds, limit=100):
    """Fetch candles and remove the still-forming candle."""
    now = int(time.time())
    start = now - seconds * (limit + 5)

    data = api_get(
        f"/api/v1/contract/kline/{symbol}",
        params={
            "interval": interval,
            "start": start,
            "end": now,
        },
    )

    if not isinstance(data, dict) or not data.get("time"):
        return pd.DataFrame()

    df = pd.DataFrame({
        "time": data.get("time", []),
        "open": data.get("open", []),
        "high": data.get("high", []),
        "low": data.get("low", []),
        "close": data.get("close", []),
        "volume": data.get("vol", []),
    })

    if df.empty:
        return df

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates("time")

    # A candle is complete only after its full interval has elapsed.
    df = df[df["time"] + seconds <= now]

    return df.tail(limit).reset_index(drop=True)


def find_pois(df):
    """Find recent confirmed swing highs/lows as simple POI levels."""
    if len(df) < 15:
        return []

    recent = df.tail(80).reset_index(drop=True)
    ranges = (recent["high"] - recent["low"]).tail(20)
    atr = float(ranges.mean()) if not ranges.empty else 0.0

    if atr <= 0:
        return []

    pois = []
    # A pivot needs two candles on both sides; this avoids using
    # unconfirmed swing points at the very end of the chart.
    for i in range(max(2, len(recent) - 60), len(recent) - 2):
        high = float(recent.loc[i, "high"])
        low = float(recent.loc[i, "low"])

        high_window = recent.loc[i - 2:i + 2, "high"]
        low_window = recent.loc[i - 2:i + 2, "low"]

        if high >= float(high_window.max()):
            pois.append({
                "side": "SHORT",
                "type": "Swing-high / supply area",
                "low": high - atr * 0.20,
                "high": high + atr * 0.10,
                "pivot_time": int(recent.loc[i, "time"]),
            })

        if low <= float(low_window.min()):
            pois.append({
                "side": "LONG",
                "type": "Swing-low / demand area",
                "low": low - atr * 0.10,
                "high": low + atr * 0.20,
                "pivot_time": int(recent.loc[i, "time"]),
            })

    # Keep the most recent levels first.
    return sorted(pois, key=lambda x: x["pivot_time"], reverse=True)


def candle_triggers(df):
    """
    Rule-based entry candidates using completed candles only.
    These are screening heuristics, not proof of market intent.
    """
    found = []
    if len(df) < 5:
        return found

    a = df.iloc[-2]  # previous completed candle
    b = df.iloc[-1]  # latest completed candle

    def bullish(c):
        return float(c["close"]) > float(c["open"])

    def bearish(c):
        return float(c["close"]) < float(c["open"])

    def body(c):
        return abs(float(c["close"]) - float(c["open"]))

    def span(c):
        return max(float(c["high"]) - float(c["low"]), 1e-12)

    # Bullish / bearish engulfing
    if (
        bearish(a) and bullish(b)
        and float(b["open"]) <= float(a["close"])
        and float(b["close"]) >= float(a["open"])
    ):
        found.append(("Bullish engulfing", "LONG"))

    if (
        bullish(a) and bearish(b)
        and float(b["open"]) >= float(a["close"])
        and float(b["close"]) <= float(a["open"])
    ):
        found.append(("Bearish engulfing", "SHORT"))

    # Two-candle reversal: prior direction followed by a strong close
    if (
        bearish(a) and bullish(b)
        and float(b["close"]) > (float(a["open"]) + float(a["close"])) / 2
    ):
        found.append(("Bullish two-bar reversal", "LONG"))

    if (
        bullish(a) and bearish(b)
        and float(b["close"]) < (float(a["open"]) + float(a["close"])) / 2
    ):
        found.append(("Bearish two-bar reversal", "SHORT"))

    # Rejection / shakeout / upthrust heuristics
    lower_wick = min(float(b["open"]), float(b["close"])) - float(b["low"])
    upper_wick = float(b["high"]) - max(float(b["open"]), float(b["close"]))

    if (
        float(b["low"]) < float(a["low"])
        and float(b["close"]) > float(a["low"])
        and lower_wick > body(b)
        and bullish(b)
    ):
        found.append(("Shakeout / spring-like rejection", "LONG"))

    if (
        float(b["high"]) > float(a["high"])
        and float(b["close"]) < float(a["high"])
        and upper_wick > body(b)
        and bearish(b)
    ):
        found.append(("Upthrust-like rejection", "SHORT"))

    # Low-volume narrow-range candle: context-dependent no-demand/supply.
    prior = df.iloc[-6:-1]
    avg_range = float((prior["high"] - prior["low"]).mean())
    avg_volume = float(prior["volume"].mean())
    current_range = span(b)
    current_volume = float(b["volume"])

    if avg_range > 0 and avg_volume > 0:
        if (
            current_range < avg_range * 0.75
            and current_volume < avg_volume * 0.75
            and bullish(b)
            and float(b["close"]) >= float(b["low"]) + current_range * 0.65
        ):
            found.append(("Possible no-supply / low-volume test", "LONG"))

        if (
            current_range < avg_range * 0.75
            and current_volume < avg_volume * 0.75
            and bearish(b)
            and float(b["close"]) <= float(b["low"]) + current_range * 0.35
        ):
            found.append(("Possible no-demand / low-volume test", "SHORT"))

    # Effort-vs-result approximation:
    # unusually high volume with a small candle body. This needs context.
    avg_vol_all = float(prior["volume"].mean())
    avg_span = float((prior["high"] - prior["low"]).mean())
    if (
        avg_vol_all > 0 and avg_span > 0
        and current_volume > avg_vol_all * 1.5
        and current_range < avg_span * 0.8
    ):
        if lower_wick > upper_wick and bullish(b):
            found.append(("Effort-vs-result: possible buying absorption", "LONG"))
        elif upper_wick > lower_wick and bearish(b):
            found.append(("Effort-vs-result: possible selling absorption", "SHORT"))

    # Simple continuation: three-candle directional sequence with a new close extreme.
    c = df.iloc[-3]
    if bullish(c) and bullish(a) and bullish(b) and float(b["close"]) > float(a["high"]):
        found.append(("Bullish continuation breakout", "LONG"))

    if bearish(c) and bearish(a) and bearish(b) and float(b["close"]) < float(a["low"]):
        found.append(("Bearish continuation breakdown", "SHORT"))

    # Deduplicate while preserving order.
    unique = []
    seen = set()
    for name, direction in found:
        if (name, direction) not in seen:
            unique.append((name, direction))
            seen.add((name, direction))

    return unique


def scan_symbol(symbol, ticker, poi_interval, poi_seconds,
                entry_interval, entry_seconds):
    """Scan the higher-timeframe POI first; fetch entry candles only at a POI."""
    price = ticker["price"]

    poi_df = get_candles(symbol, poi_interval, poi_seconds, limit=100)
    if poi_df.empty:
        return []

    pois = find_pois(poi_df)
    if not pois:
        return []

    # Avoid flooding results with old levels: retain only the nearest recent zones.
    recent_pois = pois[:12]
    rows = []

    for poi in recent_pois:
        zone_low = float(poi["low"])
        zone_high = float(poi["high"])

        # Current ticker price must be inside or very close to the zone.
        zone_width = max(zone_high - zone_low, price * 0.001)
        distance = max(zone_low - price, 0, price - zone_high)
        if distance > zone_width * 1.5:
            continue

        status = "At POI — waiting for trigger"
        triggers_text = "None confirmed"
        entry_time = None

        # Only fetch lower-timeframe candles if price is at/near this POI.
        entry_df = get_candles(
            symbol, entry_interval, entry_seconds, limit=100
        )

        if not entry_df.empty:
            entry_time = int(entry_df.iloc[-1]["time"])
            triggers = candle_triggers(entry_df)

            matching = [
                name for name, direction in triggers
                if direction == poi["side"]
            ]

            if matching:
                status = "Trigger confirmed (rule-based)"
                triggers_text = ", ".join(matching)

        rows.append({
            "Symbol": symbol,
            "Direction": poi["side"],
            "Current price": price,
            "POI type": poi["type"],
            "POI low": zone_low,
            "POI high": zone_high,
            "Distance to POI %": round(distance / price * 100, 3)
                if price else None,
            "Status": status,
            "15m / entry trigger": triggers_text,
            "POI candle time UTC": datetime.fromtimestamp(
                poi["pivot_time"], timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "Entry candle time UTC": (
                datetime.fromtimestamp(entry_time, timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                ) if entry_time else "Not checked"
            ),
        })

        # One entry-candle request per symbol is enough for this initial version.
        break

    return rows


st.title("MEXC Futures POI Scanner")
st.caption(
    "1H/selected higher-timeframe POIs + independent entry-trigger timeframe. "
    "Scanner only — it does not place orders."
)

with st.sidebar:
    st.header("Scan settings")
    poi_label = st.selectbox(
        "1. POI timeframe",
        list(POI_INTERVALS.keys()),
        index=2,
    )
    entry_label = st.selectbox(
        "2. Entry-trigger timeframe",
        list(ENTRY_INTERVALS.keys()),
        index=2,
    )
    max_contracts = st.number_input(
        "Maximum contracts per scan",
        min_value=1,
        max_value=2000,
        value=500,
        step=50,
        help=(
            "The app will attempt to scan this many eligible contracts. "
            "A very large scan can take time on free hosting."
        ),
    )
    st.caption("POI and entry timeframes can be changed independently.")

poi_interval, poi_seconds = POI_INTERVALS[poi_label]
entry_interval, entry_seconds = ENTRY_INTERVALS[entry_label]

st.write(
    f"**Current configuration:** POI = {poi_label}; "
    f"entry trigger = {entry_label}."
)

if poi_interval == entry_interval:
    st.info("Both selectors use the same timeframe. POI and trigger rules remain separate.")

if st.button("Scan MEXC futures now", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading MEXC contracts and current prices..."):
            contracts = get_contracts()
            tickers = get_tickers()

        eligible = [
            symbol for symbol in contracts
            if symbol in tickers and tickers[symbol]["price"] > 0
        ]
        eligible.sort(
            key=lambda s: tickers[s]["volume"],
            reverse=True,
        )
        eligible = eligible[:int(max_contracts)]

        st.write(
            f"Eligible USDT perpetuals returned by MEXC: **{len(contracts)}**. "
            f"Contracts selected for this scan: **{len(eligible)}**."
        )

        results = []
        errors = 0
        progress = st.progress(0)
        status = st.empty()

        for index, symbol in enumerate(eligible, start=1):
            status.write(f"Scanning {symbol} ({index}/{len(eligible)})...")
            try:
                rows = scan_symbol(
                    symbol,
                    tickers[symbol],
                    poi_interval,
                    poi_seconds,
                    entry_interval,
                    entry_seconds,
                )
                results.extend(rows)
            except Exception:
                errors += 1

            progress.progress(index / max(len(eligible), 1))
            # Keep request volume conservative for the public API.
            time.sleep(0.12)

        progress.empty()
        status.empty()

        st.session_state["scan_results"] = results
        st.session_state["scan_time"] = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.session_state["scan_errors"] = errors
        st.session_state["scan_contract_count"] = len(eligible)

    except Exception as exc:
        st.error(f"Could not load MEXC market data: {exc}")
        st.info(
            "Check whether MEXC's public futures API is reachable from "
            "your hosting region, then try again."
        )

if "scan_results" in st.session_state:
    st.caption(
        f"Last scan: {st.session_state['scan_time']} · "
        f"Contracts checked: {st.session_state['scan_contract_count']} · "
        f"Requests/errors: {st.session_state['scan_errors']}"
    )

    results = st.session_state["scan_results"]
    if results:
        frame = pd.DataFrame(results)

        confirmed = frame[
            frame["Status"] == "Trigger confirmed (rule-based)"
        ]
        waiting = frame[
            frame["Status"] == "At POI — waiting for trigger"
        ]

        st.subheader("Entry-trigger candidates")
        st.write(f"Rule-based triggers: **{len(confirmed)}**")
        if not confirmed.empty:
            st.dataframe(confirmed, use_container_width=True, hide_index=True)
        else:
            st.info("No matching entry triggers were found at the detected POIs.")

        st.subheader("Price at / near POI — waiting")
        st.write(f"Waiting candidates: **{len(waiting)}**")
        if not waiting.empty:
            st.dataframe(waiting, use_container_width=True, hide_index=True)

        csv = frame.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scan results (CSV)",
            data=csv,
            file_name="mexc_futures_poi_scan.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No POI candidates were found near the current prices. "
            "Try another timeframe or scan again later."
        )

st.divider()
st.caption(
    "Important: candle patterns are heuristic signals, not guaranteed entries. "
    "Confirm market structure, stop placement, liquidity, and risk independently. "
    "No order execution is performed."
)
