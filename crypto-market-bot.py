"""
Crypto Market Analysis Bot
A Streamlit web app for interactive crypto charts + rule-based Buy/Sell signals.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta

# Page config
st.set_page_config(
    page_title="Crypto Market Analysis Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a modern dark look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .signal-buy {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff1744, #ff5252);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 23, 68, 0.3);
    }
    .signal-hold {
        background: linear-gradient(135deg, #ff9100, #ffab40);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 145, 0, 0.3);
    }
    .metric-card {
        background: #1e1e2f;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stMetric label {
        color: #aaa !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Popular cryptos
CRYPTO_OPTIONS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "XRP": "XRP-USD",
    "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Custom ticker...": "CUSTOM",
}

selected_name = st.sidebar.selectbox("Select Cryptocurrency", list(CRYPTO_OPTIONS.keys()), index=0)

if selected_name == "Custom ticker...":
    custom = st.sidebar.text_input("Enter ticker (e.g. MATIC-USD, PEPE-USD)", value="MATIC-USD")
    ticker = custom.strip().upper() if custom else "BTC-USD"
    selected_name = ticker
else:
    ticker = CRYPTO_OPTIONS[selected_name]

# Timeframe
INTERVAL_OPTIONS = {
    "1 Hour": "1h",
    "4 Hours": "4h",
    "1 Day": "1d",
    "1 Week": "1wk",
}
interval_label = st.sidebar.selectbox("Candle Interval", list(INTERVAL_OPTIONS.keys()), index=2)
interval = INTERVAL_OPTIONS[interval_label]

# Period
PERIOD_OPTIONS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}
period_label = st.sidebar.selectbox("Lookback Period", list(PERIOD_OPTIONS.keys()), index=3)
period = PERIOD_OPTIONS[period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### Indicators")
show_ma = st.sidebar.checkbox("Moving Averages (SMA 20 / 50 / 200)", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)
show_volume = st.sidebar.checkbox("Volume", value=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 Signals are generated using classic technical rules (Golden Cross, RSI extremes, MACD crossovers, Bollinger touches).")

# Header
st.markdown(f'<div class="main-header">📈 Crypto Market Analysis Bot</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Real-time charts & rule-based Buy/Sell signals for {selected_name}</div>', unsafe_allow_html=True)

# Fetch data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

with st.spinner(f"Fetching {selected_name} data..."):
    df = fetch_data(ticker, period, interval)

if df.empty:
    st.error("No data available. Try a different period or interval.")
    st.stop()

# Calculate indicators
df = df.copy()
df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)

# Bollinger Bands
bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
df["BB_High"] = bb.bollinger_hband()
df["BB_Mid"] = bb.bollinger_mavg()
df["BB_Low"] = bb.bollinger_lband()

# RSI
df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

# MACD
macd = ta.trend.MACD(df["Close"])
df["MACD"] = macd.macd()
df["MACD_Signal"] = macd.macd_signal()
df["MACD_Hist"] = macd.macd_diff()

# Latest values
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ========== SIGNAL ENGINE ==========
def generate_signals(df: pd.DataFrame) -> dict:
    signals = []
    score = 0  # positive = bullish, negative = bearish

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    # 1. Moving Average Crossovers
    if not pd.isna(latest["SMA_50"]) and not pd.isna(latest["SMA_200"]):
        if prev["SMA_50"] <= prev["SMA_200"] and latest["SMA_50"] > latest["SMA_200"]:
            signals.append(("🟢 Golden Cross (SMA50 > SMA200)", "strong_buy"))
            score += 2
        elif prev["SMA_50"] >= prev["SMA_200"] and latest["SMA_50"] < latest["SMA_200"]:
            signals.append(("🔴 Death Cross (SMA50 < SMA200)", "strong_sell"))
            score -= 2
        elif latest["SMA_50"] > latest["SMA_200"]:
            signals.append(("🟢 Bullish MA structure (SMA50 > SMA200)", "buy"))
            score += 1
        else:
            signals.append(("🔴 Bearish MA structure (SMA50 < SMA200)", "sell"))
            score -= 1

    # Short-term MA
    if not pd.isna(latest["SMA_20"]) and not pd.isna(latest["SMA_50"]):
        if latest["Close"] > latest["SMA_20"] > latest["SMA_50"]:
            signals.append(("🟢 Price > SMA20 > SMA50 (short-term uptrend)", "buy"))
            score += 1
        elif latest["Close"] < latest["SMA_20"] < latest["SMA_50"]:
            signals.append(("🔴 Price < SMA20 < SMA50 (short-term downtrend)", "sell"))
            score -= 1

    # 2. RSI
    if not pd.isna(latest["RSI"]):
        if latest["RSI"] < 30:
            signals.append((f"🟢 RSI Oversold ({latest['RSI']:.1f}) – potential bounce", "buy"))
            score += 1.5
        elif latest["RSI"] > 70:
            signals.append((f"🔴 RSI Overbought ({latest['RSI']:.1f}) – potential pullback", "sell"))
            score -= 1.5
        elif latest["RSI"] < 45:
            signals.append((f"🟡 RSI slightly weak ({latest['RSI']:.1f})", "neutral"))
        elif latest["RSI"] > 55:
            signals.append((f"🟡 RSI slightly strong ({latest['RSI']:.1f})", "neutral"))
        else:
            signals.append((f"⚪ RSI neutral ({latest['RSI']:.1f})", "neutral"))

    # 3. MACD
    if not pd.isna(latest["MACD"]) and not pd.isna(latest["MACD_Signal"]):
        if prev["MACD"] <= prev["MACD_Signal"] and latest["MACD"] > latest["MACD_Signal"]:
            signals.append(("🟢 MACD bullish crossover", "buy"))
            score += 1.5
        elif prev["MACD"] >= prev["MACD_Signal"] and latest["MACD"] < latest["MACD_Signal"]:
            signals.append(("🔴 MACD bearish crossover", "sell"))
            score -= 1.5
        elif latest["MACD"] > latest["MACD_Signal"]:
            signals.append(("🟢 MACD above signal line", "buy"))
            score += 0.5
        else:
            signals.append(("🔴 MACD below signal line", "sell"))
            score -= 0.5

    # 4. Bollinger Bands
    if not pd.isna(latest["BB_Low"]) and not pd.isna(latest["BB_High"]):
        if latest["Close"] <= latest["BB_Low"]:
            signals.append(("🟢 Price at/near lower Bollinger Band – possible support", "buy"))
            score += 1
        elif latest["Close"] >= latest["BB_High"]:
            signals.append(("🔴 Price at/near upper Bollinger Band – possible resistance", "sell"))
            score -= 1

    # Final decision
    if score >= 2.5:
        decision = "BUY"
        confidence = min(abs(score) / 6 * 100, 95)
    elif score <= -2.5:
        decision = "SELL"
        confidence = min(abs(score) / 6 * 100, 95)
    else:
        decision = "HOLD"
        confidence = 50 + abs(score) * 5

    return {
        "decision": decision,
        "score": score,
        "confidence": round(confidence, 1),
        "reasons": signals,
    }

signal_data = generate_signals(df)

# ========== TOP METRICS ==========
col1, col2, col3, col4, col5 = st.columns(5)

price = latest["Close"]
change = latest["Close"] - prev["Close"]
change_pct = (change / prev["Close"]) * 100 if prev["Close"] != 0 else 0

col1.metric("Current Price", f"${price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
col2.metric("24h High" if interval in ["1h", "4h"] else "Period High", f"${df['High'].max():,.2f}")
col3.metric("24h Low" if interval in ["1h", "4h"] else "Period Low", f"${df['Low'].min():,.2f}")
col4.metric("RSI (14)", f"{latest['RSI']:.1f}" if not pd.isna(latest["RSI"]) else "N/A")
col5.metric("Volume", f"{latest['Volume']:,.0f}")

st.markdown("---")

# ========== SIGNAL DISPLAY ==========
sig_col1, sig_col2 = st.columns([1, 2])

with sig_col1:
    decision = signal_data["decision"]
    if decision == "BUY":
        st.markdown(f'<div class="signal-buy">🟢 BUY SIGNAL<br><small>Confidence: {signal_data["confidence"]}%</small></div>', unsafe_allow_html=True)
    elif decision == "SELL":
        st.markdown(f'<div class="signal-sell">🔴 SELL SIGNAL<br><small>Confidence: {signal_data["confidence"]}%</small></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="signal-hold">🟡 HOLD<br><small>Confidence: {signal_data["confidence"]}%</small></div>', unsafe_allow_html=True)

    st.markdown(f"**Signal Score:** `{signal_data['score']:+.1f}`")
    st.caption("Positive = Bullish | Negative = Bearish")

with sig_col2:
    st.markdown("#### Signal Reasons")
    for reason, _ in signal_data["reasons"]:
        st.markdown(f"- {reason}")

st.markdown("---")

# ========== CHART ==========
st.markdown("### 📊 Interactive Price Chart")

# Determine subplot rows
rows = 1
row_heights = [0.6]
if show_volume:
    rows += 1
    row_heights.append(0.15)
if show_rsi:
    rows += 1
    row_heights.append(0.15)
if show_macd:
    rows += 1
    row_heights.append(0.2)

# Normalize heights
total = sum(row_heights)
row_heights = [h / total for h in row_heights]

fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights,
    subplot_titles=["Price"] + (["Volume"] if show_volume else []) + (["RSI"] if show_rsi else []) + (["MACD"] if show_macd else []),
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color="#00c853",
        decreasing_line_color="#ff1744",
    ),
    row=1, col=1,
)

# Moving Averages
if show_ma:
    if not df["SMA_20"].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(color="#00d4ff", width=1.5)), row=1, col=1)
    if not df["SMA_50"].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(color="#ffab00", width=1.5)), row=1, col=1)
    if not df["SMA_200"].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], name="SMA 200", line=dict(color="#e040fb", width=2)), row=1, col=1)

# Bollinger Bands
if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], name="BB Upper", line=dict(color="rgba(100,100,255,0.4)", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], name="BB Lower", line=dict(color="rgba(100,100,255,0.4)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(100,100,255,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], name="BB Mid", line=dict(color="rgba(100,100,255,0.6)", width=1)), row=1, col=1)

current_row = 2

# Volume
if show_volume:
    colors = ["#00c853" if c >= o else "#ff1744" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.7),
        row=current_row, col=1,
    )
    current_row += 1

# RSI
if show_rsi:
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#7b2ff7", width=1.5)), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=current_row, col=1)
    current_row += 1

# MACD
if show_macd:
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#00d4ff", width=1.5)), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#ff9100", width=1.5)), row=current_row, col=1)
    colors_hist = ["#00c853" if v >= 0 else "#ff1744" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color=colors_hist, opacity=0.6), row=current_row, col=1)

# Layout
fig.update_layout(
    height=700 if rows > 2 else 550,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=40, b=40),
    hovermode="x unified",
)

fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="#333")
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#333")

st.plotly_chart(fig, use_container_width=True)

# ========== ADDITIONAL INFO ==========
with st.expander("📋 Latest Data Snapshot"):
    st.dataframe(
        df.tail(10)[["Open", "High", "Low", "Close", "Volume", "SMA_20", "SMA_50", "RSI", "MACD"]].style.format("{:.2f}"),
        use_container_width=True,
    )

with st.expander("ℹ️ How the signals work"):
    st.markdown("""
    **Rule-based Signal Engine**

    The bot combines several classic technical analysis rules:

    | Indicator | Buy Condition | Sell Condition |
    |-----------|---------------|----------------|
    | **Moving Averages** | Golden Cross (SMA50 crosses above SMA200) or Price > SMA20 > SMA50 | Death Cross or Price < SMA20 < SMA50 |
    | **RSI (14)** | RSI < 30 (oversold) | RSI > 70 (overbought) |
    | **MACD** | MACD line crosses above Signal line | MACD line crosses below Signal line |
    | **Bollinger Bands** | Price touches or goes below lower band | Price touches or goes above upper band |

    Each condition contributes a score.  
    - **Score ≥ +2.5** → **BUY**  
    - **Score ≤ -2.5** → **SELL**  
    - Otherwise → **HOLD**

    Confidence is derived from the absolute score magnitude.
    """)

st.markdown("---")
st.caption(f"Data source: Yahoo Finance • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • This is not financial advice.")
