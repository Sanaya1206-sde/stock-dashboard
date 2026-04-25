# app.py — Ultimate Stock Dashboard
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import requests

st.set_page_config(
    page_title="Stock Dashboard Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme toggle ───────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = True

def toggle_theme():
    st.session_state.dark = not st.session_state.dark

dark = st.session_state.dark
bg      = "#0f172a" if dark else "#f8fafc"
card    = "#1e293b" if dark else "#ffffff"
text    = "#f1f5f9" if dark else "#1e293b"
muted   = "#94a3b8" if dark else "#64748b"
accent  = "#6366f1"
green   = "#22c55e"
red     = "#ef4444"
amber   = "#f59e0b"
plot_t  = "plotly_dark" if dark else "plotly_white"

st.markdown(f"""
<style>
  /* Fix tab visibility */
  .stTabs [data-baseweb="tab-list"] {{
    background: {'#1e293b' if dark else '#f1f5f9'};
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border-bottom: none !important;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {muted};
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    border: none !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: {'#334155' if dark else '#ffffff'} !important;
    color: {text} !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }}
  .stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
  }}
  .stTabs [data-baseweb="tab-border"] {{
    display: none !important;
  }}
  /* Fix metric colors */
  [data-testid="stMetricValue"] {{
    color: {text} !important;
  }}
  [data-testid="stMetricDelta"] {{
    font-weight: 600;
  }}
  /* Remove white lines/borders */
  hr {{
    border-color: {'#334155' if dark else '#e2e8f0'} !important;
  }}
  .stApp {{ background:{bg}; color:{text}; }}
  .block-container {{ padding-top:1.5rem; }}
  .metric-box {{
    background:{card}; border-radius:14px; padding:18px 22px;
    border:1px solid {'#334155' if dark else '#e2e8f0'};
    margin-bottom:8px;
  }}
  .section-title {{
    font-size:1.15rem; font-weight:700;
    color:{text}; margin:18px 0 10px;
    border-left:4px solid {accent};
    padding-left:10px;
  }}
  .news-card {{
    background:{card}; border-radius:10px;
    padding:14px 18px; margin-bottom:10px;
    border:1px solid {'#334155' if dark else '#e2e8f0'};
  }}
  .pill {{
    display:inline-block; padding:3px 12px;
    border-radius:20px; font-size:0.8rem; font-weight:600;
  }}
  .pill-green {{ background:#dcfce7; color:#166534; }}
  .pill-red   {{ background:#fee2e2; color:#991b1b; }}
  .pill-gray  {{ background:#f1f5f9; color:#475569; }}
  div[data-testid="stSidebar"] {{
    background:{'#0f172a' if dark else '#f1f5f9'};
  }}
  /* Hide default streamlit header white bar */
  header[data-testid="stHeader"] {{
    background: {bg} !important;
    border-bottom: none !important;
  }}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch(ticker, period):
    df = yf.Ticker(ticker).history(period=period)
    return df[['Open','High','Low','Close','Volume']].dropna()

@st.cache_data(ttl=300)
def get_info(ticker):
    try: return yf.Ticker(ticker).info
    except: return {}

@st.cache_data(ttl=600)
def get_news(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.news[:8] if t.news else []
    except: return []

def add_indicators(df):
    # RSI
    d = df['Close'].diff()
    g = d.where(d>0,0).rolling(14).mean()
    l = (-d.where(d<0,0)).rolling(14).mean()
    df['RSI'] = 100-(100/(1+g/l))
    # MACD
    e12 = df['Close'].ewm(span=12).mean()
    e26 = df['Close'].ewm(span=26).mean()
    df['MACD']   = e12-e26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist']   = df['MACD']-df['Signal']
    # Bollinger
    df['MA20']   = df['Close'].rolling(20).mean()
    df['UB']     = df['MA20']+2*df['Close'].rolling(20).std()
    df['LB']     = df['MA20']-2*df['Close'].rolling(20).std()
    # EMA
    df['EMA9']   = df['Close'].ewm(span=9).mean()
    df['EMA21']  = df['Close'].ewm(span=21).mean()
    return df

def predict(df):
    data   = df['Close'].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1,1)).flatten()
    X, y   = [], []
    for i in range(30, len(scaled)):
        X.append(scaled[i-30:i])
        y.append(scaled[i])
    X, y   = np.array(X), np.array(y)
    split  = int(len(X)*0.8)
    model  = LinearRegression()
    model.fit(X[:split], y[:split])
    preds  = model.predict(X[split:])
    preds  = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    actual = scaler.inverse_transform(y[split:].reshape(-1,1)).flatten()
    # Future 30 days
    last30 = scaled[-30:]
    future = []
    for _ in range(30):
        p = model.predict(last30.reshape(1,-1))[0]
        future.append(p)
        last30 = np.append(last30[1:], p)
    future = scaler.inverse_transform(np.array(future).reshape(-1,1)).flatten()
    return actual, preds, future

def sentiment_color(score):
    if score > 0.1:  return "pill-green", "Positive 📈"
    if score < -0.1: return "pill-red",   "Negative 📉"
    return "pill-gray", "Neutral ➡️"

# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 📈 Stock Dashboard Pro")
    st.button("🌙 Dark" if not dark else "☀️ Light", on_click=toggle_theme)
    st.divider()

    mode = st.radio("Mode", ["Single Stock","Compare Stocks","Portfolio Tracker"])
    st.divider()

    period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=3)
    POPULAR = ["AAPL","GOOGL","MSFT","TSLA","AMZN","META","NVDA","NFLX"]
    st.markdown("**Quick pick:**")
    cols = st.columns(4)
    chosen = None
    for i,s in enumerate(POPULAR):
        if cols[i%4].button(s, key=f"b{s}"):
            chosen = s

# ════════════════════════════════════════════════════════════
# MODE 1 — Single Stock
# ════════════════════════════════════════════════════════════
if mode == "Single Stock":
    ticker = st.sidebar.text_input("Stock Symbol", value=chosen or "AAPL").upper()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📉 Technical", "🤖 AI Prediction", "📰 News", "ℹ️ Company"
    ])

    with st.spinner(f"Loading {ticker}..."):
        df   = fetch(ticker, period)
        df   = add_indicators(df)
        info = get_info(ticker)
        news = get_news(ticker)

    cur   = df['Close'].iloc[-1]
    prev  = df['Close'].iloc[-2]
    chg   = cur-prev
    pct   = (chg/prev)*100
    high  = df['Close'].max()
    low   = df['Close'].min()
    vol   = df['Volume'].iloc[-1]

    # ── TAB 1: Overview ──────────────────────────────────────
    with tab1:
        name = info.get('longName', ticker)
        sector = info.get('sector','—')
        st.markdown(f"## {name}")
        st.markdown(f"`{ticker}` &nbsp;•&nbsp; {sector}")

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Price",    f"${cur:.2f}",  f"{chg:+.2f}")
        c2.metric("Change",   f"{pct:+.2f}%")
        c3.metric("Volume",   f"{vol/1e6:.1f}M")
        c4.metric("Period High", f"${high:.2f}")
        c5.metric("Period Low",  f"${low:.2f}")

        # Candlestick + Volume
        st.markdown('<p class="section-title">Candlestick Chart + Volume</p>',
                    unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7,0.3], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            increasing_line_color=green, decreasing_line_color=red,
            name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['UB'], name="Upper Band",
            line=dict(color=red, dash="dash", width=1), opacity=0.5), row=1,col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['LB'], name="Lower Band",
            line=dict(color=green, dash="dash", width=1),
            fill='tonexty', fillcolor='rgba(99,102,241,0.05)',
            opacity=0.5), row=1,col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA20'], name="MA20",
            line=dict(color=amber, width=1.5, dash="dot")), row=1,col=1)
        colors = [green if df['Close'].iloc[i]>=df['Open'].iloc[i]
                  else red for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            marker_color=colors, name="Volume", opacity=0.7), row=2,col=1)
        fig.update_layout(template=plot_t, height=520,
                          xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Technical ─────────────────────────────────────
    with tab2:
        st.markdown('<p class="section-title">RSI — Relative Strength Index</p>',
                    unsafe_allow_html=True)
        rsi_now = df['RSI'].iloc[-1]
        rsi_cls, rsi_lbl = ("pill-red","Overbought") if rsi_now>70 \
            else ("pill-green","Oversold") if rsi_now<30 \
            else ("pill-gray","Neutral")
        st.markdown(f'RSI: **{rsi_now:.1f}** &nbsp;<span class="pill {rsi_cls}">{rsi_lbl}</span>',
                    unsafe_allow_html=True)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'],
            line=dict(color="#a855f7", width=2), name="RSI",
            fill='tozeroy', fillcolor='rgba(168,85,247,0.08)'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=red,
                          annotation_text="Overbought 70")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=green,
                          annotation_text="Oversold 30")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color=muted, opacity=0.4)
        fig_rsi.update_layout(template=plot_t, height=280, showlegend=False,
                               yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

        st.markdown('<p class="section-title">MACD</p>', unsafe_allow_html=True)
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'],
            name="MACD", line=dict(color="#3b82f6", width=2)))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'],
            name="Signal", line=dict(color=amber, width=2)))
        fig_macd.add_trace(go.Bar(x=df.index, y=df['Hist'], name="Histogram",
            marker_color=[green if v>0 else red for v in df['Hist']]))
        fig_macd.update_layout(template=plot_t, height=280)
        st.plotly_chart(fig_macd, use_container_width=True)

        st.markdown('<p class="section-title">EMA Crossover (9 vs 21)</p>',
                    unsafe_allow_html=True)
        fig_ema = go.Figure()
        fig_ema.add_trace(go.Scatter(x=df.index, y=df['Close'],
            name="Price", line=dict(color=muted, width=1), opacity=0.5))
        fig_ema.add_trace(go.Scatter(x=df.index, y=df['EMA9'],
            name="EMA 9", line=dict(color="#6366f1", width=2)))
        fig_ema.add_trace(go.Scatter(x=df.index, y=df['EMA21'],
            name="EMA 21", line=dict(color=amber, width=2)))
        fig_ema.update_layout(template=plot_t, height=280)
        st.plotly_chart(fig_ema, use_container_width=True)

    # ── TAB 3: AI Prediction ─────────────────────────────────
    with tab3:
        st.markdown('<p class="section-title">AI Price Prediction (30-day forecast)</p>',
                    unsafe_allow_html=True)
        if len(df) < 60:
            st.warning("Need at least 60 days of data. Select a longer period.")
        else:
            with st.spinner("Running AI model..."):
                actual, preds, future = predict(df)

            mape = abs((actual-preds)/actual).mean()*100
            acc  = 100-mape

            a1,a2,a3 = st.columns(3)
            a1.metric("Model Accuracy", f"{acc:.1f}%")
            a2.metric("Error Rate",     f"{mape:.1f}%")
            a3.metric("30-day Forecast",
                      f"${future[-1]:.2f}",
                      f"{((future[-1]-cur)/cur*100):+.1f}%")

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                y=actual, name="Actual",
                line=dict(color="#6366f1", width=2)))
            fig_pred.add_trace(go.Scatter(
                y=preds, name="Predicted",
                line=dict(color=amber, width=2, dash="dash")))
            # Future forecast
            start = len(actual)
            fig_pred.add_trace(go.Scatter(
                x=list(range(start, start+30)), y=future,
                name="30-day Forecast",
                line=dict(color=green, width=2, dash="dot"),
                fill='tozeroy', fillcolor='rgba(34,197,94,0.05)'))
            fig_pred.add_vline(x=start, line_dash="dash",
                               line_color=muted, opacity=0.5,
                               annotation_text="Forecast starts")
            fig_pred.update_layout(template=plot_t, height=400,
                                   xaxis_title="Days",
                                   yaxis_title="Price (USD)")
            st.plotly_chart(fig_pred, use_container_width=True)
            st.info("⚠️ AI predictions are for educational purposes only — not financial advice.")

    # ── TAB 4: News + Sentiment ──────────────────────────────
    with tab4:
        st.markdown('<p class="section-title">Latest News + Sentiment Analysis</p>',
                    unsafe_allow_html=True)
        if not news:
            st.info("No recent news found for this ticker.")
        else:
            for item in news:
                title     = item.get('title','No title')
                publisher = item.get('publisher','Unknown')
                link      = item.get('link','#')
                ts        = item.get('providerPublishTime', 0)
                date      = datetime.fromtimestamp(ts).strftime('%b %d, %Y') if ts else '—'

                # Simple keyword sentiment
                positive_words = ['surge','gain','rise','high','beat','profit',
                                  'growth','bull','up','record','strong','boost']
                negative_words = ['fall','drop','loss','crash','miss','down',
                                  'bear','decline','risk','weak','cut','warn']
                title_lower = title.lower()
                score = sum(1 for w in positive_words if w in title_lower) \
                      - sum(1 for w in negative_words if w in title_lower)
                pill_cls, pill_lbl = sentiment_color(score)

                st.markdown(f"""
                <div class="news-card">
                  <div style="display:flex;justify-content:space-between;
                              align-items:flex-start;gap:12px">
                    <div>
                      <a href="{link}" target="_blank"
                         style="color:{text};font-weight:600;
                                text-decoration:none;font-size:0.95rem">
                        {title}
                      </a>
                      <div style="color:{muted};font-size:0.8rem;margin-top:4px">
                        {publisher} &nbsp;•&nbsp; {date}
                      </div>
                    </div>
                    <span class="pill {pill_cls}" style="white-space:nowrap">
                      {pill_lbl}
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 5: Company Info ──────────────────────────────────
    with tab5:
        st.markdown('<p class="section-title">Company Overview</p>',
                    unsafe_allow_html=True)
        if info:
            c1, c2 = st.columns(2)
            fields_left = {
                "Sector":          info.get('sector','—'),
                "Industry":        info.get('industry','—'),
                "Country":         info.get('country','—'),
                "Employees":       f"{info.get('fullTimeEmployees',0):,}",
                "Website":         info.get('website','—'),
            }
            fields_right = {
                "Market Cap":      f"${info.get('marketCap',0)/1e9:.2f}B",
                "P/E Ratio":       f"{info.get('trailingPE',0):.2f}",
                "EPS":             f"${info.get('trailingEps',0):.2f}",
                "Dividend Yield":  f"{info.get('dividendYield',0)*100:.2f}%",
                "Beta":            f"{info.get('beta',0):.2f}",
            }
            with c1:
                for k,v in fields_left.items():
                    st.markdown(f"**{k}:** {v}")
            with c2:
                for k,v in fields_right.items():
                    st.markdown(f"**{k}:** {v}")

            desc = info.get('longBusinessSummary','')
            if desc:
                st.markdown('<p class="section-title">About</p>',
                            unsafe_allow_html=True)
                st.markdown(f'<p style="color:{muted};line-height:1.7">'
                            f'{desc[:800]}...</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODE 2 — Compare Stocks
# ════════════════════════════════════════════════════════════
elif mode == "Compare Stocks":
    st.markdown("## 📊 Stock Comparison")
    col1,col2,col3 = st.columns(3)
    t1 = col1.text_input("Stock 1", "AAPL").upper()
    t2 = col2.text_input("Stock 2", "MSFT").upper()
    t3 = col3.text_input("Stock 3", "GOOGL").upper()
    tickers = [t for t in [t1,t2,t3] if t]

    colors_list = ["#6366f1","#22c55e","#f59e0b"]

    with st.spinner("Fetching comparison data..."):
        dfs   = {t: add_indicators(fetch(t,period)) for t in tickers}
        infos = {t: get_info(t) for t in tickers}

    # Normalised price comparison
    st.markdown('<p class="section-title">Normalised Price (base 100)</p>',
                unsafe_allow_html=True)
    fig_cmp = go.Figure()
    for i,t in enumerate(tickers):
        norm = dfs[t]['Close'] / dfs[t]['Close'].iloc[0] * 100
        fig_cmp.add_trace(go.Scatter(
            x=dfs[t].index, y=norm, name=t,
            line=dict(color=colors_list[i], width=2)))
    fig_cmp.update_layout(template=plot_t, height=380,
                          yaxis_title="Return (base=100)")
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Side-by-side metrics
    st.markdown('<p class="section-title">Key Metrics Comparison</p>',
                unsafe_allow_html=True)
    cols = st.columns(len(tickers))
    for i,t in enumerate(tickers):
        df  = dfs[t]
        inf = infos[t]
        cur = df['Close'].iloc[-1]
        pct = (df['Close'].iloc[-1]-df['Close'].iloc[-2])/df['Close'].iloc[-2]*100
        with cols[i]:
            st.markdown(f"### {t}")
            st.metric("Price",    f"${cur:.2f}", f"{pct:+.2f}%")
            st.metric("RSI",      f"{df['RSI'].iloc[-1]:.1f}")
            st.metric("Mkt Cap",  f"${inf.get('marketCap',0)/1e9:.1f}B")
            st.metric("P/E",      f"{inf.get('trailingPE',0):.1f}")
            st.metric("52W High", f"${df['Close'].max():.2f}")

    # Volume comparison
    st.markdown('<p class="section-title">Volume Comparison</p>',
                unsafe_allow_html=True)
    fig_vol = go.Figure()
    for i,t in enumerate(tickers):
        fig_vol.add_trace(go.Bar(
            x=dfs[t].index, y=dfs[t]['Volume'],
            name=t, marker_color=colors_list[i], opacity=0.7))
    fig_vol.update_layout(template=plot_t, height=300, barmode='group')
    st.plotly_chart(fig_vol, use_container_width=True)

# ════════════════════════════════════════════════════════════
# MODE 3 — Portfolio Tracker
# ════════════════════════════════════════════════════════════
elif mode == "Portfolio Tracker":
    st.markdown("## 💼 Portfolio Tracker")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = [
            {"ticker":"AAPL","shares":10,"buy_price":150.0},
            {"ticker":"MSFT","shares":5, "buy_price":280.0},
            {"ticker":"TSLA","shares":3, "buy_price":200.0},
        ]

    # Add stock form
    with st.expander("➕ Add Stock to Portfolio"):
        pc1,pc2,pc3,pc4 = st.columns([2,2,2,1])
        new_t  = pc1.text_input("Symbol",    "NVDA").upper()
        new_s  = pc2.number_input("Shares",  min_value=0.01, value=1.0)
        new_bp = pc3.number_input("Buy Price", min_value=0.01, value=100.0)
        if pc4.button("Add", type="primary"):
            st.session_state.portfolio.append({
                "ticker":new_t,"shares":new_s,"buy_price":new_bp})
            st.success(f"Added {new_t}!")

    # Calculate portfolio
    rows   = []
    total_invested = 0
    total_current  = 0

    for item in st.session_state.portfolio:
        try:
            df  = fetch(item['ticker'], "1mo")
            cur = df['Close'].iloc[-1]
            inv = item['shares'] * item['buy_price']
            val = item['shares'] * cur
            pnl = val - inv
            pct = (pnl / inv) * 100
            total_invested += inv
            total_current  += val
            rows.append({
                "Ticker":      item['ticker'],
                "Shares":      item['shares'],
                "Buy Price":   f"${item['buy_price']:.2f}",
                "Current":     f"${cur:.2f}",
                "Invested":    f"${inv:,.2f}",
                "Value":       f"${val:,.2f}",
                "P&L":         f"${pnl:+,.2f}",
                "Return %":    f"{pct:+.1f}%",
                "_pnl":        pnl,
            })
        except:
            pass

    # Portfolio summary
    total_pnl = total_current - total_invested
    total_pct = (total_pnl/total_invested*100) if total_invested else 0

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Total Invested", f"${total_invested:,.2f}")
    s2.metric("Current Value",  f"${total_current:,.2f}",
              f"{total_pnl:+,.2f}")
    s3.metric("Total P&L",      f"${total_pnl:+,.2f}")
    s4.metric("Total Return",   f"{total_pct:+.1f}%")

    # Portfolio table
    st.markdown('<p class="section-title">Holdings</p>', unsafe_allow_html=True)
    display = pd.DataFrame([{k:v for k,v in r.items() if k!='_pnl'} for r in rows])
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Allocation pie chart
    if rows:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title">Portfolio Allocation</p>',
                        unsafe_allow_html=True)
            vals   = [float(r['Value'].replace('$','').replace(',',''))
                      for r in rows]
            labels = [r['Ticker'] for r in rows]
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=vals,
                hole=0.45,
                marker_colors=["#6366f1","#22c55e","#f59e0b",
                                "#ef4444","#a855f7","#3b82f6"]))
            fig_pie.update_layout(template=plot_t, height=320,
                                  showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown('<p class="section-title">P&L per Stock</p>',
                        unsafe_allow_html=True)
            pnls   = [r['_pnl'] for r in rows]
            clrs   = [green if p>=0 else red for p in pnls]
            fig_pnl= go.Figure(go.Bar(
                x=labels, y=pnls,
                marker_color=clrs,
                text=[f"${p:+,.1f}" for p in pnls],
                textposition='outside'))
            fig_pnl.update_layout(template=plot_t, height=320,
                                  yaxis_title="P&L ($)")
            st.plotly_chart(fig_pnl, use_container_width=True)