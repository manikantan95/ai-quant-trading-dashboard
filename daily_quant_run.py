# ==========================================
# 🚀 AI HEDGE FUND: STANDALONE DAILY ENGINE
# ==========================================
import os
import requests
import pandas as pd
import datetime as dt
import numpy as np
import json
import feedparser
import urllib.parse
import sqlite3
import io
import random
import time
from tqdm import tqdm
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe

from ta.momentum import rsi
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import volume_weighted_average_price
from openai import OpenAI

# ==========================================
# 1. API INITIALIZATION (VIA GITHUB SECRETS)
# ==========================================
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
ACCESS_TOKEN = os.environ.get('UPSTOX_TOKEN')
DB_NAME = "trading_agent_memory.db"

# ==========================================
# 2. STATEFUL MEMORY LAYER 
# ==========================================
def init_production_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, prediction_date TEXT NOT NULL, 
        run_timestamp TEXT NOT NULL, ticker TEXT NOT NULL,
        cap_class TEXT NOT NULL, limelight_score INTEGER DEFAULT 0, tech_bias TEXT NOT NULL,
        tech_confidence REAL NOT NULL, sentiment_score REAL NOT NULL, market_impact TEXT NOT NULL,
        core_catalyst TEXT, predicted_direction TEXT NOT NULL, UNIQUE(prediction_date, ticker)
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS swing_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, setup_date TEXT DEFAULT CURRENT_DATE, 
        run_timestamp TEXT NOT NULL, ticker TEXT, cap_class TEXT, time_horizon TEXT, 
        entry_price REAL, stop_loss REAL, target_price REAL, atr_at_entry REAL, 
        timeframe_days INTEGER, status TEXT DEFAULT 'ACTIVE', exit_date TEXT,
        exit_price REAL, technical_weight_used REAL, sentiment_weight_used REAL
    )""")
    conn.commit()

    # PULL HISTORY FROM GOOGLE SHEETS
    # (Crucial for GitHub Actions so we don't overwrite the cloud database with an empty local run)
    print("☁️ Pulling historical memory from Google Sheets...")
    try:
        creds_json = os.environ.get('GCP_CREDENTIALS')
        if creds_json:
            client_gs = gspread.authorize(Credentials.from_service_account_info(
                json.loads(creds_json), 
                scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            ))
            sheet = client_gs.open("Quant_Fund_DB")
            
            # Restore swing_trades
            try:
                ws_swing = sheet.worksheet("swing_trades")
                df_swing = get_as_dataframe(ws_swing).dropna(how='all')
                if not df_swing.empty:
                    df_swing.to_sql("swing_trades", conn, if_exists="replace", index=False)
            except Exception: pass
            
            # Restore daily_predictions
            try:
                ws_daily = sheet.worksheet("daily_predictions")
                df_daily = get_as_dataframe(ws_daily).dropna(how='all')
                if not df_daily.empty:
                    df_daily.to_sql("daily_predictions", conn, if_exists="replace", index=False)
            except Exception: pass
    except Exception as e:
        print(f"⚠️ Could not pull history (normal if this is the absolute first run). Error: {e}")

    conn.close()
    print("💾 Production Database Ready (Append Mode Active).")

# ==========================================
# 3. MARKET DATA & QUANT MATH
# ==========================================
def fetch_upstox_historical_data_safe(instrument_key, days_back=400):
    to_date = dt.datetime.today().strftime('%Y-%m-%d')
    from_date = (dt.datetime.today() - dt.timedelta(days=days_back)).strftime('%Y-%m-%d')
    url = f'https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_date}/{from_date}'
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {ACCESS_TOKEN}'}
    res = requests.get(url, headers=headers)

    if res.status_code == 200 and 'data' in res.json() and res.json()['data']:
        df = pd.DataFrame(res.json()['data']['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.sort_index(ascending=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
        return df
    return None

def apply_technical_indicators(df):
    if df is None or len(df) < 50: return None
    df['RSI'] = rsi(df['close'], window=14)
    macd = MACD(df['close'])
    df['MACD_Line'], df['MACD_Signal'] = macd.macd(), macd.macd_signal()
    df['SMA_20'], df['SMA_50'] = SMAIndicator(df['close'], window=20).sma_indicator(), SMAIndicator(df['close'], window=50).sma_indicator()
    df['ATR_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_High'], df['BB_Low'] = bb.bollinger_hband(), bb.bollinger_lband()
    df['VWAP'] = volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['Dev_from_20SMA'] = ((df['close'] - df['SMA_20']) / df['SMA_20']) * 100
    df.dropna(inplace=True)
    return df

# ==========================================
# 4. MACRO NEWS AGGREGATOR
# ==========================================
def fetch_upstox_news_safe(instrument_keys):
    res = requests.get('https://api.upstox.com/v2/news', params={'category': 'instrument_keys', 'instrument_keys': instrument_keys}, headers={'Accept': 'application/json', 'Authorization': f'Bearer {ACCESS_TOKEN}'})
    news = []
    if res.status_code == 200 and res.json().get('status') == 'success':
        for key, articles in res.json().get('data', {}).items():
            news.extend([{"title": a.get('heading'), "summary": a.get('summary'), "publisher": "Upstox"} for a in articles[:5]])
    return news

def fetch_google_news_safe(company, ticker):
    query = urllib.parse.quote_plus(f'"{company}" OR "{ticker}" stock market')
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en")
    return [{"title": a.title, "publisher": a.source.title if hasattr(a, 'source') else "Google"} for a in feed.entries[:5]]

def merge_news_streams(u_news, g_news):
    return "=== UPSTOX ===\n" + "".join([f"- {n['title']}: {n['summary']}\n" for n in u_news]) + "\n=== GOOGLE ===\n" + "".join([f"- {n['title']} ({n['publisher']})\n" for n in g_news])

# ==========================================
# 5. LLM INTELLIGENCE
# ==========================================
def run_technical_agent(ticker, tech_row):
    prompt = f"Analyze indicators for {ticker}:\n{tech_row.to_dict()}\nRespond purely in JSON: {{'ticker': '{ticker}', 'directional_bias': 'BULLISH/BEARISH/NEUTRAL', 'confidence_score': 0.0-1.0, 'primary_reason': '...'}}"
    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    return json.loads(res.choices[0].message.content)

def run_sentiment_agent(ticker, news_text):
    prompt = f"Analyze news for {ticker}:\n{news_text}\nRespond purely in JSON: {{'ticker': '{ticker}', 'sentiment_score': -1.0 to 1.0, 'market_impact': 'HIGH/MEDIUM/LOW', 'core_catalyst': '...', 'risk_factor': '...'}}"
    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    return json.loads(res.choices[0].message.content)

# ==========================================
# 6. CORPORATE EQUITY GATEKEEPER
# ==========================================
def build_ultimate_corporate_equity_map():
    df = pd.read_json('https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz', compression='gzip')
    df = df[df['segment'] == 'NSE_EQ'].copy()
    for kw in ['BEES', 'GILT', 'SGB', 'NIFTY', 'BANKB', 'INFRAB', 'EETF', 'ETF', 'PSUBK', 'LOWVOL', 'MIDCAP', 'SMALLCAP']:
        df = df[~df['trading_symbol'].str.contains(kw, case=False, na=False)]
    df = df[~df['trading_symbol'].str.match(r'^\d') & ~df['trading_symbol'].str.contains(r'\d{2,}$', regex=True)]
    return dict(zip(df['trading_symbol'], df['instrument_key']))

def fetch_official_nse_universe():
    headers, universe = {"User-Agent": "Mozilla/5.0"}, []
    urls = {"LARGE": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv", "MID": "https://archives.nseindia.com/content/indices/ind_niftymidcap50list.csv", "SMALL": "https://archives.nseindia.com/content/indices/ind_niftysmallcap50list.csv"}
    for cap, url in urls.items():
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            universe.extend([{'ticker': t, 'cap': cap, 'limelight_score': random.randint(30, 100)} for t in pd.read_csv(io.StringIO(res.text))['Symbol']])
    return universe

live_upstox_map = build_ultimate_corporate_equity_map()

# ==========================================
# 7. THE 100-STOCK SCANNER
# ==========================================
def run_daily_agent_pipeline():
    global live_upstox_map
    if not live_upstox_map: live_upstox_map = build_ultimate_corporate_equity_map()

    full_pool = fetch_official_nse_universe()
    if not full_pool: return

    target_100 = random.sample([s for s in full_pool if s['cap'] == 'LARGE'], 25) + random.sample([s for s in full_pool if s['cap'] == 'MID'], 38) + random.sample([s for s in full_pool if s['cap'] == 'SMALL'], 37)

    conn = sqlite3.connect(DB_NAME)
    today_str = dt.datetime.today().strftime('%Y-%m-%d')
    exact_time_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for stock in tqdm(target_100, desc="AI Processing"):
        ticker, key = stock['ticker'], live_upstox_map.get(stock['ticker'])
        if not key: continue
        try:
            df_tech = apply_technical_indicators(fetch_upstox_historical_data_safe(key))
            if df_tech is None or df_tech.empty: continue

            news_txt = merge_news_streams(fetch_upstox_news_safe(key), fetch_google_news_safe(ticker, ticker))
            t_res, s_res = run_technical_agent(ticker, df_tech.iloc[-1]), run_sentiment_agent(ticker, news_txt)
            p_dir = "UP" if (t_res['directional_bias']=="BULLISH" and s_res.get('sentiment_score',0)>0) else ("DOWN" if (t_res['directional_bias']=="BEARISH" and s_res.get('sentiment_score',0)<0) else "NEUTRAL")

            conn.execute("INSERT OR REPLACE INTO daily_predictions (prediction_date, run_timestamp, ticker, cap_class, limelight_score, tech_bias, tech_confidence, sentiment_score, market_impact, core_catalyst, predicted_direction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (today_str, exact_time_now, ticker, stock['cap'], stock['limelight_score'], t_res['directional_bias'], t_res['confidence_score'], s_res.get('sentiment_score',0), s_res.get('market_impact','LOW'), s_res.get('core_catalyst',''), p_dir))
            conn.commit()
            time.sleep(1)
        except Exception as e: print(f"Skipped {ticker}: {e}")
    conn.close()
    print("✅ Market scan complete & timestamped.")

# ==========================================
# 8. PORTFOLIO AUDITOR & SELF-LEARNING
# ==========================================
def run_post_trade_audit():
    print("🔍 Auditing Active Portfolio...")
    conn = sqlite3.connect(DB_NAME)
    active = pd.read_sql_query("SELECT * FROM swing_trades WHERE status = 'ACTIVE'", conn)
    today = dt.datetime.today()

    if active.empty:
        print("   No active swing trades to audit today.")
        conn.close()
        return

    for _, t in active.iterrows():
        key = live_upstox_map.get(t['ticker'])
        if not key: continue
        try:
            df = fetch_upstox_historical_data_safe(key, days_back=60)
            if df is None: continue
            df = df[df.index >= pd.to_datetime(t['setup_date'])]
            if df.empty: continue

            high, low = float(df['high'].max()), float(df['low'].min())
            stat, exit_p = 'ACTIVE', None

            if high >= t['target_price']: stat, exit_p = 'HIT_TARGET', t['target_price']
            elif low <= t['stop_loss']: stat, exit_p = 'HIT_STOP', t['stop_loss']
            elif (today - dt.datetime.strptime(t['setup_date'], '%Y-%m-%d')).days > t['timeframe_days']: stat, exit_p = 'TIME_EXPIRED', float(df.iloc[-1]['close'])

            if stat != 'ACTIVE':
                conn.execute("UPDATE swing_trades SET status=?, exit_date=?, exit_price=? WHERE id=?", (stat, today.strftime('%Y-%m-%d'), exit_p, t['id']))
                print(f"  [{stat}] Closed {t['ticker']} at ₹{exit_p}")
        except: pass
    conn.commit()
    conn.close()
    print("✅ Audit Complete.")

# ==========================================
# 9. MULTI-HORIZON PORTFOLIO ALLOCATOR
# ==========================================
def allocate_multi_horizon_portfolio():
    print("🧠 Allocating Unique Multi-Horizon Setups...")
    conn = sqlite3.connect(DB_NAME)
    exact_time_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        latest_date = pd.read_sql_query("SELECT MAX(prediction_date) as md FROM daily_predictions", conn).iloc[0]['md']
    except:
        latest_date = None

    if not latest_date:
        conn.close()
        return print("Run Master Loop first.")

    cands = pd.read_sql_query("SELECT * FROM daily_predictions WHERE prediction_date=? AND predicted_direction='UP'", conn, params=(latest_date,))

    if cands.empty:
        conn.close()
        return print("No bullish setups today.")

    allocated_this_run = set()

    for horiz in ['5-DAY', '10-DAY', '1-MONTH', '3-6 MONTH']:
        t_w, s_w = 0.50, 0.50
        try:
            df_hist = pd.read_sql_query("SELECT * FROM swing_trades WHERE time_horizon = ? AND status != 'ACTIVE' ORDER BY run_timestamp DESC LIMIT 10", conn, params=(horiz,))
            if len(df_hist) >= 3 and (len(df_hist[df_hist['status'] == 'HIT_TARGET']) / len(df_hist)) < 0.45:
                t_w, s_w = 0.30, 0.70 if horiz in ['5-DAY', '10-DAY'] else (0.70, 0.30)
                print(f"   ⚠️ Agent adjusted weights for {horiz} based on past historical timestamps.")
        except: pass

        cands['score'] = (cands['tech_confidence'] * t_w) + (cands['sentiment_score'] * s_w)
        filtered_cands = cands[~cands['ticker'].isin(allocated_this_run)].sort_values('score', ascending=False)

        for _, r in filtered_cands.head(2).iterrows():
            key = live_upstox_map.get(r['ticker'])
            if not key: continue

            df = apply_technical_indicators(fetch_upstox_historical_data_safe(key, 100))
            if df is None or df.empty: continue

            c, atr = float(df.iloc[-1]['close']), float(df.iloc[-1]['ATR_14'])
            sl, tp, d = (c-(1.0*atr), c+(2.0*atr), 5) if horiz == '5-DAY' else ((c-(1.5*atr), c+(3.0*atr), 10) if horiz == '10-DAY' else ((c-(2.5*atr), c+(5.0*atr), 22) if horiz == '1-MONTH' else (c*0.92, c*1.25, 90)))

            if pd.read_sql_query("SELECT COUNT(*) as c FROM swing_trades WHERE setup_date=? AND ticker=? AND time_horizon=?", conn, params=(latest_date, r['ticker'], horiz)).iloc[0]['c'] == 0:
                conn.execute("INSERT INTO swing_trades (setup_date, run_timestamp, ticker, cap_class, time_horizon, entry_price, stop_loss, target_price, atr_at_entry, timeframe_days, technical_weight_used, sentiment_weight_used) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (latest_date, exact_time_now, r['ticker'], r['cap_class'], horiz, c, sl, tp, atr, d, t_w, s_w))
                print(f"  📌 [{horiz}] {r['ticker']} | Entry: {c:.2f} | Target: {tp:.2f}")
                allocated_this_run.add(r['ticker'])

    conn.commit()
    conn.close()
    print("✅ Balanced Portfolio Allocation Complete.")

# ==========================================
# 10. TERMINAL DASHBOARD
# ==========================================
def display_institutional_dashboard():
    conn = sqlite3.connect(DB_NAME)
    print("="*75 + "\n 📊 INSTITUTIONAL MULTI-HORIZON QUANT DASHBOARD \n" + "="*75)
    print("\n🟢 ACTIVE POSITIONS:")
    df_act = pd.read_sql_query("SELECT time_horizon, setup_date, ticker, cap_class, entry_price, target_price, stop_loss FROM swing_trades WHERE status='ACTIVE' ORDER BY time_horizon", conn)
    print(df_act.to_string(index=False) if not df_act.empty else "No active positions.")
    print("\n🏆 RECENT RESOLUTIONS:")
    df_res = pd.read_sql_query("SELECT exit_date, time_horizon, ticker, status, exit_price FROM swing_trades WHERE status!='ACTIVE' ORDER BY exit_date DESC LIMIT 5", conn)
    print(df_res.to_string(index=False) if not df_res.empty else "No resolved trades yet.\n" + "="*75)
    conn.close()

# ==========================================
# 11. GOOGLE SHEETS / STREAMLIT BRIDGE
# ==========================================
def sync_memory_to_cloud():
    print("☁️ Syncing local memory to Google Sheets Cloud...")
    try:
        creds_json = os.environ.get('GCP_CREDENTIALS')
        if not creds_json: 
            return print("⚠️ GCP_CREDENTIALS not found in environment. Skipping cloud sync.")

        client = gspread.authorize(Credentials.from_service_account_info(
            json.loads(creds_json), 
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        ))
        sheet, conn = client.open("Quant_Fund_DB"), sqlite3.connect(DB_NAME)
        df_swing, df_daily = pd.read_sql_query("SELECT * FROM swing_trades", conn), pd.read_sql_query("SELECT * FROM daily_predictions", conn)
        conn.close()

        for name, df in [("swing_trades", df_swing), ("daily_predictions", df_daily)]:
            try: ws = sheet.worksheet(name)
            except gspread.exceptions.WorksheetNotFound: ws = sheet.add_worksheet(title=name, rows="5000", cols="20")
            ws.clear()
            set_with_dataframe(ws, df)

        print(f"✅ Cloud Sync Complete! Data safely mirrored to 'Quant_Fund_DB'.")
    except Exception as e: 
        print(f"❌ Failed to sync to cloud. Check credentials and sharing permissions. Error: {e}")

# ==========================================
# 12. MAIN EXECUTION TRIGGER
# ==========================================
if __name__ == "__main__":
    print("🚀 STARTING DAILY HEDGE FUND SEQUENCE...\n")
    
    init_production_database()
    run_post_trade_audit()
    run_daily_agent_pipeline()
    allocate_multi_horizon_portfolio()
    display_institutional_dashboard()
    sync_memory_to_cloud()
    
    print("\n🎉 ALL SYSTEMS GO. DAILY CYCLE COMPLETE.")
