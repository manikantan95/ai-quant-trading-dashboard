# app.py
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(page_title="AI Hedge Fund", page_icon="📈", layout="wide")
st.title("🤖 Multi-Agent Quant Dashboard")

# --- CLOUD DATABASE CONNECTION ---
# Establish connection to Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=600)
def load_cloud_data():
    # Read the data directly from your Google Sheet tabs
    df_swing = conn.read(worksheet="swing_trades", usecols=list(range(14))) 
    
    # Filter it just like we did in SQLite
    active = df_swing[df_swing['status'] == 'ACTIVE']
    history = df_swing[df_swing['status'] != 'ACTIVE']
    return active, history

df_active, df_history = load_cloud_data()

# --- TOP METRICS ROW ---
col1, col2, col3 = st.columns(3)
col1.metric("Active Positions", len(df_active))
col2.metric("Total Closed Trades", len(df_history))

if not df_history.empty:
    win_rate = (len(df_history[df_history['status'] == 'HIT_TARGET']) / len(df_history)) * 100
    col3.metric("System Win Rate", f"{win_rate:.1f}%")
else:
    col3.metric("System Win Rate", "0.0%")

st.divider()

# --- ACTIVE TRADES SECTION ---
st.subheader("🟢 Active Portfolio Allocation")
if not df_active.empty:
    display_df = df_active[['time_horizon', 'setup_date', 'ticker', 'cap_class', 'entry_price', 'target_price', 'stop_loss']]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No active trades currently held.")

st.divider()

# --- HISTORICAL TRADES SECTION ---
st.subheader("🏆 Recently Closed Trades")
if not df_history.empty:
    # Sort by the most recently exited trades
    df_history = df_history.sort_values(by='exit_date', ascending=False).head(10)
    display_hist = df_history[['exit_date', 'time_horizon', 'ticker', 'status', 'exit_price']]
    st.dataframe(display_hist, use_container_width=True, hide_index=True)
else:
    st.info("No historical trades logged yet.")
