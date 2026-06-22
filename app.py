import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
from openai import OpenAI

st.set_page_config(page_title="AI Hedge Fund", page_icon="📈", layout="wide")
st.title("🤖 Multi-Agent Quant Dashboard")

# --- SECURE CLOUD DATABASE CONNECTION ---
@st.cache_data(ttl=600)
def load_cloud_data():
    # 1. Use the VIP Master Key (GCP_CREDENTIALS)
    creds_json = st.secrets["GCP_CREDENTIALS"]
    client_gs = gspread.authorize(Credentials.from_service_account_info(
        json.loads(creds_json), 
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    ))
    
    # 2. Open the spreadsheet securely bypassing all public link blocks
    sheet = client_gs.open_by_url("https://docs.google.com/spreadsheets/d/1okyCzz15cThwRYOePRBynKZoIjuclzVdGy8nCJYpbFI/edit")
    
    # 3. Read the data
    df_swing = pd.DataFrame(sheet.worksheet("swing_trades").get_all_records())
    
    # 4. Filter for dashboard views
    active = df_swing[df_swing['status'] == 'ACTIVE']
    history = df_swing[df_swing['status'] != 'ACTIVE']
    return active, history, df_swing

# Execute the secure data pull
df_active, df_history, df_full = load_cloud_data()

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
    df_history_sorted = df_history.sort_values(by='exit_date', ascending=False).head(10)
    display_hist = df_history_sorted[['exit_date', 'time_horizon', 'ticker', 'status', 'exit_price']]
    st.dataframe(display_hist, use_container_width=True, hide_index=True)
else:
    st.info("No historical trades logged yet.")

st.divider()

# --- AI CONVERSATIONAL TERMINAL ---
st.subheader("💬 Query Your Quantitative Agent")

portfolio_context = df_full.to_string(index=False)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about active setups, recent performance, or system win rates..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    system_instruction = f"""
    You are an elite quantitative portfolio analyst assistant for a proprietary multi-agent swing trading fund.
    You have direct live read access to the complete database of historical and active positions.
    
    Here is the live database state:
    {portfolio_context}
    
    Answer the user's queries professionally, using exact metrics, tickers, and price points from the data context where applicable.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            *st.session_state.messages
        ]
    )
    
    answer = response.choices[0].message.content
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
