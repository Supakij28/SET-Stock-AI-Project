import streamlit as st
import yfinance as yf
import yahooquery as yq
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from dtaidistance import dtw
import os
import requests
import google.generativeai as genai
import json
import pytz
import textwrap
from supabase import create_client, Client
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# --- Configuration ---
SET_TZ = pytz.timezone('Asia/Bangkok')
st.set_page_config(page_title="Quant Strategy Station", layout="wide")

# --- Password Protection ---
def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Safety check for secrets access
        actual_password = os.getenv("APP_PASSWORD", "admin1234")
        try:
            if "APP_PASSWORD" in st.secrets:
                actual_password = st.secrets["APP_PASSWORD"]
        except:
            pass

        if st.session_state["password"] == actual_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Please enter the access password", type="password", on_change=password_entered, key="password"
        )
        st.info("💡 Tip: For first-time setup on Streamlit Cloud, add 'APP_PASSWORD' to your Secrets.")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Please enter the access password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

# --- Database Integration (Supabase) ---
def get_supabase_client() -> Client:
    """Initialize Supabase client from Streamlit secrets or environment variables."""
    url = ""
    key = ""
    
    # 1. Try Streamlit Secrets (Cloud)
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
    except:
        pass
        
    # 2. Try Environment Variables (Local)
    if not url or not key:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
    if not url or not key:
        st.error("🔑 Supabase credentials missing! Please set SUPABASE_URL and SUPABASE_KEY.")
        return None
        
    return create_client(url, key)

supabase = get_supabase_client()

def init_db():
    """Verify Supabase connection."""
    if supabase:
        try:
            # Simple health check
            supabase.table("scan_results").select("count", count="exact").limit(1).execute()
        except Exception as e:
            st.warning(f"⚠️ Supabase connection issue or table missing: {e}")
            st.info("💡 Make sure you have run the schema.sql in Supabase SQL Editor.")

def init_log_db():
    """Verify Supabase connection for logs."""
    pass # Managed via Supabase

def get_silent_accum_insights(limit=50):
    """
    Analyze historical SILENT ACCUM signals to find 'Time to Upside'.
    """
    if not supabase:
        return None
        
    try:
        # Get last N SILENT ACCUM signals from Supabase
        response = supabase.table("scan_results") \
            .select("ticker, scan_date, price, signal_type") \
            .eq("signal_type", "SILENT ACCUM") \
            .order("id", desc=True) \
            .limit(limit) \
            .execute()
            
        signals = pd.DataFrame(response.data)
        
        if signals.empty:
            return None
            
        results = []
        for _, sig in signals.iterrows():
            ticker = sig['ticker']
            signal_date = pd.to_datetime(sig['scan_date'])
            entry_price = sig['price']
            
            # Get historical data for this ticker
            df = get_stock_data(ticker)
            if df is not None and not df.empty:
                # Filter data from signal date onwards
                future_df = df[df.index >= signal_date].copy()
                if len(future_df) > 1:
                    # Skip the signal day itself for 'days to move' calculation
                    test_df = future_df.iloc[1:11] # Look up to 10 days ahead
                    
                    found_move = False
                    for day_idx, (idx, row) in enumerate(test_df.iterrows()):
                        # Check if high reached +1%
                        max_ret = (row['High'] / entry_price - 1) * 100
                        if max_ret >= 1.0:
                            results.append({
                                'ticker': ticker,
                                'signal_date': sig['scan_date'],
                                'days_to_move': day_idx + 1,
                                'max_gain_t5': (test_df.iloc[:5]['High'].max() / entry_price - 1) * 100 if len(test_df) >= 5 else None,
                                'win_t5': 1 if (len(test_df) >= 5 and (test_df.iloc[:5]['High'].max() / entry_price - 1) * 100 >= 1.0) else 0
                            })
                            found_move = True
                            break
                    
                    if not found_move and len(test_df) >= 5:
                        # Recorded as no move within 10 days, but still check T+5 win
                        results.append({
                            'ticker': ticker,
                            'signal_date': sig['scan_date'],
                            'days_to_move': None,
                            'max_gain_t5': (test_df.iloc[:5]['High'].max() / entry_price - 1) * 100,
                            'win_t5': 1 if (test_df.iloc[:5]['High'].max() / entry_price - 1) * 100 >= 1.0 else 0
                        })
        
        return pd.DataFrame(results) if results else None
    except Exception as e:
        st.error(f"Error in SILENT ACCUM analysis: {e}")
        return None

def save_analysis_snapshot(batch_df, market_regime):
    """Save a snapshot of the current scan results for performance tracking."""
    if batch_df.empty or not supabase:
        return
        
    try:
        run_id = datetime.now(SET_TZ).strftime("%Y%m%d_%H%M%S")
        now = datetime.now(SET_TZ)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine Session Flag
        hour = now.hour
        if 7 <= hour < 12: session_flag = "Morning"
        elif 12 <= hour < 15: session_flag = "Midday"
        else: session_flag = "Afternoon"
        
        log_entries = []
        for _, row in batch_df.iterrows():
            log_entries.append({
                "run_id": run_id,
                "timestamp": timestamp,
                "session_flag": session_flag,
                "ticker": row['Ticker'],
                "signal": row['Signal'],
                "last_price": row['Last Price'],
                "high_price": row['Day High'],
                "market_context": market_regime,
                "status": 'Pending'
            })
            
        if log_entries:
            supabase.table("trading_log").insert(log_entries).execute()
            
    except Exception as e:
        print(f"Logging Error: {e}")

def validate_performance(days_forward=3):
    """
    Automatically validate T+3 performance for pending logs using Supabase.
    """
    if not supabase:
        return 0
        
    try:
        response = supabase.table("trading_log").select("*").eq("status", "Pending").execute()
        df_pending = pd.DataFrame(response.data)
        
        if df_pending.empty:
            return 0
            
        updated_count = 0
        
        for _, row in df_pending.iterrows():
            ticker = row['ticker']
            entry_price = row['last_price']
            entry_date = pd.to_datetime(row['timestamp']).normalize()
            record_id = row['id']
            
            df_hist = get_stock_data(ticker)
            if df_hist is not None and not df_hist.empty:
                valid_index = df_hist.index[df_hist.index <= entry_date]
                if not valid_index.empty:
                    last_valid_date = valid_index[-1]
                    idx = df_hist.index.get_loc(last_valid_date)
                    
                    if idx + days_forward < len(df_hist):
                        actual_entry_high = float(df_hist['High'].iloc[idx])
                        future_data = df_hist.iloc[idx+1 : idx+1+days_forward]
                        
                        if not future_data.empty:
                            t3_close = float(future_data['Close'].iloc[-1])
                            outcome_pct = ((t3_close - entry_price) / entry_price) * 100
                            min_low = float(future_data['Low'].min())
                            max_dd = ((min_low - entry_price) / entry_price) * 100
                            
                            status = "Success" if outcome_pct > 1.0 else "Fail"
                            
                            supabase.table("trading_log").update({
                                "high_price": actual_entry_high,
                                "outcome_t3_pct": outcome_pct,
                                "max_dd_pct": max_dd,
                                "status": status,
                                "verified_date": datetime.now(SET_TZ).strftime("%Y-%m-%d")
                            }).eq("id", record_id).execute()
                            
                            updated_count += 1
                        
        return updated_count
    except Exception as e:
        print(f"Validation Error: {e}")
        return 0

def save_scan_result(data):
    """Save a single scan result to Supabase, supporting optional labeling."""
    if not supabase:
        return
        
    try:
        # Base columns map
        payload = {
            "ticker": data['ticker'],
            "scan_date": data['date'],
            "scan_time": data['time'],
            "price": data['price'],
            "bull_score": data['bull_score'],
            "bear_score": data['bear_score'],
            "score_diff": data['score_diff'],
            "signal_type": data['signal_type'],
            "market_regime": data['market_regime'],
            "relative_vol": data['rel_vol'],
            "rsi": data['rsi']
        }
        
        # Optional columns
        if 'mtf_status' in data: payload['mtf_status'] = data['mtf_status']
        if 'mtf_score' in data: payload['mtf_score'] = data['mtf_score']
        if 'conviction_score' in data: payload['conviction_score'] = data['conviction_score']
        if 'outcome_label' in data: payload['outcome_label'] = data['outcome_label']
        if 'outcome_pct' in data: payload['outcome_pct'] = data['outcome_pct']
        if 'verified_date' in data: payload['verified_date'] = data['verified_date']
            
        supabase.table("scan_results").insert(payload).execute()
    except Exception as e:
        print(f"Database Error: {e}")

def get_historical_scores(ticker, limit=5):
    """Retrieve historical scores for a ticker from Supabase."""
    if not supabase:
        return pd.DataFrame()
        
    try:
        response = supabase.table("scan_results") \
            .select("*") \
            .eq("ticker", ticker) \
            .order("scan_date", desc=True) \
            .order("scan_time", desc=True) \
            .limit(limit) \
            .execute()
        return pd.DataFrame(response.data)
    except:
        return pd.DataFrame()

def run_automated_labeling(days_forward=3, win_threshold=2.0):
    """
    Check unlabeled scan results and verify if they were Win or Loss using Supabase.
    """
    if not supabase:
        return 0
        
    try:
        response = supabase.table("scan_results").select("*").is_("outcome_label", "null").execute()
        df_unlabeled = pd.DataFrame(response.data)
        
        if df_unlabeled.empty:
            return 0
        
        updated_count = 0
        now_th = datetime.now(SET_TZ).replace(tzinfo=None)
        
        for _, row in df_unlabeled.iterrows():
            ticker = row['ticker']
            scan_date_str = row['scan_date']
            scan_price = row['price']
            record_id = row['id']
            
            scan_date = pd.to_datetime(scan_date_str).normalize()
            
            try:
                df_hist = get_stock_data(ticker)
                if df_hist is not None and not df_hist.empty:
                    valid_index = df_hist.index[df_hist.index <= scan_date]
                    if not valid_index.empty:
                        last_valid_date = valid_index[-1]
                        idx = df_hist.index.get_loc(last_valid_date)
                        
                        if idx + days_forward < len(df_hist):
                            future_data = df_hist.iloc[idx+1 : idx+1+days_forward]
                            
                            if not future_data.empty:
                                max_high = float(future_data['High'].max())
                                max_return = ((max_high - scan_price) / scan_price) * 100
                                
                                label = "Win" if max_return >= win_threshold else "Loss"
                                
                                supabase.table("scan_results").update({
                                    "outcome_label": label,
                                    "outcome_pct": max_return,
                                    "verified_date": now_th.strftime("%Y-%m-%d")
                                }).eq("id", record_id).execute()
                                updated_count += 1
            except Exception as e:
                print(f"Error labeling {ticker} (ID: {record_id}): {e}")
                
        return updated_count
    except Exception as e:
        print(f"Labeling Error: {e}")
        return 0

# Initialize DB on startup
init_db()

# Global Fix for yfinance on Windows
try:
    yf.set_tz_cache_location(None)
except:
    pass

# --- SET100 List Management ---
TICKERS_FILE = "tickers_config.json"

# Hardcoded Fallback (Original List)
SET100_TICKERS_FALLBACK = [
    "ADVANC.BK", "AMATA.BK", "AOT.BK", "AP.BK", "AWC.BK", "BAM.BK", "BANPU.BK", "BBL.BK", "BCH.BK", "BCP.BK",
    "BCPG.BK", "BDMS.BK", "BEC.BK", "BEM.BK", "BGRIM.BK", "BH.BK", "BJC.BK", "BLA.BK", "BPP.BK", "BTS.BK",
    "CBG.BK", "CENTEL.BK", "CHG.BK", "CK.BK", "CKP.BK", "COM7.BK", "CPALL.BK", "CPF.BK", "CPN.BK", "CRC.BK",
    "DELTA.BK", "DOHOME.BK", "EA.BK", "EGCO.BK", "EPG.BK", "FORTH.BK", "GLOBAL.BK", "GPSC.BK", "GULF.BK", "GUNKUL.BK",
    "HANA.BK", "HMPRO.BK", "INTUCH.BK", "IRPC.BK", "IVL.BK", "JMART.BK", "JMT.BK", "KBANK.BK", "KCE.BK", "KKP.BK",
    "KTB.BK", "KTC.BK", "LH.BK", "M.BK", "MAJOR.BK", "MEGA.BK", "MINT.BK", "MTC.BK", "OR.BK", "ORI.BK",
    "OSP.BK", "PLANB.BK", "PRM.BK", "PSL.BK", "PTG.BK", "PTT.BK", "PTTEP.BK", "PTTGC.BK", "QH.BK", "RATCH.BK",
    "RCL.BK", "SAWAD.BK", "SCB.BK", "SCC.BK", "SCGP.BK", "SINGER.BK", "SIRI.BK", "SPALI.BK", "SPRC.BK", "STA.BK",
    "STEC.BK", "STGT.BK", "TASCO.BK", "TCAP.BK", "THANI.BK", "THG.BK", "TIDLOR.BK", "TIPH.BK", "TISCO.BK", "TOP.BK",
    "TQM.BK", "TRUE.BK", "TTA.BK", "TTB.BK", "TU.BK", "VGI.BK", "WHA.BK"
]

SET100_SECTORS_FALLBACK = {
    "ADVANC.BK": "ICT", "AMATA.BK": "Property", "AOT.BK": "Transport", "AP.BK": "Property", "AWC.BK": "Property",
    "BAM.BK": "Finance", "BANPU.BK": "Energy", "BBL.BK": "Banking", "BCH.BK": "Health", "BCP.BK": "Energy",
    "BCPG.BK": "Energy", "BDMS.BK": "Health", "BEC.BK": "Media", "BEM.BK": "Transport", "BGRIM.BK": "Energy",
    "BH.BK": "Health", "BJC.BK": "Commerce", "BLA.BK": "Insurance", "BPP.BK": "Energy", "BTS.BK": "Transport",
    "CBG.BK": "Food", "CENTEL.BK": "Tourism", "CHG.BK": "Health", "CK.BK": "Construct", "CKP.BK": "Energy",
    "COM7.BK": "Commerce", "CPALL.BK": "Commerce", "CPF.BK": "Food", "CPN.BK": "Property", "CRC.BK": "Commerce",
    "DELTA.BK": "Electronic", "DOHOME.BK": "Commerce", "EA.BK": "Energy", "EGCO.BK": "Energy", "EPG.BK": "Construct",
    "FORTH.BK": "ICT", "GLOBAL.BK": "Commerce", "GPSC.BK": "Energy", "GULF.BK": "Energy", "GUNKUL.BK": "Energy",
    "HANA.BK": "Electronic", "HMPRO.BK": "Commerce", "INTUCH.BK": "ICT", "IRPC.BK": "Energy", "IVL.BK": "Petrochem",
    "JMART.BK": "Commerce", "JMT.BK": "Finance", "KBANK.BK": "Banking", "KCE.BK": "Electronic", "KKP.BK": "Banking",
    "KTB.BK": "Banking", "KTC.BK": "Finance", "LH.BK": "Property", "M.BK": "Food", "MAJOR.BK": "Media",
    "MEGA.BK": "Commerce", "MINT.BK": "Food", "MTC.BK": "Finance", "OR.BK": "Energy", "ORI.BK": "Property",
    "OSP.BK": "Food", "PLANB.BK": "Media", "PRM.BK": "Transport", "PSL.BK": "Transport", "PTG.BK": "Energy",
    "PTT.BK": "Energy", "PTTEP.BK": "Energy", "PTTGC.BK": "Petrochem", "QH.BK": "Property", "RATCH.BK": "Energy",
    "RCL.BK": "Transport", "SAWAD.BK": "Finance", "SCB.BK": "Banking", "SCC.BK": "Construct", "SCGP.BK": "Packaging",
    "SINGER.BK": "Commerce", "SIRI.BK": "Property", "SPALI.BK": "Property", "SPRC.BK": "Energy", "STA.BK": "Agri",
    "STEC.BK": "Construct", "STGT.BK": "Agri", "TASCO.BK": "Construct", "TCAP.BK": "Finance", "THANI.BK": "Finance",
    "THG.BK": "Health", "TIDLOR.BK": "Finance", "TIPH.BK": "Insurance", "TISCO.BK": "Banking", "TOP.BK": "Energy",
    "TQM.BK": "Insurance", "TRUE.BK": "ICT", "TTA.BK": "Transport", "TTB.BK": "Banking", "TU.BK": "Food",
    "VGI.BK": "Media", "WHA.BK": "Property"
}

def load_ticker_config():
    """Load tickers and sectors from JSON file, fallback to hardcoded if not exists."""
    if os.path.exists(TICKERS_FILE):
        try:
            with open(TICKERS_FILE, "r", encoding='utf-8') as f:
                config = json.load(f)
                return config.get("tickers", SET100_TICKERS_FALLBACK), config.get("sectors", SET100_SECTORS_FALLBACK)
        except:
            pass
    return SET100_TICKERS_FALLBACK, SET100_SECTORS_FALLBACK

def save_ticker_config(tickers, sectors):
    """Save tickers and sectors to JSON file."""
    try:
        with open(TICKERS_FILE, "w", encoding='utf-8') as f:
            json.dump({"tickers": tickers, "sectors": sectors, "last_update": str(datetime.now())}, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Save Config Error: {e}")
        return False

def fetch_set100_from_web():
    """
    Fetch current SET100 tickers from a reliable source (Yahoo Finance Index or similar).
    Note: Direct scraping of SET.or.th is often blocked, so we use a hybrid approach.
    """
    try:
        # Method: Use YahooQuery to get index components if possible
        # Alternatively, we can use a known reliable financial data provider URL
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
        # Try to get from a stable source like a maintained list or Yahoo Query
        # For Thailand, the symbol ^SET100.BK components aren't always available via API
        # So we try to find a table from a financial news site that is easier to scrape
        url = "https://www.set.or.th/th/market/index/set100/overview" # User's requested URL
        # Note: In a real app, you might need a more robust scraper for SET.or.th
        # For now, we'll implement a robust fallback fetcher.
        
        # Simulating a fetch for demonstration or using a robust fallback
        # Let's try to get it via yahooquery symbol search for the index
        t = yq.Ticker("^SET100.BK")
        # Unfortunately, Yahoo doesn't always provide index components for SET100 via API
        
        # Fallback: Since scraping SET.or.th directly in Streamlit/Trae might be tricky,
        # we will provide a way for the user to verify/refresh via the UI
        # and we will use the existing list as the base.
        return None # In a real implementation, return the list of tickers
    except:
        return None

# Initial Load
SET100_TICKERS, SET100_SECTORS = load_ticker_config()

# --- 0. AI Optimizer Functions ---
def get_ai_optimization(ticker, stats, trade_log, current_params, api_key):
    """Call Gemini to analyze trade performance and suggest better parameters."""
    if not api_key:
        st.sidebar.warning("Please enter Google API Key first.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        
        # 1. Dynamically find available models
        model_name = None
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Preference order
            for target in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro', 'models/gemini-pro']:
                if target in available_models:
                    model_name = target
                    break
            if not model_name and available_models:
                model_name = available_models[0]
        except Exception as list_err:
            # If list_models fails, fallback to standard names
            model_name = 'gemini-1.5-flash'
            
        # Clean the model name (remove 'models/' prefix if present)
        clean_model_name = model_name.replace('models/', '') if model_name else 'gemini-1.5-flash'
        model = genai.GenerativeModel(clean_model_name)
        
        # Prepare context
        log_summary = trade_log.tail(10).to_string() if trade_log is not None else "No trades yet."
        prompt = f"""
        As a Senior Quant Analyst, optimize this trading strategy for {ticker}.
        Current Stats: {stats}
        Current Parameters: {current_params}
        Recent Trades: {log_summary}
        
        Goal: Improve Win Rate and Total Return while minimizing Max Drawdown.
        Return ONLY a JSON object with these keys:
        - rsi_p (5-30)
        - rsi_b (10-80)
        - rsi_s (40-90)
        - ema_f (5-50)
        - ema_s (10-200)
        - rv_m (1.0-3.0)
        - reasoning (brief explanation in Thai)
        """
        
        response = model.generate_content(prompt)
        text = response.text
        # Extract JSON from potential markdown blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
            
        return json.loads(text.strip())
    except Exception as e:
        st.sidebar.error(f"AI Error: {str(e)}")
        return None

# --- 0. Market Regime Helper ---
@st.cache_data(ttl=3600)
def get_market_regime():
    """Fetch SET Index and determine if we are in a Bull or Bear market."""
    try:
        set_idx = get_stock_data("^SET.BK")
        if set_idx is not None:
            set_idx['EMA200'] = set_idx['Close'].ewm(span=200, adjust=False).mean()
            curr_price = set_idx['Close'].iloc[-1]
            ema200 = set_idx['EMA200'].iloc[-1]
            return "BULL" if curr_price > ema200 else "BEAR", curr_price
    except:
        pass
    return "UNKNOWN", 0

# --- 1. Data & Indicators ---
@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    """Fetch sector and industry info with fallback and browser-like headers."""
    # 0. Try loaded config first
    if ticker in SET100_SECTORS:
        return SET100_SECTORS[ticker]
        
    # 1. Try YahooQuery with Session Headers
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        t = yq.Ticker(ticker, session=session)
        profile = t.summary_profile
        if isinstance(profile, dict):
            data = profile.get(ticker) or next(iter(profile.values()), None)
            if isinstance(data, dict):
                s = data.get('sector') or data.get('sectorDisp')
                if s and s != 'N/A': return s
    except:
        pass
    
    # 2. Try yfinance info as last resort
    try:
        info = yf.Ticker(ticker).info
        if info and 'sector' in info:
            return info['sector']
    except:
        pass

    return 'N/A'

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        t = yq.Ticker(ticker, session=session)
        df = t.history(start="2018-01-01").reset_index()
        if "symbol" in df.columns: df = df[df["symbol"] == ticker]
        if df.empty: return None
        
        # Ensure date column is datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index("date")[["close", "volume", "open", "high", "low"]].rename(
            columns={"close": "Close", "volume": "Volume", "open": "Open", "high": "High", "low": "Low"}
        )
        return df
    except:
        return None

def get_mtf_confluence(ticker):
    """
    Perform Multi-Timeframe analysis (1H and 15M) to confirm Daily signals.
    Improved Logic: Uses EMA Cross and RSI Momentum.
    """
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        t = yq.Ticker(ticker, session=session)
        
        # 1. Check 1H Timeframe (Trend Confirmation)
        df_1h = t.history(period='7d', interval='1h').reset_index()
        if df_1h.empty: return "N/A", 0
        
        # Ensure 'close' column is used
        close_1h = df_1h['close']
        ema10_1h = close_1h.ewm(span=10, adjust=False).mean()
        ema20_1h = close_1h.ewm(span=20, adjust=False).mean()
        
        is_1h_bull = ema10_1h.iloc[-1] > ema20_1h.iloc[-1]
        is_1h_trending = close_1h.iloc[-1] > ema10_1h.iloc[-1]
        
        # 2. Check 15M Timeframe (Entry Timing)
        df_15m = t.history(period='2d', interval='15m').reset_index()
        if df_15m.empty: return "1H Only", 50 if is_1h_bull else 0
        
        close_15m = df_15m['close']
        # Simple RSI for 15M
        delta = close_15m.diff()
        gain = delta.clip(lower=0)
        loss = delta.clip(upper=0).abs()
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_15m = 100 - (100 / (1 + rs))
        
        is_15m_recovering = rsi_15m.iloc[-1] > rsi_15m.iloc[-2] and rsi_15m.iloc[-1] > 45
        is_15m_bull = close_15m.iloc[-1] > close_15m.ewm(span=10, adjust=False).mean().iloc[-1]
        
        # Calculate Confluence Score
        score = 0
        status = []
        if is_1h_bull: 
            score += 30
            status.append("1H Bull")
        if is_1h_trending:
            score += 20
            status.append("1H Trend")
        if is_15m_recovering:
            score += 30
            status.append("15M RSI Up")
        if is_15m_bull:
            score += 20
            status.append("15M Bull")
            
        final_status = " + ".join(status) if status else "No Confluence"
        return final_status, score
    except Exception as e:
        return f"Error: {str(e)[:20]}", 0

def generate_ai_trading_plan(ticker, row, api_key, ai_insights=None):
    """
    Generate a detailed Trading Plan using Google Gemini AI.
    """
    if not api_key:
        return "⚠️ กรุณาใส่ Google API Key ใน Sidebar เพื่อใช้งาน AI Trading Plan"
    
    try:
        genai.configure(api_key=api_key)
        
        # Robust model selection (matching get_ai_optimization)
        model_name = None
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Preference order
            for target in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro', 'models/gemini-pro']:
                if target in available_models:
                    model_name = target
                    break
            if not model_name and available_models:
                model_name = available_models[0]
        except:
            # If list_models fails, fallback to standard names
            model_name = 'gemini-1.5-flash'
            
        # Clean the model name (remove 'models/' prefix if present)
        clean_model_name = model_name.replace('models/', '') if model_name else 'gemini-1.5-flash'
        model = genai.GenerativeModel(clean_model_name)
        
        # Prepare context data
        context = f"""
        คุณคือผู้เชี่ยวชาญการเทรดหุ้นแนว Quant (Professional Quant Trader)
        ช่วยเขียนแผนการเทรด (Trading Plan) สำหรับหุ้น {ticker} โดยใช้ข้อมูลเทคนิคดังนี้:
        - ราคาล่าสุด: {row['Last Price']}
        - สัญญาณหลัก (Signal): {row['Signal']}
        - คะแนนฝั่งซื้อ (Bull Score): {row['Bullish Score (%)']}%
        - คะแนนฝั่งขาย (Bear Score): {row['Bearish Score (%)']}%
        - ความต่างของคะแนน (Score Diff): {row['Score Diff']}
        - การยืนยันหลายไทม์เฟรม (MTF Status): {row['MTF Conf']} (Score: {row['MTF Score']})
        - Relative Volume: {row['Relative Vol']}x
        - ความแม่นยำทางสถิติ (Pattern Consensus): {row['Pattern Consensus (%)']}%
        - สภาวะตลาด (Market Regime): {st.session_state.get('market_regime', 'N/A')}
        
        {f"ข้อมูลวิเคราะห์เพิ่มเติมจาก AI (Historical Insights): {ai_insights}" if ai_insights else ""}
        
        กรุณาเขียนแผนในรูปแบบ Markdown (ภาษาไทย) โดยมีหัวข้อดังนี้:
        1. **Strategy**: แนะนำกลยุทธ์ (เช่น Breakout, Buy on Dip, หรือ Wait)
        2. **Trade Setup**: อธิบายเหตุผลที่น่าสนใจตามข้อมูลเทคนิค
        3. **Execution Plan**:
           - Entry Zone: (ช่วงราคาที่น่าเข้าซื้อ)
           - Stop Loss: (จุดตัดขาดทุนที่เหมาะสม)
           - Take Profit: (เป้าหมายทำกำไร 1 และ 2)
        4. **Risk Management**: ข้อควรระวังสำหรับหุ้นตัวนี้
        """
        
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

def calculate_quant_indicators(df, rsi_period, ema_fast_len, ema_slow_len):
    d = df.copy()
    # RSI
    delta = d['Close'].diff()
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs()
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    d['RSI'] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))
    
    # EMAs
    d['EMA_Fast'] = d['Close'].ewm(span=ema_fast_len, adjust=False).mean()
    d['EMA_Slow'] = d['Close'].ewm(span=ema_slow_len, adjust=False).mean()
    
    # Relative Volume (RV)
    d['RV'] = d['Volume'] / (d['Volume'].rolling(20).mean() + 1e-9)
    
    # ATR (Volatility Guard)
    high_low = d['High'] - d['Low']
    high_close = (d['High'] - d['Close'].shift()).abs()
    low_close = (d['Low'] - d['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    d['ATR'] = true_range.rolling(14).mean()
    d['ATR_Avg_5'] = d['ATR'].rolling(5).mean()
    
    return d.dropna()

# --- 2. Backtesting Engine ---
def run_backtest(df, rsi_buy, rsi_sell, rv_min):
    d = df.copy()
    # Buy Signal: EMA Cross UP
    d['EMA_Cross_Up'] = (d['EMA_Fast'] > d['EMA_Slow']) & (d['EMA_Fast'].shift(1) <= d['EMA_Slow'].shift(1))
    
    # Sell Signal: EMA Cross DOWN or RSI > Sell Threshold
    d['EMA_Cross_Down'] = (d['EMA_Fast'] < d['EMA_Slow']) & (d['EMA_Fast'].shift(1) >= d['EMA_Slow'].shift(1))
    
    # Final Conditions
    buy_cond = d['EMA_Cross_Up'] & (d['RSI'] <= rsi_buy) & (d['RV'] >= rv_min)
    sell_cond = d['EMA_Cross_Down'] | (d['RSI'] >= rsi_sell)
    
    # Stats for debugging
    total_crosses = d['EMA_Cross_Up'].sum()
    rsi_met = (d['EMA_Cross_Up'] & (d['RSI'] <= rsi_buy)).sum()
    vol_met = (d['EMA_Cross_Up'] & (d['RV'] >= rv_min)).sum()
    
    debug_info = {
        'total_crosses': total_crosses,
        'rsi_met': rsi_met,
        'vol_met': vol_met
    }
    
    position = 0
    trades = []
    equity = [100000] # Starting Capital
    
    for i in range(len(d)):
        if position == 0 and buy_cond.iloc[i]:
            position = 1
            entry_price = d['Close'].iloc[i]
            entry_date = d.index[i]
        elif position == 1 and sell_cond.iloc[i]:
            position = 0
            exit_price = d['Close'].iloc[i]
            exit_date = d.index[i]
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
            trades.append({
                'Entry Date': entry_date,
                'Entry Price': entry_price,
                'Exit Date': exit_date,
                'Exit Price': exit_price,
                'Profit (%)': profit_pct
            })
            equity.append(equity[-1] * (1 + profit_pct/100))
            
    if not trades: return None, debug_info, None
    
    trade_log = pd.DataFrame(trades)
    win_rate = (len(trade_log[trade_log['Profit (%)'] > 0]) / len(trade_log)) * 100
    total_return = ((equity[-1] - 100000) / 100000) * 100
    
    # Max Drawdown
    equity_series = pd.Series(equity)
    drawdown = (equity_series.cummax() - equity_series) / equity_series.cummax()
    max_dd = drawdown.max() * 100
    
    stats = {
        'Total Return (%)': total_return,
        'Win Rate (%)': win_rate,
        'Max Drawdown (%)': max_dd,
        'Total Trades': len(trades),
        'debug': debug_info
    }
    
    equity_df = pd.DataFrame({'Trade': range(len(equity)), 'Equity': equity})
    return trade_log, stats, equity_df

# --- 3. DTW Pattern Projection ---
def get_dtw_projection(df, lookback=40, forecast=20):
    prices = df['Close'].values
    if len(prices) < lookback + forecast + 100: return None
    
    current_window = prices[-lookback:]
    scaler = StandardScaler()
    curr_norm = scaler.fit_transform(current_window.reshape(-1, 1)).flatten()
    
    matches = []
    # Scan history (skip the last part to avoid overlapping with current)
    for i in range(len(prices) - lookback - forecast - 10):
        hist_window = prices[i : i + lookback]
        hist_norm = scaler.fit_transform(hist_window.reshape(-1, 1)).flatten()
        
        # Calculate DTW distance
        dist = dtw.distance(curr_norm, hist_norm)
        matches.append({'index': i, 'dist': dist})
    
    # Get top 3 matches
    best_matches = sorted(matches, key=lambda x: x['dist'])[:3]
    
    projections = []
    for m in best_matches:
        idx = m['index']
        # Get the move AFTER the historical match
        future_prices = prices[idx + lookback : idx + lookback + forecast]
        # Normalize and scale to current price level
        base_price = prices[-1]
        start_hist_future = prices[idx + lookback - 1]
        pct_changes = future_prices / start_hist_future
        projected_path = base_price * pct_changes
        
        projections.append({
            'date_range': f"{pd.to_datetime(df.index[idx]).date()} to {pd.to_datetime(df.index[idx+lookback]).date()}",
            'path': projected_path,
            'score': 100 * (1 - m['dist']/max([x['dist'] for x in matches]))
        })
    return projections

def get_pre_breakout_scanner(df, breakout_threshold=0.05, lookback=5, mode='bullish'):
    """
    Improved Scanner: Multivariate DTW + Slope Filter + Z-score Normalization.
    - Price, Volume, and Volatility patterns are compared.
    - Global Constraint (Sakoe-Chiba) applied to DTW.
    """
    if len(df) < 100: return None
    
    # 1. Prepare Data
    df_scan = df.copy()
    df_scan['Pct_Change'] = df_scan['Close'].pct_change()
    df_scan['Volatility'] = df_scan['Close'].rolling(5).std() # Local volatility
    
    if mode == 'bullish':
        events = df_scan[df_scan['Pct_Change'] >= breakout_threshold].index.tolist()
    else:
        events = df_scan[df_scan['Pct_Change'] <= -breakout_threshold].index.tolist()
    
    patterns = []
    scaler = StandardScaler()
    
    for b_date in events:
        idx = df_scan.index.get_loc(b_date)
        if idx < lookback: continue
        
        # Extract 5 days BEFORE event
        pattern_data = df_scan.iloc[idx-lookback : idx]
        jump_price = df_scan['Close'].iloc[idx]
        
        # Multivariate Normalization (Z-score)
        p_norm = scaler.fit_transform(pattern_data[['Close']]).flatten()
        v_norm = scaler.fit_transform(pattern_data[['Volume']]).flatten()
        volat_norm = scaler.fit_transform(pattern_data[['Volatility']].fillna(0)).flatten()
        
        jump_norm = scaler.fit(pattern_data[['Close']]).transform([[jump_price]])[0][0]
        
        ohlc_raw = df_scan.iloc[idx-lookback : idx+1][['Open', 'High', 'Low', 'Close']]
        
        patterns.append({
            'date': b_date,
            'p_norm': p_norm,
            'v_norm': v_norm,
            'volat_norm': volat_norm,
            'jump_norm': jump_norm,
            'jump_pct': df_scan['Pct_Change'].iloc[idx] * 100,
            'ohlc_raw': ohlc_raw
        })
    
    if not patterns: return None
    
    # 2. Current Pattern Extraction
    current_5 = df_scan.iloc[-lookback:]
    curr_p_norm = scaler.fit_transform(current_5[['Close']]).flatten()
    curr_v_norm = scaler.fit_transform(current_5[['Volume']]).flatten()
    curr_volat_norm = scaler.fit_transform(current_5[['Volatility']].fillna(0)).flatten()
    
    # Slope Filter: Recent 2 days slope check
    # Instead of returning None, we'll just let the distance metric handle it
    # or make it very lenient (e.g., not a massive drop)
    slope_2d = (current_5['Close'].iloc[-1] - current_5['Close'].iloc[-3]) / current_5['Close'].iloc[-3]
    if mode == 'bullish' and slope_2d < -0.02: return None # Only block if dropping > 2% in 2 days
    
    results = []
    for p in patterns:
        # Multivariate DTW with Sakoe-Chiba constraint (window=2)
        dist_p = dtw.distance(curr_p_norm, p['p_norm'], window=2)
        dist_v = dtw.distance(curr_v_norm, p['v_norm'], window=2)
        dist_volat = dtw.distance(curr_volat_norm, p['volat_norm'], window=2)
        
        # Weighted Distance (Magnitude & Context)
        # w1: Price(0.5), w2: Volume(0.3), w3: Volatility(0.2)
        total_dist = (0.5 * dist_p) + (0.3 * dist_v) + (0.2 * dist_volat)
        
        # Scale historical OHLC for visualization
        base_price = current_5['Open'].iloc[0]
        hist_base = p['ohlc_raw']['Open'].iloc[0]
        ohlc_scaled = p['ohlc_raw'] * (base_price / hist_base)
        
        results.append({
            'date': p['date'],
            'jump': p['jump_pct'],
            'dist': total_dist,
            'hist_p': p['p_norm'],
            'hist_p_ext': np.append(p['p_norm'], p['jump_norm']),
            'hist_v': p['v_norm'],
            'hist_ohlc_scaled': ohlc_scaled
        })
    
    best_results = sorted(results, key=lambda x: x['dist'])[:3]
    return {
        'curr_p': curr_p_norm,
        'curr_v': curr_v_norm,
        'curr_ohlc': current_5[['Open', 'High', 'Low', 'Close']],
        'matches': best_results
    }

def validate_scanner_accuracy(df, winning_patterns, price_threshold=80.0, vol_threshold=70.0, lookback=5):
    """
    Advanced Validator: Multivariate (Price+Vol+Volatility) + Market Regime + Momentum.
    """
    if len(df) < 100 or not winning_patterns: return None
    
    scaler = StandardScaler()
    df_scan = df.copy()
    df_scan['Volatility'] = df_scan['Close'].rolling(5).std()
    df_scan['EMA_Cross_Up'] = (df_scan['EMA_Fast'] > df_scan['EMA_Slow']) & (df_scan['EMA_Fast'].shift(1) <= df_scan['EMA_Slow'].shift(1))
    df_scan['EMA_Slow_Slope'] = df_scan['EMA_Slow'].diff(3)
    df_scan['Strong_Trend'] = (df_scan['Close'] > df_scan['EMA_Slow']) & (df_scan['EMA_Fast'] > df_scan['EMA_Slow']) & (df_scan['EMA_Slow_Slope'] > 0)
    
    # Calculate return over the NEXT 3 DAYS to see if it eventually jumps
    # (More realistic for swing/ATC)
    df_scan['Next_3D_Max_Return'] = df_scan['High'].rolling(3).max().shift(-3) / df_scan['Close'] - 1
    
    # Get Market Regime for Context Filter
    regime, _ = get_market_regime()
    
    results = []
    scan_start = max(lookback, len(df_scan) - 500) 
    
    for i in range(scan_start, len(df_scan) - 3): # -3 to allow 3-day forward window
        # 1. Context Filter (Strong Trend or EMA Cross)
        has_cross = df_scan['EMA_Cross_Up'].iloc[i-lookback : i].any()
        is_trending = df_scan['Strong_Trend'].iloc[i-1]
        if not (has_cross or is_trending): continue
            
        # 2. Market Regime Adjustment: Stricter in BEAR market
        adj_p_threshold = price_threshold if regime == "BULL" else price_threshold + 5.0
        adj_v_threshold = vol_threshold if regime == "BULL" else vol_threshold + 5.0
            
        window = df_scan.iloc[i-lookback : i]
        # Slope check: Must not be a sharp drop
        win_slope = (window['Close'].iloc[-1] - window['Close'].iloc[-3]) / window['Close'].iloc[-3]
        if win_slope < -0.01: continue

        win_p_norm = scaler.fit_transform(window[['Close']]).flatten()
        win_v_norm = scaler.fit_transform(window[['Volume']]).flatten()
        win_volat_norm = scaler.fit_transform(window[['Volatility']].fillna(0)).flatten()
        
        best_sim_p = 0
        best_sim_v = 0
        
        for p in winning_patterns:
            # Multivariate DTW with constraint
            dist_p = dtw.distance(win_p_norm, p['price_pattern'], window=2)
            dist_v = dtw.distance(win_v_norm, p['vol_pattern'], window=2)
            
            sim_p = max(0, 100 - (dist_p * 20))
            sim_v = max(0, 100 - (dist_v * 20))
            
            if sim_p > best_sim_p: 
                best_sim_p = sim_p
                best_sim_v = sim_v
        
        # 3. Hybrid Filtering (Price + Vol + Trend)
        if best_sim_p >= adj_p_threshold and best_sim_v >= adj_v_threshold:
            next_ret = df_scan['Next_3D_Max_Return'].iloc[i-1]
            overall_sim = (best_sim_p * 0.6) + (best_sim_v * 0.4)
            
            results.append({
                'Date': df_scan.index[i-1],
                'Overall Sim': overall_sim,
                'Price Sim': best_sim_p,
                'Vol Sim': best_sim_v,
                'Next Day Return (%)': next_ret * 100,
                'Result': '✅ Hit' if next_ret > 0.01 else '❌ Miss' # Win if > 1% gain in 3 days
            })
            
    if not results: return None
    
    res_df = pd.DataFrame(results)
    total = len(res_df)
    hits_df = res_df[res_df['Result'] == '✅ Hit']
    misses_df = res_df[res_df['Result'] == '❌ Miss']
    
    hit_rate = (len(hits_df) / total) * 100
    avg_profit = hits_df['Next Day Return (%)'].mean() if not hits_df.empty else 0
    avg_loss = misses_df['Next Day Return (%)'].mean() if not misses_df.empty else 0
    expectancy = (hit_rate/100 * avg_profit) + ((1 - hit_rate/100) * avg_loss)
    
    return {
        'summary': {
            'Total Signals': total,
            'Hit Rate': hit_rate,
            'Avg Profit': avg_profit,
            'Avg Loss': avg_loss,
            'Expectancy': expectancy,
            'RR Ratio': abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        },
        'log': res_df.sort_values(by='Date', ascending=False)
    }

# --- Quant Helper Functions ---
def calculate_conviction_score(ticker, signal, similarity, mtf_score, regime, perf_stats, rsi=50, rel_vol=1.0, score_velocity=0, sector_rs=0, price_change=0):
    """
    Calculate the conviction score for a stock based on various metrics.
    Supports Strategy Classification and Sector Relative Strength (SRS).
    """
    # 1. Score Persistence
    hist_scores = get_historical_scores(ticker, limit=5)
    persistence = "Flat"
    if not hist_scores.empty and len(hist_scores) >= 2:
        latest = hist_scores['bull_score'].iloc[0]
        prev = hist_scores['bull_score'].iloc[-1]
        if latest > prev: persistence = "Rising 📈"
        elif latest < prev: persistence = "Falling 📉"
    
    # 2. Global Signal Win Rate
    sig_stats = perf_stats.get(signal, {})
    sig_win_rate = sig_stats.get('Win_Rate', 0)
    sig_total = sig_stats.get('Total', 0)
    
    formula_score = 0
    reasons = []
    warnings = []
    strategy = "SWING" # Default strategy
    
    # --- Strategy Logic: Identify DAY TRADE (Potential Spike) ---
    is_spike_potential = (score_velocity > 10) or (rel_vol > 1.8) or (signal in ['PRE-FLY', 'GOLDEN BUY'])
    
    if is_spike_potential:
        strategy = "DAY TRADE (SPIKE)"
        if rsi > 65:
            warnings.append("⚠️ DAY TRADE Risk: High RSI (Spike & Drop potential)")
        if persistence == "Falling 📉":
            warnings.append("⚠️ DAY TRADE Risk: Weakening Momentum")
    
    # [BONUS] Market Regime
    if regime == 'BULL': 
        formula_score += 20
        reasons.append("Market BULL (+20)")
        
    # [BONUS] Signal Quality
    if signal in ['PRE-FLY', 'GOLDEN BUY']: 
        formula_score += 30
        reasons.append(f"Signal {signal} (+30)")
    elif signal == 'SILENT ACCUM':
        formula_score += 15
        reasons.append("Signal Accumulation (+15)")
        
    # [BONUS] Sector Relative Strength (SRS)
    # SRS = Price Change - Sector Avg.
    # Logic: Leading Star (+20: หุ้นบวกสวนกลุ่มลบ), Outperformer (+10: >กลุ่ม 0.5%), Underperformer (-10), Laggard (-20)
    sector_avg = price_change - sector_rs
    
    if price_change > 0 and sector_avg < 0:
        formula_score += 20
        reasons.append("Leading Star 🌟 (Up while Sector Down) (+20)")
    elif sector_rs > 0.5:
        formula_score += 10
        reasons.append(f"Outperformer (RS: {sector_rs:+.1f}%) (+10)")
    elif sector_rs < -1.0: # Strict Filter: Penalty increased
        formula_score -= 25
        warnings.append(f"Underperformer ⚠️ (RS: {sector_rs:+.1f}%) (-25)")
        strategy = "WAIT (Weak Sector Flow)"
    elif sector_rs < -0.5:
        formula_score -= 10
        warnings.append(f"Underperformer (RS: {sector_rs:+.1f}%) (-10)")
    elif price_change < 0 and sector_avg > 0:
        formula_score -= 20
        warnings.append("Laggard ⚠️ (Down while Sector Up) (-20)")

    # [BONUS] Database-Backed Signal Performance
    if sig_total >= 5:
        if sig_win_rate >= 60:
            formula_score += 15
            reasons.append(f"High Reliability Signal (+15)")
        elif sig_win_rate < 40:
            formula_score -= 25
            warnings.append(f"Low Reliability Signal ({sig_win_rate:.1f}% Win Rate) (-25)")
        
    # [BONUS] Score Persistence
    if persistence == "Rising 📈": 
        formula_score += 20
        reasons.append("Score Rising (+20)")
        
    # [BONUS] Pattern Similarity
    if similarity >= 90: 
        formula_score += 30
        reasons.append("Pattern Match > 90% (+30)")
    elif similarity >= 80:
        formula_score += 15
        reasons.append("Pattern Match > 80% (+15)")
        
    # [BONUS] MTF Confirmation
    if mtf_score >= 80:
        formula_score += 20
        reasons.append("Strong MTF Conf (+20)")
    elif mtf_score >= 60:
        formula_score += 10
        reasons.append("Moderate MTF Conf (+10)")

    # [AUDIT INSIGHT] RSI Sweet Spot (40-55) - The "Dark Horse" Zone
    if 40 <= rsi <= 55:
        formula_score += 15
        reasons.append("Audit: RSI Sweet Spot (+15)")

    # [PENALTY] Overextended / Spike & Fade Protection
    # If RSI is too high or price jumped too much today, it might be a trap
    if rsi > 70:
        formula_score -= 20
        warnings.append("Overbought RSI (>70) (-20)")
    if rsi > 80:
        formula_score -= 30
        warnings.append("Extreme Overbought RSI (>80) (-30)")

    # [PENALTY] Risky Signals & Negative Persistence
    if signal in ['WAIT', 'WAIT (DOWNTREND)', 'WAIT (BEARISH TRAP)', 'FADING MOMENTUM', 'CONFLICT (HIGH RISK)']:
        formula_score -= 50
        warnings.append(f"Neutral/Bearish Signal: {signal} (-50)")
    
    if persistence == "Falling 📉":
        formula_score -= 15
        warnings.append("Momentum is Weakening (-15)")

    if signal == 'REJECTION WICK':
        penalty = -60
        if regime == 'BULL': penalty += 10
        if rsi < 50: penalty += 25
        if rsi > 70: penalty -= 40
        if rel_vol > 1.8: penalty -= 20
        
        formula_score += penalty
        if penalty < 0:
            warnings.append(f"REJECTION WICK ({penalty})")
        else:
            reasons.append(f"Wick at Support/Neutral RSI ({penalty})")
            
    elif signal == 'PIN BAR (SUPPORT)':
        formula_score += 25
        reasons.append("Bullish Pin Bar at Support (+25)")
            
    elif signal == 'CONFLICT (HIGH RISK)':
        formula_score -= 20
        warnings.append("CONFLICT SIGNAL (-20)")
        
    return formula_score, strategy, reasons, warnings

def get_recovery_signals(ticker, df):
    """
    Detects Bottom Fishing / Recovery signals for stocks that have been falling for a while.
    Focuses on Oversold RSI and Downside Exhaustion.
    """
    if df is None or len(df) < 20:
        return None
    
    d = df.tail(20).copy()
    rsi_curr = d['RSI'].iloc[-1]
    rsi_prev = d['RSI'].iloc[-2]
    close_curr = d['Close'].iloc[-1]
    
    # 1. Oversold Check
    is_oversold = rsi_curr < 35 or rsi_prev < 35
    is_rsi_turning = rsi_curr > rsi_prev and rsi_curr > 30 # Turning up from bottom
    
    # 2. Downside Exhaustion (Price falling for at least 3 days)
    # Check if last 3-5 days are mostly negative
    recent_returns = d['Close'].pct_change().tail(5)
    negative_days = (recent_returns < 0).sum()
    is_exhausted = negative_days >= 3
    
    # 3. Bottoming Candlesticks
    c_open = d['Open'].iloc[-1]
    c_high = d['High'].iloc[-1]
    c_low = d['Low'].iloc[-1]
    c_close = d['Close'].iloc[-1]
    body_size = abs(c_close - c_open)
    lower_wick = min(c_open, c_close) - c_low
    is_bullish_pin = lower_wick > (body_size * 1.5) and c_close > c_low
    
    # 4. Volume Spike at Bottom
    avg_vol = d['Volume'].rolling(10).mean().iloc[-1]
    curr_vol = d['Volume'].iloc[-1]
    is_vol_spike = curr_vol > (avg_vol * 1.5)
    
    # Calculate Recovery Score (0-100)
    score = 0
    reasons = []
    
    if is_oversold: 
        score += 40
        reasons.append("Oversold (RSI < 35)")
    if is_rsi_turning:
        score += 20
        reasons.append("RSI Turning Up")
    if is_exhausted:
        score += 15
        reasons.append("Price Exhaustion (3+ Days Drop)")
    if is_bullish_pin:
        score += 15
        reasons.append("Bullish Pin Bar (Support)")
    if is_vol_spike:
        score += 10
        reasons.append("Volume Spike at Bottom")
        
    if score >= 50: # Minimum score to be considered a recovery candidate
        return {
            'ticker': ticker,
            'recovery_score': score,
            'rsi': rsi_curr,
            'reasons': reasons,
            'price': close_curr,
            'is_pin': is_bullish_pin
        }
    return None

def generate_unified_report(batch_df, regime):
    """
    Combines scan results, persistence, and reliability into a single analysis dataframe.
    Now supports Intraday Memory (HMPRO Fix) and Signal Tier Sorting.
    """
    if batch_df.empty:
        return pd.DataFrame()
    
    # 1. Filter candidates (Score Diff > 5 AND NOT a 'WAIT' signal)
    # Filter out neutral/bearish signals from the main analysis to focus on quality
    ignored_signals = ['WAIT', 'WAIT (DOWNTREND)', 'WAIT (BEARISH TRAP)', 'FADING MOMENTUM', 'CONFLICT (HIGH RISK)']
    candidates = batch_df[
        (batch_df['Score Diff'] > 5) & 
        (~batch_df['Signal'].isin(ignored_signals))
    ].copy()
    
    if candidates.empty:
        # Fallback to show something if no perfect matches, but with lower score diff
        candidates = batch_df[batch_df['Score Diff'] > 0].copy()
        
    # Get signal performance stats
    perf_stats = get_signal_performance_stats()

    # --- NEW: Fetch Intraday Memory (signals from earlier today) ---
    today_str = datetime.now(SET_TZ).strftime("%Y-%m-%d")
    intraday_memory = {}
    if supabase:
        try:
            response = supabase.table("trading_log") \
                .select("ticker, signal") \
                .like("timestamp", f"{today_str}%") \
                .execute()
            today_logs = pd.DataFrame(response.data)
            
            if not today_logs.empty:
                for _, log_row in today_logs.iterrows():
                    t = log_row['ticker']
                    sig = log_row['signal']
                    if t not in intraday_memory: intraday_memory[t] = set()
                    intraday_memory[t].add(sig)
        except:
            pass
    
    pos_signals = ['BUY', 'GOLDEN BUY', 'PRE-FLY', 'PIN BAR (SUPPORT)', 'SILENT ACCUM']
    
    report_data = []
    # Analyze Top 20 by Score Diff
    for _, row in candidates.head(20).iterrows():
        ticker = row['Ticker']
        similarity = row['Pattern Consensus (%)']
        mtf_score = row.get('MTF Score', 0)
        rsi = row.get('RSI', 50)
        rel_vol = row.get('Relative Vol', 1.0)
        score_velocity = row.get('Score Velocity', 0)
        sector_rs = row.get('Sector_RS', 0)
        price_change = row.get('% Change', 0)
        atr = row.get('ATR', 0)
        last_price = row.get('Last Price', 0)
        
        # Dynamic Stop Loss: Last Price - (2 * ATR)
        stop_loss = last_price - (2 * atr) if atr > 0 else last_price * 0.95
        
        formula_score, strategy, reasons, warnings = calculate_conviction_score(
            ticker, row['Signal'], similarity, mtf_score, regime, perf_stats, 
            rsi, rel_vol, score_velocity, sector_rs=sector_rs, price_change=price_change
        )
        
        # Strict SRS Filter: Skip if significantly underperforming
        if sector_rs < -1.5:
            continue
        
        # --- Intraday Memory Check (The HMPRO Fix) ---
        past_signals = intraday_memory.get(ticker, set())
        has_positive_past = any(ps in pos_signals for ps in past_signals)
        
        if has_positive_past and row['Signal'] not in pos_signals:
            formula_score += 15 # Intraday Bonus
            reasons.append(f"Earlier Intraday Strength (+15): {list(past_signals)}")
            warnings.append(f"⚠️ Signal changed from {list(past_signals)} to {row['Signal']} at Close")

        # --- Signal Tier Sorting (Stricter) ---
        # Tier 1: Current Positive Signal AND Rising/Flat Persistence
        # Tier 2: Positive Signal with Falling Persistence OR Past Positive Signal
        # Tier 3: Others
        persistence_val = "Rising 📈" if score_velocity > 0 else "Falling 📉"
        
        tier = 3
        if row['Signal'] in pos_signals:
            tier = 1 if persistence_val == "Rising 📈" else 2
        elif has_positive_past:
            tier = 2
        
        # Penalize if signal is high risk or fading
        if row['Signal'] in ['REJECTION WICK', 'FADING MOMENTUM', 'CONFLICT (HIGH RISK)']:
            tier = 3

        hist_scores = get_historical_scores(ticker, limit=5)
        trend_data = hist_scores['bull_score'].tolist()[::-1] if not hist_scores.empty else []
        
        ticker_labeled = hist_scores[hist_scores['outcome_label'].notnull()] if not hist_scores.empty else pd.DataFrame()
        ticker_win_rate = 0
        if not ticker_labeled.empty:
            ticker_win_rate = (len(ticker_labeled[ticker_labeled['outcome_label'] == 'Win']) / len(ticker_labeled)) * 100
            
        sig_stats = perf_stats.get(row['Signal'], {})
        sig_win_rate = sig_stats.get('Win_Rate', 0)
            
        report_data.append({
            'Ticker': ticker,
            'Signal': row['Signal'],
            'Strategy': strategy,
            'Stop_Loss': stop_loss,
            'Persistence': "Rising 📈" if score_velocity > 0 else "Falling 📉",
            'Score_Trend': trend_data,
            'Similarity': similarity,
            'Ticker_Win_Rate': ticker_win_rate,
            'Signal_Win_Rate': sig_win_rate,
            'MTF_Score': mtf_score,
            'Conviction_Score': formula_score,
            'Signal_Tier': tier,
            'Why': reasons,
            'Warnings': warnings,
            'Price': row['Last Price'],
            'Intraday_History': list(past_signals)
        })
    
    # Sort by: 1. Signal Tier (1 is highest), 2. Conviction Score
    return pd.DataFrame(report_data).sort_values(['Signal_Tier', 'Conviction_Score'], ascending=[True, False])


# --- 7. SET100 Batch Scanner ---
def get_signal_performance_stats():
    """Calculate Win Rate for each signal type from Supabase."""
    if not supabase:
        return {}
        
    try:
        response = supabase.table("scan_results") \
            .select("signal_type, outcome_label") \
            .is_("outcome_label", "not.null") \
            .execute()
        df = pd.DataFrame(response.data)
        
        if df.empty:
            return {}
        
        perf = df.groupby('signal_type').agg(
            Total=('outcome_label', 'count'),
            Wins=('outcome_label', lambda x: (x == 'Win').sum())
        )
        perf['Win_Rate'] = (perf['Wins'] / perf['Total']) * 100
        return perf.to_dict('index')
    except Exception as e:
        print(f"Stats Error: {e}")
        return {}

def run_set100_batch_scan(tickers, target_date=None):
    """
    Scan all tickers for bullish and bearish matches with Conservative Logic.
    Supports historical scanning if target_date is provided.
    """
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Pre-fetch all sectors in bulk
    status_text.text("🔄 Initializing Sector Information...")
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        bulk_t = yq.Ticker(tickers, session=session)
        all_profiles = bulk_t.summary_profile
    except:
        all_profiles = {}
    
    # Get performance stats once for scoring
    perf_stats = get_signal_performance_stats()
    pos_count = 0
    neg_count = 0
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"Scanning {ticker} ({i+1}/{len(tickers)})...")
            df_full = get_stock_data(ticker)
            if df_full is not None and len(df_full) > 100:
                # If target_date is provided, slice data to that date
                if target_date:
                    t_ts = pd.Timestamp(target_date).normalize()
                    # Slice using .loc for robust DatetimeIndex handling
                    df_raw = df_full.loc[:t_ts].copy()
                    df_future = df_full.loc[t_ts:].iloc[1:].copy() # Future starts after t_ts
                else:
                    df_raw = df_full.copy()
                    df_future = pd.DataFrame()

                if len(df_raw) < 100: continue

                # Calculate indicators for ATR/Volatility check
                df = calculate_quant_indicators(df_raw, 14, 10, 50)
                
                # Bullish/Bearish Scan
                bullish = get_pre_breakout_scanner(df_raw, mode='bullish')
                bearish = get_pre_breakout_scanner(df_raw, mode='bearish')
                
                bull_score = 0
                bull_jump = 0
                if bullish and bullish['matches']:
                    best_bull = bullish['matches'][0]
                    bull_score = max(0, 100 - (best_bull['dist'] * 20))
                    bull_jump = best_bull['jump']
                
                bear_score = 0
                bear_jump = 0
                if bearish and bearish['matches']:
                    best_bear = bearish['matches'][0]
                    bear_score = max(0, 100 - (best_bear['dist'] * 20))
                    bear_jump = best_bear['jump']
                
                # Calculate Relative Volume
                avg_vol = df_raw['Volume'].rolling(5).mean().iloc[-1]
                curr_vol = df_raw['Volume'].iloc[-1]
                rel_vol = curr_vol / avg_vol if avg_vol > 0 else 1.0
                
                # Calculate % Change
                pct_change = ((df_raw['Close'].iloc[-1] - df_raw['Close'].iloc[-2]) / df_raw['Close'].iloc[-2]) * 100
                if pct_change > 0: pos_count += 1
                elif pct_change < 0: neg_count += 1
                
                # ATC Risk
                day_high = df_raw['High'].iloc[-1]
                last_price = df_raw['Close'].iloc[-1]
                atc_risk = ((day_high - last_price) / day_high) * 100 if day_high > 0 else 0
                
                # Volatility Guard
                atr_now = df['ATR'].iloc[-1]
                atr_avg5 = df['ATR_Avg_5'].iloc[-1]
                high_vol = atr_now > (atr_avg5 * 1.2)
                
                # NEW: Volume Compression (บีบตัวของวอลุ่มก่อนระเบิด)
                # เช็คว่าวอลุ่มวันนี้ต่ำกว่าค่าเฉลี่ย 5 วัน และอยู่ในช่วงสะสม (0.6 - 1.0)
                vol_avg5 = df['Volume'].rolling(5).mean().iloc[-1]
                curr_vol = df['Volume'].iloc[-1]
                is_vol_compressed = curr_vol < vol_avg5 and 0.6 <= rel_vol <= 1.0
                
                # Calculate Pattern Consensus (Hybrid Hit Rate + Trend Quality)
                # MOVED UP to avoid NameError in Signal Logic
                consensus = 0
                try:
                    # 1. Historical Pattern Match (Hit Rate)
                    df_temp = df.copy()
                    df_temp['Pct_Change'] = df_temp['Close'].pct_change()
                    events = df_temp[df_temp['Pct_Change'] >= 0.05].index.tolist()
                    
                    winning_pats = []
                    for b_date in events:
                        idx = df_temp.index.get_loc(b_date)
                        if idx < 5: continue
                        p_data = df_temp.iloc[idx-5 : idx]
                        p_norm = StandardScaler().fit_transform(p_data[['Close']]).flatten()
                        v_norm = StandardScaler().fit_transform(p_data[['Volume']]).flatten()
                        winning_pats.append({'price_pattern': p_norm, 'vol_pattern': v_norm})
                    
                    val = validate_scanner_accuracy(df, winning_pats, price_threshold=80.0, vol_threshold=70.0)
                    hit_rate = val['summary']['Hit Rate'] if val else 0
                    
                    # 2. Current Trend Quality (0-100)
                    ema_f_curr = df['EMA_Fast'].iloc[-1]
                    ema_s_curr = df['EMA_Slow'].iloc[-1]
                    close_curr = df['Close'].iloc[-1]
                    ema_s_slope = (df['EMA_Slow'].iloc[-1] - df['EMA_Slow'].iloc[-5]) / df['EMA_Slow'].iloc[-5] * 100
                    
                    trend_score = 0
                    if close_curr > ema_s_curr: trend_score += 40
                    if ema_f_curr > ema_s_curr: trend_score += 40
                    if ema_s_slope > 0: trend_score += 20
                    
                    if hit_rate > 0:
                        consensus = (hit_rate * 0.6) + (trend_score * 0.4)
                    else:
                        consensus = trend_score
                except:
                    consensus = 0

                # Fetch Sector
                sector = 'N/A'
                if isinstance(all_profiles, dict) and ticker in all_profiles:
                    p_data = all_profiles[ticker]
                    if isinstance(p_data, dict):
                        sector = p_data.get('sector') or p_data.get('sectorDisp') or 'N/A'
                
                if sector == 'N/A':
                    sector = get_stock_info(ticker)

                # --- 1. Advanced Candlestick Analysis (The BCP Fix) ---
                c_open = df_raw['Open'].iloc[-1]
                c_high = df_raw['High'].iloc[-1]
                c_low = df_raw['Low'].iloc[-1]
                c_close = df_raw['Close'].iloc[-1]
                body_size = abs(c_close - c_open)
                upper_wick = c_high - max(c_open, c_close)
                lower_wick = min(c_open, c_close) - c_low
                
                # Refined Detection: 
                # - REJECTION WICK: Long upper wick at high price levels (Potential Reversal)
                # - PIN BAR: Long lower wick at low price levels (Potential Support)
                is_rejection_wick = upper_wick > (body_size * 1.5) and (c_high > last_price)
                is_pin_bar = lower_wick > (body_size * 1.5) and (c_close > c_open) # Bullish Pin Bar
                
                # --- 2. Score Velocity (The Momentum Trend) ---
                prev_bull_score = 0
                if len(df_raw) > 2:
                    # Briefly re-run scanner for yesterday (Simplified)
                    df_prev = df_raw.iloc[:-1]
                    bull_prev = get_pre_breakout_scanner(df_prev, mode='bullish')
                    if bull_prev and bull_prev['matches']:
                        prev_bull_score = max(0, 100 - (bull_prev['matches'][0]['dist'] * 20))
                
                score_velocity = bull_score - prev_bull_score if prev_bull_score > 0 else 0
                is_fading_momentum = score_velocity < -5.0 # Score dropping fast
                
                # --- Signal Logic V7 (The Multi-Variable Guard) ---
                score_diff = bull_score - bear_score
                m_regime, _ = get_market_regime()
                
                # Logic Insights:
                # 1. BCP Case: Rejection at resistance (EMA50) with long wick.
                # 2. Score Trend: Even if score is high, if it's falling (Velocity < 0), be careful.
                # 3. Conflict: Bull and Bear scores both high (>70) = Indecision.
                
                rsi_curr = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                ema_fast = df['EMA_Fast'].iloc[-1]
                ema_slow = df['EMA_Slow'].iloc[-1]
                
                # Dynamic Thresholds
                is_pre_fly = 12.0 <= score_diff <= 22.0
                is_caution = score_diff > 25.0
                is_momentum_strong = rel_vol > 1.5
                # SILENT ACCUM: Price + (Volume Compression OR Tight Rel Vol) + Low ATC Risk
                is_silent_accum = (pct_change > 0 and (is_vol_compressed or 0.8 <= rel_vol <= 1.2) and atc_risk < 0.5)
                is_rr_good = bull_jump > abs(bear_jump)
                
                # --- NEW: Multi-Timeframe Confirmation (MTF) ---
                mtf_status = "N/A"
                mtf_score = 0
                # Only check MTF for Live Scan (target_date is None) to save time and data
                # OPTIMIZATION: Only fetch MTF if Bullish Score is decent (> 40)
                if target_date is None and bull_score > 40:
                    mtf_status, mtf_score = get_mtf_confluence(ticker)
                
                # Refined Filters (The BCP & CPF Balance)
                is_blow_off = (rsi_curr > 72 and rel_vol > 1.8 and pct_change > 4.5) or (rsi_curr > 82)
                is_conflict_zone = (bull_score > 70 and bear_score > 70)
                is_bearish_trap = (bear_score > bull_score) or (bear_score > 80.0)
                is_overextended = (last_price > ema_fast * 1.06) 
                is_downtrend = last_price < ema_slow # Below EMA50
                
                signal = 'WAIT'
                
                # Signal Assignment with Priority
                if is_rejection_wick:
                    signal = 'REJECTION WICK'
                elif is_pin_bar and is_downtrend:
                    signal = 'PIN BAR (SUPPORT)'
                elif is_fading_momentum and score_diff < 15:
                    signal = 'FADING MOMENTUM'
                elif is_blow_off or is_overextended:
                    signal = 'TAKE PROFIT / WAIT'
                elif is_bearish_trap and score_diff < 5:
                    signal = 'WAIT (BEARISH TRAP)'
                elif is_conflict_zone:
                    signal = 'CONFLICT (HIGH RISK)'
                elif pct_change > 0:
                    if is_pre_fly and is_momentum_strong and is_rr_good:
                        signal = 'PRE-FLY'
                    elif is_silent_accum and score_diff > 5:
                        signal = 'SILENT ACCUM'
                    elif is_caution:
                        signal = 'CAUTION'
                    elif 2.0 <= score_diff <= 15.0 and is_momentum_strong and is_rr_good:
                        if is_downtrend:
                            signal = 'BOTTOM PLAY'
                        else:
                            signal = 'GOLDEN BUY'
                    elif bull_score > 80 and score_diff > 15 and is_momentum_strong:
                        signal = 'BUY'
                    elif is_downtrend:
                        signal = 'WAIT (DOWNTREND)'
                elif is_downtrend:
                    signal = 'WAIT (DOWNTREND)'

                # Outcome Calculation (Historical Only)
                outcome = "N/A"
                max_dd = 0.0
                outcome_data = {}
                if target_date and not df_future.empty:
                    # Calculate return and max drawdown over next 3 days
                    next_3d = df_future.iloc[:3]
                    if not next_3d.empty:
                        # NEW: Use Final Close (Conservative) and Max Potential
                        final_ret = (next_3d['Close'].iloc[-1] / last_price - 1) * 100
                        max_high_ret = (next_3d['High'].max() / last_price - 1) * 100
                        max_dd = (next_3d['Low'].min() / last_price - 1) * 100
                        outcome = f"{final_ret:+.1f}% ({max_high_ret:+.1f}%)"
                        
                        # Prepare labeling data for immediate save
                        outcome_data['outcome_label'] = "Win" if max_high_ret >= 2.0 else "Loss"
                        outcome_data['outcome_pct'] = max_high_ret
                        outcome_data['verified_date'] = datetime.now(SET_TZ).strftime("%Y-%m-%d")

                if not df_raw.empty:
                    # If live scan (target_date is None), use current time. If historical, use candle date.
                    if target_date:
                        last_update_val = df_raw.index[-1].strftime("%Y-%m-%d %H:%M") if hasattr(df_raw.index[-1], 'strftime') else str(df_raw.index[-1])
                        db_date = df_raw.index[-1].strftime("%Y-%m-%d")
                        db_time = "00:00:00"
                    else:
                        now_th = datetime.now(SET_TZ)
                        last_update_val = now_th.strftime("%Y-%m-%d %H:%M")
                        db_date = now_th.strftime("%Y-%m-%d")
                        db_time = now_th.strftime("%H:%M:%S")

                    # --- NEW: Recovery Signal Check (Bottom Fishing) ---
                    recovery_data = get_recovery_signals(ticker, df)
                    # Note: strategy will be updated later in the post-processing loop

                    # Capture basic info for SRS calculation
                    results.append({
                        'Ticker': ticker,
                        'Pattern Consensus (%)': consensus,
                        'Sector': sector,
                        'Last Price': last_price,
                        'Day High': day_high,
                        'ATR': atr_now,
                        '% Change': pct_change,
                        'Relative Vol': rel_vol,
                        'MTF Conf': mtf_status,
                        'MTF Score': mtf_score,
                        'ATC Risk (%)': atc_risk,
                        'Bullish Score (%)': bull_score,
                        'Expected Jump (%)': bull_jump,
                        'Bearish Score (%)': bear_score,
                        'Expected Drop (%)': bear_jump,
                        'Score Diff': score_diff,
                        'Signal': signal,
                        'Strategy': 'SWING', # Temporary
                        'Vol Alert': '⚠️ High Vol' if high_vol else 'Normal',
                        'Outcome (3D)': outcome,
                        'Max DD (3D)': f"{max_dd:+.1f}%",
                        'RSI': rsi_curr,
                        'Score Velocity': score_velocity,
                        'm_regime': m_regime,
                        'perf_stats': perf_stats,
                        'db_date': db_date,
                        'db_time': db_time,
                        'outcome_data': outcome_data,
                        'last_update_val': last_update_val,
                        'recovery_data': recovery_data
                    })
        except Exception as e:
            continue
        progress_bar.progress((i + 1) / len(tickers))
    
    status_text.text("📊 Calculating Sector Relative Strength (SRS)...")
    if results:
        df_results = pd.DataFrame(results)
        
        # 1. Calculate Sector Averages
        sector_avgs = df_results.groupby('Sector')['% Change'].mean().to_dict()
        
        final_results = []
        for _, row in df_results.iterrows():
            ticker = row['Ticker']
            s_avg = sector_avgs.get(row['Sector'], 0)
            srs_val = row['% Change'] - s_avg
            
            # 2. Re-calculate Conviction Score with SRS
            conviction_score, strategy, reasons, warnings = calculate_conviction_score(
                ticker, row['Signal'], row['Pattern Consensus (%)'], 
                row['MTF Score'], row['m_regime'], row['perf_stats'], 
                row['RSI'], row['Relative Vol'], row['Score Velocity'],
                sector_rs=srs_val, price_change=row['% Change']
            )
            
            # Update recovery data with final strategy
            rec_data = row['recovery_data']
            if rec_data:
                rec_data['actual_strategy'] = strategy
            
            # 3. Final entry for the report
            entry = {
                'Ticker': ticker,
                'Pattern Consensus (%)': row['Pattern Consensus (%)'],
                'Sector': row['Sector'],
                'Last Price': row['Last Price'],
                'Day High': row['Day High'],
                '% Change': row['% Change'],
                'Sector_RS': srs_val, # NEW
                'Relative Vol': row['Relative Vol'],
                'MTF Conf': row['MTF Conf'],
                'MTF Score': row['MTF Score'],
                'ATC Risk (%)': row['ATC Risk (%)'],
                'Bullish Score (%)': row['Bullish Score (%)'],
                'Expected Jump (%)': row['Expected Jump (%)'],
                'Bearish Score (%)': row['Bearish Score (%)'],
                'Expected Drop (%)': row['Expected Drop (%)'],
                'Score Diff': row['Score Diff'],
                'Signal': row['Signal'],
                'Strategy': strategy,
                'Conviction_Score': conviction_score,
                'Why': reasons,
                'Warnings': warnings,
                'Vol Alert': row['Vol Alert'],
                'Outcome (3D)': row['Outcome (3D)'],
                'Max DD (3D)': row['Max DD (3D)'],
                'Last Update': row['last_update_val'],
                'RSI': row['RSI'],
                'Score Velocity': row['Score Velocity'],
                'Recovery_Data': rec_data
            }
            final_results.append(entry)
            
            # 4. Save to Database
            db_data = {
                'ticker': ticker,
                'date': row['db_date'],
                'time': row['db_time'],
                'price': row['Last Price'],
                'bull_score': row['Bullish Score (%)'],
                'bear_score': row['Bearish Score (%)'],
                'score_diff': row['Score Diff'],
                'signal_type': row['Signal'],
                'market_regime': row['m_regime'],
                'rel_vol': row['Relative Vol'],
                'rsi': row['RSI'],
                'mtf_status': row['MTF Conf'],
                'mtf_score': row['MTF Score'],
                'conviction_score': conviction_score,
                'sector_rs': srs_val # Optional: add to DB if schema supports
            }
            db_data.update(row['outcome_data'])
            save_scan_result(db_data)
        
        results = final_results

    status_text.text("Scan Complete!")
    
    # NEW: Save analysis snapshot for performance tracking (only for live scans)
    if target_date is None and results:
        m_regime, _ = get_market_regime()
        save_analysis_snapshot(pd.DataFrame(results), m_regime)
        
    return pd.DataFrame(results), pos_count, neg_count

# --- Main App ---
# 1. Background Auto-Labeling (Update the brain before scanning)
if 'auto_labeled' not in st.session_state:
    with st.spinner("🧠 Updating Brain (Auto-Labeling)..."):
        try:
            # Run existing labeling
            count_orig = run_automated_labeling()
            # Run NEW automated performance validation
            count_new = validate_performance()
            
            total_updated = count_orig + (count_new if count_new else 0)
            if total_updated > 0:
                st.toast(f"✅ AI Brain Updated: {total_updated} results verified!", icon="🧠")
            st.session_state['auto_labeled'] = True
        except:
            pass

# 2. Market Regime Header ---
regime, set_price = get_market_regime()
regime_color = "lime" if regime == "BULL" else ("red" if regime == "BEAR" else "gray")
st.sidebar.markdown(f"""
### 📊 Market Regime: :{regime_color}[{regime}]
- **SET Index:** {set_price:,.2f}
- **Status:** {'ตลาดเป็นใจ (Buy on Dip)' if regime == 'BULL' else 'ระวังตัว (Cash is King)'}
""", unsafe_allow_html=True)

# API Key at the top for global use
user_api_key = st.sidebar.text_input("🔑 Google API Key", type="password", help="Needed for AI Trading Plan and Optimization")
st.session_state['api_key'] = user_api_key

# --- Sidebar: SET100 Multi-Scanner ---
st.sidebar.divider()
st.sidebar.header("🔍 SET100 Multi-Scanner")

# --- NEW: Dynamic Ticker Management ---
with st.sidebar.expander("⚙️ Ticker Management", expanded=False):
    st.write("จัดการรายชื่อหุ้นในดัชนี SET100")
    if st.button("🔄 Update SET100 List", use_container_width=True):
        with st.spinner("Fetching latest tickers..."):
            # Try to fetch or at least refresh sector info for new tickers
            # For SET100, we can use a simpler approach: allow user to input/edit
            # or use a reliable source if available.
            # Here we will trigger a refresh of sectors for current tickers
            updated_sectors = SET100_SECTORS.copy()
            for t_code in SET100_TICKERS:
                if t_code not in updated_sectors or updated_sectors[t_code] == 'N/A':
                    sector = get_stock_info(t_code)
                    if sector != 'N/A':
                        updated_sectors[t_code] = sector
            
            if save_ticker_config(SET100_TICKERS, updated_sectors):
                st.success("Config updated and saved!")
                st.rerun()
    
    st.info(f"Current Tickers: {len(SET100_TICKERS)}")
    # Allow manual entry for missing tickers
    new_tickers_raw = st.text_area("Edit Tickers (Comma separated)", value=", ".join(SET100_TICKERS), height=150)
    if st.button("💾 Save Manual Changes"):
        new_list = [t.strip().upper() for t in new_tickers_raw.split(",") if t.strip()]
        # Ensure .BK suffix
        new_list = [t if ".BK" in t else f"{t}.BK" for t in new_list]
        
        # Auto-fetch sectors for new ones
        updated_sectors = SET100_SECTORS.copy()
        with st.spinner("Fetching sectors for new tickers..."):
            for t_code in new_list:
                if t_code not in updated_sectors:
                    updated_sectors[t_code] = get_stock_info(t_code)
        
        if save_ticker_config(new_list, updated_sectors):
            st.success("Tickers list updated!")
            st.rerun()

run_set100 = st.sidebar.button("🚀 Run SET100 Batch Scan", use_container_width=True)

# --- NEW: Historical Multi-Scan Backtest ---
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Historical Batch Backtest")
backtest_date = st.sidebar.date_input("Select Historical Date", datetime.now(SET_TZ) - timedelta(days=5))
run_historical = st.sidebar.button("📊 Run Historical Scan", use_container_width=True)

# Persistent storage for scan results to prevent re-scanning on UI interaction
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = None

if run_set100 or run_historical:
    st.session_state['batch_results'] = None  # Clear old results
    st.header("🏆 SET100 Scanner Leaderboard")
    
    t_date = backtest_date if run_historical else None
    msg = f"Historical Scan for {t_date}" if t_date else "Searching for stocks with High Bullish Match and Low Danger Zone..."
    st.info(msg)
    
    batch_df, pos_count, neg_count = run_set100_batch_scan(SET100_TICKERS, target_date=t_date)
    st.session_state['batch_results'] = {
        'df': batch_df,
        'pos': pos_count,
        'neg': neg_count,
        'is_historical': True if t_date else False
    }
    st.rerun() # Refresh to clean up scanning status and display results from state

if st.session_state['batch_results'] is not None:
    res = st.session_state['batch_results']
    batch_df = res['df']
    pos_count = res['pos']
    neg_count = res['neg']
    is_hist = res.get('is_historical', False)
    
    if not batch_df.empty:
        # --- GLOBAL CSS FOR COMPACT CARDS ---
        st.markdown(textwrap.dedent("""
            <style>
            .compact-card {
                background-color: #ffffff;
                padding: 12px;
                border-radius: 12px;
                border: 1px solid #f0f0f0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                font-family: 'Inter', sans-serif;
                margin-bottom: 12px;
            }
            .card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 6px;
            }
            .header-left {
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .dot-indicator {
                height: 8px;
                width: 8px;
                border-radius: 50%;
            }
            .ticker-name {
                font-size: 1.1rem;
                font-weight: 800;
                color: #000 !important;
            }
            .status-pill {
                padding: 2px 8px;
                background-color: #f8f9fa;
                color: #6c757d !important;
                border-radius: 12px;
                font-size: 0.6rem;
                font-weight: 600;
            }
            .score-container {
                display: flex;
                align-items: baseline;
                gap: 8px;
                margin: 6px 0;
            }
            .score-label {
                font-size: 0.65rem;
                color: #888 !important;
            }
            .score-big {
                font-size: 1.5rem;
                font-weight: 900;
                color: #000 !important;
            }
            .signal-badge {
                display: inline-block;
                padding: 3px 10px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.75rem;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 8px;
                margin-top: 10px;
                padding-top: 8px;
                border-top: 1px solid #f0f0f0;
            }
            .stat-item {
                text-align: center;
            }
            .stat-lbl {
                font-size: 0.55rem;
                color: #999 !important;
            }
            .stat-val {
                font-size: 0.8rem;
                font-weight: 700;
                color: #333 !important;
            }
            </style>
        """), unsafe_allow_html=True)

        # --- SIGNALS BY CATEGORY ---
        if 'Signal' in batch_df.columns:
            st.subheader("📊 Signals by Category")
            st.caption("🌐 **Sector Relative Strength (SRS):** วิเคราะห์เปรียบเทียบหุ้นกับค่าเฉลี่ยของกลุ่มอุตสาหกรรม เพื่อหาหุ้นที่ 'แข็งแกร่งกว่าตลาด' (Outperformer)")
            sig_counts = batch_df['Signal'].value_counts().reset_index()
            sig_counts.columns = ['Signal', 'Count']
            
            # Use small columns for a compact overview
            num_cols = min(len(sig_counts), 6)
            s_cols = st.columns(num_cols)
            for i, (_, s_row) in enumerate(sig_counts.iterrows()):
                s_cols[i % num_cols].metric(s_row['Signal'], s_row['Count'])
            
            # --- SILENT ACCUM CLUSTER DETECTION ---
            sa_count = sig_counts[sig_counts['Signal'] == 'SILENT ACCUM']['Count'].values[0] if 'SILENT ACCUM' in sig_counts['Signal'].values else 0
            if sa_count >= 3:
                st.info(f"🔵 **Smart Money Accumulation Cluster Detected!**  \nพบหุ้น SET100 ติดสัญญาณ `SILENT ACCUM` พร้อมกัน **{sa_count} ตัว**  \n*แนวโน้ม: ตลาดมีโอกาสเกิด Reversal ขาขึ้นในระยะสั้น (Confidence: High)*")
                st.caption("💡 **Feature Insight:** ระบบตรวจพบการเก็บของพร้อมกันในหลายตัว (Cluster) ซึ่งเป็นสัญญาณบ่งชี้ Market Breadth ว่าเงินทุนกำลังไหลเข้าสะสมหุ้นในกลุ่ม SET100")
        
        st.divider()

        # --- MAIN UI TABS ---
        main_tabs = st.tabs(["🚀 Execution Center", "🔬 Quant Research Lab", "📜 Admin & History"])
        
        with main_tabs[0]: # Execution Center
            report_tabs = st.tabs(["🚀 Unified Report", "💎 Bottom Fishing", "📊 Market Breadth"])
            
            with report_tabs[0]: # Unified Report
                if not is_hist:
                    st.info("สรุปผลการวิเคราะห์เชิงปริมาณ (Search + Analyze + Persistence + Backtest)")
                    st.caption("🔍 **ระบบคัดกรองอัจฉริยะ:** รวม 3 กลยุทธ์ใหม่ (1) **Volume Compression** ตรวจจับวอลุ่มแห้งก่อนระเบิด (2) **Sector Flow Filter** คัดเฉพาะหุ้นที่แข็งแกร่งกว่ากลุ่ม (SRS) และ (3) **Dynamic Stop Loss** ปรับตามความผันผวนจริง (ATR)")
                    unified_df = generate_unified_report(batch_df, regime)
                    
                    if not unified_df.empty:
                        # Sorting: High Conviction first, then positive signals
                        high_strength = unified_df[unified_df['Conviction_Score'] >= 40].copy()
                        pos_signals = ['BUY', 'GOLDEN BUY', 'PRE-FLY', 'PIN BAR (SUPPORT)', 'SILENT ACCUM']
                        early_birds = unified_df[(unified_df['Conviction_Score'] < 40) & (unified_df['Signal'].isin(pos_signals))].copy()
                        top_conviction = pd.concat([high_strength, early_birds]).head(20)
                        
                        if not top_conviction.empty:
                            st.success(f"🔥 พบหุ้นน่าสนใจ {len(top_conviction)} ตัว (จัดลำดับตามคะแนนและความมั่นใจ)")
                            
                            for idx, (i, row) in enumerate(top_conviction.iterrows()):
                                is_early_bird = row['Conviction_Score'] < 40
                                
                                # Dot & Label Logic
                                if is_early_bird:
                                    dot_color = "#10b981" # Emerald
                                    s_label = "EARLY ENTRY"
                                else:
                                    dot_color = "#3b82f6" if "SWING" in row['Strategy'] else "#f59e0b"
                                    s_label = row['Strategy']

                                # Signal Styling
                                sig_val = row['Signal']
                                sig_bg = "#f3f4f6"; sig_fg = "#4b5563"; sig_border = "none"
                                if sig_val in ['BUY', 'GOLDEN BUY', 'PRE-FLY']: sig_bg = "#dcfce7"; sig_fg = "#166534"
                                elif sig_val == 'REJECTION WICK': sig_bg = "#111827"; sig_fg = "#ffffff"
                                elif sig_val == 'SILENT ACCUM': sig_bg = "#ecfdf5"; sig_fg = "#065f46"
                                elif sig_val == 'CONFLICT (HIGH RISK)': sig_bg = "#fee2e2"; sig_fg = "#991b1b"
                                elif sig_val == 'PIN BAR (SUPPORT)': sig_bg = "#dcfce7"; sig_fg = "#166534"; sig_border = "1px solid #166534"
                                
                                # Card Content
                                intraday_html = ""
                                if 'Intraday_History' in row and row['Intraday_History']:
                                    past_sigs = ", ".join(row['Intraday_History'])
                                    intraday_html = f'<div class="intraday-alert" style="font-size: 0.65rem; color: #f59e0b; margin-bottom: 4px;">⚡ <b>Intraday:</b> {past_sigs}</div>'

                                # Build card HTML
                                srs_val = row.get('Sector_RS', 0)
                                srs_color = "#166534" if srs_val > 0 else ("#991b1b" if srs_val < 0 else "#4b5563")
                                stop_loss = row.get('Stop_Loss', 0)
                                
                                card_html = f"""
                                <div class="compact-card">
                                    <div class="card-header">
                                        <div class="header-left">
                                            <div class="dot-indicator" style="background-color: {dot_color};"></div>
                                            <div class="ticker-name">{row['Ticker']}</div>
                                        </div>
                                        <div class="status-pill">{s_label}</div>
                                    </div>
                                    {intraday_html}
                                    <div class="score-container">
                                        <div class="score-label">Score</div>
                                        <div class="score-big">{row['Conviction_Score']}</div>
                                    </div>
                                    <div class="signal-badge" style="background-color: {sig_bg}; color: {sig_fg}; border: {sig_border};">{sig_val}</div>
                                    <div class="stats-grid">
                                        <div class="stat-item">
                                            <div class="stat-lbl">SECTOR RS</div>
                                            <div class="stat-val" style="color: {srs_color}; font-weight: 700;">{srs_val:+.1f}%</div>
                                        </div>
                                        <div class="stat-item">
                                            <div class="stat-lbl">STOP LOSS</div>
                                            <div class="stat-val" style="color: #991b1b;">{stop_loss:.2f}</div>
                                        </div>
                                        <div class="stat-item">
                                            <div class="stat-lbl">MTF CONF</div>
                                            <div class="stat-val">{row['MTF_Score']}</div>
                                        </div>
                                    </div>
                                </div>
                                """
                                st.markdown(card_html, unsafe_allow_html=True)
                                
                                with st.expander(f"Details: {row['Ticker']}", expanded=False):
                                    st.write(f"✅ {row['Why']}")
                                    if row['Warnings']: st.warning(row['Warnings'])
                                    if user_api_key:
                                        if st.button(f"AI Plan: {row['Ticker']}", key=f"tab_unified_btn_{row['Ticker']}"):
                                            st.markdown(generate_ai_trading_plan(row['Ticker'], batch_df[batch_df['Ticker']==row['Ticker']].iloc[0], user_api_key))
                        
                        with st.expander("🔍 View All Unified Candidates", expanded=False):
                            st.dataframe(unified_df, use_container_width=True)
                    else:
                        st.info("ℹ️ ไม่พบหุ้นที่เข้าเกณฑ์ Unified")
            
            with report_tabs[1]: # Bottom Fishing
                st.info("หุ้นที่ Oversold และเริ่มมีสัญญาณกลับตัว (Bottom Fishing)")
                recovery_list = [row['Recovery_Data'] for _, row in batch_df.iterrows() if 'Recovery_Data' in row and row['Recovery_Data'] is not None]
                
                if recovery_list:
                    recovery_df = pd.DataFrame(recovery_list).sort_values('recovery_score', ascending=False)
                    st.success(f"🎯 พบหุ้นที่มีโอกาสกลับตัว {len(recovery_df)} ตัว")
                    
                    for idx, r_row in recovery_df.iterrows():
                        r_dot_color = "#8b5cf6" 
                        r_sig_bg = "#f5f3ff"; r_sig_fg = "#5b21b6"
                        reasons_html = "".join([f'<div style="font-size: 0.75rem; color: #5b21b6; margin-bottom: 2px;">• {reason}</div>' for reason in r_row['reasons']])
                        
                        # Compact Card for Recovery
                        act_sig = r_row.get('actual_signal', 'N/A')
                        act_strat = r_row.get('actual_strategy', 'N/A')
                        
                        r_card_html = f"""<div class="compact-card"><div class="card-header"><div class="header-left"><div class="dot-indicator" style="background-color: {r_dot_color};"></div><div class="ticker-name">{r_row['ticker']}</div></div><div class="status-pill recovery">RECOVERY</div></div><div class="score-container"><div class="score-label">Score</div><div class="score-big">{r_row['recovery_score']}</div></div><div style="display: flex; flex-direction: column; gap: 4px;"><div class="signal-badge" style="background-color: {r_sig_bg}; color: {r_sig_fg};">OVERSOLD RECOVERY</div><div style="font-size: 0.75rem; font-weight: 600; color: #7c3aed;">Signal: {act_sig}</div><div style="font-size: 0.75rem; font-weight: 600; color: #4b5563;">Strategy: {act_strat}</div></div><div style="margin-top: 10px; padding: 6px; background-color: #fdfcff; border-radius: 8px; border: 1px dashed #ddd6fe;">{reasons_html}</div><div class="stats-grid"><div class="stat-item"><div class="stat-lbl">RSI</div><div class="stat-val">{r_row['rsi']:.1f}</div></div><div class="stat-item"><div class="stat-lbl">PRICE</div><div class="stat-val">{r_row['price']:.2f}</div></div><div class="stat-item"><div class="stat-lbl">PIN BAR</div><div class="stat-val">{'✅' if r_row['is_pin'] else '❌'}</div></div></div></div>"""
                        st.markdown(r_card_html, unsafe_allow_html=True)
                        
                        with st.expander(f"Recovery Analysis: {r_row['ticker']}"):
                            st.write(f"🔍 **ทำไมถึงติดโผ:** {', '.join(r_row['reasons'])}")
                            if user_api_key:
                                if st.button(f"Recovery AI Plan: {r_row['ticker']}", key=f"tab_recovery_btn_{r_row['ticker']}"):
                                    dummy_row = {'Last Price': r_row['price'], 'Signal': 'RECOVERY (OVERSOLD)', 'Bullish Score (%)': r_row['recovery_score'], 'Bearish Score (%)': 0, 'Score Diff': r_row['recovery_score'], 'MTF Conf': 'N/A', 'MTF Score': 0, 'Relative Vol': 1.0, 'Pattern Consensus (%)': 50}
                                    st.markdown(generate_ai_trading_plan(r_row['ticker'], dummy_row, user_api_key))
                else:
                    st.info("ℹ️ ยังไม่พบหุ้น Oversold")
            
            with report_tabs[2]: # Market Breadth
                st.subheader(f"📊 Market Breadth: หุ้นบวก {pos_count} | หุ้นลบ {neg_count}")
                if 'Signal' in batch_df.columns:
                    sig_counts = batch_df['Signal'].value_counts().reset_index()
                    st.dataframe(sig_counts, use_container_width=True)

        with main_tabs[1]: # Research Lab
            lab_tabs = st.tabs(["🎯 Hit Rate Accuracy", "🤖 AI Insights", "🔍 Missed Opportunities", "📈 Performance Summary", "💎 SILENT ACCUM Insight"])
            
            if supabase:
                try:
                    # Get labeled data from Supabase
                    response = supabase.table("scan_results") \
                        .select("*") \
                        .not_.is_("outcome_label", "null") \
                        .order("id", desc=True) \
                        .limit(200) \
                        .execute()
                    labeled_df = pd.DataFrame(response.data)
                except Exception as e:
                    st.error(f"Error fetching research data: {e}")
                    labeled_df = pd.DataFrame()

                with lab_tabs[0]: # Hit Rate Accuracy
                    st.info("🎯 การวัดผลความแม่นยำแยกตามประเภทของสัญญาณ (Signal Type)")
                    if not labeled_df.empty:
                        # Calculate Win Rate per Signal Type
                        sig_perf = labeled_df.groupby('signal_type').agg(
                            Total_Signals=('outcome_label', 'count'),
                            Wins=('outcome_label', lambda x: (x == 'Win').sum())
                        )
                        sig_perf['Win Rate (%)'] = (sig_perf['Wins'] / sig_perf['Total_Signals']) * 100
                        sig_perf = sig_perf.sort_values('Win Rate (%)', ascending=False)
                        
                        perf_col1, perf_col2 = st.columns([2, 1])
                        fig_perf = go.Figure(go.Bar(
                            x=sig_perf.index, y=sig_perf['Win Rate (%)'],
                            text=sig_perf['Win Rate (%)'].apply(lambda x: f"{x:.1f}%"),
                            textposition='auto',
                            marker=dict(color=sig_perf['Win Rate (%)'], colorscale='Greens', showscale=False)
                        ))
                        fig_perf.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), yaxis_title="Win Rate (%)")
                        perf_col1.plotly_chart(fig_perf, use_container_width=True)
                        perf_col2.write("**Detailed Stats**")
                        perf_col2.dataframe(sig_perf.style.format({'Win Rate (%)': '{:.1f}%'}), use_container_width=True)
                    else:
                        st.warning("ยังไม่มีข้อมูลที่ Label แล้ว กรุณารอ 3 วันทำการหลังการสแกน")
                
                with lab_tabs[1]: # AI Insights
                    st.info("🤖 วิเคราะห์ว่าตัวแปรไหนมีผลต่อการ 'Win' มากที่สุด")
                    if len(labeled_df) >= 10:
                        try:
                            features = ['bull_score', 'bear_score', 'score_diff', 'relative_vol', 'rsi']
                            X = labeled_df[features].fillna(0)
                            y = labeled_df['outcome_label'].apply(lambda x: 1 if x == 'Win' else 0)
                            rf = RandomForestClassifier(n_estimators=100, random_state=42)
                            rf.fit(X, y)
                            importance_df = pd.DataFrame({'Feature': features, 'Importance (%)': rf.feature_importances_ * 100}).sort_values('Importance (%)', ascending=False)
                            
                            fi_col1, fi_col2 = st.columns([2, 1])
                            fig_fi = go.Figure(go.Bar(x=importance_df['Importance (%)'], y=importance_df['Feature'], orientation='h', marker=dict(color='royalblue')))
                            fig_fi.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20), xaxis_title="Importance (%)")
                            fi_col1.plotly_chart(fig_fi, use_container_width=True)
                            
                            win_only = labeled_df[labeled_df['outcome_label'] == 'Win']
                            top_feature = importance_df.iloc[0]['Feature']
                            avg_val = win_only[top_feature].mean()
                            st.session_state['ai_insights_summary'] = f"ตัวแปรที่แม่นที่สุดคือ {top_feature} โดยหุ้นที่ Win ส่วนใหญ่มีค่าเฉลี่ยอยู่ที่ {avg_val:.2f}"
                            fi_col2.info(f"**Key Takeaway:**\nตัวแปรที่แม่นที่สุดคือ **{top_feature}**\nหุ้นที่ Win ส่วนใหญ่มีค่าเฉลี่ย {top_feature} อยู่ที่ **{avg_val:.2f}**")
                        except Exception as e:
                            st.error(f"Error in AI Analysis: {e}")
                    else:
                        st.warning(f"ต้องการข้อมูลที่ Label แล้วอย่างน้อย 10 รายการ (ปัจจุบันมี {len(labeled_df)})")
                
                with lab_tabs[2]: # Missed Opportunities
                    st.info("🔍 วิเคราะห์หุ้นที่ 'Win' แต่ระบบไม่ได้แนะนำเป็น High Conviction (คะแนน < 60)")
                    if not labeled_df.empty:
                        missed_df = labeled_df[(labeled_df['outcome_label'] == 'Win') & (labeled_df['conviction_score'] < 60)].copy()
                        if not missed_df.empty:
                            st.warning(f"พบหุ้น 'ม้ามืด' {len(missed_df)} รายการ")
                            avg_rsi_missed = missed_df['rsi'].mean()
                            avg_vol_missed = missed_df['relative_vol'].mean()
                            common_signal = missed_df['signal_type'].mode().iloc[0] if not missed_df['signal_type'].empty else "N/A"
                            ma1, ma2, ma3 = st.columns(3)
                            ma1.metric("Avg RSI of Missed", f"{avg_rsi_missed:.1f}")
                            ma2.metric("Avg Vol of Missed", f"{avg_vol_missed:.1f}x")
                            ma3.metric("Common Signal", common_signal)
                            st.dataframe(missed_df[['ticker', 'scan_date', 'signal_type', 'conviction_score', 'outcome_pct', 'rsi', 'relative_vol']], use_container_width=True)
                        else:
                            st.success("✨ ยังไม่พบหุ้นม้ามืดที่หลุดรอดไป")
                    else:
                        st.info("ยังไม่มีข้อมูลการวัดผล")
                
                with lab_tabs[3]: # Performance Summary
                    st.info("📈 สรุปความแม่นยำของสูตรการสแกน T+3 Performance")
                    try:
                        response = supabase.table("trading_log") \
                            .select("*") \
                            .neq("status", "Pending") \
                            .execute()
                        verified_logs = pd.DataFrame(response.data)
                        
                        if not verified_logs.empty:
                            accuracy = (len(verified_logs[verified_logs['status'] == 'Success']) / len(verified_logs)) * 100
                            p1, p2, p3 = st.columns(3)
                            p1.metric("Overall Accuracy", f"{accuracy:.1f}%")
                            p2.metric("Avg Outcome (T+3)", f"{verified_logs['outcome_t3_pct'].mean():+.2f}%")
                            p3.metric("Avg Max Drawdown", f"{verified_logs['max_dd_pct'].mean():+.2f}%")
                            
                            session_stats = verified_logs.groupby('session_flag').agg(Total=('status', 'count'), Wins=('status', lambda x: (x == 'Success').sum()))
                            session_stats['Accuracy (%)'] = (session_stats['Wins'] / session_stats['Total']) * 100
                            fig_session = go.Figure(go.Bar(x=session_stats.index, y=session_stats['Accuracy (%)'], text=session_stats['Accuracy (%)'].apply(lambda x: f"{x:.1f}%"), textposition='auto', marker=dict(color=session_stats['Accuracy (%)'], colorscale='Viridis')))
                            fig_session.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
                            st.plotly_chart(fig_session, use_container_width=True)
                        else:
                            st.info("กำลังรอรวบรวมข้อมูล T+3")
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                with lab_tabs[4]: # SILENT ACCUM Insight
                    st.info("💎 เจาะลึกพฤติกรรมหุ้น SILENT ACCUM: วัดระยะเวลาการฟื้นตัวและโอกาสชนะ")
                    st.caption("📈 **Feature Insight:** วิเคราะห์สถิติย้อนหลังของสัญญาณ SILENT ACCUM เพื่อหาค่าเฉลี่ยจำนวนวันที่ราคามักจะ 'ระเบิด' (Days to Move) และอัตราการชนะ (Win Rate) ภายใน 5 วัน")
                    sa_data = get_silent_accum_insights(limit=100)
                    
                    if sa_data is not None and not sa_data.empty:
                        # 1. Overview Metrics
                        avg_days = sa_data['days_to_move'].mean()
                        win_rate_t5 = (sa_data['win_t5'].sum() / len(sa_data)) * 100
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Avg. Days to Move", f"{avg_days:.1f} Days")
                        m2.metric("Win Rate (T+5)", f"{win_rate_t5:.1f}%")
                        m3.metric("Sample Size", f"{len(sa_data)} Signals")
                        
                        # 2. Distribution Chart
                        st.write("### 📊 Distribution of Days to Move (+1% Upside)")
                        dist_df = sa_data['days_to_move'].value_counts().sort_index().reset_index()
                        dist_df.columns = ['Days', 'Frequency']
                        
                        fig_sa = go.Figure(go.Bar(
                            x=dist_df['Days'], y=dist_df['Frequency'],
                            text=dist_df['Frequency'], textposition='auto',
                            marker=dict(color='#10b981')
                        ))
                        fig_sa.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), xaxis_title="Trading Days", yaxis_title="Number of Cases")
                        st.plotly_chart(fig_sa, use_container_width=True)
                        
                        # 3. Recent Cases
                        st.write("### 📜 Recent SILENT ACCUM Cases")
                        st.dataframe(sa_data[['ticker', 'signal_date', 'days_to_move', 'max_gain_t5']].head(10).style.format({'max_gain_t5': '{:.2f}%'}), use_container_width=True)
                    else:
                        st.warning("ยังไม่มีข้อมูล SILENT ACCUM เพียงพอสำหรับการวิเคราะห์ (ต้องการข้อมูลในฐานข้อมูลอย่างน้อย 1 รายการ)")

        with main_tabs[2]: # Admin & History
            st.subheader("🏆 Leaderboard & History")
            # Sorting desired columns to the front
            cols = batch_df.columns.tolist()
            desired_order = ["Ticker", "Signal", "Pattern Consensus (%)", "Last Price", "Day High"]
            actual_order = [c for c in desired_order if c in batch_df.columns]
            batch_df = batch_df[actual_order + [c for c in cols if c not in actual_order]]
            
            with st.expander("🔍 View Scanner Leaderboard (Table)", expanded=True):
                st.dataframe(batch_df, use_container_width=True)
            
            st.divider()
            st.subheader("📜 Database & Labeling")
            
            l1, l2 = st.columns([1, 2])
            if l1.button("🏷️ Run Automated Labeling", use_container_width=True):
                with st.spinner("Updating labels..."):
                    count = run_automated_labeling()
                    st.success(f"Updated {count} records!") if count > 0 else st.info("No new records to label.")

            with st.expander("View Saved Scan History", expanded=False):
                if supabase:
                    try:
                        response = supabase.table("scan_results") \
                            .select("*") \
                            .order("id", desc=True) \
                            .limit(500) \
                            .execute()
                        history_df = pd.DataFrame(response.data)
                        
                        if not history_df.empty:
                            def style_outcome(val):
                                if val == 'Win': return 'color: green; font-weight: bold'
                                if val == 'Loss': return 'color: red'
                                return ''
                            st.dataframe(history_df.style.map(style_outcome, subset=['outcome_label']), use_container_width=True)
                        
                        st.write("### 📊 Quick Insights from DB")
                        c1, c2, c3 = st.columns(3)
                        if not labeled_df.empty:
                            win_rate = (len(labeled_df[labeled_df['outcome_label'] == 'Win']) / len(labeled_df)) * 100
                            c2.metric("Win Rate (Labeled)", f"{win_rate:.1f}%")
                        
                        unique_tickers = history_df['ticker'].unique()
                        selected_h_ticker = c3.selectbox("Select Ticker for Score Trend", unique_tickers, key="hist_ticker_select")
                        if selected_h_ticker:
                            ticker_history = history_df[history_df['ticker'] == selected_h_ticker].sort_values('id')
                            c3.line_chart(ticker_history.set_index('scan_date')['bull_score'])
                    except Exception as e:
                        st.error(f"Error loading history: {e}")
            
            st.divider()
            st.subheader("📤 Export & Printing Tools")
            ex1, ex2, ex3 = st.columns(3)
            
            # Formatting and Styling for Export
            def style_batch(styler):
                def highlight_best(row):
                    styles = [''] * len(row)
                    if 'BUY' in str(row['Signal']): styles = ['background-color: rgba(34, 197, 94, 0.2)'] * len(row)
                    elif row['Bearish Score (%)'] > 85: styles = ['background-color: rgba(239, 68, 68, 0.2)'] * len(row)
                    return styles
                styler.apply(highlight_best, axis=1)
                styler.map(lambda x: 'color: lime; font-weight: bold' if 'BUY' in str(x) else ('color: red; font-weight: bold' if x == 'SELL' else 'color: gray'), subset=['Signal'])
                styler.format({'Last Price': '{:.2f}', '% Change': '{:+.2f}%', 'Relative Vol': '{:.2f}x', 'MTF Score': '{:.0f}', 'Pattern Consensus (%)': '{:.1f}%', 'Bullish Score (%)': '{:.1f}%', 'Bearish Score (%)': '{:.1f}%', 'Score Diff': '{:.1f}'})
                return styler

            styled_export = style_batch(batch_df.style)
            html_buffer = styled_export.to_html()
            
            ex1.checkbox("📸 Full-Length View (For PDF)", key="admin_full_view")
            
            full_html = f"<html><body><h2>🏆 SET100 Quant Report</h2>{html_buffer}</body></html>"
            ex2.download_button("📄 Download HTML Report", data=full_html, file_name=f"SET100_Report_{datetime.now(SET_TZ).strftime('%Y%m%d_%H%M')}.html", mime="text/html", use_container_width=True)
            
            csv = batch_df.to_csv(index=False).encode('utf-8-sig')
            ex3.download_button("Excel/CSV Export", data=csv, file_name=f"SET100_Data_{datetime.now(SET_TZ).strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", use_container_width=True)
                
# --- Sidebar: Strategy Builder ---
st.sidebar.header("🛠 Quant Strategy Builder")

# Initialize AI Optimization values in session state
if 'ai_params' not in st.session_state:
    st.session_state['ai_params'] = None

if 'stock_list' not in st.session_state:
    st.session_state['stock_list'] = ["KTC.BK", "AMATA.BK", "RCL.BK", "CPF.BK"]

with st.sidebar.expander("📥 Data Management", expanded=True):
    new_ticker = st.text_input("➕ Add Ticker", "").upper()
    if new_ticker:
        if not new_ticker.endswith(".BK") and "." not in new_ticker: new_ticker += ".BK"
        if new_ticker not in st.session_state['stock_list']:
            st.session_state['stock_list'].append(new_ticker)
    selected_ticker = st.selectbox("Select Stock", st.session_state['stock_list'])
    fetch_btn = st.button("🚀 Fetch Data", type="primary", use_container_width=True)

if fetch_btn:
    with st.spinner("Downloading..."):
        df_raw = get_stock_data(selected_ticker)
        if df_raw is not None:
            st.session_state['df_raw'] = df_raw
            st.session_state['active_ticker'] = selected_ticker
        else:
            st.error("Failed to load data.")

st.sidebar.divider()

if 'df_raw' in st.session_state:
    if st.sidebar.button("🤖 AI Strategy Optimizer", use_container_width=True):
        if not user_api_key:
            st.sidebar.warning("Please enter Google API Key above.")
        else:
            # We need current stats and trade log for AI. 
            current_df = calculate_quant_indicators(st.session_state['df_raw'], 14, 5, 20)
            _, current_stats, _ = run_backtest(current_df, 30, 70, 1.2)
            
            with st.sidebar.status("AI is analyzing...", expanded=True) as status:
                ai_rec = get_ai_optimization(
                    st.session_state['active_ticker'], 
                    current_stats, 
                    None, 
                    {"rsi_p": 14, "rsi_b": 30, "rsi_s": 70, "ema_f": 5, "ema_s": 20, "rv_m": 1.2},
                    user_api_key
                )
            if ai_rec:
                st.session_state['ai_params'] = ai_rec
                status.update(label="✅ AI Optimization Complete!", state="complete")
                st.sidebar.info(f"💡 AI Recommendation: {ai_rec['reasoning']}")
            else:
                status.update(label="❌ AI Optimization Failed", state="error")

with st.sidebar.form("strategy_params"):
    st.subheader("⚙️ Buy/Sell Parameters")
    
    # Use AI values if available, else use defaults
    p = st.session_state['ai_params'] if st.session_state['ai_params'] else {}
    
    rsi_p = st.slider("RSI Period", 5, 30, p.get('rsi_p', 14))
    rsi_b = st.slider("Buy Threshold (RSI <=)", 10, 80, p.get('rsi_b', 50))
    rsi_s = st.slider("Sell Threshold (RSI >=)", 40, 90, p.get('rsi_s', 70))
    ema_f = st.number_input("Fast EMA", 5, 50, p.get('ema_f', 10))
    ema_s = st.number_input("Slow EMA", 10, 200, p.get('ema_s', 50))
    rv_m = st.slider("Min Rel. Volume", 1.0, 3.0, p.get('rv_m', 1.5), 0.1)
    
    st.divider()
    st.subheader("🔍 Scanner Settings")
    min_sim = st.slider("Min Similarity Threshold (%)", 50, 95, 80)
    scanner_mode = st.radio("Scanner Mode", ["Bullish (Breakout)", "Bearish (Danger Zone)"], horizontal=True)
    
    apply_btn = st.form_submit_button("🔄 Apply & Backtest", use_container_width=True)

# --- Main Dashboard ---
if 'df_raw' in st.session_state:
    df = calculate_quant_indicators(st.session_state['df_raw'], rsi_p, ema_f, ema_s)
    trade_log, stats, equity_curve = run_backtest(df, rsi_b, rsi_s, rv_m)
    
    st.title(f"📈 {st.session_state['active_ticker']} Strategy Station")
    
    # Volatility Guard Alert
    atr_now = df['ATR'].iloc[-1]
    atr_avg5 = df['ATR_Avg_5'].iloc[-1]
    if atr_now > (atr_avg5 * 1.2):
        st.warning(f"⚠️ **High Volatility Alert**: ATR ({atr_now:.2f}) is 20%+ above 5-day average ({atr_avg5:.2f}). Exercise caution!")
    
    # 1. Performance Metrics
    if stats and 'Total Return (%)' in stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{stats['Total Return (%)']:.2f}%")
        m2.metric("Win Rate", f"{stats['Win Rate (%)']:.1f}%")
        m3.metric("Max Drawdown", f"-{stats['Max Drawdown (%)']:.2f}%", delta_color="inverse")
        m4.metric("Total Trades", stats['Total Trades'])
    elif stats and 'total_crosses' in stats:
        st.info("No trades executed with current parameters. See 'Strategy Breakdown' below for details.")
    
    # 2. Main Chart (Candlestick + Signals + Projection)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Candlesticks
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name=f"EMA {ema_f}", line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name=f"EMA {ema_s}", line=dict(color='blue', width=1)), row=1, col=1)
    
    # Buy/Sell Arrows
    if trade_log is not None:
        fig.add_trace(go.Scatter(x=trade_log['Entry Date'], y=trade_log['Entry Price'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='Buy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=trade_log['Exit Date'], y=trade_log['Exit Price'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=rsi_s, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=rsi_b, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. DTW Projection Chart
    st.subheader("🔮 Pattern Matching Projection (Next 20 Days)")
    projections = get_dtw_projection(df)
    if projections:
        proj_fig = go.Figure()
        # Last 40 days of actual price
        last_40 = df['Close'].iloc[-40:]
        proj_fig.add_trace(go.Scatter(x=list(range(-40, 0)), y=last_40.values, name="Actual Price", line=dict(color='black', width=3)))
        
        colors = ['rgba(34, 197, 94, 0.6)', 'rgba(59, 130, 246, 0.6)', 'rgba(168, 85, 247, 0.6)']
        for i, p in enumerate(projections):
            proj_fig.add_trace(go.Scatter(x=list(range(0, 20)), y=p['path'], name=f"Match {i+1} ({p['date_range']})", line=dict(dash='dot', color=colors[i])))
            
        proj_fig.update_layout(height=400, template="plotly_white", xaxis_title="Days from Today", yaxis_title="Price")
        st.plotly_chart(proj_fig, use_container_width=True)
    
    # 4. Trade Log & Equity Curve
    c_log, c_eq = st.columns([0.6, 0.4])
    with c_log:
        st.subheader("📜 Detailed Trade Log")
        if trade_log is not None:
            st.dataframe(trade_log.style.format({'Profit (%)': '{:.2f}%', 'Entry Price': '{:.2f}', 'Exit Price': '{:.2f}'}), use_container_width=True)
        else:
            st.info("No trades executed with current parameters.")
            # Show why no trades happened
            if stats and 'total_crosses' in stats: # case when run_backtest returned (None, debug_info, None)
                d_info = stats
            elif isinstance(stats, dict) and 'debug' in stats: # case when it returned (log, stats, curve) but we check here just in case
                d_info = stats['debug']
            else:
                d_info = None
            
            if d_info:
                st.warning(f"""
                🔎 **Strategy Breakdown:**
                - Total EMA Crosses found: {d_info['total_crosses']}
                - Met RSI <= {rsi_b}: {d_info['rsi_met']}
                - Met Volume >= {rv_m}: {d_info['vol_met']}
                
                *Tip: Try increasing 'Buy Threshold' or decreasing 'Min Rel. Volume' to see more trades.*
                """)
            
    with c_eq:
        st.subheader("📈 Equity Growth")
        if equity_curve is not None:
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(x=equity_curve['Trade'], y=equity_curve['Equity'], fill='tozeroy', line=dict(color='green')))
            eq_fig.update_layout(height=350, template="plotly_white", title="Capital: 100k Base")
            st.plotly_chart(eq_fig, use_container_width=True)

    # 5. Pattern Scanner
    st.divider()
    mode_label = "🚀 Pre-Breakout Pattern Scanner" if scanner_mode == "Bullish (Breakout)" else "🚩 Danger Zone Scanner"
    st.subheader(f"{mode_label} (Last 5 Days vs History)")
    
    scan_mode_val = 'bullish' if scanner_mode == "Bullish (Breakout)" else 'bearish'
    with st.spinner(f"Scanning for historical patterns..."):
        scan_results = get_pre_breakout_scanner(df, mode=scan_mode_val)
        
    if scan_results:
        # 3 Columns for charts: Price, Volume, Candlestick
        scan_fig = make_subplots(
            rows=1, cols=3, 
            subplot_titles=("Price Pattern Similarity", "Volume Flow Similarity", "Candlestick Comparison"),
            column_widths=[0.3, 0.3, 0.4]
        )
        
        # Color configuration based on mode
        main_color = 'green' if scan_mode_val == 'bullish' else 'red'
        jump_color = 'lime' if scan_mode_val == 'bullish' else 'crimson'
        
        # Current Patterns (5 points)
        scan_fig.add_trace(go.Scatter(x=list(range(5)), y=scan_results['curr_p'], name="Current Price", line=dict(color='black', width=3)), row=1, col=1)
        scan_fig.add_trace(go.Scatter(x=list(range(5)), y=scan_results['curr_v'], name="Current Volume", line=dict(color='gray', width=3, dash='dash')), row=1, col=2)
        
        # Historical Best Match
        best = scan_results['matches'][0]
        # 1. Historical Pattern (Day 0-4)
        scan_fig.add_trace(go.Scatter(
            x=list(range(5)), 
            y=best['hist_p'], 
            name=f"Best Match History ({pd.to_datetime(best['date']).date()})", 
            line=dict(color=main_color, dash='dot')
        ), row=1, col=1)
        
        # 2. Breakout/Drop Projection (Day 4 to 5) - Highlighted
        jump_label = f"{best['jump']:+.1f}%"
        scan_fig.add_trace(go.Scatter(
            x=[4, 5], 
            y=[best['hist_p_ext'][4], best['hist_p_ext'][5]], 
            name="Historical Move",
            mode='lines+markers+text',
            text=["", jump_label],
            textposition="top center",
            line=dict(color=jump_color, width=5),
            marker=dict(size=8, color=jump_color)
        ), row=1, col=1)
        
        scan_fig.add_trace(go.Scatter(x=list(range(5)), y=best['hist_v'], name=f"Best Match Volume", line=dict(color='blue', dash='dot')), row=1, col=2)
        
        # 3. Candlestick Comparison (Col 3)
        # Current Candlesticks
        curr_ohlc = scan_results['curr_ohlc']
        scan_fig.add_trace(go.Candlestick(
            x=list(range(5)),
            open=curr_ohlc['Open'],
            high=curr_ohlc['High'],
            low=curr_ohlc['Low'],
            close=curr_ohlc['Close'],
            name="Current Candles"
        ), row=1, col=3)
        
        # Historical Best Match Candlesticks (Scaled and Transparent)
        hist_ohlc = best['hist_ohlc_scaled']
        scan_fig.add_trace(go.Candlestick(
            x=list(range(6)),
            open=hist_ohlc['Open'],
            high=hist_ohlc['High'],
            low=hist_ohlc['Low'],
            close=hist_ohlc['Close'],
            name="Historical Match Candles",
            increasing_line_color='rgba(34, 197, 94, 0.3)',
            decreasing_line_color='rgba(239, 68, 68, 0.3)',
            increasing_fillcolor='rgba(34, 197, 94, 0.1)',
            decreasing_fillcolor='rgba(239, 68, 68, 0.1)'
        ), row=1, col=3)
        
        scan_fig.update_layout(height=450, template="plotly_white", showlegend=True, xaxis3_rangeslider_visible=False)
        st.plotly_chart(scan_fig, use_container_width=True)
        
        # Matching Summary Column (below charts for more space)
        st.write("### 📊 Matching Summary")
        m_cols = st.columns(3)
        for i, m in enumerate(scan_results['matches']):
            similarity = max(0, 100 - (m['dist'] * 20)) 
            sim_color = "green" if similarity > 80 else "orange"
            jump_val_color = "green" if m['jump'] > 0 else "red"
            with m_cols[i]:
                st.markdown(f"""
                **Match #{i+1}: {pd.to_datetime(m['date']).date()}**
                - Similarity Score: :{sim_color}[{similarity:.1f}%]
                - Historical Move: :{jump_val_color}[{m['jump']:+.1f}%]
                """)
        
        if scan_results['matches'][0]['dist'] < 1.0:
            msg = "🔥 HIGH CONVICTION: Pattern highly resembles a historical breakout!" if scan_mode_val == 'bullish' else "⚠️ WARNING: Pattern highly resembles a historical CRASH!"
            if scan_mode_val == 'bullish':
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.info("ℹ️ Pattern is being monitored. No high-conviction match yet.")
        
        # 6. Scanner Accuracy Validator
        st.divider()
        st.subheader("✅ Scanner Accuracy Validator")
        st.write(f"Test how often this {min_sim}%+ Similarity pattern actually leads to a {'price jump' if scan_mode_val == 'bullish' else 'price drop'} in the past.")
        
        if st.button("🔍 Validate Scanner Accuracy", use_container_width=True):
            # Extract patterns based on current mode
            df_temp = df.copy()
            df_temp['Pct_Change'] = df_temp['Close'].pct_change()
            
            if scan_mode_val == 'bullish':
                events = df_temp[df_temp['Pct_Change'] >= 0.05].index.tolist()
            else:
                events = df_temp[df_temp['Pct_Change'] <= -0.05].index.tolist()
                
            winning_pats = []
            for b_date in events:
                idx = df_temp.index.get_loc(b_date)
                if idx < 5: continue
                p_data = df_temp.iloc[idx-5 : idx]
                p_norm = StandardScaler().fit_transform(p_data[['Close']]).flatten()
                v_norm = StandardScaler().fit_transform(p_data[['Volume']]).flatten()
                winning_pats.append({'price_pattern': p_norm, 'vol_pattern': v_norm})

            with st.spinner("Validating historical signals..."):
                val_results = validate_scanner_accuracy(df, winning_pats, price_threshold=min_sim, vol_threshold=70.0)
            
            if val_results:
                s = val_results['summary']
                
                # Big Metrics Row 1
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Signals", s['Total Signals'])
                m2.metric("Hit Rate (%)", f"{s['Hit Rate']:.1f}%")
                m3.metric("Expectancy", f"{s['Expectancy']:.2f}%", help="Expected profit per trade")
                
                # Big Metrics Row 2
                m4, m5, m6 = st.columns(3)
                m4.metric("Avg Profit", f"+{s['Avg Profit']:.2f}%")
                m5.metric("Avg Loss", f"{s['Avg Loss']:.2f}%")
                m6.metric("Risk/Reward Ratio", f"{s['RR Ratio']:.2f}", help="Avg Profit / Avg Loss")
                
                st.write("#### 📜 Historical Accuracy Log")
                
                # Formatting function for the table
                def style_log(row):
                    styles = [''] * len(row)
                    # Highlight red if loss is worse than -3%
                    if row['Next Day Return (%)'] < -3:
                        styles = ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
                    return styles

                # Apply styling
                styled_df = val_results['log'].style.apply(style_log, axis=1).map(
                    lambda x: 'color: green' if x == '✅ Hit' else 'color: red', 
                    subset=['Result']
                ).format({
                    'Overall Sim': '{:.1f}%',
                    'Price Sim': '{:.1f}%', 
                    'Vol Sim': '{:.1f}%', 
                    'Next Day Return (%)': '{:.2f}%'
                })
                
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info(f"Not enough signals found with Price Similarity > {min_sim}% and Vol Similarity > 70% in history.")
    else:
        st.info("No historical breakout events (>5%) found for this stock to compare.")

else:
    st.info("👋 Welcome! Please select a stock and click '🚀 Fetch Data' in the sidebar to start your analysis.")

