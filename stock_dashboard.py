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
import time
import pickle
import hashlib
from pathlib import Path
import google.generativeai as genai
import json
import pytz
import textwrap
import backtest_engine
import signal_engine
import paper_trading
from supabase import create_client, Client
from dotenv import load_dotenv
from scanner_engine import (
    calculate_quant_indicators, calculate_wvf, get_pre_breakout_scanner, 
    get_recovery_signals, core_strategy_scanner, calculate_conviction_score, 
    get_market_regime as get_engine_market_regime, validate_scanner_accuracy, 
    get_mtf_confluence, get_signal_performance_stats
)

# Load .env for local development
load_dotenv()

# --- Configuration ---
SET_TZ = pytz.timezone('Asia/Bangkok')
CACHE_DIR = Path(".cache/stock_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# User-Agent list for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'
]

def get_disk_cache(ticker):
    """Load data from disk if it exists and is not too old (1 hour)."""
    try:
        cache_file = CACHE_DIR / f"{ticker.replace('^', '_')}.pkl"
        if cache_file.exists():
            if (time.time() - cache_file.stat().st_mtime) < 3600:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
    except:
        pass
    return None

def save_disk_cache(ticker, df):
    """Save data to disk."""
    try:
        cache_file = CACHE_DIR / f"{ticker.replace('^', '_')}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
    except:
        pass
st.set_page_config(page_title="Quant Strategy Station", layout="wide")

# --- Utility Functions ---
def safe_float(val, default=0.0):
    """Safely convert value to float, handling None, NaN, and non-numeric strings."""
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

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

def load_best_params():
    """Load optimized parameters from Supabase or local fallback."""
    params_dict = {}
    
    # 1. Try Supabase
    if supabase:
        try:
            resp = supabase.table("strategy_params").select("*").execute()
            if resp.data:
                for row in resp.data:
                    params_dict[row['ticker']] = json.loads(row['params_json'])
                return params_dict
        except:
            pass # Table might not exist
            
    # 2. Try Local Fallback
    try:
        if os.path.exists("best_params.json"):
            with open("best_params.json", "r") as f:
                return json.load(f)
    except:
        pass
        
    return params_dict

def save_best_params(ticker, params):
    """Save optimized parameters to Supabase and local fallback."""
    # 1. Save Local
    try:
        current_params = load_best_params()
        current_params[ticker] = params
        with open("best_params.json", "w") as f:
            json.dump(current_params, f, indent=4)
    except Exception as e:
        st.error(f"Error saving params locally: {e}")
        
    # 2. Save Supabase
    if supabase:
        try:
            # Upsert logic
            payload = {
                "ticker": ticker,
                "params_json": json.dumps(params),
                "updated_at": datetime.now(SET_TZ).isoformat()
            }
            supabase.table("strategy_params").upsert(payload, on_conflict="ticker").execute()
        except Exception as e:
            # Table might not exist, skip silently
            pass

# --- Initialization & Authentication Flow ---
def check_password():
    """Returns True if the user has the correct password with Institutional Grade UI."""
    
    # 1. Persistent Authentication via Session State or Query Params (for Mobile Refresh)
    if st.session_state.get("authenticated", False):
        return True
        
    if st.query_params.get("auth") == "true":
        st.session_state["authenticated"] = True
        return True

    # 2. Inject Custom CSS for Font, UI Styling, and Mobile Pull-to-Refresh Disable
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&family=Sarabun:wght@300;400;600&display=swap');
        
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: 'Prompt', 'Sarabun', sans-serif !important;
            overscroll-behavior-y: none !important;
            touch-action: pan-x pan-y;
        }
        
        .login-card {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 2.5rem;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            max-width: 500px;
            margin: 2rem auto;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }
        
        .login-header {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .login-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }
        
        input {
            font-family: 'Prompt', sans-serif !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 3. Login UI with st.form for Mobile UX Stability
    with st.container():
        st.markdown("""
            <div class="login-card">
                <div class="login-header">🔒 Stock AI Quantitative Terminal</div>
                <div class="login-subtitle">ระบบวิเคราะห์หุ้นและคัดกรองสัญญาณ Pre-Breakout / DTW Confluence</div>
            </div>
        """, unsafe_allow_html=True)
        
        _, col, _ = st.columns([1, 2, 1])
        with col:
            with st.form("login_form", clear_on_submit=False):
                st.subheader("🔒 Authentication Required")
                user_password = st.text_input("Please enter the access password", type="password", placeholder="Enter password to unlock...")
                submit_button = st.form_submit_button("เข้าสู่ระบบ (Unlock Terminal)", use_container_width=True, type="primary")

                if submit_button:
                    # Fetch and Clean Secret Password
                    target_pass = str(st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", "admin1234"))).strip()
                    entered_pass = str(user_password).strip()

                    if entered_pass == target_pass:
                        st.session_state["authenticated"] = True
                        st.query_params["auth"] = "true" # Set flag for refresh persistence
                        st.rerun()
                    else:
                        st.error("😞 Password incorrect / รหัสผ่านไม่ถูกต้อง")

    return False

# 1. Block main app rendering until authenticated
if not check_password():
    st.stop()

# 2. System Initialization (After login)
with st.spinner("Initializing system, please wait..."):
    # Initialize Database Connection
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

def get_silent_accum_insights(limit=100, ticker_filter=None, deduplicate=True):
    """
    Analyze historical SILENT ACCUM signals from both manual and auto scan results.
    Strictly follows First-Signal-Wins logic per ticker per day.
    """
    if not supabase:
        return None
        
    try:
        # Calculate start date for 90 days history
        now_th = datetime.now(SET_TZ)
        start_dt = (now_th - timedelta(days=90)).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date_str = start_dt.strftime('%Y-%m-%d')
        
        # Initialize empty dataframes
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        
        # 1. Fetch from scan_results (Manual)
        query1 = supabase.table("scan_results") \
            .select("ticker, scan_date, scan_time, price, signal_type, bull_score") \
            .eq("signal_type", "SILENT ACCUM") \
            .gte("scan_date", start_date_str)
            
        if ticker_filter:
            query1 = query1.eq("ticker", ticker_filter)
            
        res1 = query1.execute()
        
        if res1.data:
            df1 = pd.DataFrame(res1.data)
            # Standardize for Concatenation
            df1['source'] = 'manual'
            df1['scan_type'] = 'MANUAL_SCAN'
            # Normalize timestamp to Asia/Bangkok
            df1['full_timestamp'] = pd.to_datetime(df1['scan_date'].astype(str) + ' ' + df1['scan_time'].fillna('00:00:00').astype(str))
            df1['full_timestamp'] = df1['full_timestamp'].dt.tz_localize(None)
            df1 = df1.rename(columns={'scan_date': 'signal_date_raw', 'signal_type': 'signal', 'bull_score': 'score', 'scan_time': 'signal_time'})
            df1['signal_date'] = df1['full_timestamp'].dt.date
        
        # 2. Fetch from auto_scan_results (Auto)
        query2 = supabase.table("auto_scan_results") \
            .select("ticker, scanned_at, close_price, signal, strategy, is_silent_accum, score, scan_type") \
            .eq("signal", "SILENT ACCUM") \
            .gte("scanned_at", start_dt.isoformat())
            
        if ticker_filter:
            query2 = query2.eq("ticker", ticker_filter)
            
        res2 = query2.execute()
            
        if res2.data:
            df2 = pd.DataFrame(res2.data)
            if not df2.empty:
                # Extra safety filter
                df2 = df2[df2['signal'].fillna('').str.upper() == 'SILENT ACCUM'].copy()
                if not df2.empty:
                    df2['source'] = 'auto'
                    # Normalize timestamp to Asia/Bangkok
                    df2['full_timestamp'] = pd.to_datetime(df2['scanned_at'], utc=True).dt.tz_convert(SET_TZ).dt.tz_localize(None)
                    df2['signal_date'] = df2['full_timestamp'].dt.date
                    df2['signal_time'] = df2['full_timestamp'].dt.strftime('%H:%M:%S')
                    df2 = df2.rename(columns={'close_price': 'price'})
                    if 'scan_type' not in df2.columns:
                        df2['scan_type'] = 'AUTO_SCAN'
        
        # 3. Correct Union & Deduplication Logic
        combined = pd.concat([df1, df2], ignore_index=True)
        if combined.empty:
            return None
            
        # Ensure data types are consistent for deduplication
        combined['ticker'] = combined['ticker'].astype(str).str.strip().str.upper()
        combined['full_timestamp'] = pd.to_datetime(combined['full_timestamp'], errors='coerce')
        combined['signal_date'] = pd.to_datetime(combined['signal_date']).dt.date
        
        if deduplicate:
            # [STRICT IMMUTABLE LOG] Sort by timestamp ASCENDING and keep FIRST of day per ticker
            combined = combined.sort_values(by='full_timestamp', ascending=True)
            combined = combined.drop_duplicates(subset=['ticker', 'signal_date'], keep='first')
        
        # 4. Sorting Final Output for Display
        combined = combined.sort_values(by=['signal_date', 'full_timestamp'], ascending=[False, False])
        
        # Apply row limit only for overview mode
        if limit and not ticker_filter:
            signals = combined.head(limit)
        else:
            signals = combined
            
        # 5. Performance Calculation Loop
        unique_tickers = signals['ticker'].unique()
        ticker_data_cache = {}
        
        results = []
        for _, sig in signals.iterrows():
            ticker = sig['ticker']
            signal_date = pd.to_datetime(sig['signal_date']).date()
            entry_price = sig['price']
            
            # Fetch stock data (cached per ticker)
            if ticker not in ticker_data_cache:
                ticker_data_cache[ticker] = get_stock_data(ticker)
                
            df = ticker_data_cache[ticker]
            
            # Initialize default result row
            res_row = {
                'ticker': ticker,
                'signal_date': sig['signal_date'],
                'signal_time': sig.get('signal_time', '00:00:00'),
                'scan_type': sig.get('scan_type', 'N/A'),
                'full_timestamp': sig['full_timestamp'],
                'score': sig['score'],
                'days_to_move': None,
                'max_gain_t5': None,
                'win_t5': 0
            }
            
            if df is not None and not df.empty:
                # Normalize index for date comparison
                df_norm = df.copy()
                df_norm.index = pd.to_datetime(df_norm.index).tz_localize(None).normalize().date
                
                # Get future price data starting from signal date
                future_df = df_norm[df_norm.index >= signal_date].copy()
                
                if len(future_df) > 1:
                    # test_df starts from T+1
                    test_df = future_df.iloc[1:11] # Up to T+10
                    
                    # 1. Calculate Days to Move (+1% Upside)
                    found_move = False
                    for day_idx, (idx, row) in enumerate(test_df.iterrows()):
                        max_ret = (row['High'] / entry_price - 1) * 100
                        if max_ret >= 1.0:
                            res_row['days_to_move'] = day_idx + 1
                            found_move = True
                            break
                    
                    # 2. Calculate Max Gain and Win within T+5
                    if len(test_df) > 0:
                        t5_window = test_df.head(5)
                        max_gain = (t5_window['High'].max() / entry_price - 1) * 100
                        res_row['max_gain_t5'] = max_gain
                        res_row['win_t5'] = 1 if max_gain >= 1.0 else 0
            
            results.append(res_row)
        
        final_df = pd.DataFrame(results)
        if not final_df.empty:
            final_df = final_df.sort_values(by=['signal_date', 'full_timestamp'], ascending=[False, False])
            final_df = final_df.reset_index(drop=True)
            
        return final_df
    except Exception as e:
        st.error(f"Error in SILENT ACCUM analysis: {e}")
        import traceback
        print(traceback.format_exc())
        return None
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
            # STRICT RULE: Remove 'id' from entries to allow auto-increment PK
            for entry in log_entries:
                entry.pop('id', None)
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

def check_recent_signal_exists(ticker, signal_type, window_minutes=15):
    """
    Check if a signal for this ticker has been recorded in the last X minutes.
    Used for debouncing auto/manual scan inserts.
    """
    if not supabase:
        return False
    try:
        now_utc = datetime.now(pytz.utc)
        threshold = (now_utc - timedelta(minutes=window_minutes)).isoformat()
        
        res = supabase.table("auto_scan_results") \
            .select("id") \
            .eq("ticker", ticker) \
            .eq("signal", signal_type) \
            .gte("scanned_at", threshold) \
            .limit(1) \
            .execute()
            
        return len(res.data) > 0
    except Exception as e:
        print(f"Error checking recent signal: {e}")
        return False

def save_scan_result(data):
    """Save a single scan result to Supabase, supporting optional labeling."""
    if not supabase:
        return False
        
    try:
        # Sanitized Data: Convert to Native Python Types and handle NaN
        def clean_val(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            if isinstance(v, (np.integer, np.floating)):
                return v.item()
            if isinstance(v, np.bool_):
                return bool(v)
            return v

        payload = {
            "ticker": clean_val(data.get('ticker')),
            "scan_date": clean_val(data.get('date')),
            "scan_time": clean_val(data.get('time')),
            "price": clean_val(data.get('price')),
            "bull_score": clean_val(data.get('bull_score')),
            "bear_score": clean_val(data.get('bear_score')),
            "score_diff": clean_val(data.get('score_diff')),
            "signal_type": clean_val(data.get('signal_type')),
            "market_regime": clean_val(data.get('market_regime')),
            "relative_vol": clean_val(data.get('rel_vol')),
            "rsi": clean_val(data.get('rsi'))
        }
        
        # Optional columns
        if 'mtf_status' in data: payload['mtf_status'] = clean_val(data['mtf_status'])
        if 'mtf_score' in data: payload['mtf_score'] = clean_val(data['mtf_score'])
        if 'conviction_score' in data: payload['conviction_score'] = clean_val(data['conviction_score'])
        if 'outcome_label' in data: payload['outcome_label'] = clean_val(data['outcome_label'])
        if 'outcome_pct' in data: payload['outcome_pct'] = clean_val(data['outcome_pct'])
        if 'verified_date' in data: payload['verified_date'] = clean_val(data['verified_date'])
        
        # STRICT RULE: Remove 'id' to allow Supabase auto-increment PK
        payload.pop('id', None)
        
        # Debug Logging: Print 1st payload sample to console
        if not hasattr(save_scan_result, "_logged_sample"):
            print(f"DEBUG: Sample Payload for Supabase: {json.dumps(payload, indent=2, default=str)}")
            save_scan_result._logged_sample = True
            
        res = supabase.table("scan_results").insert(payload).execute()
        
        # Check for errors in response
        if hasattr(res, 'error') and res.error:
            st.error(f"❌ บันทึกลง Supabase ล้มเหลว ({payload['ticker']}): {res.error}")
            return False
            
        # --- NEW: Manual Scan Integration for SILENT ACCUM ---
        # If this is a SILENT ACCUM signal, also record to auto_scan_results for intraday tracking
        if payload['signal_type'] == 'SILENT ACCUM':
            # 15-Minute Debounce Check
            if not check_recent_signal_exists(payload['ticker'], 'SILENT ACCUM', window_minutes=15):
                auto_payload = {
                    'ticker': payload['ticker'],
                    'scanned_at': datetime.now(SET_TZ).isoformat(), # Use Bangkok Time
                    'score': payload.get('conviction_score', payload.get('bull_score')),
                    'signal': 'SILENT ACCUM',
                    'strategy': data.get('strategy', 'ACCUMULATION'),
                    'sector': data.get('sector', 'N/A'),
                    'close_price': payload['price'],
                    'change_percent': data.get('change_percent'),
                    'volume': data.get('volume'),
                    'rsi': payload['rsi'],
                    'is_recovery': data.get('is_recovery', False),
                    'is_pinbar': data.get('is_pinbar', False),
                    'is_silent_accum': True,
                    'scan_type': 'MANUAL_SCAN'
                }
                # Clean and Insert
                cleaned_auto = {k: clean_val(v) for k, v in auto_payload.items()}
                supabase.table("auto_scan_results").insert(cleaned_auto).execute()
            
        return True
    except Exception as e:
        st.error(f"❌ บันทึกลง Supabase ล้มเหลว ({data.get('ticker', 'Unknown')}): {e}")
        return False

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

def fetch_latest_scan_results():
    """Retrieve the most recent batch scan results from either manual or auto tables."""
    if not supabase:
        return None, 0, 0
    try:
        # 1. Get latest from manual scan_results
        latest_manual = supabase.table("scan_results") \
            .select("scan_date, scan_time") \
            .order("scan_date", desc=True) \
            .order("scan_time", desc=True) \
            .limit(1) \
            .execute()
            
        manual_dt = None
        if latest_manual.data:
            m = latest_manual.data[0]
            try:
                manual_dt = datetime.strptime(f"{m['scan_date']} {m['scan_time']}", "%Y-%m-%d %H:%M:%S")
                manual_dt = SET_TZ.localize(manual_dt)
            except:
                manual_dt = None

        # 2. Get latest from auto_scan_results
        latest_auto = supabase.table("auto_scan_results") \
            .select("scanned_at") \
            .order("scanned_at", desc=True) \
            .limit(1) \
            .execute()
            
        auto_dt = None
        if latest_auto.data:
            auto_dt = pd.to_datetime(latest_auto.data[0]['scanned_at']).tz_convert(SET_TZ)

        # 3. Decide which one to load (prefer newest)
        use_auto = False
        if auto_dt and manual_dt:
            use_auto = auto_dt > manual_dt
        elif auto_dt:
            use_auto = True
            
        if use_auto:
            # Fetch from auto_scan_results
            response = supabase.table("auto_scan_results") \
                .select("*") \
                .eq("scanned_at", latest_auto.data[0]['scanned_at']) \
                .execute()
            
            if not response.data: return None, 0, 0
            
            df = pd.DataFrame(response.data)
            df = df.rename(columns={
                'ticker': 'Ticker',
                'close_price': 'Last Price',
                'score': 'Bullish Score (%)',
                'signal': 'Signal',
                'strategy': 'Strategy',
                'rsi': 'RSI',
                'change_percent': '% Change',
                'rel_vol': 'Relative Vol' # Corrected mapping
            })
            
            # If 'rel_vol' is missing (older schema), set to N/A instead of mapping absolute volume
            if 'Relative Vol' not in df.columns:
                df['Relative Vol'] = 1.0
            
            # Map Sector
            df['Sector'] = df['Ticker'].map(SET100_SECTORS)
            
            # Fill missing columns for UI compatibility
            df['Bearish Score (%)'] = 0
            df['Score Diff'] = df['Bullish Score (%)']
            df['MTF Score'] = 0
            df['MTF Conf'] = 'N/A'
            df['ATC Risk (%)'] = 0
            df['Expected Jump (%)'] = 0
            df['Expected Drop (%)'] = 0
            df['Outcome (3D)'] = 'N/A'
            df['Max DD (3D)'] = '0.0%'
            df['Last Update'] = auto_dt.strftime("%Y-%m-%d %H:%M")
            df['Sector_RS'] = 0
            df['Conviction_Score'] = df['Bullish Score (%)']
            df['Why'] = 'Latest Auto Scan'
            df['Warnings'] = ''
            df['Vol Alert'] = 'Normal'
            df['Score Velocity'] = 0
            
            pos_count = len(df[df['% Change'] > 0])
            neg_count = len(df[df['% Change'] < 0])
            return df, pos_count, neg_count
        else:
            # Fetch from manual scan_results
            if not latest_manual.data: return None, 0, 0
            
            date = latest_manual.data[0]['scan_date']
            time = latest_manual.data[0]['scan_time']
            
            response = supabase.table("scan_results") \
                .select("*") \
                .eq("scan_date", date) \
                .eq("scan_time", time) \
                .execute()
                
            if not response.data: return None, 0, 0
                
            df = pd.DataFrame(response.data)
            df = df.rename(columns={
                'ticker': 'Ticker',
                'price': 'Last Price',
                'bull_score': 'Bullish Score (%)',
                'bear_score': 'Bearish Score (%)',
                'score_diff': 'Score Diff',
                'signal_type': 'Signal',
                'mtf_score': 'MTF Score',
                'conviction_score': 'Conviction_Score',
                'relative_vol': 'Relative Vol',
                'rsi': 'RSI',
                'market_regime': 'Market_Regime'
            })
            
            df['Sector'] = df['Ticker'].map(SET100_SECTORS)
            df['Last Update'] = f"{date} {time[:5]}"
            
            # These columns might be missing in older manual results
            if 'Score Diff' not in df.columns: df['Score Diff'] = df['Bullish Score (%)']
            
            pos_count = len(df[df['Score Diff'] > 0])
            neg_count = len(df[df['Score Diff'] < 0])
            
            return df, pos_count, neg_count
            
    except Exception as e:
        st.error(f"⚠️ Error loading latest scan results: {e}")
        return None, 0, 0

def fetch_historical_signals(signal_type, lookback_days=30):
    """Fetch historical signals of a specific type from Supabase."""
    if not supabase:
        return pd.DataFrame()
        
    try:
        cutoff_date = (datetime.now(SET_TZ) - timedelta(days=lookback_days)).isoformat()
        
        # Query auto_scan_results
        resp = supabase.table("auto_scan_results") \
            .select("*") \
            .eq("signal", signal_type) \
            .gte("scanned_at", cutoff_date) \
            .order("scanned_at", desc=True) \
            .execute()
            
        df = pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
        
        if not df.empty:
            df = df.rename(columns={
                'ticker': 'Ticker',
                'close_price': 'Last Price',
                'score': 'Conviction_Score',
                'scanned_at': 'Last Update',
                'signal': 'Signal',
                'strategy': 'Strategy',
                'change_percent': '% Change',
                'rel_vol': 'Relative Vol'
            })
            # Handle RV formatting and missing column
            if 'Relative Vol' not in df.columns:
                df['Relative Vol'] = 1.0
            
            # Format dates
            df['Last Update'] = pd.to_datetime(df['Last Update']).dt.tz_convert(SET_TZ).dt.strftime("%Y-%m-%d %H:%M")
            
            # Add required columns for UI compatibility
            df['Expected Jump (%)'] = 0.0 # Placeholder
            df['Expected Drop (%)'] = -1.0 # Placeholder
            
        return df
    except Exception as e:
        st.error(f"Error fetching historical signals: {e}")
        return pd.DataFrame()

def fetch_market_scan_results():
    """
    [HYBRID MANDATE] Retrieve the latest results from both scan_results (Manual) 
    and auto_scan_results (Auto) tables, choosing the newest for each ticker.
    """
    if not supabase:
        return pd.DataFrame()
    
    df_auto = pd.DataFrame()
    df_manual = pd.DataFrame()
    
    # 1. Fetch from auto_scan_results (Auto)
    try:
        auto_res = supabase.table("auto_scan_results") \
            .select("*") \
            .order("scanned_at", desc=True) \
            .limit(300) \
            .execute()
        
        if auto_res.data:
            df_auto = pd.DataFrame(auto_res.data)
            df_auto['source'] = 'Auto'
            # Standardize scanned_at to naive Bangkok time
            df_auto['scanned_at'] = pd.to_datetime(df_auto['scanned_at'], errors='coerce')
            if df_auto['scanned_at'].dt.tz is not None:
                df_auto['scanned_at'] = df_auto['scanned_at'].dt.tz_convert(SET_TZ).dt.tz_localize(None)
            
            # Standardize Column Names
            df_auto = df_auto.rename(columns={
                'close_price': 'close_price', # already correct
                'score': 'bull_score'
            })
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถดึงข้อมูลจาก auto_scan_results: {e}")
        df_auto = pd.DataFrame()

    # 2. Fetch from scan_results (Manual)
    try:
        manual_res = supabase.table("scan_results") \
            .select("*") \
            .order("scan_date", desc=True) \
            .limit(300) \
            .execute()
        
        if manual_res.data:
            df_manual = pd.DataFrame(manual_res.data)
            df_manual['source'] = 'Manual'
            
            # Standardize scanned_at
            d_col = 'scan_date' if 'scan_date' in df_manual.columns else ('date' if 'date' in df_manual.columns else None)
            t_col = 'scan_time' if 'scan_time' in df_manual.columns else ('time' if 'time' in df_manual.columns else None)
            
            if d_col and t_col:
                df_manual['scanned_at'] = pd.to_datetime(df_manual[d_col].astype(str) + ' ' + df_manual[t_col].fillna('00:00:00').astype(str), errors='coerce')
            elif d_col:
                df_manual['scanned_at'] = pd.to_datetime(df_manual[d_col].astype(str), errors='coerce')
            else:
                c_col = 'created_at' if 'created_at' in df_manual.columns else None
                df_manual['scanned_at'] = pd.to_datetime(df_manual[c_col], errors='coerce') if c_col else pd.Timestamp.now()
            
            if df_manual['scanned_at'].dt.tz is not None:
                df_manual['scanned_at'] = df_manual['scanned_at'].dt.tz_localize(None)

            # Standardize Column Names
            df_manual = df_manual.rename(columns={
                'signal_type': 'signal',
                'price': 'close_price'
            })
            
            # Ensure missing columns exist for concat
            for col in ['strategy', 'change_percent', 'volume', 'sector', 'rsi']:
                if col not in df_manual.columns:
                    df_manual[col] = None
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถดึงข้อมูลจาก scan_results: {e}")
        df_manual = pd.DataFrame()

    # 3. Hybrid Aggregation
    if df_auto.empty and df_manual.empty:
        return pd.DataFrame()
        
    # Standardize columns to avoid InvalidIndexError
    cols = ['ticker', 'signal', 'score', 'strategy', 'close_price', 'change_percent', 'rsi', 'scanned_at', 'source', 'is_pinbar', 'is_silent_accum', 'relative_vol', 'bull_score', 'conviction_score']
    
    def clean_df(df):
        if df.empty:
            return pd.DataFrame(columns=cols)
        # 1. Clean Duplicate Columns
        df = df.loc[:, ~df.columns.duplicated()].copy()
        # 2. Reset Index
        df.reset_index(drop=True, inplace=True)
        # 3. Ensure 'score' exists
        if 'score' not in df.columns:
            df['score'] = df.get('conviction_score', df.get('bull_score', df.get('score')))
        
        # 4. Fill missing columns
        for c in cols:
            if c not in df.columns:
                if c == 'relative_vol':
                    df[c] = df.get('relative_vol', df.get('rel_vol'))
                elif c == 'bull_score':
                    df[c] = df.get('bull_score', df.get('score'))
                elif c == 'conviction_score':
                    df[c] = df.get('conviction_score', df.get('score'))
                else:
                    df[c] = None
        return df[cols]

    df_auto_clean = clean_df(df_auto)
    df_manual_clean = clean_df(df_manual)
    
    combined = pd.concat([df_auto_clean, df_manual_clean], ignore_index=True)

    # 4. Final Processing & Deduplicate
    if not combined.empty:
        # Standardize score column again to be safe
        if 'conviction_score' in combined.columns:
            combined['score'] = combined['conviction_score'].fillna(combined['score'])
        
        # Ensure 'signal' exists
        if 'signal' not in combined.columns:
            combined['signal'] = 'WAIT'

        # Sort and Deduplicate
        combined = combined.sort_values(by='scanned_at', ascending=False)
        combined = combined.drop_duplicates(subset=['ticker'], keep='first')
        
    return combined

def fetch_ticker_combined_history(ticker, days=90):
    """Retrieve historical scan signals from both scan_results and auto_scan_results with strict ticker filtering."""
    if not supabase:
        return pd.DataFrame()
    try:
        # Standardize ticker for query
        clean_ticker = ticker.strip().upper()
        base_ticker = clean_ticker.replace('.BK', '')
        bk_ticker = base_ticker + '.BK'
        ticker_list = list(set([base_ticker, bk_ticker]))
        
        # Use start of day for query to be safe
        now_th = datetime.now(SET_TZ)
        start_dt = (now_th - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = start_dt.isoformat()
        
        # 1. Fetch from scan_results - STRICT SQL FILTERING
        res1 = supabase.table("scan_results") \
            .select("ticker, scan_date, signal_type, bull_score, price") \
            .in_("ticker", ticker_list) \
            .gte("scan_date", start_date[:10]) \
            .execute()
        
        df1 = pd.DataFrame(res1.data)
        if not df1.empty:
            df1 = df1.rename(columns={
                'scan_date': 'scanned_at', 
                'signal_type': 'signal', 
                'bull_score': 'score',
                'price': 'close_price'
            })
            df1['scanned_at'] = pd.to_datetime(df1['scanned_at']).dt.tz_localize(None)
            df1['source'] = 'manual'
        
        # 2. Fetch from auto_scan_results - STRICT SQL FILTERING
        res2 = supabase.table("auto_scan_results") \
            .select("ticker, scanned_at, signal, strategy, score, close_price, is_silent_accum, rsi, volume") \
            .in_("ticker", ticker_list) \
            .gte("scanned_at", start_date) \
            .execute()
            
        df2 = pd.DataFrame(res2.data)
        if not df2.empty:
            df2['scanned_at'] = pd.to_datetime(df2['scanned_at']).dt.tz_convert(SET_TZ).dt.tz_localize(None)
            df2['source'] = 'auto'
            
        # Combine
        combined = pd.concat([df1, df2], ignore_index=True)
        if combined.empty:
            return pd.DataFrame()
            
        # Normalize ticker column for consistent deduplication
        combined['ticker'] = combined['ticker'].str.strip().str.upper()
            
        # Sort and deduplicate by date, ticker and signal type
        combined['date_only'] = combined['scanned_at'].dt.date
        combined = combined.sort_values('scanned_at', ascending=True)
        # Use keep='first' as per "First Signal Wins" logic for intraday signals
        combined = combined.drop_duplicates(subset=['ticker', 'date_only', 'signal'], keep='first')
        
        return combined.drop(columns=['date_only'])
    except Exception as e:
        print(f"Error fetching combined history for {ticker}: {e}")
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
        # Use silent=True to avoid cluttering UI with index fetch errors
        set_idx = get_stock_data("^SET.BK", silent=True)
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
    """Fetch sector and industry info with robust browser-like headers."""
    if not ticker: return 'N/A'
    if ticker in SET100_SECTORS:
        return SET100_SECTORS[ticker]
        
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': USER_AGENTS[0]})
        t = yq.Ticker(ticker, session=session)
        profile = t.summary_profile
        if isinstance(profile, dict):
            data = profile.get(ticker) or next(iter(profile.values()), None)
            if isinstance(data, dict):
                s = data.get('sector') or data.get('sectorDisp')
                if s and s != 'N/A': return s
    except:
        pass
        
    try:
        info = yf.Ticker(ticker).info
        if info and 'sector' in info:
            return info['sector']
    except:
        pass

    return 'N/A'

@st.cache_data(ttl=600)
def get_stock_data(ticker, silent=False):
    """
    Fetch historical stock data with Multi-Tier Caching (Memory + Disk).
    """
    if not ticker:
        return None
        
    clean_ticker = ticker.strip().upper()
    if not clean_ticker.endswith('.BK') and not clean_ticker.startswith('^'):
        clean_ticker = f"{clean_ticker}.BK"
    
    # Tier 2: Disk Cache
    cached_df = get_disk_cache(clean_ticker)
    if cached_df is not None:
        return cached_df

    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Randomize User-Agent
            ua = USER_AGENTS[attempt % len(USER_AGENTS)]
            headers = {'User-Agent': ua}
            
            # 1. Try YahooQuery
            session = requests.Session()
            session.headers.update(headers)
            t = yq.Ticker(clean_ticker, session=session)
            df = t.history(start="2018-01-01")
            
            if df is not None and not df.empty:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                else:
                    df = df.reset_index()
                    
                if "symbol" in df.columns:
                    df = df[df["symbol"] == clean_ticker].copy()
                
                if not df.empty:
                    col_map = {"close": "Close", "volume": "Volume", "open": "Open", "high": "High", "low": "Low"}
                    available_cols = [c for c in col_map.keys() if c in df.columns]
                    if "date" in df.columns and len(available_cols) >= 3:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index("date")
                        df = df[available_cols].rename(columns=col_map)
                        
                        df = df.sort_index()
                        df = df[~df.index.duplicated(keep='last')]
                        df = df.ffill().bfill()
                        
                        if df.index.tz is not None:
                            df.index = df.index.tz_convert(SET_TZ).tz_localize(None)
                        else:
                            df.index = df.index.tz_localize(None)
                        
                        needed = ["Open", "High", "Low", "Close", "Volume"]
                        if all(c in df.columns for c in needed):
                            save_disk_cache(clean_ticker, df)
                            return df
            
            # 2. Fallback to yfinance
            yf_ticker = yf.Ticker(clean_ticker)
            df_yf = yf_ticker.history(period="2y")
            
            if df_yf is None or df_yf.empty:
                df_yf = yf_ticker.history(period="max")
                
            if df_yf is not None and not df_yf.empty:
                needed = ["Open", "High", "Low", "Close", "Volume"]
                df_yf = df_yf[[c for c in needed if c in df_yf.columns]].copy()
                df_yf = df_yf.sort_index()
                df_yf = df_yf[~df_yf.index.duplicated(keep='last')]
                df_yf = df_yf.ffill().bfill()
                
                if df_yf.index.tz is not None:
                    df_yf.index = df_yf.index.tz_convert(SET_TZ).tz_localize(None)
                else:
                    df_yf.index = df_yf.index.tz_localize(None)
                    
                if all(c in df_yf.columns for c in needed):
                    save_disk_cache(clean_ticker, df_yf)
                    return df_yf
            
            raise ValueError(f"Empty data")
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            elif not silent:
                if "^SET.BK" in clean_ticker or "SET.BK" in clean_ticker:
                    pass
                else:
                    # Suppress UI alert clutter
                    print(f"Failed to fetch {clean_ticker}: {e}")
                
    return None

def batch_get_stock_data(tickers):
    """
    Download multiple tickers in one go using yf.download to reduce API hits.
    Saves to disk cache for future use.
    """
    clean_tickers = []
    for t in tickers:
        ct = t.strip().upper()
        if not ct.endswith('.BK') and not ct.startswith('^'):
            ct = f"{ct}.BK"
        clean_tickers.append(ct)
    
    # Filter out tickers already in disk cache
    missing_tickers = [t for t in clean_tickers if get_disk_cache(t) is None]
    
    if missing_tickers:
        try:
            # Download missing tickers in one batch
            # Note: yf.download returns a MultiIndex if multiple tickers are passed
            data = yf.download(missing_tickers, period="2y", group_by='ticker', threads=True, progress=False)
            
            for t in missing_tickers:
                try:
                    if len(missing_tickers) == 1:
                        df = data
                    else:
                        df = data[t]
                    
                    if df is not None and not df.empty:
                        needed = ["Open", "High", "Low", "Close", "Volume"]
                        df = df[[c for c in needed if c in df.columns]].copy()
                        df = df.sort_index()
                        df = df[~df.index.duplicated(keep='last')]
                        df = df.ffill().bfill()
                        
                        if df.index.tz is not None:
                            df.index = df.index.tz_convert(SET_TZ).tz_localize(None)
                        else:
                            df.index = df.index.tz_localize(None)
                            
                        save_disk_cache(t, df)
                except:
                    continue
        except Exception as e:
            print(f"Batch download failed: {e}")
    
    # Return all requested data (from memory/disk)
    return {t: get_stock_data(t, silent=True) for t in clean_tickers}


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
        - ความแม่นยำทางสถิติ (Pattern Consensus): {row.get('Pattern Consensus (%)', 0)}%
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



# --- Quant Helper Functions ---


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
        (batch_df['Score Diff'].apply(lambda x: safe_float(x) > 5)) & 
        (~batch_df['Signal'].isin(ignored_signals))
    ].copy()
    
    if candidates.empty:
        # Fallback to show something if no perfect matches, but with lower score diff
        candidates = batch_df[batch_df['Score Diff'].apply(lambda x: safe_float(x) > 0)].copy()
        
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
        similarity = safe_float(row.get('Pattern Consensus (%)', 0))
        mtf_score = safe_float(row.get('MTF Score', 0))
        rsi = safe_float(row.get('RSI', 50))
        rel_vol = safe_float(row.get('Relative Vol', 1.0))
        score_velocity = safe_float(row.get('Score Velocity', 0))
        sector_rs = safe_float(row.get('Sector_RS', 0))
        price_change = safe_float(row.get('% Change', 0))
        atr = safe_float(row.get('ATR', 0))
        last_price = safe_float(row.get('Last Price', 0))
        
        # Dynamic Stop Loss: Last Price - (2 * ATR)
        stop_loss = safe_float(last_price) - (2 * safe_float(atr)) if safe_float(atr) > 0 else safe_float(last_price) * 0.95
        
        formula_score, strategy, reasons, warnings = calculate_conviction_score(
            ticker, row['Signal'], similarity, mtf_score, regime, perf_stats, 
            rsi, rel_vol, score_velocity, sector_rs=sector_rs, price_change=price_change
        )
        
        # Ensure formula_score is a valid number
        formula_score = safe_float(formula_score)
        
        # Strict SRS Filter: Skip if significantly underperforming
        if safe_float(sector_rs) < -1.5:
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
        if not ticker_labeled.empty and len(ticker_labeled) > 0:
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
            'Intraday_History': list(past_signals),
            'Sector_RS': sector_rs
        })
    
    # Sort by: 1. Signal Tier (1 is highest), 2. Conviction Score
    df_report = pd.DataFrame(report_data)
    
    if df_report.empty:
        # Return empty DataFrame with expected columns to avoid downstream KeyErrors
        return pd.DataFrame(columns=[
            'Ticker', 'Signal', 'Strategy', 'Stop_Loss', 'Persistence', 
            'Score_Trend', 'Similarity', 'Ticker_Win_Rate', 'Signal_Win_Rate', 
            'MTF_Score', 'Conviction_Score', 'Signal_Tier', 'Why', 'Warnings', 
            'Price', 'Intraday_History', 'Sector_RS'
        ])
        
    # Standardize column name just in case
    if 'signal_tier' in df_report.columns and 'Signal_Tier' not in df_report.columns:
        df_report = df_report.rename(columns={'signal_tier': 'Signal_Tier'})
    
    if 'Signal_Tier' not in df_report.columns:
        df_report['Signal_Tier'] = 3 # Default to Tier 3
        
    return df_report.sort_values(['Signal_Tier', 'Conviction_Score'], ascending=[True, False])


# --- 7. SET100 Batch Scanner ---
# Removed local get_signal_performance_stats as it is now in scanner_engine.py

def run_set100_batch_scan(tickers, target_date=None):
    """
    Scan all tickers for bullish and bearish matches with Conservative Logic.
    Supports historical scanning if target_date is provided.
    """
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # NEW: Create a uniform timestamp for this batch run to allow grouping in DB
    batch_now = datetime.now(SET_TZ)
    batch_date = batch_now.strftime("%Y-%m-%d")
    batch_time = batch_now.strftime("%H:%M:%S")
    batch_update_str = batch_now.strftime("%Y-%m-%d %H:%M")
    
    # Reset debug logger for save_scan_result
    if hasattr(save_scan_result, "_logged_sample"):
        delattr(save_scan_result, "_logged_sample")
    
    # Pre-fetch all sectors in bulk
    status_text.text("🔄 Initializing Sector & Price Data...")
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': USER_AGENTS[0]})
        bulk_t = yq.Ticker(tickers, session=session)
        all_profiles = bulk_t.summary_profile
    except:
        all_profiles = {}
    
    # NEW: Batch Download Price Data for all tickers to reduce API hits
    try:
        all_data = batch_get_stock_data(tickers)
    except Exception as e:
        print(f"Batch data fetch failed: {e}")
        all_data = {}
    
    # Get performance stats once for scoring
    perf_stats = get_signal_performance_stats(supabase)
    pos_count = 0
    neg_count = 0
    success_count = 0
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"Scanning {ticker} ({i+1}/{len(tickers)})...")
            
            # Use pre-fetched data if available
            clean_t = ticker.strip().upper()
            if not clean_t.endswith('.BK') and not clean_t.startswith('^'):
                clean_t = f"{clean_t}.BK"
            
            df_full = all_data.get(clean_t)
            
            if df_full is None:
                # Fallback to individual fetch if batch missed it
                df_full = get_stock_data(ticker, silent=True)
            
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

                # Unified Scanner Call (V7)
                scan_res = core_strategy_scanner(ticker, df_raw, target_date=target_date)
                if not scan_res: continue
                
                # Extract values for report
                bull_score = scan_res['bull_score']
                bear_score = scan_res['bear_score']
                score_diff = scan_res['score_diff']
                signal = scan_res['signal']
                last_price = scan_res['close_price']
                pct_change = scan_res['change_percent']
                rsi_curr = scan_res['rsi']
                rel_vol = scan_res['rel_vol']
                atc_risk = scan_res['atc_risk']
                consensus = scan_res['consensus']
                mtf_status = scan_res['mtf_status']
                mtf_score = scan_res['mtf_score']
                recovery_data = scan_res['recovery_data']
                atr_now = scan_res['atr_now']
                high_vol = scan_res['high_vol']
                bull_jump = scan_res['bull_jump']
                bear_jump = scan_res['bear_jump']
                score_velocity = scan_res['score_velocity']
                
                if pct_change > 0: pos_count += 1
                elif pct_change < 0: neg_count += 1
                
                day_high = df_raw['High'].iloc[-1]
                m_regime, _ = get_market_regime()

                # Fetch Sector
                sector = 'N/A'
                if isinstance(all_profiles, dict) and ticker in all_profiles:
                    p_data = all_profiles[ticker]
                    if isinstance(p_data, dict):
                        sector = p_data.get('sector') or p_data.get('sectorDisp') or 'N/A'
                
                if sector == 'N/A':
                    sector = get_stock_info(ticker)

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
                    # If live scan (target_date is None), use batch time. If historical, use candle date.
                    if target_date:
                        last_update_val = df_raw.index[-1].strftime("%Y-%m-%d %H:%M") if hasattr(df_raw.index[-1], 'strftime') else str(df_raw.index[-1])
                        db_date = df_raw.index[-1].strftime("%Y-%m-%d")
                        db_time = "00:00:00"
                    else:
                        last_update_val = batch_update_str
                        db_date = batch_date
                        db_time = batch_time


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
                ticker, row['Signal'], row.get('Pattern Consensus (%)', 0), 
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
                'Pattern Consensus (%)': row.get('Pattern Consensus (%)', 0),
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
            if save_scan_result(db_data):
                success_count += 1
        
        results = final_results
        if success_count > 0:
            st.success(f"✅ บันทึกผลสแกนลงตาราง scan_results จำนวน {success_count} รายการเรียบร้อยแล้ว")
        else:
            st.warning("⚠️ ไม่สามารถบันทึกข้อมูลลง Supabase ได้ โปรดตรวจสอบ Error บนหน้าจอ")

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
    # NEW: Try to fetch latest scan from Supabase on startup/refresh
    with st.spinner("🔄 Loading Latest Scan Results from Supabase..."):
        df_latest, pos_l, neg_l = fetch_latest_scan_results()
        if df_latest is not None:
            st.session_state['batch_results'] = {
                'df': df_latest,
                'pos': pos_l,
                'neg': neg_l,
                'is_historical': False,
                'source': 'Supabase Persistence'
            }
            st.toast("✅ Latest scan results loaded from Supabase!", icon="💾")

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

# --- Session State Data Loading ---
# Initialize defaults to prevent NameError in tabs
batch_df = pd.DataFrame()
pos_count = 0
neg_count = 0
is_hist = False
regime = "NEUTRAL"

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

        # --- Helper Functions for Radar ---
def get_radar_performance_stats(supabase):
    """Calculate Win Rate and Avg Return from trading_log (Last 30 days)."""
    if not supabase: return {}
    try:
        thirty_days_ago = (datetime.now(SET_TZ) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        response = supabase.table("trading_log") \
            .select("signal, status, outcome_t3_pct") \
            .gte("timestamp", thirty_days_ago) \
            .execute()
        
        if not response.data: return {}
            
        df_perf = pd.DataFrame(response.data)
        if df_perf.empty: return {}
            
        def standardize_signal(sig):
            sig = str(sig).upper()
            if 'BUY' in sig or 'BREAKOUT' in sig: return 'BUY / BREAKOUT'
            if 'PULLBACK' in sig or 'PIN BAR' in sig: return 'PULLBACK / PIN BAR'
            if 'SILENT' in sig: return 'SILENT ACCUM'
            return 'OTHER'
            
        df_perf['signal_cat'] = df_perf['signal'].apply(standardize_signal)
        
        stats = df_perf.groupby('signal_cat').agg(
            Win_Rate=('status', lambda x: (x == 'Success').sum() / len(x) * 100 if len(x) > 0 else 0),
            Avg_Return=('outcome_t3_pct', 'mean')
        ).to_dict('index')
        return stats
    except:
        return {}

# --- 2-Tier Navigation ---
st.sidebar.header("🕹️ Strategy Navigator")
main_category = st.sidebar.radio(
    "Main Category",
    ["🎯 Trading & Daily Operations", "🌋 Market Insights & Analytics", "🧪 Quant Lab & Administration"],
    key="main_nav"
)

# Initialize data if needed for breadth/regime
if 'batch_results' in st.session_state and st.session_state['batch_results'] is not None:
    batch_df = st.session_state['batch_results']['df']
    pos_count = st.session_state['batch_results']['pos']
    neg_count = st.session_state['batch_results']['neg']
else:
    batch_df = pd.DataFrame()
    pos_count = 0
    neg_count = 0

if main_category == "🎯 Trading & Daily Operations":
    sub_tabs = st.tabs(["🎯 Signal Command Center", "🚀 Unified Scanner & Pattern Radar"])
    
    with sub_tabs[0]: # 🎯 Signal Command Center
        try:
            st.title("🎯 Trading Signal Command Center")
            st.markdown("""
                <style>
                .stDataFrame td {
                    font-size: 15px !important;
                }
                </style>
            """, unsafe_allow_html=True)
            st.info("📅 **Daily Action Plan:** แผนการเทรดรายวันสำหรับพรุ่งนี้ (T+1 Open) อ้างอิงจาก Optimized Parameters ล่าสุด")
            
            # 1. Executive Summary Cards
            ticker_list = SET100_TICKERS
            with st.spinner("Evaluating daily signals..."):
                all_data = batch_get_stock_data(ticker_list)
                # Fetch optimized parameters
                optimized_params = load_best_params()
                signal_results = signal_engine.evaluate_daily_signals(all_data, optimized_params_dict=optimized_params)
                entry_orders_df = signal_results['new_entries']
                
                # Active Positions Logic - Load from Supabase if available
                if 'active_positions' not in st.session_state or st.session_state.get('active_positions') is None:
                    if supabase:
                        try:
                            # Fetch positions with 'Pending' (System) or 'OPEN' (Paper) status from trading_log
                            resp = supabase.table("trading_log").select("*").or_("status.eq.Pending,status.eq.OPEN").execute()
                            if resp.data:
                                df_active = pd.DataFrame(resp.data)
                                # Map Supabase columns: ticker -> ticker, entry_price -> entry_price, timestamp -> entry_date
                                # System signals use 'last_price' as entry estimate, Paper trades use 'entry_price'
                                if 'entry_price' in df_active.columns:
                                    df_active['entry_price'] = df_active['entry_price'].fillna(df_active['last_price'])
                                else:
                                    df_active['entry_price'] = df_active['last_price']
                                    
                                st.session_state['active_positions'] = df_active[['ticker', 'entry_price', 'timestamp']].rename(columns={
                                    'ticker': 'ticker',
                                    'entry_price': 'entry_price',
                                    'timestamp': 'entry_date'
                                })
                            else:
                                st.session_state['active_positions'] = pd.DataFrame(columns=['ticker', 'entry_price', 'entry_date'])
                        except Exception as e:
                            st.error(f"Error fetching positions from Supabase: {e}")
                            st.session_state['active_positions'] = pd.DataFrame(columns=['ticker', 'entry_price', 'entry_date'])
                    else:
                        st.session_state['active_positions'] = pd.DataFrame(columns=['ticker', 'entry_price', 'entry_date'])
                
                active_pos_df = st.session_state['active_positions']
                exit_control_df = signal_engine.check_active_positions(active_pos_df, all_data, optimized_params_dict=optimized_params)
            
            col_sc1, col_sc2, col_sc3 = st.columns(3)
            col_sc1.metric("Today's New Buy Signals", len(entry_orders_df))
            col_sc2.metric("Active Open Positions", len(active_pos_df))
            
            tp_sl_reached = 0
            if not exit_control_df.empty:
                tp_sl_reached = len(exit_control_df[exit_control_df['Action Required'].str.contains('SELL', na=False)])
            col_sc3.metric("TP / SL Reached Today", tp_sl_reached)
            
            # --- Paper Trading Portfolio Metrics ---
            st.divider()
            st.markdown("### 💼 Paper Trading Portfolio Summary")
            portfolio_metrics = paper_trading.get_paper_portfolio_metrics(supabase)
            if portfolio_metrics:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Portfolio Value", f"{portfolio_metrics['total_portfolio_value']:,.2f} THB")
                m2.metric("Win Rate %", f"{portfolio_metrics['win_rate']}%")
                m3.metric("Realized PnL", f"{portfolio_metrics['realized_pnl']:,.2f} THB", delta=portfolio_metrics['realized_pnl'])
                m4.metric("Un-realized PnL", f"{portfolio_metrics['unrealized_pnl']:,.2f} THB", delta=portfolio_metrics['unrealized_pnl'])
            
            st.divider()
            
            # 2. Daily Action Plan Tables
            st.markdown("### 🟢 Entry Orders for Tomorrow (T+1 Open)")
            st.caption("ออเดอร์ที่เตรียมเข้าซื้อที่ราคาเปิดในวันทำการถัดไป (Zero Look-Ahead Bias)")
            
            if not entry_orders_df.empty:
                # Add columns for Paper Trading selection
                display_entry_df = entry_orders_df.copy()
                
                # Show the signals table with institutional formatting
                formatted_entry_df = display_entry_df.style.format({
                    'Calculated Entry Price (Open)': '{:,.2f}',
                    'Dynamic Stop Loss': '{:,.2f}',
                    'Dynamic Take Profit': '{:,.2f}',
                    'R:R Ratio': '{:,.2f}'
                }).set_properties(**{'font-weight': 'bold', 'font-size': '15px'})
                
                st.dataframe(formatted_entry_df, use_container_width=True, hide_index=True)
                
                # Paper Trading Execution Form
                with st.expander("🟢 Execute Paper Buy Orders", expanded=False):
                    buy_col1, buy_col2, buy_col3 = st.columns([2, 2, 1])
                    selected_ticker = buy_col1.selectbox("Select Ticker to Buy", options=display_entry_df['Ticker'].tolist())
                    
                    # Get signal type for the selected ticker
                    sig_type = display_entry_df[display_entry_df['Ticker'] == selected_ticker]['Signal Type'].values[0]
                    entry_px = display_entry_df[display_entry_df['Ticker'] == selected_ticker]['Calculated Entry Price (Open)'].values[0]
                    
                    quantity = buy_col2.number_input(f"Quantity for {selected_ticker}", min_value=100, value=1000, step=100)
                    
                    if buy_col3.button("🟢 Execute Buy", use_container_width=True):
                        success, msg = paper_trading.execute_paper_buy(supabase, selected_ticker, entry_px, quantity, sig_type)
                        if success:
                            st.success(msg)
                            st.session_state['active_positions'] = None # Reset to force refresh
                            st.rerun()
                        else:
                            st.error(msg)

                csv_orders = entry_orders_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Download Tomorrow_Orders.csv",
                    csv_orders,
                    f"Tomorrow_Orders_{datetime.now(SET_TZ).strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key='download-orders-command-center'
                )
            else:
                st.info("ℹ️ ยังไม่มีสัญญาณซื้อใหม่ในวันนี้")
            
            st.divider()
            c_exit1, c_exit2 = st.columns([8, 2])
            c_exit1.markdown("### 🔴 Exit & Risk Control Center")
            
            if c_exit2.button("🔄 Refresh Live Prices", use_container_width=True):
                # Clear all stock data caches to force fresh fetch
                st.cache_data.clear()
                # Clear disk cache for active tickers to ensure live data
                if 'active_positions' in st.session_state and st.session_state['active_positions'] is not None:
                    active_tickers = st.session_state['active_positions']['ticker'].tolist()
                    for t in active_tickers:
                        ct = t.strip().upper()
                        if not ct.endswith('.BK') and not ct.startswith('^'):
                            ct = f"{ct}.BK"
                        cache_file = CACHE_DIR / f"{ct.replace('^', '_')}.pkl"
                        if cache_file.exists():
                            try:
                                cache_file.unlink()
                            except:
                                pass
                st.rerun()

            st.caption("ติดตามสถานะออเดอร์ที่เปิดอยู่ และตรวจสอบจุดตัดขาดทุน/ทำกำไร")
            
            if not exit_control_df.empty:
                def style_exit_table(row):
                    action = str(row.get('Action Required', ''))
                    # Institutional Grade Styling: Bold and Larger Font
                    base_style = 'font-weight: bold; font-size: 15px;'
                    if 'SELL' in action:
                        color = 'rgba(239, 68, 68, 0.2)' if '(SL)' in action else 'rgba(34, 197, 94, 0.2)'
                        return [f'background-color: {color}; {base_style}'] * len(row)
                    return [base_style] * len(row)
                
                # Format numbers and apply institutional styles
                formatted_exit_df = exit_control_df.style.apply(style_exit_table, axis=1).format({
                    'Entry Price': '{:,.2f}',
                    'Current Price': '{:,.2f}',
                    'Target TP': '{:,.2f}',
                    'Target SL': '{:,.2f}',
                    'Current PnL %': '{:+.2f}%'
                })
                
                st.dataframe(formatted_exit_df, use_container_width=True, hide_index=True)
                
                # Paper Trading Exit Form
                with st.expander("🔴 Execute Paper Sell / Close Positions", expanded=False):
                    # We need the Supabase IDs to close positions
                    if supabase:
                        resp = supabase.table("trading_log").select("id, ticker, entry_price, status").eq("status", "OPEN").execute()
                        if resp.data:
                            open_pos_df = pd.DataFrame(resp.data)
                            sell_col1, sell_col2, sell_col3 = st.columns([2, 2, 1])
                            
                            pos_to_sell = sell_col1.selectbox(
                                "Select Position to Close", 
                                options=open_pos_df['id'].tolist(),
                                format_func=lambda x: f"{open_pos_df[open_pos_df['id']==x]['ticker'].values[0]} (Entry: {open_pos_df[open_pos_df['id']==x]['entry_price'].values[0]})"
                            )
                            
                            # Get current price for the selected ticker if possible
                            ticker_to_sell = open_pos_df[open_pos_df['id'] == pos_to_sell]['ticker'].values[0]
                            current_px = all_data[ticker_to_sell]['Close'].iloc[-1] if ticker_to_sell in all_data else 0.0
                            
                            exit_px = sell_col2.number_input(f"Exit Price for {ticker_to_sell}", value=float(current_px))
                            
                            if sell_col3.button("🔴 Execute Sell", use_container_width=True):
                                success, msg = paper_trading.execute_paper_sell(supabase, pos_to_sell, exit_px)
                                if success:
                                    st.success(msg)
                                    st.session_state['active_positions'] = None # Reset to force refresh
                                    st.rerun()
                                else:
                                    st.error(msg)
                        else:
                            st.info("No open paper positions to close.")

                csv_exits = exit_control_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Download Exit_Orders.csv",
                    csv_exits,
                    f"Exit_Orders_{datetime.now(SET_TZ).strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key='download-exits-command-center'
                )
            else:
                st.info("ℹ️ ยังไม่มีออเดอร์ที่เปิดสถานะอยู่ (Active Positions)")
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดใน Command Center: {e}")

    with sub_tabs[1]: # 🚀 Unified Scanner & Pattern Radar
        # ... (Unified Scanner logic moved here)
        try:
            st.subheader("🚀 Unified Scanner & Pattern Radar")
            st.caption("🔍 **ระบบคัดกรองอัจฉริยะ:** รวม 3 กลยุทธ์ใหม่ (1) **Volume Compression** ตรวจจับวอลุ่มแห้งก่อนระเบิด (2) **Sector Flow Filter** คัดเฉพาะหุ้นที่แข็งแกร่งกว่ากลุ่ม (SRS) และ (3) **Dynamic Stop Loss** ปรับตามความผันผวนจริง (ATR)")
            
            # Ensure batch_df is available
            if not batch_df.empty:
                unified_df = generate_unified_report(batch_df, regime)
                
                if not unified_df.empty:
                    # Safe Data Clean-up before Rendering
                    unified_df = unified_df.fillna({
                        'Conviction_Score': 0, 
                        'Sector_RS': 0, 
                        'Stop_Loss': 0, 
                        'Similarity': 0,
                        'Pattern Consensus (%)': 0,
                        'Ticker_Win_Rate': 0,
                        'Signal_Win_Rate': 0,
                        'MTF_Score': 0,
                        'Signal_Tier': 3
                    })
                    
                    # Ensure Signal_Tier exists for filtering
                    if 'Signal_Tier' not in unified_df.columns:
                        unified_df['Signal_Tier'] = 3
                        
                    # Sorting: High Conviction first, then positive signals
                    # Use safe_float for robust numerical comparison
                    high_strength = unified_df[unified_df['Conviction_Score'].apply(lambda x: safe_float(x) >= 40)].copy()
                    pos_signals = ['BUY', 'GOLDEN BUY', 'PRE-FLY', 'PIN BAR (SUPPORT)', 'SILENT ACCUM']
                    early_birds = unified_df[(unified_df['Conviction_Score'].apply(lambda x: safe_float(x) < 40)) & (unified_df['Signal'].isin(pos_signals))].copy()
                    top_conviction = pd.concat([high_strength, early_birds]).head(20)
                    
                    if not top_conviction.empty:
                        st.success(f"🔥 พบหุ้นน่าสนใจ {len(top_conviction)} ตัว (จัดลำดับตามคะแนนและความมั่นใจ)")
                        
                        for idx, (i, row) in enumerate(top_conviction.iterrows()):
                            # Safe Numerical Handling for Card Logic
                            conv_score = safe_float(row.get('Conviction_Score', 0))
                            is_early_bird = conv_score < 40
                            
                            # Dot & Label Logic
                            if is_early_bird:
                                dot_color = "#10b981" # Emerald
                                s_label = "EARLY ENTRY"
                            else:
                                dot_color = "#3b82f6" if "SWING" in str(row.get('Strategy', '')) else "#f59e0b"
                                s_label = row.get('Strategy', 'Unknown')

                            # Signal Styling
                            sig_val = row.get('Signal', 'WAIT')
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

                            # Build card HTML with Safe Numerical Formatting
                            srs_val = safe_float(row.get('Sector_RS', 0))
                            srs_color = "#166534" if srs_val > 0 else ("#991b1b" if srs_val < 0 else "#4b5563")
                            stop_loss = safe_float(row.get('Stop_Loss', 0))
                            # Try to get Similarity first, then Pattern Consensus (%)
                            pat_consensus = safe_float(row.get('Similarity', row.get('Pattern Consensus (%)', 0)))
                            
                            card_html = f'<div class="compact-card"><div class="card-header"><div class="header-left"><div class="dot-indicator" style="background-color: {dot_color};"></div><div class="ticker-name">{row["Ticker"]}</div></div><div class="status-pill">{s_label}</div></div>{intraday_html}<div class="score-container"><div class="score-label">Score</div><div class="score-big">{conv_score:.0f}</div></div><div class="signal-badge" style="background-color: {sig_bg}; color: {sig_fg}; border: {sig_border};">{sig_val}</div><div class="stats-grid"><div class="stat-item"><div class="stat-lbl">SECTOR RS</div><div class="stat-val" style="color: {srs_color}; font-weight: 700;">{srs_val:+.1f}%</div></div><div class="stat-item"><div class="stat-lbl">STOP LOSS</div><div class="stat-val" style="color: #991b1b;">{stop_loss:.2f}</div></div><div class="stat-item"><div class="stat-lbl">PATTERN</div><div class="stat-val">{pat_consensus:.1f}%</div></div></div></div>'
                            # Clean HTML indentation and render
                            clean_card_html = textwrap.dedent(card_html).strip()
                            st.markdown(clean_card_html, unsafe_allow_html=True)
                            
                            with st.expander(f"Details: {row['Ticker']}", expanded=False):
                                st.write(f"✅ {row['Why']}")
                                if row['Warnings']: st.warning(row['Warnings'])
                                if user_api_key:
                                    if st.button(f"AI Plan: {row['Ticker']}", key=f"tab_unified_btn_{row['Ticker']}"):
                                        st.markdown(generate_ai_trading_plan(row['Ticker'], batch_df[batch_df['Ticker']==row['Ticker']].iloc[0], user_api_key))
                        
                        st.divider()
                        st.subheader("📋 ตารางสรุปรวม (Summary Table)")
                        # Safe Column Selection
                        u_cols = {
                            'Ticker': 'Ticker', 
                            'Conviction_Score': 'Score', 
                            'Signal': 'Signal', 
                            'Strategy': 'Strategy', 
                            'Sector_RS': 'Sector RS', 
                            'Stop_Loss': 'Stop Loss', 
                            'Similarity': 'Pattern'
                        }
                        unified_summary = top_conviction[[c for c in u_cols.keys() if c in top_conviction.columns]].copy()
                        unified_summary.rename(columns=u_cols, inplace=True)
                        st.dataframe(unified_summary, use_container_width=True)
                    
                    with st.expander("🔍 View All Unified Candidates", expanded=False):
                        st.dataframe(unified_df, use_container_width=True)
                else:
                    st.info("ℹ️ ไม่พบหุ้นที่เข้าเกณฑ์ Unified")
            else:
                st.warning("กรุณาทำการสแกนหุ้นก่อนเพื่อดูรายงาน Unified Report")
            
            # --- Bottom Fishing ---
            st.divider()
            st.info("💎 หุ้นที่ Oversold และเริ่มมีสัญญาณกลับตัว (Bottom Fishing)")
            st.caption("🎯 **Feature Insight:** ค้นหาหุ้นที่มี RSI ต่ำกว่า 35 และเริ่มมีแรงซื้อกลับ (RSI Turning Up) พร้อม Candlestick รูปแบบ Bullish Pin Bar เพื่อหาจังหวะต้นเทรนด์")
            
            # [HYBRID MANDATE] Use latest results from both sources
            hybrid_results = fetch_market_scan_results()
            
            if not hybrid_results.empty:
                # 1. Filtering for Oversold
                hybrid_results['rsi'] = pd.to_numeric(hybrid_results['rsi'], errors='coerce')
                oversold_df = hybrid_results[hybrid_results['rsi'] <= 35].copy()
                
                if not oversold_df.empty:
                    oversold_df = oversold_df.sort_values('rsi', ascending=True)
                    st.success(f"🎯 พบหุ้น Oversold (RSI <= 35) จำนวน {len(oversold_df)} ตัว")
                    
                    # Display cards
                    for idx, row in oversold_df.iterrows():
                        r_dot_color = "#8b5cf6" 
                        r_sig_bg = "#f5f3ff"; r_sig_fg = "#5b21b6"
                        
                        # Build dynamic reasons based on available data
                        reasons = []
                        rsi_val = safe_float(row.get('rsi', 50))
                        if rsi_val < 30: reasons.append("Extreme Oversold (RSI < 30)")
                        elif rsi_val <= 35: reasons.append("Oversold Zone (RSI <= 35)")
                        if row.get('is_pinbar'): reasons.append("Bullish Pin Bar Detected")
                        if row.get('signal') == 'BUY': reasons.append("Positive Buy Signal")
                        
                        reasons_html = "".join([f'<div style="font-size: 0.75rem; color: #5b21b6; margin-bottom: 2px;">• {reason}</div>' for reason in reasons])
                        
                        # Card Content
                        r_card_html = f"""
                        <div class="compact-card">
                            <div class="card-header">
                                <div class="header-left">
                                    <div class="dot-indicator" style="background-color: {r_dot_color};"></div>
                                    <div class="ticker-name">{row['ticker']}</div>
                                </div>
                                <div class="status-pill recovery">OVERSOLD</div>
                            </div>
                            <div class="score-container">
                                <div class="score-label">Score</div>
                                <div class="score-big">{int(row['score']) if pd.notna(row['score']) else 0}</div>
                            </div>
                            <div style="display: flex; flex-direction: column; gap: 4px;">
                                <div class="signal-badge" style="background-color: {r_sig_bg}; color: {r_sig_fg};">RSI: {row['rsi']:.1f}</div>
                                <div style="font-size: 0.75rem; font-weight: 600; color: #7c3aed;">Signal: {row['signal']}</div>
                                <div style="font-size: 0.75rem; font-weight: 600; color: #4b5563;">Strategy: {row['strategy'] if row['strategy'] else 'N/A'}</div>
                            </div>
                            <div style="margin-top: 10px; padding: 6px; background-color: #fdfcff; border-radius: 8px; border: 1px dashed #ddd6fe;">
                                {reasons_html}
                            </div>
                            <div class="stats-grid">
                                <div class="stat-item"><div class="stat-lbl">RSI</div><div class="stat-val">{row['rsi']:.1f}</div></div>
                                <div class="stat-item"><div class="stat-lbl">PRICE</div><div class="stat-val">{row['close_price']:.2f}</div></div>
                                <div class="stat-item"><div class="stat-lbl">PIN BAR</div><div class="stat-val">{'✅' if row.get('is_pinbar') else '❌'}</div></div>
                            </div>
                        </div>
                        """
                        # Clean HTML indentation and render
                        clean_r_card_html = textwrap.dedent(r_card_html).strip()
                        st.markdown(clean_r_card_html, unsafe_allow_html=True)
                        
                        with st.expander(f"Analysis: {row['ticker']}"):
                            st.write(f"🔍 **เหตุผลที่ติดโผ:** {', '.join(reasons)}")
                            if user_api_key:
                                if st.button(f"Oversold AI Plan: {row['ticker']}", key=f"tab_oversold_btn_{row['ticker']}"):
                                    dummy_row = {'Last Price': row['close_price'], 'Signal': row['signal'], 'Bullish Score (%)': row['score'], 'Bearish Score (%)': 0, 'Score Diff': row['score'], 'MTF Conf': 'N/A', 'MTF Score': 0, 'Relative Vol': 1.0, 'Pattern Consensus (%)': 50}
                                    st.markdown(generate_ai_trading_plan(row['ticker'], dummy_row, user_api_key))
                    
                    st.divider()
                    st.subheader("📋 ตารางสรุปหุ้น Oversold (Summary Table)")
                    summary_cols = ['ticker', 'rsi', 'close_price', 'signal', 'strategy', 'source', 'scanned_at']
                    st.dataframe(oversold_df[summary_cols].rename(columns={
                        'ticker': 'Ticker',
                        'rsi': 'RSI',
                        'close_price': 'Price',
                        'signal': 'Signal',
                        'strategy': 'Strategy',
                        'source': 'Source',
                        'scanned_at': 'Scanned At'
                    }), use_container_width=True)
                else:
                    st.info("ℹ️ ยังไม่พบหุ้น Oversold (RSI <= 35)")
            else:
                st.info("ℹ️ ยังไม่มีข้อมูลการสแกนในระบบ (กรุณากด Run SET100 Batch Scan หรือรอระบบ Auto Scan)")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดใน Unified Scanner: {e}")

elif main_category == "🌋 Market Insights & Analytics":
    sub_tabs = st.tabs(["💙 Silent Accumulation Scanner", "🌋 WVF Market Bottom Analysis", "📊 Market Breadth & Regime"])
    
    with sub_tabs[0]: # 💙 Silent Accumulation Scanner
        try:
            st.subheader("💙 Silent Accumulation Scanner")
            st.info("💙 **Silent Accumulation:** ตรวจจับหุ้นที่มีการสะสมของราคาอย่างเงียบเชียบ โดยมีลักษณะราคาบวกเล็กน้อย วอลุ่มลดลงหรือคงที่ และมีความเสี่ยงต่ำ (ATC Risk < 0.5%)")
            
            # 1. Historical Filter
            c1, c2 = st.columns([2, 3])
            sa_lookback_mode = c1.selectbox("เลือกช่วงเวลาย้อนหลัง", 
                                          ["Latest Scan Only", "Past 7 Days", "Past 30 Days", "All History"],
                                          key="sa_lookback_filter")
            
            # 2. Data Preparation
            sa_df = pd.DataFrame()
            if sa_lookback_mode == "Latest Scan Only":
                if not batch_df.empty:
                    sa_df = batch_df[batch_df['Signal'] == 'SILENT ACCUM'].copy()
                else:
                    st.info("ℹ️ ยังไม่มีข้อมูลการสแกนล่าสุด (กรุณากด Run SET100 Batch Scan ใน Sidebar)")
            else:
                days_map = {"Past 7 Days": 7, "Past 30 Days": 30, "All History": 365}
                lookback_days = days_map.get(sa_lookback_mode, 7)
                with st.spinner(f"⏳ Loading historical signals for {sa_lookback_mode}..."):
                    sa_df = fetch_historical_signals('SILENT ACCUM', lookback_days=lookback_days)
            
            if not sa_df.empty:
                # Calculate R:R Ratio (Defensive)
                def calc_rr(row):
                    jump = safe_float(row.get('Expected Jump (%)', 0))
                    drop = abs(safe_float(row.get('Expected Drop (%)', 1)))
                    return round(jump / drop if drop != 0 else 0, 2)
                
                sa_df['R:R Ratio'] = sa_df.apply(calc_rr, axis=1)
                
                # Format Numeric Columns for Display
                sa_df['Relative Vol'] = sa_df['Relative Vol'].apply(lambda x: round(safe_float(x), 2))
                sa_df['Conviction_Score'] = sa_df['Conviction_Score'].apply(lambda x: round(safe_float(x), 2))
                
                # Sort by Signal Date (Newest First) and Accumulation Score (Highest First)
                sa_df['Sort_Date'] = pd.to_datetime(sa_df['Last Update'], errors='coerce')
                sa_df = sa_df.sort_values(by=['Sort_Date', 'Conviction_Score'], ascending=[False, False])
                
                # Display metrics
                st.write(f"🔥 พบหุ้นเข้าเงื่อนไข Silent Accumulation ({sa_lookback_mode}) ทั้งหมด **{len(sa_df)}** ตัว")
                
                # Table display
                display_cols = ['Ticker', 'Last Update', 'Conviction_Score', 'Last Price', 'Relative Vol', 'R:R Ratio']
                st.dataframe(sa_df[display_cols].rename(columns={
                    'Ticker': 'Ticker',
                    'Last Update': 'Signal Date',
                    'Conviction_Score': 'Accumulation Score',
                    'Last Price': 'Close Price',
                    'Relative Vol': 'Volume Ratio (RV)',
                    'R:R Ratio': 'R:R Ratio'
                }), use_container_width=True, hide_index=True)
                
                # CSV Download
                csv_sa = sa_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Download Silent_Accum_Results.csv",
                    csv_sa,
                    f"Silent_Accum_{sa_lookback_mode.replace(' ', '_')}_{datetime.now(SET_TZ).strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key='download-sa-scanner-hist'
                )
            elif sa_lookback_mode != "Latest Scan Only":
                st.info(f"ℹ️ ไม่พบข้อมูลสัญญาณ Silent Accumulation ในช่วง {sa_lookback_mode}")
        except Exception as e:
            st.error(f"Error in Silent Accum Scanner: {e}")

    with sub_tabs[1]: # 🌋 WVF Market Bottom Analysis
        try:
            st.subheader("🌋 Market Bottom Analysis (Williams Vix Fix)")
            st.info("🌋 **Williams Vix Fix (WVF):** เครื่องมือจับจุดกลับตัวที่ฐาน (Market Bottom) โดยวัดความผันผวนของราคาเทียบกับ High ในรอบ Lookback หาก WVF พุ่งทะลุ Upper Bollinger Band จะเกิดสัญญาณ Climax Spike")
            
            # 1. WVF Control Panel
            with st.expander("⚙️ WVF Parameters & Control Panel", expanded=False):
                c1, c2, c3, c4, c5 = st.columns(5)
                wvf_lookback = c1.number_input("Lookback Period", 10, 100, 22)
                wvf_bb_len = c2.number_input("BB Length", 10, 100, 20)
                
                sensitivity_options = {
                    "High Sensitivity (BB StdDev = 1.2)": 1.2,
                    "Medium Sensitivity (BB StdDev = 1.5)": 1.5,
                    "Strict Climax (BB StdDev = 2.0)": 2.0
                }
                wvf_sensitivity = c3.selectbox("Sensitivity Level", list(sensitivity_options.keys()), index=1)
                wvf_bb_mult = sensitivity_options[wvf_sensitivity]
                
                wvf_percentile = c4.slider("Percentile High Threshold", 0.5, 0.99, 0.85, 0.05)
                
                scan_range_options = {
                    "วันล่าสุด (Latest Day)": 1,
                    "ย้อนหลัง 5 วันทำการ": 5,
                    "ย้อนหลัง 20 วันทำการ (1 เดือน)": 20,
                    "เลือกวันที่เจาะจง (Custom Date)": 0
                }
                wvf_scan_mode = c5.selectbox("ช่วงเวลาสแกน", list(scan_range_options.keys()))
                
                wvf_scan_days = scan_range_options[wvf_scan_mode]
                wvf_target_date = None
                if wvf_scan_mode == "เลือกวันที่เจาะจง (Custom Date)":
                    wvf_target_date = st.date_input("เลือกวันที่ต้องการสแกน", datetime.now(SET_TZ).date())
                
                wvf_scan_btn = st.button("🚀 Run WVF Market Scan", type="primary", use_container_width=True)
                
                st.divider()
                col_v1, col_v2 = st.columns(2)
                show_silent_accum = col_v1.checkbox("แสดงสัญญาณ Silent Accumulation บนกราฟ", value=True)
                wvf_panel_ratio = col_v2.slider("ปรับความสูงพาเนล WVF", 0.15, 0.6, 0.35, 0.05)

            # 2. WVF Scanner Table
            st.write(f"### 🔍 WVF Bottom Climax Scanner ({wvf_scan_mode})")
            
            tickers = SET100_TICKERS
            wvf_results = []
            
            if wvf_scan_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, ticker in enumerate(tickers):
                    status_text.text(f"Scanning {ticker}...")
                    df_wvf = get_stock_data(ticker)
                    if df_wvf is not None and len(df_wvf) > wvf_lookback:
                        df_wvf = calculate_wvf(df_wvf, wvf_lookback, wvf_bb_len, wvf_bb_mult, wvf_percentile)
                        
                        if wvf_target_date:
                            target_dt = pd.to_datetime(wvf_target_date).date()
                            matches = df_wvf[df_wvf.index.date == target_dt]
                            if not matches.empty and matches.iloc[0]['Is_WVF_Spike']:
                                row = matches.iloc[0]
                                wvf_results.append({
                                    'Ticker': ticker,
                                    'Signal Date': row.name.strftime('%Y-%m-%d'),
                                    'Days Ago': (datetime.now(SET_TZ).date() - row.name.date()).days,
                                    'Price': row['Close'],
                                    'WVF Value': round(row['WVF'], 2),
                                    'Upper BB': round(row['WVF_Upper'], 2),
                                    'Signal': '🌋 BOTTOM CLIMAX'
                                })
                        else:
                            recent_df = df_wvf.tail(wvf_scan_days)
                            spikes = recent_df[recent_df['Is_WVF_Spike']]
                            for date, row in spikes.iterrows():
                                wvf_results.append({
                                    'Ticker': ticker,
                                    'Signal Date': date.strftime('%Y-%m-%d'),
                                    'Days Ago': (datetime.now(SET_TZ).date() - date.date()).days,
                                    'Price': row['Close'],
                                    'WVF Value': round(row['WVF'], 2),
                                    'Upper BB': round(row['WVF_Upper'], 2),
                                    'Signal': '🌋 BOTTOM CLIMAX'
                                })
                    progress_bar.progress((i + 1) / len(tickers))
                
                status_text.empty()
                progress_bar.empty()
                
                if wvf_results:
                    st.session_state['wvf_scan_results'] = pd.DataFrame(wvf_results)
                    st.success(f"พบสัญญาณ WVF Climax ทั้งหมด {len(wvf_results)} จุด!")
                else:
                    st.session_state['wvf_scan_results'] = pd.DataFrame()
                    st.info(f"ไม่พบหุ้นที่เกิดสัญญาณ WVF Climax")

            if 'wvf_scan_results' in st.session_state and not st.session_state['wvf_scan_results'].empty:
                st.dataframe(st.session_state['wvf_scan_results'].sort_values('Signal Date', ascending=False), use_container_width=True)
            
            # Ticker Selector
            signaled_tickers = []
            if 'wvf_scan_results' in st.session_state and not st.session_state['wvf_scan_results'].empty:
                signaled_tickers = st.session_state['wvf_scan_results']['Ticker'].unique().tolist()
            
            sorted_tickers = signaled_tickers + [t for t in tickers if t not in signaled_tickers]
            ticker_display_map = {t: (f"🔥 {t} (Climax Signal)" if t in signaled_tickers else t) for t in sorted_tickers}
            
            selected_wvf_ticker = st.selectbox("เลือกหุ้นเพื่อวิเคราะห์ (Select Ticker to Analyze):", options=sorted_tickers, format_func=lambda x: ticker_display_map.get(x, x), key="wvf_ticker_selector_new")

            if selected_wvf_ticker:
                with st.spinner(f"Loading {selected_wvf_ticker} chart..."):
                    df_chart = get_stock_data(selected_wvf_ticker)
                    if df_chart is not None and len(df_chart) > wvf_lookback:
                        df_chart = calculate_wvf(df_chart, wvf_lookback, wvf_bb_len, wvf_bb_mult, wvf_percentile)
                        
                        # Spike First Trigger Logic
                        df_chart['Is_WVF_First_Trigger'] = False
                        last_wvf_idx = -10
                        for i in range(len(df_chart)):
                            if df_chart['Is_WVF_Spike'].iloc[i]:
                                if i - last_wvf_idx >= 5:
                                    df_chart.iloc[i, df_chart.columns.get_loc('Is_WVF_First_Trigger')] = True
                                    last_wvf_idx = i
                                    
                        df_plot = df_chart.tail(250).copy()
                        
                        # Silent Accum Integration
                        sa_insights = get_silent_accum_insights(ticker_filter=selected_wvf_ticker, deduplicate=False)
                        df_plot['Is_Silent_Accum'] = False
                        if sa_insights is not None and not sa_insights.empty:
                            sig_dates = pd.to_datetime(sa_insights['signal_date']).dt.date.unique().tolist()
                            df_plot.loc[[d.date() in sig_dates for d in df_plot.index], 'Is_Silent_Accum'] = True
                        
                        # --- TRADINGVIEW STYLE CHART ---
                        latest = df_plot.iloc[-1]
                        st.markdown(f"""
                            <div style='background-color: rgba(15, 23, 42, 0.9); padding: 12px; border-radius: 6px; border-left: 4px solid #089981; margin-bottom: 15px;'>
                                <span style='color: #94a3b8; font-size: 0.85rem;'>{latest.name.strftime('%d %b %Y')}</span>
                                <div style='margin-top: 5px; display: flex; gap: 15px; color: white;'>
                                    <span><b>O:</b> {latest['Open']:.2f}</span>
                                    <span><b>H:</b> {latest['High']:.2f}</span>
                                    <span><b>L:</b> {latest['Low']:.2f}</span>
                                    <span><b>C:</b> {latest['Close']:.2f}</span>
                                    <span style='color: #00FF00;'><b>WVF:</b> {latest['WVF']:.2f}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[1-wvf_panel_ratio, wvf_panel_ratio])
                        
                        # Candlestick
                        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price", increasing_line_color='#089981', decreasing_line_color='#F23645', increasing_fillcolor='#089981', decreasing_fillcolor='#F23645', hoverinfo='none'), row=1, col=1)
                        
                        # Markers (Size 14 Triangle-up)
                        wvf_spikes = df_plot[df_plot['Is_WVF_First_Trigger']]
                        fig.add_trace(go.Scatter(x=wvf_spikes.index, y=wvf_spikes['Low'] * 0.985, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00FF66', line=dict(width=1, color='black')), name='▲ WVF Climax Signal', hoverinfo='none'), row=1, col=1)
                        
                        if show_silent_accum:
                            sa_spikes = df_plot[df_plot['Is_Silent_Accum']]
                            fig.add_trace(go.Scatter(x=sa_spikes.index, y=sa_spikes['Low'] * 0.97, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#FF8C00', line=dict(width=1, color='black')), name='▲ Silent Accum Signal', hoverinfo='none'), row=1, col=1)
                        
                        # WVF Bars
                        colors = ['#00FF00' if spike else '#363A45' for spike in df_plot['Is_WVF_Spike']]
                        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['WVF'], marker_color=colors, name='WVF Value', showlegend=False, hoverinfo='none'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['WVF_Upper'], line=dict(color='rgba(173, 255, 47, 0.7)', width=1.5, dash='dash'), name='Upper BB', hoverinfo='none'), row=2, col=1)
                        
                        # Layout
                        fig.update_layout(height=800, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=50, b=10), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=0.98), hovermode="x", dragmode='pan')
                        fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, spikecolor='gray', spikedash='dash', matches='x', showgrid=True, gridcolor='rgba(128, 128, 128, 0.15)')
                        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeselector=dict(buttons=list([dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"), dict(count=1, label="1Y", step="year", stepmode="backward"), dict(step="all", label="ALL")]), bgcolor="rgba(54, 58, 69, 0.8)", activecolor="#089981", y=1.02, x=0.01), range=[df_plot.index[-120], df_plot.index[-1]], row=1, col=1)
                        fig.update_yaxes(autorange="reversed", fixedrange=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.15)', row=2, col=1)
                        fig.update_yaxes(fixedrange=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.15)', row=1, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'responsive': True, 'displayModeBar': False})
        except Exception as e:
            st.error(f"Error in WVF Analysis: {e}")

    with sub_tabs[2]: # 📊 Market Breadth & Regime
        try:
            st.subheader(f"📊 Market Breadth: หุ้นบวก {pos_count} | หุ้นลบ {neg_count}")
            st.caption("📈 **Market Breadth:** สรุปภาพรวมความแข็งแกร่งของตลาด SET100")
            
            if not batch_df.empty and 'Signal' in batch_df.columns:
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    st.write("### Signal Distribution")
                    sig_counts = batch_df['Signal'].value_counts().reset_index()
                    sig_counts.columns = ['Signal', 'Count']
                    st.dataframe(sig_counts, use_container_width=True, hide_index=True)
                with c_m2:
                    st.write("### Regime Analysis")
                    if 'Regime' in batch_df.columns:
                        st.info(f"Current Market Regime: **{batch_df['Regime'].iloc[0]}**")
                    else:
                        st.info("Regime data not available.")
        except Exception as e:
            st.error(f"Error in Market Breadth: {e}")

elif main_category == "🧪 Quant Lab & Administration":
    sub_tabs = st.tabs(["🧪 Backtest & Optimizer Engine", "⚙️ Admin & History Logs"])
    
    with sub_tabs[0]: # 🧪 Backtest & Optimizer Engine
        try:
            st.subheader("🧪 WVF Strategy Backtest & Grid Search Optimizer")
            st.info("🧪 **Backtest Engine:** ทดสอบย้อนหลังและหาค่าพารามิเตอร์ที่เหมาะสมที่สุด (Optimization) สำหรับกลยุทธ์ WVF Bottom Climax โดยไม่มี Look-ahead bias")
            
            # 1. Selection & Parameters
            with st.expander("⚙️ Backtest Settings & Optimizer Grid", expanded=True):
                col_bt1, col_bt2, col_bt3 = st.columns(3)
                
                with col_bt1:
                    st.markdown("### 📊 Asset & Capital")
                    bt_ticker = st.selectbox("Select Stock for Backtest", st.session_state.get('set100_tickers', ["ADVANC.BK"]), key="bt_ticker_new")
                    initial_cap = st.number_input("Initial Capital (THB)", 10000, 1000000, 100000, 10000)
                    comm_rate = st.number_input("Commission (%)", 0.0, 1.0, 0.157, 0.01) / 100
                    slip_rate = st.number_input("Slippage (%)", 0.0, 1.0, 0.10, 0.01) / 100
                    
                with col_bt2:
                    st.markdown("### 🌋 WVF Parameters")
                    lookback_range = st.slider("Lookback Period Range", 10, 60, (20, 30), 2)
                    bb_mult_range = st.slider("BB StdDev Range", 1.0, 3.0, (1.5, 2.5), 0.25)
                    st.divider()
                    st.markdown("### 🔍 Entry Filters")
                    use_trend_filter = st.checkbox("Enable Trend Filter (Price > EMA)", value=True, key="bt_trend_filter")
                    ema_trend_val = st.number_input("EMA Period", 20, 250, 200)
                    
                with col_bt3:
                    st.markdown("### 🚪 Exit Strategy")
                    exit_choice = st.selectbox("Exit Rule", [
                        "StopLoss_TakeProfit", 
                        "RSI_Overbought", 
                        "Trailing_Stop",
                        "Fixed Holding Days",
                        "Indicator Exit (RSI/BB)"
                    ])
                    
                    if exit_choice == "Fixed Holding Days":
                        exit_val = st.slider("Days to Hold", 1, 30, 10)
                        exit_type = 'days'
                    elif exit_choice == "StopLoss_TakeProfit":
                        tp = st.slider("Take Profit (%)", 1.0, 30.0, 10.0) / 100
                        sl = st.slider("Stop Loss (%)", 1.0, 20.0, 5.0) / 100
                        exit_type = 'StopLoss_TakeProfit'
                        exit_val = (tp, sl)
                    elif exit_choice == "RSI_Overbought":
                        exit_val = st.slider("RSI Exit Level", 50, 90, 65)
                        exit_type = 'RSI_Overbought'
                    elif exit_choice == "Trailing_Stop":
                        exit_val = st.slider("ATR Multiplier", 1.0, 5.0, 2.0, 0.5)
                        exit_type = 'Trailing_Stop'
                    else:
                        exit_val = 0
                        exit_type = 'indicator_exit'

                st.divider()
                col_btn1, col_btn2 = st.columns(2)
                run_bt = col_btn1.button("🚀 Run Single Backtest", use_container_width=True, type="primary")
                run_opt = col_btn2.button("🔍 Run Grid Search Optimizer", use_container_width=True)

            # 2. Execution Logic
            if run_bt or run_opt:
                df_bt = get_stock_data(bt_ticker)
                if df_bt is not None and not df_bt.empty:
                    if run_bt:
                        params = {
                            'initial_capital': initial_cap,
                            'commission': comm_rate,
                            'slippage': slip_rate,
                            'lookback': lookback_range[0] if isinstance(lookback_range, tuple) else lookback_range,
                            'bb_mult': bb_mult_range[0] if isinstance(bb_mult_range, tuple) else bb_mult_range,
                            'use_trend_filter': use_trend_filter,
                            'ema_trend_period': ema_trend_val,
                            'exit_type': exit_type,
                            'exit_value': exit_val if exit_type in ['days', 'RSI_Overbought', 'Trailing_Stop'] else 0
                        }
                        if exit_type == 'StopLoss_TakeProfit':
                            params['take_profit'] = tp
                            params['stop_loss'] = sl
                        elif exit_type == 'RSI_Overbought':
                            params['rsi_exit'] = exit_val
                        elif exit_type == 'Trailing_Stop':
                            params['atr_mult'] = exit_val
                        
                        with st.spinner("Running backtest..."):
                            results = backtest_engine.run_wvf_backtest(df_bt, params)
                        
                        if results:
                            summary = results['summary']
                            st.write("### 📈 Performance Summary")
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("Net Profit", f"{summary.get('Net Profit (%)', 0)}%")
                            m2.metric("Win Rate", f"{summary.get('Win Rate (%)', 0)}%")
                            m3.metric("Max Drawdown", f"{summary.get('Max Drawdown (%)', 0)}%")
                            m4.metric("Sharpe Ratio", summary.get('Sharpe Ratio', 0))
                            m5.metric("Total Trades", summary.get('Total Trades', 0))
                            
                            st.write("### 📊 Equity Curve")
                            fig_equity = go.Figure()
                            fig_equity.add_trace(go.Scatter(x=results['equity_curve'].index, y=results['equity_curve']['Equity'], name='Strategy Equity', line=dict(color='#00FF66')))
                            fig_equity.update_layout(template='plotly_dark', height=400, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig_equity, use_container_width=True)
                            
                            st.write("### 📜 Trade Execution Log")
                            st.dataframe(results['trade_log'], use_container_width=True)
                            csv_bt = results['trade_log'].to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Trade Log (CSV)", csv_bt, f"trades_{bt_ticker}.csv", "text/csv")
                    
                    elif run_opt:
                        grid = {
                            'lookback': list(range(lookback_range[0], lookback_range[1] + 1, 2)),
                            'bb_mult': list(np.arange(bb_mult_range[0], bb_mult_range[1] + 0.1, 0.25)),
                            'use_trend_filter': [use_trend_filter],
                            'ema_trend_period': [ema_trend_val],
                            'exit_type': [exit_type]
                        }
                        if exit_type == 'StopLoss_TakeProfit':
                            grid['stop_loss'] = [0.03, 0.05, 0.07]
                            grid['take_profit'] = [0.08, 0.12, 0.15]
                        elif exit_type == 'RSI_Overbought':
                            grid['rsi_exit'] = [60, 65, 70]
                        elif exit_type == 'Trailing_Stop':
                            grid['atr_mult'] = [1.5, 2.0, 2.5, 3.0]
                        
                        with st.spinner(f"Optimizing strategy..."):
                            opt_results = backtest_engine.optimize_wvf_strategy(df_bt, grid)
                        
                        if opt_results is not None and not opt_results.empty:
                            st.write("### 🏆 Optimization Results (Ranked by Sharpe)")
                            st.dataframe(opt_results.head(10), use_container_width=True)
                            best = opt_results.iloc[0]
                            st.success(f"✅ Best Parameters: Sharpe={best['Sharpe']} | Profit: {best['Net Profit (%)']}% | Trades: {best['Trades']}")
                            
                            # Add Save Button
                            best_params_to_save = {
                                'lookback': int(best['lookback']),
                                'bb_mult': float(best['bb_mult']),
                                'use_trend_filter': bool(best['use_trend_filter']),
                                'ema_trend_period': int(best['ema_trend_period']),
                                'exit_type': str(best['exit_type'])
                            }
                            # Add exit values based on type
                            if best['exit_type'] == 'StopLoss_TakeProfit':
                                best_params_to_save['take_profit'] = float(best['take_profit'])
                                best_params_to_save['stop_loss'] = float(best['stop_loss'])
                            elif best['exit_type'] == 'RSI_Overbought':
                                best_params_to_save['rsi_exit'] = float(best['rsi_exit'])
                            elif best['exit_type'] == 'Trailing_Stop':
                                best_params_to_save['atr_mult'] = float(best['atr_mult'])
                            
                            if st.button("💾 Apply & Save these Best Parameters for Signal Engine", type="primary"):
                                save_best_params(bt_ticker, best_params_to_save)
                                st.success(f"🚀 Parameters for {bt_ticker} saved successfully! Signal Engine will now use these settings.")
                else:
                    st.error("ไม่สามารถโหลดข้อมูลหุ้นสำหรับการทดสอบได้")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในระบบ Backtest: {e}")

    with sub_tabs[1]: # ⚙️ Admin & History Logs
        try:
            st.subheader("⚙️ Admin & History Logs")
            st.caption("📜 **Database Persistence:** ดึงข้อมูลประวัติการสแกนและผลแพ้ชนะย้อนหลังจาก Supabase")
            
            if supabase:
                with st.expander("🔍 View Saved Scan History", expanded=True):
                    response = supabase.table("scan_results").select("*").order("id", desc=True).limit(200).execute()
                    history_df = pd.DataFrame(response.data)
                    if not history_df.empty:
                        st.dataframe(history_df, use_container_width=True)
                    else:
                        st.info("No history records found.")
                
                st.divider()
                st.subheader("🛠️ Maintenance Tools")
                col_adm1, col_adm2 = st.columns(2)
                if col_adm1.button("🏷️ Run Automated Labeling", use_container_width=True):
                    with st.spinner("Updating labels..."):
                        count = run_automated_labeling()
                        st.success(f"Updated {count} records!")
                if col_adm2.button("🧹 Clear Local Cache", use_container_width=True):
                    st.cache_data.clear()
                    st.success("Local cache cleared!")
        except Exception as e:
            st.error(f"Error in Admin Tools: {e}")


# --- End of Application ---
