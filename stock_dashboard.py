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
from scanner_engine import (
    calculate_quant_indicators, 
    get_pre_breakout_scanner, 
    get_recovery_signals, 
    core_strategy_scanner,
    calculate_conviction_score,
    get_market_regime as get_engine_market_regime,
    validate_scanner_accuracy,
    get_mtf_confluence,
    get_signal_performance_stats
)

# Load .env for local development
load_dotenv()

# --- Configuration ---
SET_TZ = pytz.timezone('Asia/Bangkok')
st.set_page_config(page_title="Quant Strategy Station", layout="wide")

# --- Password Protection ---
def check_password():
    # 1. ถ้าผ่านการล็อกอินอยู่แล้ว ให้ให้ผ่านทันที
    if st.session_state.get("password_correct", False):
        return True

    # 2. ฟังก์ชันตรวจสอบเมื่อมีการกด Enter/ส่งข้อมูลในช่องรหัสผ่าน
    def password_entered():
        user_input = str(st.session_state.get("password_input", "")).strip()
        target_password = str(st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", "admin1234"))).strip()
        
        if user_input == target_password:
            st.session_state["password_correct"] = True
            st.session_state["password_error"] = False
        else:
            st.session_state["password_correct"] = False
            st.session_state["password_error"] = True

    # 3. แสดงฟอร์มกรอกรหัสผ่าน
    st.text_input(
        "Please enter the access password",
        type="password",
        on_change=password_entered,
        key="password_input"
    )
    
    # 4. แสดง Error Message เฉพาะกรณีที่พิมพ์ผิดจริงๆ เท่านั้น
    if st.session_state.get("password_error", False):
        st.error("😕 Password incorrect")
        
    return False

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
                'volume': 'Relative Vol' 
            })
            
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
    cols = ['ticker', 'signal', 'score', 'strategy', 'close_price', 'change_percent', 'rsi', 'scanned_at', 'source', 'is_pinbar', 'is_silent_accum']
    
    def clean_df(df):
        if df.empty:
            return pd.DataFrame(columns=cols)
        # 1. Clean Duplicate Columns
        df = df.loc[:, ~df.columns.duplicated()].copy()
        # 2. Reset Index
        df.reset_index(drop=True, inplace=True)
        # 3. Ensure 'score' exists
        if 'score' not in df.columns:
            df['score'] = df.get('conviction_score', df.get('bull_score'))
        
        # 4. Fill missing columns
        for c in cols:
            if c not in df.columns:
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
    """
    Fetch historical stock data with robust isolation and fallback.
    Ensures the data returned is strictly for the requested ticker and has a clean Date index.
    """
    if not ticker:
        return None
        
    # Clean ticker: Strip spaces, ensure .BK suffix for Thai stocks if missing
    clean_ticker = ticker.strip().upper()
    if not clean_ticker.endswith('.BK') and not clean_ticker.startswith('^'):
        clean_ticker = f"{clean_ticker}.BK"
        
    try:
        # 1. Try YahooQuery with strict symbol filtering
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        t = yq.Ticker(clean_ticker, session=session)
        df = t.history(start="2018-01-01")
        
        if df is not None and not df.empty:
            # Handle YahooQuery's MultiIndex or SingleIndex return
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            else:
                df = df.reset_index()
                
            # STRICT FILTERING: Ensure we only have the requested ticker's data
            # YahooQuery sometimes returns other symbols if passed a list or via internal mapping
            if "symbol" in df.columns:
                df = df[df["symbol"] == clean_ticker].copy()
            
            if not df.empty:
                # Standardize columns and index
                # YahooQuery uses lowercase column names: [date, symbol, close, volume, open, high, low, ...]
                col_map = {
                    "close": "Close", 
                    "volume": "Volume", 
                    "open": "Open", 
                    "high": "High", 
                    "low": "Low"
                }
                
                # Verify required columns exist
                available_cols = [c for c in col_map.keys() if c in df.columns]
                if "date" in df.columns and len(available_cols) >= 3:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index("date")
                    df = df[available_cols].rename(columns=col_map)
                    
                    # Sort and drop duplicates in index
                    df = df.sort_index()
                    df = df[~df.index.duplicated(keep='last')]
                    
                    # Ensure index is Naive Bangkok Time
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert(SET_TZ).tz_localize(None)
                    else:
                        df.index = df.index.tz_localize(None)
                        
                    return df

        # 2. Fallback to yfinance if YahooQuery fails or returns invalid data
        print(f"⚠️ YahooQuery fallback for {clean_ticker}...")
        yf_ticker = yf.Ticker(clean_ticker)
        # Fetch longer period to ensure we have enough data for indicators
        df_yf = yf_ticker.history(period="max") 
        
        if df_yf is not None and not df_yf.empty:
            # Ensure standard OHLCV column names and drop extra columns like Dividends
            needed = ["Open", "High", "Low", "Close", "Volume"]
            df_yf = df_yf[[c for c in needed if c in df_yf.columns]].copy()
            
            # Sort and deduplicate
            df_yf = df_yf.sort_index()
            df_yf = df_yf[~df_yf.index.duplicated(keep='last')]
            
            # Ensure index is naive datetime
            if df_yf.index.tz is not None:
                df_yf.index = df_yf.index.tz_convert(SET_TZ).tz_localize(None)
            else:
                df_yf.index = df_yf.index.tz_localize(None)
                
            return df_yf
            
        return None
    except Exception as e:
        print(f"Critical Error fetching {clean_ticker}: {e}")
        return None
    except Exception as e:
        print(f"❌ Critical error fetching data for {ticker}: {e}")
        return None


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
        similarity = row.get('Pattern Consensus (%)', 0)
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
            'Intraday_History': list(past_signals),
            'Sector_RS': sector_rs
        })
    
    # Sort by: 1. Signal Tier (1 is highest), 2. Conviction Score
    return pd.DataFrame(report_data).sort_values(['Signal_Tier', 'Conviction_Score'], ascending=[True, False])


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
    status_text.text("🔄 Initializing Sector Information...")
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        bulk_t = yq.Ticker(tickers, session=session)
        all_profiles = bulk_t.summary_profile
    except:
        all_profiles = {}
    
    # Get performance stats once for scoring
    perf_stats = get_signal_performance_stats(supabase)
    pos_count = 0
    neg_count = 0
    success_count = 0
    
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

        # --- MAIN UI TABS ---
        main_tabs = st.tabs([
            "🚀 Unified Report", 
            "💎 Bottom Fishing", 
            "📊 Market Breadth", 
            "📜 Admin & History", 
            "💎 SILENT ACCUM Insight",
            "📊 Market Scan Results (SET100)",
            "🛠️ Advanced Tools / More Features"
        ])
        
        with main_tabs[0]: # Unified Report
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
                            pat_consensus = row.get('Pattern Consensus (%)', 0)
                            
                            card_html = f'<div class="compact-card"><div class="card-header"><div class="header-left"><div class="dot-indicator" style="background-color: {dot_color};"></div><div class="ticker-name">{row["Ticker"]}</div></div><div class="status-pill">{s_label}</div></div>{intraday_html}<div class="score-container"><div class="score-label">Score</div><div class="score-big">{row["Conviction_Score"]}</div></div><div class="signal-badge" style="background-color: {sig_bg}; color: {sig_fg}; border: {sig_border};">{sig_val}</div><div class="stats-grid"><div class="stat-item"><div class="stat-lbl">SECTOR RS</div><div class="stat-val" style="color: {srs_color}; font-weight: 700;">{srs_val:+.1f}%</div></div><div class="stat-item"><div class="stat-lbl">STOP LOSS</div><div class="stat-val" style="color: #991b1b;">{stop_loss:.2f}</div></div><div class="stat-item"><div class="stat-lbl">PATTERN</div><div class="stat-val">{pat_consensus:.1f}%</div></div></div></div>'
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
        
        with main_tabs[1]: # Bottom Fishing
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
                        if row['rsi'] < 30: reasons.append("Extreme Oversold (RSI < 30)")
                        elif row['rsi'] <= 35: reasons.append("Oversold Zone (RSI <= 35)")
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
        
        with main_tabs[2]: # Market Breadth
            st.subheader(f"📊 Market Breadth: หุ้นบวก {pos_count} | หุ้นลบ {neg_count}")
            st.caption("📈 **Market Breadth:** สรุปภาพรวมความแข็งแกร่งของตลาด SET100 ผ่านจำนวนหุ้นที่บวกและลบ เพื่อดูทิศทางกระแสเงินทุน (Money Flow)")
            
            # --- SIGNALS BY CATEGORY ---
            if 'Signal' in batch_df.columns:
                st.divider()
                st.markdown("### 📊 Signals by Category")
                st.caption("🌐 **Sector Relative Strength (SRS):** วิเคราะห์เปรียบเทียบหุ้นกับค่าเฉลี่ยของกลุ่มอุตสาหกรรม เพื่อหาหุ้นที่ 'แข็งแกร่งกว่าตลาด' (Outperformer)")
                sig_counts = batch_df['Signal'].value_counts().reset_index()
                sig_counts.columns = ['Signal', 'Count']
                
                # Use small columns for a compact overview
                num_cols = min(len(sig_counts), 6)
                s_cols = st.columns(num_cols)
                for i, (_, s_row) in enumerate(sig_counts.iterrows()):
                    s_cols[i % num_cols].metric(s_row['Signal'], s_row['Count'])
                
                st.dataframe(sig_counts, use_container_width=True)
                
                # --- SILENT ACCUM CLUSTER DETECTION ---
                sa_count = sig_counts[sig_counts['Signal'] == 'SILENT ACCUM']['Count'].values[0] if 'SILENT ACCUM' in sig_counts['Signal'].values else 0
                if sa_count >= 3:
                    st.info(f"🔵 **Smart Money Accumulation Cluster Detected!**  \nพบหุ้น SET100 ติดสัญญาณ `SILENT ACCUM` พร้อมกัน **{sa_count} ตัว**  \n*แนวโน้ม: ตลาดมีโอกาสเกิด Reversal ขาขึ้นในระยะสั้น (Confidence: High)*")
                    st.caption("💡 **Feature Insight:** ระบบตรวจพบการเก็บของพร้อมกันในหลายตัว (Cluster) ซึ่งเป็นสัญญาณบ่งชี้ Market Breadth ว่าเงินทุนกำลังไหลเข้าสะสมหุ้นในกลุ่ม SET100")

        with main_tabs[3]: # Admin & History
            st.subheader("🏆 Leaderboard & History (Supabase)")
            st.caption("📜 **Data Persistence:** ดึงข้อมูลประวัติการสแกนและผลแพ้ชนะ (Win/Loss) ย้อนหลังโดยตรงจากฐานข้อมูล Cloud (Supabase)")
            # Sorting desired columns to the front
            cols = batch_df.columns.tolist()
            desired_order = ["Ticker", "Signal", "Pattern Consensus (%)", "Last Price", "Day High"]
            actual_order = [c for c in desired_order if c in batch_df.columns]
            display_df = batch_df[actual_order + [c for c in cols if c not in actual_order]]
            
            with st.expander("🔍 View Scanner Leaderboard (Table)", expanded=True):
                st.dataframe(display_df, use_container_width=True)
            
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
                        # Load labeled data for metrics
                        labeled_df = pd.DataFrame()
                        try:
                            l_resp = supabase.table("scan_results").select("*").not_.is_("outcome_label", "null").execute()
                            labeled_df = pd.DataFrame(l_resp.data)
                        except: pass
                        
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

            styled_export = style_batch(display_df.style)
            html_buffer = styled_export.to_html()
            
            ex1.checkbox("📸 Full-Length View (For PDF)", key="admin_full_view")
            
            full_html = f"<html><body><h2>🏆 SET100 Quant Report</h2>{html_buffer}</body></html>"
            ex2.download_button("📄 Download HTML Report", data=full_html, file_name=f"SET100_Report_{datetime.now(SET_TZ).strftime('%Y%m%d_%H%M')}.html", mime="text/html", use_container_width=True)
            
            csv = display_df.to_csv(index=False).encode('utf-8-sig')
            ex3.download_button("Excel/CSV Export", data=csv, file_name=f"SET100_Data_{datetime.now(SET_TZ).strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", use_container_width=True)

        with main_tabs[4]: # SILENT ACCUM Insight
            st.info("💎 เจาะลึกพฤติกรรมหุ้น SILENT ACCUM: วัดระยะเวลาการฟื้นตัวและโอกาสชนะ")
            st.caption("📈 **Feature Insight:** วิเคราะห์สถิติย้อนหลังของสัญญาณ SILENT ACCUM เพื่อหาค่าเฉลี่ยจำนวนวันที่ราคามักจะ 'ระเบิด' (Days to Move) และอัตราการชนะ (Win Rate) ภายใน 5 วัน")
            
            # 0. Display Control
            row_limit = st.slider("จำนวนรายการที่แสดงผลล่าสุด", min_value=10, max_value=200, value=30, step=10, key="sa_row_limit")
            
            sa_data = get_silent_accum_insights(limit=row_limit)
            
            if sa_data is not None and not sa_data.empty:
                # 1. Overview Metrics
                # Only calculate metrics for rows that have days_to_move (historical)
                hist_sa = sa_data[sa_data['days_to_move'].notna()]
                
                if not hist_sa.empty:
                    avg_days = hist_sa['days_to_move'].mean()
                    win_rate_t5 = (hist_sa['win_t5'].sum() / len(hist_sa)) * 100
                else:
                    avg_days = 0
                    win_rate_t5 = 0
                    
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
                st.write(f"### 📜 Recent SILENT ACCUM Cases (Top {row_limit})")
                st.dataframe(
                    sa_data[['ticker', 'signal_date', 'score', 'days_to_move', 'max_gain_t5']]
                    .style.format({
                        'max_gain_t5': '{:.2f}%',
                        'days_to_move': '{:.0f}',
                        'score': '{:.1f}'
                    }, na_rep='Pending'), 
                    use_container_width=True
                )

                st.divider()
                # 4. Single Ticker Analysis
                st.divider()
                st.write("### 🔍 SILENT ACCUM Single Ticker Analysis")
                
                # Fetch all unique tickers that have SILENT ACCUM signals to populate selectbox
                all_sa_tickers = []
                if sa_data is not None and not sa_data.empty:
                    all_sa_tickers = sorted(sa_data['ticker'].unique().tolist())
                
                sel_sa_ticker = st.selectbox("เลือกหุ้นเพื่อดูประวัติ SILENT ACCUM รายตัว", all_sa_tickers, key="sa_ticker_select_new")
                
                if sel_sa_ticker:
                    # Fetch full historical analysis for THIS specific ticker (Intraday Timeline)
                    with st.spinner(f"กำลังวิเคราะห์ประวัติ SILENT ACCUM สำหรับ {sel_sa_ticker}..."):
                        # [INTRADAY TIMELINE] Set deduplicate=False to see all signals for this ticker
                        ticker_sa = get_silent_accum_insights(limit=None, ticker_filter=sel_sa_ticker, deduplicate=False)
                    
                    if ticker_sa is not None and not ticker_sa.empty:
                        # 1. Price Chart with SILENT ACCUM Markers (Full Width)
                        st.write(f"**Price Chart with SILENT ACCUM Markers: {sel_sa_ticker}**")
                        # ... (Rest of the chart logic stays same as it uses normalized dates for markers)
                        # ...
                        # (I will keep the chart code as it was in my previous successful edit)
                        with st.spinner(f"ดึงข้อมูลกราฟสำหรับ {sel_sa_ticker}..."):
                            hist_price_raw = get_stock_data(sel_sa_ticker)
                            
                            if hist_price_raw is not None and not hist_price_raw.empty:
                                # Standardize for plotting (Last 180 days)
                                df_plot = hist_price_raw.tail(180).copy()
                                # Ensure index is naive datetime for Plotly and marker alignment
                                if df_plot.index.tz is not None:
                                    df_plot.index = df_plot.index.tz_convert(SET_TZ).tz_localize(None)
                                
                                # 1. Create Subplots: Price (Candlestick) + Volume
                                fig = make_subplots(
                                    rows=2, cols=1, 
                                    shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    row_heights=[0.7, 0.3]
                                )
                                
                                # Candlestick
                                fig.add_trace(go.Candlestick(
                                    x=df_plot.index, 
                                    open=df_plot['Open'], 
                                    high=df_plot['High'], 
                                    low=df_plot['Low'], 
                                    close=df_plot['Close'], 
                                    name='Price'
                                ), row=1, col=1)
                                
                                # Volume
                                fig.add_trace(go.Bar(
                                    x=df_plot.index, 
                                    y=df_plot['Volume'], 
                                    name='Volume', 
                                    marker_color='rgba(100, 100, 100, 0.5)'
                                ), row=2, col=1)
                                
                                # 2. Add SILENT ACCUM Markers (Pin to Low Price)
                                # Alignment: Convert signal dates to naive date objects for comparison
                                sig_dates = pd.to_datetime(ticker_sa['signal_date']).dt.date.unique().tolist()
                                df_plot_dates = df_plot.index.date
                                
                                # Filter rows in df_plot that match a signal date
                                marker_mask = [d in sig_dates for d in df_plot_dates]
                                markers = df_plot[marker_mask].copy()
                                
                                if not markers.empty:
                                    fig.add_trace(go.Scatter(
                                        x=markers.index, 
                                        y=markers['Low'] * 0.98, 
                                        mode='markers', 
                                        marker=dict(
                                            symbol='triangle-up', 
                                            size=15, 
                                            color='#3b82f6', 
                                            line=dict(width=2, color='white')
                                        ), 
                                        name='SILENT ACCUM Signal', 
                                        hovertemplate='<b>SILENT ACCUM</b><br>Date: %{x}<br>Price: %{y:.2f}'
                                    ), row=1, col=1)
                                
                                # Final Layout Update
                                fig.update_layout(
                                    height=650, 
                                    margin=dict(t=30, b=30, l=30, r=30), 
                                    template='plotly_dark', 
                                    xaxis_rangeslider_visible=False, 
                                    showlegend=True, 
                                    legend=dict(
                                        orientation="h", 
                                        yanchor="bottom", 
                                        y=1.02, 
                                        xanchor="right", 
                                        x=1
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"⚠️ ไม่สามารถดึงข้อมูลราคาของ {sel_sa_ticker} จาก Yahoo Finance ได้ในขณะนี้ กรุณาลองใหม่อีกครั้งหรือตรวจสอบ Ticker")
                        
                        # 2. Intraday Signal History Table (Below Chart)
                        st.write(f"**Intraday Signal History: {sel_sa_ticker}**")
                        st.caption("🕒 **Intraday Tracking:** แสดงประวัติสัญญาณทุกรอบเวลาที่เกิดขึ้น (Debounced 15-min)")
                        
                        # Prepare display dataframe
                        display_sa = ticker_sa[['signal_date', 'signal_time', 'score', 'scan_type', 'days_to_move', 'max_gain_t5']].copy()
                        display_sa = display_sa.rename(columns={
                            'signal_date': 'Date',
                            'signal_time': 'Time',
                            'score': 'Score',
                            'scan_type': 'Type',
                            'days_to_move': 'Days to Move',
                            'max_gain_t5': 'Max Gain (T+5)'
                        })
                        
                        st.dataframe(
                            display_sa.style.format({
                                'Max Gain (T+5)': '{:.2f}%',
                                'Days to Move': '{:.0f}',
                                'Score': '{:.1f}'
                            }, na_rep='-'),
                            use_container_width=True
                        )
                    else:
                        st.info(f"ไม่พบประวัติสัญญาณ SILENT ACCUM สำหรับ {sel_sa_ticker} ในช่วง 90 วันที่ผ่านมา")

            else:
                st.warning("ยังไม่มีข้อมูล SILENT ACCUM เพียงพอสำหรับการวิเคราะห์")


        with main_tabs[5]: # Market Scan Results (Hybrid)
            st.info("📊 Market Scan Results (SET100)")
            st.caption("🕒 **Hybrid View:** แสดงผลการวิเคราะห์ล่าสุดของหุ้นแต่ละตัว โดยรวมข้อมูลจากทั้งการสแกนอัตโนมัติ (Auto) และการสแกนด้วยตนเอง (Manual)")
            
            combined_df = fetch_market_scan_results()
            
            if not combined_df.empty:
                # 1. Metric Summary
                last_scan = combined_df['scanned_at'].max()
                buy_count = len(combined_df[combined_df['signal'] == 'BUY'])
                wait_count = len(combined_df[combined_df['signal'].str.contains('WAIT', na=False)])
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Stocks", len(combined_df))
                m2.metric("BUY Signals", buy_count)
                m3.metric("WAIT Signals", wait_count)
                m4.metric("Latest Update", last_scan.strftime("%H:%M:%S"))
                
                st.divider()
                
                # 2. Filters
                f1, f2, f3 = st.columns([1, 1, 2])
                sig_options = ["ALL"] + sorted(combined_df['signal'].dropna().unique().tolist())
                sel_sig = f1.selectbox("Filter Signal", sig_options, key="mkt_sig_filter")
                
                strat_options = ["ALL"] + sorted(combined_df['strategy'].dropna().unique().tolist())
                sel_strat = f2.selectbox("Filter Strategy", strat_options, key="mkt_strat_filter")
                
                search_ticker = f3.text_input("🔍 Ticker Search", "", key="mkt_ticker_search").upper()
                
                # Apply Filters
                filtered_mkt = combined_df.copy()
                if sel_sig != "ALL":
                    filtered_mkt = filtered_mkt[filtered_mkt['signal'] == sel_sig]
                if sel_strat != "ALL":
                    filtered_mkt = filtered_mkt[filtered_mkt['strategy'] == sel_strat]
                if search_ticker:
                    filtered_mkt = filtered_mkt[filtered_mkt['ticker'].str.contains(search_ticker)]
                
                # 3. Display Dataframe with Styling
                def style_mkt_scan(styler):
                    def highlight_buy(row):
                        return ['background-color: rgba(34, 197, 94, 0.15)' if row['signal'] == 'BUY' else '' for _ in row]
                    
                    styler.apply(highlight_buy, axis=1)
                    styler.format({
                        'price': '{:.2f}',
                        'close_price': '{:.2f}',
                        'change_percent': '{:+.2f}%',
                        'score': '{:.1f}',
                        'bull_score': '{:.1f}',
                        'rsi': '{:.1f}',
                        'volume': '{:,.0f}'
                    }, na_rep='N/A')
                    return styler

                st.subheader(f"📋 Market Results ({len(filtered_mkt)} stocks)")
                if not filtered_mkt.empty:
                    # Map price if needed
                    if 'price' in filtered_mkt.columns:
                        filtered_mkt['close_price'] = filtered_mkt['price']
                        
                    # Reorder columns for readability
                    display_cols = [
                        'ticker', 'signal', 'score', 'strategy', 'close_price', 
                        'change_percent', 'rsi', 'volume', 'source', 'scanned_at'
                    ]
                    actual_display = [c for c in display_cols if c in filtered_mkt.columns]
                    st.dataframe(style_mkt_scan(filtered_mkt[actual_display].style), use_container_width=True)
                else:
                    st.warning("ไม่พบข้อมูลตามเงื่อนไขที่กรอง")

                # --- NEW SECTION: Historical Signal Analysis ---
                st.divider()
                st.subheader("📈 Stock Historical Signal Analysis")
                st.caption("📊 **Historical Analysis:** เจาะลึกประวัติสัญญาณเทรดและแนวโน้มราคาย้อนหลัง 90 วัน (Hybrid Data)")
                
                all_tickers = sorted([str(t) for t in combined_df['ticker'].dropna().unique().tolist()])
                sel_hist_ticker = st.selectbox("เลือกหุ้นเพื่อดูประวัติสัญญาณ", all_tickers, key="mkt_hist_ticker_select")
                
                if sel_hist_ticker:
                    with st.spinner(f"Loading historical data for {sel_hist_ticker}..."):
                        # Fetch price data
                        hist_price_raw = get_stock_data(sel_hist_ticker)
                        if hist_price_raw is not None:
                            # 1. Prepare Price Data
                            hist_price = calculate_quant_indicators(hist_price_raw, 14, 10, 50)
                            hist_price = hist_price.tail(90).copy()
                            # Convert index to string 'YYYY-MM-DD' for exact matching
                            hist_price['date_str'] = hist_price.index.strftime('%Y-%m-%d')
                            
                            # 2. Fetch & Prepare Signal Data
                            hist_signals = fetch_ticker_combined_history(sel_hist_ticker, days=90)
                            
                            if not hist_signals.empty:
                                # Standardize signal dates to 'YYYY-MM-DD' strings
                                hist_signals['signal_date_str'] = pd.to_datetime(hist_signals['scanned_at']).dt.strftime('%Y-%m-%d')
                                
                                # Unify Signal Naming for Mapping (Excluding SILENT ACCUM)
                                def clean_signal_name(row):
                                    sig = str(row.get('signal', '')).upper()
                                    strat = str(row.get('strategy', '')).upper()
                                    is_silent = bool(row.get('is_silent_accum', False))
                                    
                                    # EXCLUDE SILENT ACCUM (Moved to dedicated tab)
                                    if 'SILENT' in sig or 'SILENT' in strat or is_silent: return None
                                    
                                    if 'BUY' in sig or 'BREAKOUT' in sig: return 'BUY / BREAKOUT'
                                    if 'PULLBACK' in sig or 'RETEST' in sig or 'PIN BAR' in sig: return 'PULLBACK / PIN BAR'
                                    if 'MOMENTUM' in sig or 'VOLUME' in sig or 'RECOVERY' in sig: return 'MOMENTUM / VOL'
                                    
                                    # Default all other bearish/unknown to SELL / WARNING
                                    return 'SELL / WARNING'

                                hist_signals['display_signal'] = hist_signals.apply(clean_signal_name, axis=1)
                                
                                # Dynamic Signal Selection Controls
                                # Filter out None/NaN and get unique sorted list
                                available_signals = sorted([str(s) for s in hist_signals['display_signal'].dropna().unique().tolist()])
                                selected_display_signals = st.multiselect(
                                    "🎯 เลือกประเภทสัญญาณที่ต้องการแสดง (Multi-Signal Overlay)", 
                                    available_signals, 
                                    default=available_signals,
                                    key="hist_sig_multiselect"
                                )
                                
                                # Filter signals by selected types and ticker
                                clean_sel_ticker = sel_hist_ticker.strip().upper()
                                base_sel_ticker = clean_sel_ticker.replace('.BK', '')
                                hist_signals['ticker_clean'] = hist_signals['ticker'].str.strip().str.upper().str.replace('.BK', '')
                                
                                filtered_signals = hist_signals[
                                    (hist_signals['display_signal'].isin(selected_display_signals)) & 
                                    (hist_signals['ticker_clean'] == base_sel_ticker)
                                ].copy()
                                
                                # FINAL DEDUPLICATION: Ensure 1 marker per Category per Day
                                filtered_signals = filtered_signals.sort_values('scanned_at', ascending=True)
                                filtered_signals = filtered_signals.drop_duplicates(
                                    subset=['signal_date_str', 'display_signal'], 
                                    keep='first'
                                )
                                
                                # Print Debug Summary to Streamlit (Temporary Check)
                                st.caption(f"🔍 DEBUG: Found {len(filtered_signals)} technical signals for {sel_hist_ticker} (Last 90 days)")
                            else:
                                filtered_signals = pd.DataFrame()
                                selected_display_signals = []

                            # Create Plotly Chart
                            fig_hist = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                            
                            # 1. Candlestick (Use Datetime Index for X-axis)
                            fig_hist.add_trace(go.Candlestick(
                                x=hist_price.index,
                                open=hist_price['Open'],
                                high=hist_price['High'],
                                low=hist_price['Low'],
                                close=hist_price['Close'],
                                name="Price"
                            ), row=1, col=1)
                            
                            # 2. Add Multi-Signal Markers (Overlay using Exact Date Matching)
                            # Optimized Color & Symbol Mapping for Technical Signals
                            SIGNAL_STYLE = {
                                'BUY / BREAKOUT': {'symbol': 'triangle-up', 'color': '#10b981', 'size': 14, 'label': '🟢 BUY / BREAKOUT'},
                                'SELL / WARNING': {'symbol': 'triangle-down', 'color': '#ef4444', 'size': 14, 'label': '🔴 SELL / WARNING'},
                                'PULLBACK / PIN BAR': {'symbol': 'diamond', 'color': '#f59e0b', 'size': 12, 'label': '🟡 PULLBACK / PIN BAR'},
                                'MOMENTUM / VOL': {'symbol': 'square', 'color': '#a855f7', 'size': 12, 'label': '🟣 MOMENTUM / VOL'}
                            }

                            if not filtered_signals.empty:
                                # Join signals with price data on date string to get correct OHLC positions
                                df_markers = hist_price.merge(
                                    filtered_signals, 
                                    left_on='date_str', 
                                    right_on='signal_date_str', 
                                    how='inner'
                                )
                                
                                if not df_markers.empty:
                                    # Ensure the merged dataframe has the original datetime index for plotting
                                    df_markers.index = pd.to_datetime(df_markers['date_str'])
                                    
                                    for sig_name in selected_display_signals:
                                        sig_group = df_markers[df_markers['display_signal'] == sig_name]
                                        if sig_group.empty: continue
                                        
                                        style = SIGNAL_STYLE.get(sig_name, SIGNAL_STYLE['SELL / WARNING'])
                                        
                                        # Strict Y-axis Alignment based on Signal Type
                                        y_pos = []
                                        for _, m_row in sig_group.iterrows():
                                            # Buy-side signals -> Below candle
                                            if sig_name in ['BUY / BREAKOUT', 'PULLBACK / PIN BAR', 'MOMENTUM / VOL']:
                                                y_pos.append(m_row['Low'] * 0.98)
                                            # Sell-side/Warning/Other signals -> Above candle
                                            else:
                                                y_pos.append(m_row['High'] * 1.02)
                                        
                                        hover_texts = []
                                        for _, m_row in sig_group.iterrows():
                                            h_rsi = f"RSI: {m_row.get('rsi', 0):.1f}" if pd.notna(m_row.get('rsi')) else ""
                                            h_vol = f"Vol: {m_row.get('volume', 0):,.0f}" if pd.notna(m_row.get('volume')) else ""
                                            score = m_row.get('score', 'N/A')
                                            
                                            # Use specific signal name for tooltip if category is SELL / WARNING
                                            display_title = m_row.get('signal', sig_name) if sig_name == 'SELL / WARNING' else sig_name
                                            hover_texts.append(f"<b>{display_title}</b><br>Score: {score}<br>{h_rsi}<br>{h_vol}")

                                        fig_hist.add_trace(go.Scatter(
                                            x=sig_group.index, 
                                            y=y_pos,
                                            mode='markers',
                                            marker=dict(
                                                symbol=style['symbol'], 
                                                size=style['size'], 
                                                color=style['color'],
                                                line=dict(width=1, color='white') # Add white border for visibility
                                            ),
                                            name=style['label'],
                                            text=hover_texts,
                                            hovertemplate="%{text}<extra></extra>",
                                            showlegend=True
                                        ), row=1, col=1)
                            
                            # 3. RSI Subplot
                            fig_hist.add_trace(go.Scatter(
                                x=hist_price.index, y=hist_price['RSI'],
                                name="RSI", line=dict(color='#8b5cf6', width=2)
                            ), row=2, col=1)
                            
                            fig_hist.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
                            fig_hist.add_hline(y=30, line_dash="dash", line_color="#10b981", row=2, col=1)
                            
                            # Dark Theme Style
                            fig_hist.update_layout(
                                height=650,
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0.05)',
                                xaxis_rangeslider_visible=False,
                                margin=dict(t=50, b=50, l=50, r=50),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(fig_hist, use_container_width=True)
                        else:
                            st.error(f"ไม่สามารถดึงข้อมูลราคาของ {sel_hist_ticker} ได้")
            else:
                st.info("ℹ️ ยังไม่มีข้อมูลการสแกนในระบบ (กรุณากด Run SET100 Batch Scan หรือรอระบบ Auto Scan)")

        with main_tabs[6]: # Advanced Tools / More Features
            st.info("🛠️ Advanced Tools & Strategy Builder")
            st.caption("⚙️ **Advanced Features:** รวมเครื่องมือวิเคราะห์เชิงลึก เช่น การปรับจูน Parameter ด้วย AI, ระบบทดสอบย้อนหลัง (Backtest) และห้องทดลองรูปแบบราคา (Pattern Lab)")
            
            # Sub-tabs for Advanced Tools
            adv_tabs = st.tabs(["⚙️ Strategy Builder", "📈 Performance Dashboard", "🔮 Pattern Lab"])
            
            with adv_tabs[0]: # Strategy Builder
                st.subheader("🛠 Quant Strategy Builder")
                st.caption("🔧 **Strategy Builder:** ปรับแต่งเงื่อนไขการซื้อขายด้วยตัวเอง หรือให้ AI ช่วยคำนวณค่าที่เหมาะสมที่สุด (Optimizer)")
                
                # Initialize AI Optimization values in session state
                if 'ai_params' not in st.session_state:
                    st.session_state['ai_params'] = None
                
                if 'stock_list' not in st.session_state:
                    st.session_state['stock_list'] = ["KTC.BK", "AMATA.BK", "RCL.BK", "CPF.BK"]
                
                col_data, col_ai = st.columns(2)
                with col_data:
                    st.markdown("### 📥 Data Management")
                    new_ticker = st.text_input("➕ Add Ticker", "", key="adv_add_ticker").upper()
                    if new_ticker:
                        if not new_ticker.endswith(".BK") and "." not in new_ticker: new_ticker += ".BK"
                        if new_ticker not in st.session_state['stock_list']:
                            st.session_state['stock_list'].append(new_ticker)
                    selected_ticker = st.selectbox("Select Stock", st.session_state['stock_list'], key="adv_select_ticker")
                    fetch_btn = st.button("🚀 Fetch Data", type="primary", use_container_width=True, key="adv_fetch_btn")
                
                if fetch_btn:
                    with st.spinner("Downloading..."):
                        df_raw_new = get_stock_data(selected_ticker)
                        if df_raw_new is not None:
                            st.session_state['df_raw'] = df_raw_new
                            st.session_state['active_ticker'] = selected_ticker
                            st.rerun()
                        else:
                            st.error("Failed to load data.")
                
                with col_ai:
                    st.markdown("### 🤖 AI Strategy Optimizer")
                    if 'df_raw' in st.session_state:
                        if st.button("Run AI Optimizer", use_container_width=True, key="adv_ai_btn"):
                            if not user_api_key:
                                st.warning("Please enter Google API Key in the sidebar.")
                            else:
                                current_df = calculate_quant_indicators(st.session_state['df_raw'], 14, 5, 20)
                                _, current_stats, _ = run_backtest(current_df, 30, 70, 1.2)
                                
                                with st.status("AI is analyzing...", expanded=True) as status:
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
                                    st.info(f"💡 AI Recommendation: {ai_rec['reasoning']}")
                                else:
                                    status.update(label="❌ AI Optimization Failed", state="error")
                    else:
                        st.info("Select a stock and fetch data to use AI Optimizer.")

                st.divider()
                with st.form("adv_strategy_params"):
                    st.subheader("⚙️ Buy/Sell Parameters")
                    p = st.session_state['ai_params'] if st.session_state['ai_params'] else {}
                    
                    c1, c2, c3 = st.columns(3)
                    rsi_p = c1.slider("RSI Period", 5, 30, p.get('rsi_p', 14))
                    rsi_b = c2.slider("Buy Threshold (RSI <=)", 10, 80, p.get('rsi_b', 50))
                    rsi_s = c3.slider("Sell Threshold (RSI >=)", 40, 90, p.get('rsi_s', 70))
                    
                    c4, c5, c6 = st.columns(3)
                    ema_f = c4.number_input("Fast EMA", 5, 50, p.get('ema_f', 10))
                    ema_s = c5.number_input("Slow EMA", 10, 200, p.get('ema_s', 50))
                    rv_m = c6.slider("Min Rel. Volume", 1.0, 3.0, p.get('rv_m', 1.5), 0.1)
                    
                    st.markdown("### 🔍 Scanner Settings")
                    min_sim = st.slider("Min Similarity Threshold (%)", 50, 95, 80)
                    scanner_mode = st.radio("Scanner Mode", ["Bullish (Breakout)", "Bearish (Danger Zone)"], horizontal=True)
                    
                    apply_btn = st.form_submit_button("🔄 Apply & Backtest", use_container_width=True)

            with adv_tabs[1]: # Performance Dashboard
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
                    
                    # 2. Main Chart
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name=f"EMA {ema_f}", line=dict(color='orange', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name=f"EMA {ema_s}", line=dict(color='blue', width=1)), row=1, col=1)
                    if trade_log is not None:
                        fig.add_trace(go.Scatter(x=trade_log['Entry Date'], y=trade_log['Entry Price'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='Buy'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=trade_log['Exit Date'], y=trade_log['Exit Price'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell'), row=1, col=1)
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
                    with c_eq:
                        st.subheader("📈 Equity Growth")
                        if equity_curve is not None:
                            eq_fig = go.Figure()
                            eq_fig.add_trace(go.Scatter(x=equity_curve['Trade'], y=equity_curve['Equity'], fill='tozeroy', line=dict(color='green')))
                            eq_fig.update_layout(height=350, template="plotly_white", title="Capital: 100k Base")
                            st.plotly_chart(eq_fig, use_container_width=True)
                else:
                    st.info("Select a stock in 'Strategy Builder' tab to see performance.")

            with adv_tabs[2]: # Pattern Lab
                if 'df_raw' in st.session_state:
                    mode_label = "🚀 Pre-Breakout Pattern Scanner" if scanner_mode == "Bullish (Breakout)" else "🚩 Danger Zone Scanner"
                    st.subheader(f"{mode_label} (Last 5 Days vs History)")
                    scan_mode_val = 'bullish' if scanner_mode == "Bullish (Breakout)" else 'bearish'
                    with st.spinner(f"Scanning for historical patterns..."):
                        scan_results = get_pre_breakout_scanner(df, mode=scan_mode_val)
                    if scan_results:
                        scan_fig = make_subplots(rows=1, cols=3, subplot_titles=("Price Pattern Similarity", "Volume Flow Similarity", "Candlestick Comparison"), column_widths=[0.3, 0.3, 0.4])
                        main_color = 'green' if scan_mode_val == 'bullish' else 'red'
                        jump_color = 'lime' if scan_mode_val == 'bullish' else 'crimson'
                        scan_fig.add_trace(go.Scatter(x=list(range(5)), y=scan_results['curr_p'], name="Current Price", line=dict(color='black', width=3)), row=1, col=1)
                        scan_fig.add_trace(go.Scatter(x=list(range(5)), y=scan_results['curr_v'], name="Current Volume", line=dict(color='gray', width=3, dash='dash')), row=1, col=2)
                        best = scan_results['matches'][0]
                        scan_fig.add_trace(go.Scatter(x=list(range(5)), y=best['hist_p'], name=f"Best Match History ({pd.to_datetime(best['date']).date()})", line=dict(color=main_color, dash='dot')), row=1, col=1)
                        jump_label = f"{best['jump']:+.1f}%"
                        scan_fig.add_trace(go.Scatter(x=[4, 5], y=[best['hist_p_ext'][4], best['hist_p_ext'][5]], name="Historical Move", mode='lines+markers+text', text=["", jump_label], textposition="top center", line=dict(color=jump_color, width=5), marker=dict(size=8, color=jump_color)), row=1, col=1)
                        scan_fig.add_trace(go.Scatter(x=list(range(5)), y=best['hist_v'], name=f"Best Match Volume", line=dict(color='blue', dash='dot')), row=1, col=2)
                        curr_ohlc = scan_results['curr_ohlc']
                        scan_fig.add_trace(go.Candlestick(x=list(range(5)), open=curr_ohlc['Open'], high=curr_ohlc['High'], low=curr_ohlc['Low'], close=curr_ohlc['Close'], name="Current Candles"), row=1, col=3)
                        hist_ohlc = best['hist_ohlc_scaled']
                        scan_fig.add_trace(go.Candlestick(x=list(range(6)), open=hist_ohlc['Open'], high=hist_ohlc['High'], low=hist_ohlc['Low'], close=hist_ohlc['Close'], name="Historical Match Candles", increasing_line_color='rgba(34, 197, 94, 0.3)', decreasing_line_color='rgba(239, 68, 68, 0.3)', increasing_fillcolor='rgba(34, 197, 94, 0.1)', decreasing_fillcolor='rgba(239, 68, 68, 0.1)'), row=1, col=3)
                        scan_fig.update_layout(height=450, template="plotly_white", showlegend=True, xaxis3_rangeslider_visible=False)
                        st.plotly_chart(scan_fig, use_container_width=True)
                        st.write("### 📊 Matching Summary")
                        m_cols = st.columns(3)
                        for i, m in enumerate(scan_results['matches']):
                            similarity = max(0, 100 - (m['dist'] * 20)) 
                            sim_color = "green" if similarity > 80 else "orange"
                            jump_val_color = "green" if m['jump'] > 0 else "red"
                            with m_cols[i]:
                                st.markdown(f"**Match #{i+1}: {pd.to_datetime(m['date']).date()}**\n- Similarity Score: :{sim_color}[{similarity:.1f}%]\n- Historical Move: :{jump_val_color}[{m['jump']:+.1f}%]")
                        
                        st.divider()
                        st.subheader("✅ Scanner Accuracy Validator")
                        if st.button("🔍 Validate Scanner Accuracy", use_container_width=True, key="adv_validate_btn"):
                            df_temp = df.copy()
                            df_temp['Pct_Change'] = df_temp['Close'].pct_change()
                            events = df_temp[df_temp['Pct_Change'] >= 0.05].index.tolist() if scan_mode_val == 'bullish' else df_temp[df_temp['Pct_Change'] <= -0.05].index.tolist()
                            winning_pats = []
                            for b_date in events:
                                idx = df_temp.index.get_loc(b_date)
                                if idx < 5: continue
                                p_data = df_temp.iloc[idx-5 : idx]
                                winning_pats.append({'price_pattern': StandardScaler().fit_transform(p_data[['Close']]).flatten(), 'vol_pattern': StandardScaler().fit_transform(p_data[['Volume']]).flatten()})
                            with st.spinner("Validating historical signals..."):
                                val_results = validate_scanner_accuracy(df, winning_pats, price_threshold=min_sim, vol_threshold=70.0)
                            if val_results:
                                s = val_results['summary']
                                m1, m2, m3, m4, m5, m6 = st.columns(6)
                                m1.metric("Signals", s['Total Signals'])
                                m2.metric("Hit Rate", f"{s['Hit Rate']:.1f}%")
                                m3.metric("Expectancy", f"{s['Expectancy']:.2f}%")
                                m4.metric("Avg Prof", f"+{s['Avg Profit']:.2f}%")
                                m5.metric("Avg Loss", f"{s['Avg Loss']:.2f}%")
                                m6.metric("RR Ratio", f"{s['RR Ratio']:.2f}")
                                st.dataframe(val_results['log'].style.map(lambda x: 'color: green' if x == '✅ Hit' else 'color: red', subset=['Result']), use_container_width=True)
                else:
                    st.info("Select a stock in 'Strategy Builder' tab to see Pattern Lab.")
                
# --- End of Application ---

