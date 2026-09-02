import yfinance as yf
import yahooquery as yq
import pandas as pd
import numpy as np
import os
import requests
import json
import sys
import os

# Add root directory to sys.path to import scanner_engine correctly in GitHub Actions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, time
import pytz
import holidays
from supabase import create_client, Client
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from scanner_engine import core_strategy_scanner, calculate_conviction_score, get_market_regime, get_signal_performance_stats

# Load environment variables
load_dotenv()

# Constants
SET_TZ = pytz.timezone('Asia/Bangkok')
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

SET100_TICKERS = [
    "ADVANC.BK", "AMATA.BK", "AOT.BK", "AP.BK", "AWC.BK", "BAM.BK", "BANPU.BK", "BBL.BK", "BCH.BK", "BCP.BK",
    "BCPG.BK", "BDMS.BK", "BEM.BK", "BGRIM.BK", "BH.BK", "BJC.BK", "BKI.BK", "BLA.BK", "BPP.BK", "BTS.BK",
    "CBG.BK", "CENTEL.BK", "CHG.BK", "CK.BK", "CKP.BK", "COM7.BK", "CPALL.BK", "CPF.BK", "CPN.BK", "CRC.BK",
    "DELTA.BK", "DOHOME.BK", "EA.BK", "EGCO.BK", "GLOBAL.BK", "GPSC.BK", "GULF.BK", "GUNKUL.BK", "HANA.BK", "HMPRO.BK",
    "ICHI.BK", "INTUCH.BK", "IRPC.BK", "IVL.BK", "JMART.BK", "JMT.BK", "KBANK.BK", "KCE.BK", "KEX.BK", "KKP.BK",
    "KTB.BK", "KTC.BK", "LH.BK", "M.BK", "MAJOR.BK", "MBK.BK", "MEGA.BK", "MINT.BK", "MTC.BK", "OR.BK",
    "ORI.BK", "OSP.BK", "PLANB.BK", "PRM.BK", "PSH.BK", "PSL.BK", "PTG.BK", "PTT.BK", "PTTEP.BK", "PTTGC.BK",
    "QH.BK", "RATCH.BK", "RCL.BK", "RS.BK", "SAWAD.BK", "SCB.BK", "SCC.BK", "SCGP.BK", "SINGER.BK", "SIRI.BK",
    "SPALI.BK", "SPRC.BK", "STA.BK", "STARK.BK", "STEC.BK", "STGT.BK", "SUPER.BK", "TASCO.BK", "TCAP.BK", "THANI.BK",
    "THG.BK", "TIDLOR.BK", "TIPH.BK", "TISCO.BK", "TOP.BK", "TQM.BK", "TRUE.BK", "TTB.BK", "TU.BK", "VGI.BK", "WHA.BK"
]

def is_market_open():
    # ดึงเวลาปัจจุบันตามเขตเวลาประเทศไทย
    tz = pytz.timezone('Asia/Bangkok')
    now = datetime.now(tz)
    
    # Check if this is a manual run (bypass time check)
    is_manual = os.getenv("IS_MANUAL_RUN", "false").lower() == "true"
    if is_manual:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] ⚡ Manual Run Detected: Bypassing market time check.")
        return True

    # 1. ตรวจสอบวันเสาร์ (5) หรือวันอาทิตย์ (6)
    if now.weekday() >= 5:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] วันเสาร์-อาทิตย์ (ตลาดปิด)")
        return False

    # 2. ตรวจสอบวันหยุดนักขัตฤกษ์ของประเทศไทย
    th_holidays = holidays.Thailand(years=now.year)
    if now.date() in th_holidays:
        holiday_name = th_holidays.get(now.date())
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] วันหยุดนักขัตฤกษ์: {holiday_name} (ตลาดปิด)")
        return False

    # 3. ตรวจสอบเวลาเปิด-ปิดตลาด (จันทร์-ศุกร์ 10.00-12.30 น. และ 14.30-16.30 น.)
    current_time = now.time()
    morning_open = time(10, 0)
    morning_close = time(12, 30)
    afternoon_open = time(14, 30)
    afternoon_close = time(16, 30)
    
    is_open_hours = (morning_open <= current_time <= morning_close) or (afternoon_open <= current_time <= afternoon_close)
    if not is_open_hours:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] อยู่นอกเวลาทำการตลาด (10:00-12:30, 14:30-16:30)")
        return False

    return True

def clean_record(record):
    """Clean record for JSON compliance by converting to native Python types."""
    cleaned = {}
    for k, v in record.items():
        if pd.isna(v):
            cleaned[k] = None
        elif isinstance(v, (bool, np.bool_)):
            cleaned[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            cleaned[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            if math.isinf(v) or math.isnan(v):
                cleaned[k] = None
            else:
                cleaned[k] = float(v)
        else:
            cleaned[k] = str(v)
    return cleaned

def scan_single_ticker(ticker, scanned_at):
    """Helper function to scan a single ticker using Unified Scanner Engine."""
    try:
        t = yq.Ticker(ticker)
        # Use a longer history to match Manual Scan (stock_dashboard.py)
        df_raw = t.history(start="2018-01-01").reset_index()
        if df_raw.empty: return None
        
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.set_index("date")[["close", "volume", "open", "high", "low"]].rename(
            columns={"close": "Close", "volume": "Volume", "open": "Open", "high": "High", "low": "Low"}
        )
        
        # 🚀 Unified Scan Engine (Single Source of Truth)
        # Return full scan_res to allow post-processing (SRS, Conviction) in run_scanner
        scan_res = core_strategy_scanner(ticker, df_raw, target_date=None, mtf_check=True)
        if not scan_res: return None

        # Fetch Sector Info
        sector = "N/A"
        try:
            profile = t.summary_profile.get(ticker, {})
            sector = profile.get('sector', 'N/A')
        except:
            pass
        
        # Add sector to scan_res
        scan_res['sector'] = sector
        return scan_res

    except Exception as e:
        print(f"❌ Error scanning {ticker}: {e}")
    return None

def check_recent_signal_exists(ticker, signal_type, window_minutes=15):
    """Check if a signal for this ticker has been recorded in the last X minutes."""
    if not supabase: return False
    try:
        now_utc = datetime.now(pytz.utc)
        threshold = (now_utc - timedelta(minutes=window_minutes)).isoformat()
        res = supabase.table("auto_scan_results") \
            .select("id") \
            .eq("ticker", ticker) \
            .eq("signal", signal_type) \
            .gte("scanned_at", threshold) \
            .limit(1).execute()
        return len(res.data) > 0
    except: return False

def run_scanner():
    print(f"--- 🚀 Auto Market Scanner Started at {datetime.now(SET_TZ)} ---")
    
    # [STRICT IMMUTABLE LOG] Use current execution time for all records in this batch
    now = datetime.now(SET_TZ)
    scanned_at = now.isoformat()

    print(f"📡 Fetching and analyzing {len(SET100_TICKERS)} tickers in parallel...")
    results = []
    
    # Use ThreadPoolExecutor for faster parallel scanning
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_single_ticker, ticker, scanned_at): ticker for ticker in SET100_TICKERS}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    print(f"✅ Scan complete. Found {len(results)} valid results.")

    if not results:
        print("⚠️ No results to process.")
        return

    # --- Unified Post-Processing (SRS & Conviction Score) ---
    print("📊 Calculating Sector Relative Strength (SRS) and Conviction Scores...")
    df_results = pd.DataFrame(results)
    
    # 1. Calculate Sector Averages
    sector_avgs = df_results.groupby('sector')['change_percent'].mean().to_dict()
    
    # 2. Market Regime & Performance Stats
    regime, _ = get_market_regime()
    perf_stats = get_signal_performance_stats(supabase)
    
    final_payloads = []
    for _, row in df_results.iterrows():
        ticker = row['ticker']
        
        # [DEBOUNCE CHECK] If SILENT ACCUM, check if recently recorded
        if row['is_silent_accum'] and check_recent_signal_exists(ticker, 'SILENT ACCUM', window_minutes=15):
            print(f"   ⏩ Skipping {ticker} (SILENT ACCUM already recorded in last 15m)")
            continue

        s_avg = sector_avgs.get(row['sector'], 0)
        srs_val = row['change_percent'] - s_avg
        
        # Unified Conviction Score Calculation
        conviction_score, strategy, reasons, warnings = calculate_conviction_score(
            ticker, row['signal'], row.get('consensus', 0), 
            row['mtf_score'], regime, perf_stats, 
            row['rsi'], row['rel_vol'], row['score_velocity'],
            sector_rs=srs_val, price_change=row['change_percent']
        )
        
        # Prepare final payload for auto_scan_results
        # [STRICT IMMUTABLE LOG] Always use the scan execution time (scanned_at)
        payload = {
            'ticker': ticker,
            'scanned_at': scanned_at,
            'score': float(conviction_score),
            'bull_score': float(row['bull_score']), 
            'signal': row['signal'],
            'strategy': strategy, 
            'sector': row['sector'],
            'close_price': float(row['close_price']),
            'change_percent': float(row['change_percent']),
            'volume': float(row['volume']),
            'rsi': float(row['rsi']),
            'is_recovery': bool(row['is_recovery']),
            'is_pinbar': bool(row['is_pinbar']),
            'is_silent_accum': bool(row['is_silent_accum']),
            'scan_type': 'AUTO_SCAN'
        }
        
        # [STRICT DB SCHEMA FILTERING]
        allowed_keys = [
            'ticker', 'scanned_at', 'score', 'signal', 'strategy', 'sector',
            'close_price', 'change_percent', 'volume', 'rsi', 'is_recovery',
            'is_pinbar', 'is_silent_accum', 'scan_type'
        ]
        
        filtered_payload = {k: payload.get(k) for k in allowed_keys}
        final_payloads.append(clean_record(filtered_payload))

    if final_payloads and supabase:
        try:
            # [STRICT IMMUTABLE LOG] Use INSERT instead of UPSERT to prevent overwriting historical data
            print(f"📤 Inserting {len(final_payloads)} new results to Supabase table 'auto_scan_results'...")
            response = supabase.table("auto_scan_results").insert(final_payloads).execute()
            
            if hasattr(response, 'data') and response.data:
                print(f"✅ Successfully inserted {len(response.data)} new records to Supabase!")
            else:
                print("✅ Insertion completed.")
                
        except Exception as e:
            print(f"❌ Supabase Error during insert: {e}")

    else:
        if not supabase:
            print("⚠️ Supabase credentials not found. Results not uploaded.")
        if not results:
            print("ℹ️ No results found to upload.")

if __name__ == "__main__":
    if not is_market_open():
        print("ยกเลิกการสแกนหุ้นเนื่องจากตลาดปิดทำการ")
        sys.exit(0)

    print("ตลาดเปิดทำการ เริ่มกระบวนการสแกนหุ้น...")
    run_scanner()
