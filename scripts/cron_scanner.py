import yfinance as yf
import yahooquery as yq
import pandas as pd
import numpy as np
import os
import requests
import json
from datetime import datetime, time
import pytz
from supabase import create_client, Client
from dotenv import load_dotenv
from dtaidistance import dtw
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

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
    """Check if Thai stock market is currently open."""
    now = datetime.now(SET_TZ)
    # Mon-Fri only
    if now.weekday() >= 5:
        return False
    
    current_time = now.time()
    morning_open = time(10, 0)
    morning_close = time(12, 30)
    afternoon_open = time(14, 30)
    afternoon_close = time(16, 30)
    
    is_open_hours = (morning_open <= current_time <= morning_close) or (afternoon_open <= current_time <= afternoon_close)
    
    if not is_open_hours:
        return False
    
    # Check ^SET to detect holidays
    try:
        set_idx = yf.Ticker("^SET.BK").history(period="1d")
        if set_idx.empty or len(set_idx) == 0:
            return False
        # Check if the data is from today (or very recent)
        last_date = set_idx.index[-1].astimezone(SET_TZ).date()
        if last_date != now.date():
            return False
    except:
        return False
        
    return True

def calculate_quant_indicators(df):
    d = df.copy()
    # RSI
    delta = d['Close'].diff()
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs()
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    d['RSI'] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))
    
    # EMAs
    d['EMA_Fast'] = d['Close'].ewm(span=10, adjust=False).mean()
    d['EMA_Slow'] = d['Close'].ewm(span=50, adjust=False).mean()
    
    # Relative Volume (RV)
    d['RV'] = d['Volume'] / (d['Volume'].rolling(20).mean() + 1e-9)
    
    # ATR
    high_low = d['High'] - d['Low']
    high_close = (d['High'] - d['Close'].shift()).abs()
    low_close = (d['Low'] - d['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    d['ATR'] = true_range.rolling(14).mean()
    
    return d.dropna()

def get_pre_breakout_scanner(df, mode='bullish'):
    if len(df) < 100: return None
    df_scan = df.copy()
    df_scan['Pct_Change'] = df_scan['Close'].pct_change()
    df_scan['Volatility'] = df_scan['Close'].rolling(5).std()
    
    threshold = 0.05
    lookback = 5
    
    if mode == 'bullish':
        events = df_scan[df_scan['Pct_Change'] >= threshold].index.tolist()
    else:
        events = df_scan[df_scan['Pct_Change'] <= -threshold].index.tolist()
    
    patterns = []
    scaler = StandardScaler()
    
    for b_date in events:
        idx = df_scan.index.get_loc(b_date)
        if idx < lookback: continue
        pattern_data = df_scan.iloc[idx-lookback : idx]
        p_norm = scaler.fit_transform(pattern_data[['Close']]).flatten()
        v_norm = scaler.fit_transform(pattern_data[['Volume']]).flatten()
        patterns.append({'date': b_date, 'p_norm': p_norm, 'v_norm': v_norm, 'jump_pct': df_scan['Pct_Change'].iloc[idx] * 100})
    
    if not patterns: return None
    
    current_5 = df_scan.iloc[-lookback:]
    curr_p_norm = scaler.fit_transform(current_5[['Close']]).flatten()
    curr_v_norm = scaler.fit_transform(current_5[['Volume']]).flatten()
    
    results = []
    for p in patterns:
        dist_p = dtw.distance(curr_p_norm, p['p_norm'], window=2)
        dist_v = dtw.distance(curr_v_norm, p['v_norm'], window=2)
        total_dist = (0.7 * dist_p) + (0.3 * dist_v)
        results.append({'date': p['date'], 'jump': p['jump_pct'], 'dist': total_dist})
    
    return sorted(results, key=lambda x: x['dist'])[:3]

def get_recovery_signals(ticker, df):
    if df is None or len(df) < 20: return None
    d = df.tail(20).copy()
    rsi_curr = d['RSI'].iloc[-1]
    rsi_prev = d['RSI'].iloc[-2]
    is_oversold = rsi_curr < 35 or rsi_prev < 35
    is_rsi_turning = rsi_curr > rsi_prev and rsi_curr > 30
    
    score = 0
    if is_oversold: score += 40
    if is_rsi_turning: score += 20
    
    if score >= 50:
        return {'ticker': ticker, 'recovery_score': score, 'rsi': rsi_curr, 'price': d['Close'].iloc[-1]}
    return None

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
    """Helper function to scan a single ticker for parallel execution."""
    try:
        t = yq.Ticker(ticker)
        df_raw = t.history(period="2y").reset_index()
        if df_raw.empty: return None
        
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.set_index("date")[["close", "volume", "open", "high", "low"]].rename(
            columns={"close": "Close", "volume": "Volume", "open": "Open", "high": "High", "low": "Low"}
        )
        
        df = calculate_quant_indicators(df_raw)
        if df is None or len(df) < 50:
            return None

        # --- 1. Basic Price Action ---
        c_open = df_raw['Open'].iloc[-1]
        c_high = df_raw['High'].iloc[-1]
        c_low = df_raw['Low'].iloc[-1]
        c_close = df_raw['Close'].iloc[-1]
        body_size = abs(c_close - c_open)
        upper_wick = c_high - max(c_open, c_close)
        lower_wick = min(c_open, c_close) - c_low
        
        is_pinbar = lower_wick > (body_size * 1.5) and (c_close > c_open)
        change_percent = ((c_close / df_raw['Close'].iloc[-2]) - 1) * 100 if len(df_raw) > 1 else 0

        # --- 2. Advanced Signal Logic ---
        bullish = get_pre_breakout_scanner(df, mode='bullish')
        score = 0
        is_silent_accum = False
        if bullish:
            best = bullish[0]
            score = max(0, 100 - (best['dist'] * 20))
            if 5 < score < 15 and change_percent > 0:
                is_silent_accum = True
        
        recovery = get_recovery_signals(ticker, df)
        is_recovery = True if recovery else False
        
        signal = "WAIT"
        strategy = "SWING"
        
        if is_silent_accum:
            signal = "SILENT ACCUM"
            strategy = "ACCUMULATION"
        elif is_recovery:
            signal = "RECOVERY"
            strategy = "BOTTOM FISHING"
        elif is_pinbar:
            signal = "PIN BAR"
            strategy = "REVERSAL"
        elif score > 80:
            signal = "BUY"
            strategy = "BREAKOUT"

        # --- 3. Sector Info ---
        sector = "N/A"
        try:
            profile = t.summary_profile.get(ticker, {})
            sector = profile.get('sector', 'N/A')
        except:
            pass

        # --- 4. Prepare Payload ---
        full_payload = {
            'ticker': ticker,
            'scanned_at': scanned_at,
            'score': float(score),
            'signal': signal,
            'strategy': strategy,
            'sector': sector,
            'close_price': float(c_close),
            'change_percent': float(change_percent),
            'volume': float(df_raw['Volume'].iloc[-1]),
            'rsi': float(df['RSI'].iloc[-1]),
            'is_recovery': is_recovery,
            'is_pinbar': is_pinbar,
            'is_silent_accum': is_silent_accum
        }

        # [STRICT PAYLOAD FILTERING]
        allowed_keys = [
            'ticker', 'scanned_at', 'score', 'signal', 'strategy', 'sector',
            'close_price', 'change_percent', 'volume', 'rsi', 'is_recovery',
            'is_pinbar', 'is_silent_accum'
        ]
        
        filtered_payload = {k: full_payload[k] for k in allowed_keys if k in full_payload}
        return clean_record(filtered_payload)

    except Exception as e:
        print(f"❌ Error scanning {ticker}: {e}")
    return None

def run_scanner():
    print(f"--- 🚀 Auto Market Scanner Started at {datetime.now(SET_TZ)} ---")
    
    # Check if this is a manual run (bypass time check)
    is_manual = os.getenv("IS_MANUAL_RUN", "false").lower() == "true"
    
    if is_manual:
        print("⚡ Manual Run Detected: Bypassing market time check.")
    elif not is_market_open():
        print("⏸️ Market is closed or holiday. Skipping scan.")
        return

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

    if results and supabase:
        try:
            print(f"📤 Uploading {len(results)} results to Supabase table 'auto_scan_results'...")
            # Upsert with strict payload filtering already applied in scan_single_ticker
            response = supabase.table("auto_scan_results").upsert(results).execute()
            
            if hasattr(response, 'data') and response.data:
                print(f"✅ Successfully uploaded {len(response.data)} records to Supabase!")
            else:
                print("✅ Upload completed (Check Supabase for verification).")
                
        except Exception as e:
            print(f"❌ Supabase Error during upload: {e}")
    else:
        if not supabase:
            print("⚠️ Supabase credentials not found. Results not uploaded.")
        if not results:
            print("ℹ️ No results found to upload.")

if __name__ == "__main__":
    run_scanner()
