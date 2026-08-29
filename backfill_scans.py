import yfinance as yf
import yahooquery as yq
import pandas as pd
import numpy as np
import os
import sys
import math
import time
from datetime import datetime, timedelta, date
import pytz
import holidays
from supabase import create_client, Client
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from scanner_engine import core_strategy_scanner, calculate_conviction_score, get_signal_performance_stats

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
    "BCPG.BK", "BDMS.BK", "BEM.BK", "BGRIM.BK", "BH.BK", "BJC.BK", "BLA.BK", "BPP.BK", "BTS.BK",
    "CBG.BK", "CENTEL.BK", "CHG.BK", "CK.BK", "CKP.BK", "COM7.BK", "CPALL.BK", "CPF.BK", "CPN.BK", "CRC.BK",
    "DELTA.BK", "DOHOME.BK", "EA.BK", "EGCO.BK", "GLOBAL.BK", "GPSC.BK", "GULF.BK", "GUNKUL.BK", "HANA.BK", "HMPRO.BK",
    "ICHI.BK", "IRPC.BK", "IVL.BK", "JMART.BK", "JMT.BK", "KBANK.BK", "KCE.BK", "KKP.BK",
    "KTB.BK", "KTC.BK", "LH.BK", "M.BK", "MAJOR.BK", "MBK.BK", "MEGA.BK", "MINT.BK", "MTC.BK", "OR.BK",
    "ORI.BK", "OSP.BK", "PLANB.BK", "PRM.BK", "PSH.BK", "PSL.BK", "PTG.BK", "PTT.BK", "PTTEP.BK", "PTTGC.BK",
    "QH.BK", "RATCH.BK", "RCL.BK", "RS.BK", "SAWAD.BK", "SCB.BK", "SCC.BK", "SCGP.BK", "SINGER.BK", "SIRI.BK",
    "SPALI.BK", "SPRC.BK", "STA.BK", "STGT.BK", "SUPER.BK", "TASCO.BK", "TCAP.BK", "THANI.BK",
    "THG.BK", "TIDLOR.BK", "TIPH.BK", "TISCO.BK", "TOP.BK", "TQM.BK", "TRUE.BK", "TTB.BK", "TU.BK", "VGI.BK", "WHA.BK"
]

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

def get_historical_market_regime(set_df, target_date):
    """Determine market regime for a specific historical date."""
    try:
        # Convert index to date for comparison if it isn't already
        d = set_df[set_df.index <= pd.Timestamp(target_date)].copy()
        if d.empty: return "UNKNOWN", 0
        
        d['EMA200'] = d['Close'].ewm(span=200, adjust=False).mean()
        curr_price = d['Close'].iloc[-1]
        ema200 = d['EMA200'].iloc[-1]
        return "BULL" if curr_price > ema200 else "BEAR", curr_price
    except Exception as e:
        print(f"Error calculating historical regime for {target_date}: {e}")
        return "UNKNOWN", 0

def fetch_bulk_data(tickers, start_date="2020-01-01"):
    """Fetch historical data for all tickers and SET Index in bulk."""
    print(f"📡 Downloading historical data for {len(tickers)} tickers and SET Index...")
    data = {}
    
    # Add SET Index to the list
    all_tickers = tickers + ["^SET.BK"]
    
    # Download in chunks to avoid timeout
    chunk_size = 20
    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i:i + chunk_size]
        print(f"   Downloading chunk {i//chunk_size + 1}...")
        try:
            batch_data = yf.download(chunk, start=start_date, group_by='ticker', progress=False)
            for ticker in chunk:
                try:
                    if ticker in batch_data:
                        # yfinance can return a series for single ticker or a dataframe for multiple
                        # ensure we handle both
                        if isinstance(batch_data[ticker], pd.DataFrame):
                            df = batch_data[ticker].dropna(subset=['Close'])
                        else:
                            df = batch_data.dropna(subset=['Close']) # Fallback for single ticker
                            
                        if not df.empty:
                            data[ticker] = df
                        else:
                            print(f"   ⚠️ No data found for {ticker}")
                    else:
                        print(f"   ⚠️ {ticker} missing from download results")
                except Exception as e:
                    print(f"   ⚠️ Error processing {ticker} in batch: {e}")
        except Exception as e:
            print(f"   ❌ Error downloading chunk {i//chunk_size + 1}: {e}")
            
    return data

def get_trading_days(start_date_str, end_date_str):
    """Get a list of Thai trading days between start and end dates."""
    start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    th_holidays = holidays.Thailand(years=[start.year, end.year])
    
    trading_days = []
    curr = start
    while curr <= end:
        # Check weekend
        if curr.weekday() < 5:
            # Check holiday
            if curr not in th_holidays:
                trading_days.append(curr)
        curr += timedelta(days=1)
    return trading_days

def backfill():
    print("--- 🚀 Backfill Scan Process Started ---")
    
    start_date = "2026-08-15"
    end_date = "2026-08-28"
    
    trading_days = get_trading_days(start_date, end_date)
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"📈 Total trading days to backfill: {len(trading_days)}")
    
    if not trading_days:
        print("❌ No trading days found in range.")
        return

    # Fetch data
    all_data = fetch_bulk_data(SET100_TICKERS)
    if "^SET.BK" not in all_data:
        print("❌ Could not fetch SET Index data. Aborting.")
        return
    
    set_df = all_data["^SET.BK"]
    
    # Fetch current perf stats once (or we could fetch historical ones if needed)
    perf_stats = get_signal_performance_stats(supabase)
    
    # Get sector info once
    print("🏢 Fetching sector info...")
    sector_map = {}
    def fetch_sector(ticker):
        try:
            t = yq.Ticker(ticker)
            return ticker, t.summary_profile.get(ticker, {}).get('sector', 'N/A')
        except:
            return ticker, 'N/A'
            
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_sector, ticker): ticker for ticker in SET100_TICKERS}
        for future in as_completed(futures, timeout=120): # Add timeout to prevent hanging
            ticker = futures[future]
            try:
                ticker, sector = future.result()
                sector_map[ticker] = sector
            except Exception as e:
                print(f"   ⚠️ Error fetching sector for {ticker}: {e}")
                sector_map[ticker] = 'N/A'

    for backfill_date in trading_days:
        date_str = backfill_date.strftime("%Y-%m-%d")
        print(f"\n▶️ Processing Date: {date_str} ...")
        
        # Calculate market regime for this day
        regime, set_close = get_historical_market_regime(set_df, backfill_date)
        print(f"   Market Regime: {regime} (SET: {set_close:.2f})")
        
        day_results = []
        
        for ticker in SET100_TICKERS:
            if ticker not in all_data:
                continue
                
            ticker_df = all_data[ticker]
            # Slice data up to the current backfill_date
            # Ensure target_date is inclusive
            df_slice = ticker_df[ticker_df.index <= pd.Timestamp(backfill_date)].copy()
            
            if len(df_slice) < 100:
                continue
                
            # Check if we actually have data for this specific day (ticker might have been suspended)
            if df_slice.index[-1].date() != backfill_date:
                continue
                
            try:
                # 🚀 Unified Scan Engine
                # target_date is used for MTF check bypass
                scan_res = core_strategy_scanner(ticker, df_slice, target_date=backfill_date, mtf_check=False)
                if not scan_res:
                    continue
                
                scan_res['sector'] = sector_map.get(ticker, 'N/A')
                day_results.append(scan_res)
            except Exception as e:
                print(f"   ❌ Error scanning {ticker} on {date_str}: {e}")

        if not day_results:
            print(f"   ⚠️ No valid signals found for {date_str}")
            continue
            
        # --- Post-Processing (SRS & Conviction) ---
        df_day = pd.DataFrame(day_results)
        sector_avgs = df_day.groupby('sector')['change_percent'].mean().to_dict()
        
        final_payloads = []
        for _, row in df_day.iterrows():
            ticker = row['ticker']
            s_avg = sector_avgs.get(row['sector'], 0)
            srs_val = row['change_percent'] - s_avg
            
            # Conviction Score
            conv_score, strategy, reasons, warnings = calculate_conviction_score(
                ticker, row['signal'], row.get('consensus', 0),
                row['mtf_score'], regime, perf_stats,
                row['rsi'], row['rel_vol'], row['score_velocity'],
                sector_rs=srs_val, price_change=row['change_percent']
            )
            
            # Prepare payload for 'scan_results' table
            payload = {
                'ticker': ticker,
                'scan_date': date_str,
                'scan_time': "16:30:00", # EOD
                'price': float(row['close_price']),
                'bull_score': float(row['bull_score']),
                'bear_score': float(row['bear_score']),
                'score_diff': float(row['score_diff']),
                'signal_type': row['signal'],
                'market_regime': regime,
                'relative_vol': float(row['rel_vol']),
                'rsi': float(row['rsi']),
                'mtf_status': row['mtf_status'],
                'mtf_score': float(row['mtf_score']),
                'conviction_score': float(conv_score),
                'outcome_label': None,
                'outcome_pct': None,
                'verified_date': None
            }
            
            # Remove 'id' if exists (Safety Rule)
            payload.pop('id', None)
            final_payloads.append(clean_record(payload))
            
        if final_payloads and supabase:
            try:
                print(f"   📤 Uploading {len(final_payloads)} records to Supabase...")
                # Insert batch for the day
                response = supabase.table("scan_results").insert(final_payloads).execute()
                print(f"   ✅ Done! Uploaded {len(final_payloads)} records for {date_str}")
            except Exception as e:
                print(f"   ❌ Supabase Upload Error for {date_str}: {e}")
        
        # Throttling to avoid API rate limits
        time.sleep(1)

    print("\n--- ✨ Backfill Complete! ---")

if __name__ == "__main__":
    backfill()
