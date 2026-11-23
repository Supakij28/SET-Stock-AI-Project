import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import backtest_engine
import pytz

SET_TZ = pytz.timezone('Asia/Bangkok')

def validate_and_normalize_price(ticker, df):
    """
    Validates historical OHLCV data against yfinance fast_info['lastPrice'].
    If a significant discrepancy (>15%) is found, normalizes the historical data.
    """
    if df is None or df.empty:
        return df
    
    try:
        # 1. Get Live Price from yfinance fast_info
        clean_ticker = ticker.strip().upper()
        if not clean_ticker.endswith('.BK') and not clean_ticker.startswith('^'):
            clean_ticker = f"{clean_ticker}.BK"
            
        t = yf.Ticker(clean_ticker)
        # Try different live price attributes
        live_price = t.fast_info.get('lastPrice') or t.fast_info.get('regularMarketPrice')
        
        if not live_price or live_price <= 0:
            return df
            
        # 2. Compare with Last Historical Close
        last_hist_close = df.iloc[-1]['Close']
        discrepancy_pct = abs(live_price - last_hist_close) / last_hist_close
        
        # 3. If discrepancy > 15%, normalize historical data
        if discrepancy_pct > 0.15:
            # Check if trigger date is also stale (more than 3 days old)
            last_date = df.index[-1]
            days_diff = (datetime.now().date() - last_date.date()).days
            
            if days_diff > 3 or "BANPU" in clean_ticker: # Force for known bugged tickers
                ratio = live_price / last_hist_close
                cols_to_scale = ['Open', 'High', 'Low', 'Close']
                df[cols_to_scale] = df[cols_to_scale] * ratio
                # Volume scaling is not strictly necessary but keeps it proportional
                # df['Volume'] = df['Volume'] / ratio 
                
                print(f"⚠️ Price Discrepancy detected for {ticker} ({discrepancy_pct:.2%}). Scaled by {ratio:.4f}")
    except Exception as e:
        print(f"Sanity check failed for {ticker}: {e}")
        
    return df

def evaluate_daily_signals(ticker_data_dict, optimized_params_dict=None):
    """
    Evaluate all tickers for new entry signals and exit conditions for existing positions.
    ticker_data_dict: {ticker: df}
    optimized_params_dict: {ticker: params} or None for defaults
    """
    new_entries = []
    exits_required = []
    
    default_params = {
        'lookback': 22,
        'bb_mult': 2.0,
        'use_trend_filter': True,
        'ema_trend_period': 200,
        'exit_type': 'StopLoss_TakeProfit',
        'take_profit': 0.08,
        'stop_loss': 0.03,
        'commission': 0.00157,
        'slippage': 0.0010
    }
    
    for ticker, df in ticker_data_dict.items():
        if df is None or len(df) < 200:
            continue
            
        # --- DATA SANITY CHECK & NORMALIZATION ---
        df = validate_and_normalize_price(ticker, df)
        
        # Use optimized params if available, else default
        params = default_params.copy()
        if optimized_params_dict and ticker in optimized_params_dict:
            params.update(optimized_params_dict[ticker])
        
        # Calculate Indicators using the backtest_engine logic
        d = backtest_engine.calculate_indicators(
            df, 
            lookback=params.get('lookback', 22),
            bb_mult=params.get('bb_mult', 2.0),
            ema_period=params.get('ema_trend_period', 200)
        )
        
        if params.get('exit_type') in ['rsi', 'indicator_exit', 'RSI_Overbought']:
            d['RSI'] = backtest_engine.calculate_rsi(d['Close'])
            
        d = d.dropna()
        if len(d) < 2: continue
        
        last_row = d.iloc[-1]
        
        # Enhanced Classification from backtest_engine
        wvf_status = last_row.get('WVF_Status', '⚪ NORMAL')
        
        # 1. EVALUATE ALL TICKERS (Signal at T Close -> Plan for T+1 Open)
        entry_price = last_row['Close'] 
        sl_pct = params.get('stop_loss', 0.03)
        tp_pct = params.get('take_profit', 0.08)
        
        # Trend Filter check (still useful for scoring or filtering)
        trend_ok = last_row['Close'] > last_row['EMA_Trend'] if params.get('use_trend_filter') else True
        
        new_entries.append({
            'Ticker': ticker,
            'Signal Type': 'WVF Scanner',
            'WVF Bar Status': wvf_status,
            'Trend': 'UP' if trend_ok else 'DOWN',
            'Trigger Date': d.index[-1].strftime('%Y-%m-%d'),
            'Calculated Entry Price (Open)': round(entry_price, 2),
            'Dynamic Stop Loss': round(entry_price * (1 - sl_pct), 2),
            'Dynamic Take Profit': round(entry_price * (1 + tp_pct), 2),
            'R:R Ratio': round(tp_pct / sl_pct, 2) if sl_pct > 0 else 0
        })
        
    return {
        'new_entries': pd.DataFrame(new_entries),
        'summary': {
            'new_buy_signals': len([e for e in new_entries if 'ENTRY' in e['WVF Bar Status']]),
            'active_positions': 0, # To be handled by UI/DB
            'tp_sl_reached': 0     # To be handled by UI/DB
        }
    }

def get_robust_live_price(ticker):
    """
    Fetches a reliable live price for a ticker using multiple methods and sanity checks.
    1. Try Intraday 5m history (most reliable).
    2. Try Daily history (fallback).
    3. Try fast_info with sanity check against history.
    """
    clean_ticker = ticker.strip().upper()
    if not clean_ticker.endswith('.BK') and not clean_ticker.startswith('^'):
        clean_ticker = f"{clean_ticker}.BK"
        
    try:
        t = yf.Ticker(clean_ticker)
        
        # Method 1: Intraday 5m (Last bar with volume)
        hist_intraday = t.history(period="1d", interval="5m", auto_adjust=False)
        if not hist_intraday.empty:
            valid_bars = hist_intraday[hist_intraday['Volume'] > 0]
            if not valid_bars.empty:
                return valid_bars['Close'].iloc[-1]
        
        # Method 2: Daily History (Last 5 days to ensure we get a close)
        hist_daily = t.history(period="5d", interval="1d", auto_adjust=False)
        if not hist_daily.empty:
            last_close = hist_daily['Close'].iloc[-1]
            
            # Method 3: fast_info with Sanity Check
            # fast_info can be bugged for adjusted tickers (like BANPU 14.60 vs 13.25)
            fast_price = t.fast_info.get('lastPrice') or t.fast_info.get('regularMarketPrice')
            
            if fast_price and fast_price > 0:
                # Check discrepancy between fast_info and history close
                # If difference > 5%, reject fast_info and use history
                if abs(fast_price - last_close) / last_close > 0.05:
                    print(f"⚠️ Rejected fast_info price ({fast_price}) for {clean_ticker}. Using history close ({last_close})")
                    return last_close
                return fast_price
            
            return last_close
            
    except Exception as e:
        import traceback
        print(f"❌ Error fetching robust price for {ticker}: {e}")
        traceback.print_exc()
        
    return 0.0

def check_active_positions(positions_df, ticker_data_dict, optimized_params_dict=None):
    """
    Check active positions against their SL/TP or other exit criteria.
    positions_df: DataFrame with ['ticker', 'entry_price', 'entry_date']
    """
    results = []
    
    default_params = {
        'take_profit': 0.08,
        'stop_loss': 0.03,
        'exit_type': 'StopLoss_TakeProfit'
    }
    
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(columns=['Ticker', 'Entry Price', 'Current Price', 'Current PnL %', 'Target TP', 'Target SL', 'Action Required'])

    for _, pos in positions_df.iterrows():
        ticker = pos['ticker']
        entry_price = pos['entry_price']
        
        # Robust price fetching
        current_price = get_robust_live_price(ticker)
            
        # SAFETY GUARD: If live price fetching fails, fallback to entry_price 
        # to ensure the order is NOT hidden from the dashboard.
        is_stale = False
        if current_price <= 0.0:
            current_price = entry_price
            is_stale = True
            print(f"⚠️ Falling back to entry_price for {ticker} due to fetch error.")
            
        params = default_params.copy()
        if optimized_params_dict and ticker in optimized_params_dict:
            params.update(optimized_params_dict[ticker])
            
        pnl_pct = (current_price - entry_price) / entry_price
        
        target_tp = entry_price * (1 + params.get('take_profit', 0.08))
        target_sl = entry_price * (1 - params.get('stop_loss', 0.03))
        
        action = "HOLD"
        if not is_stale:
            if current_price >= target_tp:
                action = "SELL AT OPEN (TP)"
            elif current_price <= target_sl:
                action = "SELL AT OPEN (SL)"
        else:
            action = "FETCH ERROR (HOLD)"
            
        results.append({
            'Ticker': ticker,
            'Entry Price': round(entry_price, 2),
            'Current Price': round(current_price, 2),
            'Current PnL %': round(pnl_pct * 100, 2),
            'Target TP': round(target_tp, 2),
            'Target SL': round(target_sl, 2),
            'Action Required': action
        })
        
    return pd.DataFrame(results)
