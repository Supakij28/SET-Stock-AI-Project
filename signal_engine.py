import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import backtest_engine
import pytz

SET_TZ = pytz.timezone('Asia/Bangkok')

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
        
        # 1. NEW ENTRY SIGNAL (Evaluate at T Close -> Plan for T+1 Open)
        # Condition: WVF Spike AND (Optional) Price > EMA 200
        if last_row.get('Is_WVF_Spike', False):
            trend_ok = True
            if params.get('use_trend_filter'):
                trend_ok = last_row['Close'] > last_row['EMA_Trend']
            
            if trend_ok:
                # Zero Look-Ahead Bias: Signal at T Close, Entry at T+1 Open
                # We use T Close as an estimate for T+1 Open in the plan
                entry_price = last_row['Close'] 
                sl_pct = params.get('stop_loss', 0.03)
                tp_pct = params.get('take_profit', 0.08)
                
                new_entries.append({
                    'Ticker': ticker,
                    'Signal Type': 'WVF Bottom Climax',
                    'Trigger Date': d.index[-1].strftime('%Y-%m-%d'),
                    'Calculated Entry Price (Open)': round(entry_price, 2),
                    'Dynamic Stop Loss': round(entry_price * (1 - sl_pct), 2),
                    'Dynamic Take Profit': round(entry_price * (1 + tp_pct), 2),
                    'R:R Ratio': round(tp_pct / sl_pct, 2) if sl_pct > 0 else 0
                })
        
    return {
        'new_entries': pd.DataFrame(new_entries),
        'summary': {
            'new_buy_signals': len(new_entries),
            'active_positions': 0, # To be handled by UI/DB
            'tp_sl_reached': 0     # To be handled by UI/DB
        }
    }

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
    
    for _, pos in positions_df.iterrows():
        ticker = pos['ticker']
        entry_price = pos['entry_price']
        
        # Ensure ticker format is correct for yfinance
        clean_ticker = ticker.strip().upper()
        if not clean_ticker.endswith('.BK') and not clean_ticker.startswith('^'):
            clean_ticker = f"{clean_ticker}.BK"
        
        current_price = 0.0
        
        # 1. Try to get live price from ticker_data_dict (Batch data)
        if clean_ticker in ticker_data_dict:
            df = ticker_data_dict[clean_ticker]
            if df is not None and not df.empty:
                current_price = df.iloc[-1]['Close']
        
        # 2. If price seems stale or not in dict, try yfinance fast_info for REAL-TIME
        if current_price == 0.0:
            try:
                t = yf.Ticker(clean_ticker)
                current_price = t.fast_info['lastPrice']
            except:
                pass
            
        if current_price == 0.0:
            continue
            
        params = default_params.copy()
        if optimized_params_dict and ticker in optimized_params_dict:
            params.update(optimized_params_dict[ticker])
            
        pnl_pct = (current_price - entry_price) / entry_price
        
        target_tp = entry_price * (1 + params.get('take_profit', 0.08))
        target_sl = entry_price * (1 - params.get('stop_loss', 0.03))
        
        action = "HOLD"
        if current_price >= target_tp:
            action = "SELL AT OPEN (TP)"
        elif current_price <= target_sl:
            action = "SELL AT OPEN (SL)"
            
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
