import pandas as pd
import numpy as np
from datetime import datetime
import itertools

def calculate_indicators(df, lookback=22, bb_length=20, bb_mult=2.0, percentile_high=0.85, ema_period=200):
    """
    Calculate WVF, EMA, and ATR for filtering and exits.
    """
    d = df.copy()
    if len(d) < max(lookback, bb_length, ema_period):
        return d
        
    # WVF Calculation
    d['Highest_Close'] = d['Close'].rolling(window=lookback).max()
    d['WVF'] = ((d['Highest_Close'] - d['Low']) / d['Highest_Close']) * 100
    
    d['WVF_SMA'] = d['WVF'].rolling(window=bb_length).mean()
    d['WVF_Std'] = d['WVF'].rolling(window=bb_length).std()
    d['WVF_Upper'] = d['WVF_SMA'] + (bb_mult * d['WVF_Std'])
    d['WVF_High_Range'] = d['WVF'].rolling(window=lookback).max() * percentile_high
    
    d['Is_WVF_Spike'] = (d['WVF'] >= d['WVF_Upper']) | (d['WVF'] >= d['WVF_High_Range'])
    
    # Trend Filter
    d['EMA_Trend'] = d['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # ATR for Trailing Stop
    high_low = d['High'] - d['Low']
    high_close = (d['High'] - d['Close'].shift()).abs()
    low_close = (d['Low'] - d['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    d['ATR'] = true_range.rolling(14).mean()
    
    # Bollinger Bands for Price Exit
    d['Price_SMA'] = d['Close'].rolling(window=20).mean()
    d['Price_Std'] = d['Close'].rolling(window=20).std()
    d['Price_Upper_BB'] = d['Price_SMA'] + (2.0 * d['Price_Std'])
    
    return d

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def run_wvf_backtest(df, params):
    """
    Backtest Engine with Zero Look-Ahead Bias and Enhanced Risk Management.
    """
    # 1. Setup Data
    ema_p = params.get('ema_trend_period', 200)
    d = calculate_indicators(df, 
                             lookback=params.get('lookback', 22), 
                             bb_mult=params.get('bb_mult', 2.0),
                             ema_period=ema_p)
    
    if params.get('exit_type') in ['rsi', 'indicator_exit']:
        d['RSI'] = calculate_rsi(d['Close'], period=14)
    
    d = d.dropna()
    if len(d) == 0:
        return None
        
    initial_capital = params.get('initial_capital', 100000.0)
    comm = params.get('commission', 0.00157)
    slip = params.get('slippage', 0.0010)
    friction = comm + slip
    
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    trades = []
    equity_curve = []
    
    # Trailing stop variables
    max_price_since_entry = 0
    
    # Iterate through days
    for i in range(len(d)):
        current_row = d.iloc[i]
        current_date = d.index[i]
        current_close = current_row['Close']
        
        # Equity Tracking
        current_equity = capital
        if position > 0:
            current_equity = position * current_close
        equity_curve.append({'Date': current_date, 'Equity': current_equity})
        
        # EXIT LOGIC
        if position > 0:
            exit_signal = False
            exit_reason = ""
            
            # Update trailing stop high
            max_price_since_entry = max(max_price_since_entry, current_close)
            
            # A. Stop Loss / Take Profit
            if params['exit_type'] == 'profit_stop' or params['exit_type'] == 'StopLoss_TakeProfit':
                pnl_pct = (current_close - entry_price) / entry_price
                target = params.get('take_profit', 0.10)
                stop = params.get('stop_loss', 0.05)
                
                if pnl_pct >= target:
                    exit_signal = True
                    exit_reason = f"Take Profit ({target*100}%)"
                elif pnl_pct <= -stop:
                    exit_signal = True
                    exit_reason = f"Stop Loss ({-stop*100}%)"
            
            # B. Trailing Stop (ATR based)
            elif params['exit_type'] == 'Trailing_Stop':
                atr_mult = params.get('atr_mult', 2.0)
                trailing_stop_price = max_price_since_entry - (atr_mult * current_row['ATR'])
                if current_close <= trailing_stop_price:
                    exit_signal = True
                    exit_reason = f"Trailing Stop (ATR {atr_mult})"
            
            # C. Indicator Exit (RSI / BB)
            elif params['exit_type'] == 'rsi' or params['exit_type'] == 'RSI_Overbought':
                rsi_val = params.get('rsi_exit', 65)
                if current_row['RSI'] >= rsi_val:
                    exit_signal = True
                    exit_reason = f"RSI Exit ({rsi_val})"
            
            elif params['exit_type'] == 'indicator_exit':
                if current_row['RSI'] >= 70 or current_close >= current_row['Price_Upper_BB']:
                    exit_signal = True
                    exit_reason = "Indicator Exit (RSI/BB)"
            
            # D. Fixed Days (Fallback)
            elif params['exit_type'] == 'days':
                days_held = (current_date - entry_date).days
                if days_held >= params.get('exit_value', 10):
                    exit_signal = True
                    exit_reason = f"Fixed Days ({params.get('exit_value')})"

            if exit_signal:
                if i + 1 < len(d):
                    next_open = d.iloc[i+1]['Open']
                    exit_price = next_open * (1 - friction)
                    pnl = (exit_price - entry_price) / entry_price
                    capital = position * exit_price
                    trades.append({
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': d.index[i+1],
                        'Exit Price': exit_price,
                        'Trade PnL %': pnl * 100,
                        'Exit Reason': exit_reason
                    })
                    position = 0
                    entry_price = 0
                    entry_date = None
        
        # ENTRY LOGIC
        if position == 0 and i + 1 < len(d):
            # Primary Signal: WVF Spike
            if current_row['Is_WVF_Spike']:
                # Trend Filter
                use_trend = params.get('use_trend_filter', False)
                trend_ok = True
                if use_trend:
                    trend_ok = current_close > current_row['EMA_Trend']
                
                if trend_ok:
                    next_row = d.iloc[i+1]
                    entry_date = d.index[i+1]
                    entry_price = next_row['Open'] * (1 + friction)
                    position = capital / entry_price
                    capital = 0
                    max_price_since_entry = entry_price
                
    # Summary
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    trade_log = pd.DataFrame(trades)
    
    if len(trades) == 0:
        return {
            'summary': {'Net Profit (%)': 0, 'Total Trades': 0, 'Win Rate (%)': 0, 'Max Drawdown (%)': 0, 'Sharpe Ratio': 0},
            'equity_curve': equity_df,
            'trade_log': trade_log
        }
        
    net_profit_pct = (equity_df['Equity'].iloc[-1] - initial_capital) / initial_capital * 100
    returns = equity_df['Equity'].pct_change().dropna()
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    
    rolling_max = equity_df['Equity'].cummax()
    drawdown = (equity_df['Equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    win_rate = (trade_log['Trade PnL %'] > 0).mean() * 100
    
    summary = {
        'Net Profit (%)': round(net_profit_pct, 2),
        'Total Trades': len(trades),
        'Win Rate (%)': round(win_rate, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Final Capital': round(equity_df['Equity'].iloc[-1], 2)
    }
    
    return {
        'summary': summary,
        'equity_curve': equity_df,
        'trade_log': trade_log
    }

def optimize_wvf_strategy(df, grid):
    keys = grid.keys()
    combinations = list(itertools.product(*(grid[k] for k in keys)))
    results = []
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        params['commission'] = 0.00157
        params['slippage'] = 0.0010
        params['initial_capital'] = 100000.0
        
        res = run_wvf_backtest(df, params)
        if res and res['summary']['Total Trades'] > 0:
            summary = res['summary']
            results.append({
                **params,
                'Net Profit (%)': summary['Net Profit (%)'],
                'Sharpe': summary['Sharpe Ratio'],
                'MDD (%)': summary['Max Drawdown (%)'],
                'Win Rate (%)': summary['Win Rate (%)'],
                'Trades': summary['Total Trades']
            })
            
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Sharpe', ascending=False).reset_index(drop=True)
        
    return results_df
