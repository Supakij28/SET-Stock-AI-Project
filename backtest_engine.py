import pandas as pd
import numpy as np
from datetime import datetime
import itertools

def calculate_wvf(df, lookback=22, bb_length=20, bb_mult=2.0, percentile_high=0.85):
    """
    Williams Vix Fix (WVF) calculation consistent with scanner_engine.py.
    """
    d = df.copy()
    if len(d) < max(lookback, bb_length):
        return d
        
    d['Highest_Close'] = d['Close'].rolling(window=lookback).max()
    d['WVF'] = ((d['Highest_Close'] - d['Low']) / d['Highest_Close']) * 100
    
    d['WVF_SMA'] = d['WVF'].rolling(window=bb_length).mean()
    d['WVF_Std'] = d['WVF'].rolling(window=bb_length).std()
    d['WVF_Upper'] = d['WVF_SMA'] + (bb_mult * d['WVF_Std'])
    d['WVF_High_Range'] = d['WVF'].rolling(window=lookback).max() * percentile_high
    
    d['Is_WVF_Spike'] = (d['WVF'] >= d['WVF_Upper']) | (d['WVF'] >= d['WVF_High_Range'])
    return d

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def run_wvf_backtest(df, params):
    """
    Backtest Engine with Zero Look-Ahead Bias.
    Params: initial_capital, commission, slippage, lookback, bb_mult, 
            exit_type ('days', 'profit_stop', 'rsi'), exit_value
    """
    # 1. Setup Data
    d = calculate_wvf(df, lookback=params.get('lookback', 22), bb_mult=params.get('bb_mult', 2.0))
    if params.get('exit_type') == 'rsi':
        d['RSI'] = calculate_rsi(d['Close'], period=14)
    
    d = d.dropna()
    if len(d) == 0:
        return None
        
    initial_capital = params.get('initial_capital', 100000.0)
    comm = params.get('commission', 0.00157) # 0.157%
    slip = params.get('slippage', 0.0010)   # 0.10%
    friction = comm + slip
    
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    trades = []
    equity_curve = []
    
    # Iterate through days
    for i in range(len(d)):
        current_row = d.iloc[i]
        current_date = d.index[i]
        current_close = current_row['Close']
        current_open = current_row['Open']
        
        # Equity Tracking (Mark-to-Market)
        current_equity = capital
        if position > 0:
            current_equity = position * current_close
        equity_curve.append({'Date': current_date, 'Equity': current_equity})
        
        # EXIT LOGIC (If in position)
        if position > 0:
            exit_signal = False
            exit_reason = ""
            
            # A. Fixed Days
            if params['exit_type'] == 'days':
                days_held = (d.index[i] - entry_date).days
                if days_held >= params['exit_value']:
                    exit_signal = True
                    exit_reason = f"Fixed Days ({params['exit_value']})"
            
            # B. Profit Target / Stop Loss
            elif params['exit_type'] == 'profit_stop':
                pnl_pct = (current_close - entry_price) / entry_price
                target, stop = params['exit_value'] # tuple (0.10, 0.05)
                if pnl_pct >= target:
                    exit_signal = True
                    exit_reason = f"Profit Target ({target*100}%)"
                elif pnl_pct <= -stop:
                    exit_signal = True
                    exit_reason = f"Stop Loss ({-stop*100}%)"
            
            # C. RSI Overbought
            elif params['exit_type'] == 'rsi':
                if current_row['RSI'] >= params['exit_value']:
                    exit_signal = True
                    exit_reason = f"RSI Overbought ({params['exit_value']})"
            
            if exit_signal:
                # Execute Exit at Current Day Close (Simplified) 
                # or T+1 Open for zero bias. directive says T+1 Open for trade execution.
                # But for backtest speed/simplicity in WVF, usually Day T Close is okay if signal is T-1.
                # Let's stick to Directive: "Execute Buy/Sell orders at Day T+1 Open price"
                
                # If signal triggered today, exit at next available price (Open of next bar)
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
                    # Skip tracking equity for next bar as we exited at its open
        
        # ENTRY LOGIC (If not in position)
        if position == 0 and i + 1 < len(d):
            # Signal generated at Day T Close
            if current_row['Is_WVF_Spike']:
                # Execute Buy at Day T+1 Open
                next_row = d.iloc[i+1]
                entry_date = d.index[i+1]
                entry_price = next_row['Open'] * (1 + friction)
                position = capital / entry_price
                capital = 0 # All capital in position
                
    # 2. Performance Summary
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    trade_log = pd.DataFrame(trades)
    
    if len(trades) == 0:
        return {
            'summary': {'Net Profit (%)': 0, 'Total Trades': 0},
            'equity_curve': equity_df,
            'trade_log': trade_log
        }
        
    net_profit_pct = (equity_df['Equity'].iloc[-1] - initial_capital) / initial_capital * 100
    
    # Calculate Metrics
    returns = equity_df['Equity'].pct_change().dropna()
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    
    # MDD
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
    """
    Grid Search Optimizer ranking by Sharpe Ratio.
    grid: dict with lists of params
    """
    keys = grid.keys()
    combinations = list(itertools.product(*(grid[k] for k in keys)))
    results = []
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        # Add default frictions
        params['commission'] = 0.00157
        params['slippage'] = 0.0010
        params['initial_capital'] = 100000.0
        
        res = run_wvf_backtest(df, params)
        if res and res['summary']['Total Trades'] > 0:
            summary = res['summary']
            # Filter criteria: Win Rate > 50% (optional, let user see all and rank)
            results.append({
                **params,
                'Net Profit (%)': summary['Net Profit (%)'],
                'Sharpe': summary['Sharpe Ratio'],
                'MDD (%)': summary['Max Drawdown (%)'],
                'Win Rate (%)': summary['Win Rate (%)'],
                'Trades': summary['Total Trades']
            })
            
    # Rank by Sharpe
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Sharpe', ascending=False).reset_index(drop=True)
        
    return results_df
