import pandas as pd
import numpy as np
from datetime import datetime
import pytz

SET_TZ = pytz.timezone('Asia/Bangkok')
STARTING_CAPITAL = 1_000_000.0 # Default starting capital for paper portfolio

def execute_paper_buy(supabase, ticker, entry_price, quantity, signal_type="WVF"):
    """
    Inserts a 'PAPER_OPEN' position into trading_log with defensive column handling.
    """
    if not supabase:
        return False, "Supabase client not initialized"
    
    try:
        now = datetime.now(SET_TZ)
        # Minimal payload based on user's confirmed added columns
        payload = {
            "ticker": ticker,
            "entry_price": float(entry_price),
            "last_price": float(entry_price),
            "status": "PAPER_OPEN",
            "signal": signal_type,
            "timestamp": now.isoformat(),
            "run_id": f"PAPER_{now.strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Optional columns that might exist if full migration was run
        optional_fields = {
            "action": "BUY",
            "quantity": int(quantity) if int(quantity) > 0 else 1000,
            "entry_date": now.strftime('%Y-%m-%d')
        }
        
        # Try to insert with full payload first, then fallback
        try:
            full_payload = {**payload, **optional_fields}
            resp = supabase.table("trading_log").insert(full_payload).execute()
            if resp.data: return True, f"Successfully executed Paper Buy for {ticker}"
        except Exception:
            # Fallback to minimal payload
            resp = supabase.table("trading_log").insert(payload).execute()
            if resp.data: return True, f"Successfully executed Paper Buy (Minimal) for {ticker}"
            
        return False, "Failed to insert record into Supabase"
    except Exception as e:
        return False, str(e)

def execute_paper_sell(supabase, position_id, exit_price, exit_reason="MANUAL"):
    """
    Closes a 'PAPER_OPEN' position with defensive column handling.
    """
    if not supabase:
        return False, "Supabase client not initialized"
    
    try:
        # 1. Fetch original position
        resp = supabase.table("trading_log").select("*").eq("id", position_id).single().execute()
        if not resp.data:
            return False, "Position not found"
        
        pos = resp.data
        # Use entry_price if available, fallback to last_price
        entry_price = float(pos.get('entry_price', pos.get('last_price', 0)))
        quantity = int(pos.get('quantity', 1000))
        if quantity <= 0: quantity = 1000
        
        # 2. Calculate PnL
        pnl_pct = (float(exit_price) - entry_price) / entry_price * 100 if entry_price > 0 else 0
        realized_pnl_thb = (float(exit_price) - entry_price) * quantity
        now = datetime.now(SET_TZ)
        
        # 3. Update payload
        update_payload = {
            "exit_price": float(exit_price),
            "status": "PAPER_CLOSED",
            "verified_date": now.strftime('%Y-%m-%d') # Fallback for exit_date
        }
        
        # Optional fields
        optional_updates = {
            "pnl_percent": round(pnl_pct, 2),
            "exit_reason": exit_reason,
            "action": "SELL",
            "exit_date": now.strftime('%Y-%m-%d'),
            "outcome_t3_pct": round(pnl_pct, 2) # Fallback for pnl_percent
        }
        
        # Try to update with as many fields as possible
        try:
            full_update = {**update_payload, **optional_updates}
            resp_upd = supabase.table("trading_log").update(full_update).eq("id", position_id).execute()
            if resp_upd.data: return True, f"Successfully closed position for {pos['ticker']}"
        except Exception:
            # Fallback to minimal update
            resp_upd = supabase.table("trading_log").update(update_payload).eq("id", position_id).execute()
            if resp_upd.data: return True, f"Successfully closed position (Minimal) for {pos['ticker']}"
            
        return False, "Failed to update record in Supabase"
            
    except Exception as e:
        return False, str(e)

def get_paper_portfolio_metrics(supabase, live_prices_dict=None):
    """
    Calculates portfolio metrics from trading_log.
    live_prices_dict: Optional dict {ticker: latest_close} to get accurate unrealized PnL
    """
    if not supabase:
        return None
    
    try:
        # Fetch all paper trades (status PAPER_OPEN or PAPER_CLOSED)
        resp = supabase.table("trading_log").select("*").or_("status.eq.PAPER_OPEN,status.eq.PAPER_CLOSED").execute()
        
        if not resp.data:
            return {
                "total_portfolio_value": STARTING_CAPITAL,
                "win_rate": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "open_count": 0,
                "closed_count": 0
            }
        
        df = pd.DataFrame(resp.data)
        
        # Defensive handling for missing columns
        for col in ['status', 'entry_price', 'exit_price', 'quantity', 'pnl_percent', 'last_price']:
            if col not in df.columns:
                df[col] = 0.0 if col != 'status' else 'UNKNOWN'
        
        # Ensure numeric types
        df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce').fillna(0)
        df['exit_price'] = pd.to_numeric(df['exit_price'], errors='coerce').fillna(0)
        df['last_price'] = pd.to_numeric(df['last_price'], errors='coerce').fillna(0)
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(1000)
        df['quantity'] = df['quantity'].replace(0, 1000)
        
        # Update last_price with live prices if provided
        if live_prices_dict:
            for i, row in df.iterrows():
                ticker = row['ticker']
                clean_t = ticker.strip().upper()
                if not clean_t.endswith('.BK') and not clean_t.startswith('^'):
                    clean_t = f"{clean_t}.BK"
                
                if clean_t in live_prices_dict:
                    live_df = live_prices_dict[clean_t]
                    if live_df is not None and not live_df.empty:
                        df.at[i, 'last_price'] = live_df.iloc[-1]['Close']
        
        # Realized PnL (Closed trades)
        closed_trades = df[df['status'] == 'PAPER_CLOSED'].copy()
        realized_pnl = 0.0
        win_rate = 0.0
        if not closed_trades.empty:
            closed_trades['trade_pnl'] = (closed_trades['exit_price'] - closed_trades['entry_price']) * closed_trades['quantity']
            realized_pnl = closed_trades['trade_pnl'].sum()
            
            # Use trade_pnl > 0 for win rate
            win_count = (closed_trades['trade_pnl'] > 0).sum()
            win_rate = (win_count / len(closed_trades)) * 100
            
        # Unrealized PnL (Open trades)
        open_trades = df[df['status'] == 'PAPER_OPEN'].copy()
        unrealized_pnl = 0.0
        if not open_trades.empty:
            open_trades['trade_unrealized'] = (open_trades['last_price'] - open_trades['entry_price']) * open_trades['quantity']
            unrealized_pnl = open_trades['trade_unrealized'].sum()
            
        return {
            "total_portfolio_value": round(STARTING_CAPITAL + realized_pnl + unrealized_pnl, 2),
            "win_rate": round(win_rate, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "open_count": len(open_trades),
            "closed_count": len(closed_trades)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            "total_portfolio_value": STARTING_CAPITAL,
            "win_rate": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_count": 0,
            "closed_count": 0
        }
