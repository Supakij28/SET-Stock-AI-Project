import pandas as pd
import numpy as np
from datetime import datetime
import pytz

SET_TZ = pytz.timezone('Asia/Bangkok')

def execute_paper_buy(supabase, ticker, entry_price, quantity, signal_type="WVF"):
    """
    Inserts an 'OPEN' position into trading_log with defensive column handling.
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
            "status": "OPEN",
            "signal": signal_type,
            "timestamp": now.isoformat(),
            "run_id": f"PAPER_{now.strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Optional columns that might exist if full migration was run
        optional_fields = {
            "action": "BUY",
            "quantity": int(quantity),
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
    Closes an 'OPEN' position with defensive column handling.
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
        
        # 2. Calculate PnL
        pnl_pct = (float(exit_price) - entry_price) / entry_price * 100 if entry_price > 0 else 0
        now = datetime.now(SET_TZ)
        
        # 3. Update payload
        update_payload = {
            "exit_price": float(exit_price),
            "status": "CLOSED",
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

def get_paper_portfolio_metrics(supabase):
    """
    Calculates portfolio metrics from trading_log.
    """
    if not supabase:
        return None
    
    try:
        # Fetch all paper trades (status OPEN or CLOSED)
        resp = supabase.table("trading_log").select("*").or_("status.eq.OPEN,status.eq.CLOSED").execute()
        if not resp.data:
            return {
                "total_portfolio_value": 0.0,
                "win_rate": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "open_count": 0,
                "closed_count": 0
            }
        
        df = pd.DataFrame(resp.data)
        
        # Defensive handling for missing columns (should not happen after migration)
        required_cols = ['status', 'entry_price', 'exit_price', 'quantity', 'pnl_percent', 'last_price']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col != 'status' else 'UNKNOWN'
        
        # Realized PnL (Closed trades)
        closed_trades = df[df['status'] == 'CLOSED']
        realized_pnl = 0.0
        win_rate = 0.0
        if not closed_trades.empty:
            # PnL in THB = (exit - entry) * quantity
            realized_pnl = ((pd.to_numeric(closed_trades['exit_price'], errors='coerce').fillna(0) - 
                             pd.to_numeric(closed_trades['entry_price'], errors='coerce').fillna(0)) * 
                            pd.to_numeric(closed_trades['quantity'], errors='coerce').fillna(0)).sum()
            
            pnl_pcts = pd.to_numeric(closed_trades['pnl_percent'], errors='coerce').fillna(0)
            win_rate = (pnl_pcts > 0).mean() * 100 if len(pnl_pcts) > 0 else 0
            
        # Unrealized PnL (Open trades)
        open_trades = df[df['status'] == 'OPEN']
        unrealized_pnl = 0.0
        total_open_value = 0.0
        if not open_trades.empty:
            entry_pxs = pd.to_numeric(open_trades['entry_price'], errors='coerce').fillna(0)
            last_pxs = pd.to_numeric(open_trades['last_price'], errors='coerce').fillna(0)
            quants = pd.to_numeric(open_trades['quantity'], errors='coerce').fillna(0)
            
            unrealized_pnl = ((last_pxs - entry_pxs) * quants).sum()
            total_open_value = (last_pxs * quants).sum()
            
        return {
            "total_portfolio_value": round(total_open_value + realized_pnl, 2),
            "win_rate": round(win_rate, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "open_count": len(open_trades),
            "closed_count": len(closed_trades)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
