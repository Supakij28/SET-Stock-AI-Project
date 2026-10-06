import pandas as pd
import numpy as np
from datetime import datetime
import pytz

SET_TZ = pytz.timezone('Asia/Bangkok')

def execute_paper_buy(supabase, ticker, entry_price, quantity, signal_type="WVF"):
    """
    Inserts an 'OPEN' position into trading_log.
    """
    if not supabase:
        return False, "Supabase client not initialized"
    
    try:
        now = datetime.now(SET_TZ)
        payload = {
            "ticker": ticker,
            "action": "BUY",
            "entry_price": float(entry_price),
            "last_price": float(entry_price), # Initial last_price
            "quantity": int(quantity),
            "entry_date": now.strftime('%Y-%m-%d'),
            "timestamp": now.isoformat(),
            "status": "OPEN",
            "signal": signal_type,
            "run_id": f"PAPER_{now.strftime('%Y%m%d_%H%M%S')}"
        }
        
        resp = supabase.table("trading_log").insert(payload).execute()
        if resp.data:
            return True, f"Successfully executed Paper Buy for {ticker}"
        else:
            return False, "Failed to insert record into Supabase"
    except Exception as e:
        return False, str(e)

def execute_paper_sell(supabase, position_id, exit_price, exit_reason="MANUAL"):
    """
    Closes an 'OPEN' position, calculates PnL, and updates status to 'CLOSED'.
    """
    if not supabase:
        return False, "Supabase client not initialized"
    
    try:
        # 1. Fetch original position
        resp = supabase.table("trading_log").select("*").eq("id", position_id).single().execute()
        if not resp.data:
            return False, "Position not found"
        
        pos = resp.data
        entry_price = float(pos['entry_price'])
        quantity = int(pos['quantity'])
        
        # 2. Calculate PnL
        pnl_pct = (float(exit_price) - entry_price) / entry_price * 100
        now = datetime.now(SET_TZ)
        
        # 3. Update to CLOSED
        update_payload = {
            "exit_price": float(exit_price),
            "exit_date": now.strftime('%Y-%m-%d'),
            "status": "CLOSED",
            "pnl_percent": round(pnl_pct, 2),
            "exit_reason": exit_reason,
            "action": "SELL"
        }
        
        resp_upd = supabase.table("trading_log").update(update_payload).eq("id", position_id).execute()
        if resp_upd.data:
            return True, f"Successfully closed position for {pos['ticker']} with {pnl_pct:.2f}% PnL"
        else:
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
        
        # Realized PnL (Closed trades)
        closed_trades = df[df['status'] == 'CLOSED']
        realized_pnl = 0.0
        win_rate = 0.0
        if not closed_trades.empty:
            # PnL in THB = (exit - entry) * quantity
            # Assuming commission is handled outside for simplicity, or we can add it here
            realized_pnl = ((closed_trades['exit_price'] - closed_trades['entry_price']) * closed_trades['quantity']).sum()
            win_rate = (closed_trades['pnl_percent'] > 0).mean() * 100
            
        # Unrealized PnL (Open trades)
        open_trades = df[df['status'] == 'OPEN']
        unrealized_pnl = 0.0
        total_open_value = 0.0
        if not open_trades.empty:
            # We need current price to calculate unrealized PnL accurately
            # For now, we use 'last_price' which should be updated by a separate engine
            unrealized_pnl = ((open_trades['last_price'] - open_trades['entry_price']) * open_trades['quantity']).sum()
            total_open_value = (open_trades['last_price'] * open_trades['quantity']).sum()
            
        return {
            "total_portfolio_value": total_open_value + realized_pnl, # Simplified
            "win_rate": round(win_rate, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "open_count": len(open_trades),
            "closed_count": len(closed_trades)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
