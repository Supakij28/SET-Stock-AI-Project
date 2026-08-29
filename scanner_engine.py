import pandas as pd
import numpy as np
import requests
import yahooquery as yq
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from datetime import datetime, time, timedelta
import pytz

# Constants
SET_TZ = pytz.timezone('Asia/Bangkok')

def calculate_quant_indicators(df, rsi_period=14, ema_fast=10, ema_slow=50):
    """Unified Indicator Calculation Engine."""
    d = df.copy()
    
    # RSI
    delta = d['Close'].diff()
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs()
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    d['RSI'] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))
    
    # EMAs
    d['EMA_Fast'] = d['Close'].ewm(span=ema_fast, adjust=False).mean()
    d['EMA_Slow'] = d['Close'].ewm(span=ema_slow, adjust=False).mean()
    
    # Relative Volume (RV)
    d['RV'] = d['Volume'] / (d['Volume'].rolling(20).mean() + 1e-9)
    
    # ATR
    high_low = d['High'] - d['Low']
    high_close = (d['High'] - d['Close'].shift()).abs()
    low_close = (d['Low'] - d['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    d['ATR'] = true_range.rolling(14).mean()
    d['ATR_Avg_5'] = d['ATR'].rolling(5).mean()
    
    return d.dropna()

def get_pre_breakout_scanner(df, breakout_threshold=0.05, lookback=5, mode='bullish'):
    """Improved Scanner: Multivariate DTW + Slope Filter + Z-score Normalization."""
    if len(df) < 100: return {'matches': [], 'summary': {'count': 0}}
    
    df_scan = df.copy()
    df_scan['Pct_Change'] = df_scan['Close'].pct_change()
    df_scan['Volatility'] = df_scan['Close'].rolling(5).std()
    
    if mode == 'bullish':
        events = df_scan[df_scan['Pct_Change'] >= breakout_threshold].index.tolist()
    else:
        events = df_scan[df_scan['Pct_Change'] <= -breakout_threshold].index.tolist()
    
    patterns = []
    scaler = StandardScaler()
    
    for b_date in events:
        try:
            idx = df_scan.index.get_loc(b_date)
            if idx < lookback: continue
            pattern_data = df_scan.iloc[idx-lookback : idx]
            p_norm = scaler.fit_transform(pattern_data[['Close']]).flatten()
            v_norm = scaler.fit_transform(pattern_data[['Volume']]).flatten()
            patterns.append({
                'date': b_date, 
                'p_norm': p_norm, 
                'v_norm': v_norm, 
                'price_pattern': p_norm, # For consensus
                'vol_pattern': v_norm,   # For consensus
                'jump': df_scan['Pct_Change'].iloc[idx] * 100
            })
        except:
            continue
    
    if not patterns: return {'matches': [], 'summary': {'count': 0}}
    
    current_5 = df_scan.iloc[-lookback:]
    curr_p_norm = scaler.fit_transform(current_5[['Close']]).flatten()
    curr_v_norm = scaler.fit_transform(current_5[['Volume']]).flatten()
    
    matches = []
    for p in patterns:
        dist_p = dtw.distance(curr_p_norm, p['p_norm'], window=2)
        dist_v = dtw.distance(curr_v_norm, p['v_norm'], window=2)
        total_dist = (0.7 * dist_p) + (0.3 * dist_v)
        matches.append({'date': p['date'], 'dist': total_dist, 'jump': p['jump']})
    
    sorted_matches = sorted(matches, key=lambda x: x['dist'])
    return {
        'matches': sorted_matches[:3], 
        'summary': {'count': len(patterns)},
        'patterns': patterns # Include all patterns for consensus calculation
    }

def get_recovery_signals(ticker, df):
    """Detects Bottom Fishing / Recovery signals."""
    if df is None or len(df) < 20: return None
    d = df.tail(20).copy()
    rsi_curr = d['RSI'].iloc[-1]
    rsi_prev = d['RSI'].iloc[-2]
    
    is_oversold = rsi_curr < 35 or rsi_prev < 35
    is_rsi_turning = rsi_curr > rsi_prev and rsi_curr > 30
    
    recent_returns = d['Close'].pct_change().tail(5)
    negative_days = (recent_returns < 0).sum()
    is_exhausted = negative_days >= 3
    
    c_open, c_high, c_low, c_close = d['Open'].iloc[-1], d['High'].iloc[-1], d['Low'].iloc[-1], d['Close'].iloc[-1]
    body_size = abs(c_close - c_open)
    lower_wick = min(c_open, c_close) - c_low
    is_bullish_pin = lower_wick > (body_size * 1.5) and c_close > c_low
    
    score = 0
    reasons = []
    if is_oversold: 
        score += 40
        reasons.append("Oversold (RSI < 35)")
    if is_rsi_turning: 
        score += 20
        reasons.append("RSI Turning Up")
    if is_exhausted: 
        score += 15
        reasons.append("Selling Exhaustion")
    if is_bullish_pin: 
        score += 25
        reasons.append("Bullish Pin Bar")
    
    if score >= 50:
        return {
            'ticker': ticker,
            'recovery_score': score, 
            'rsi': rsi_curr, 
            'price': c_close, 
            'is_pin': is_bullish_pin,
            'reasons': reasons
        }
    return None

def get_signal_performance_stats(supabase=None):
    """Calculate Win Rate for each signal type from Supabase (Unified)."""
    if not supabase:
        return {}
        
    try:
        response = supabase.table("scan_results") \
            .select("signal_type, outcome_label") \
            .not_.is_("outcome_label", "null") \
            .execute()
        df = pd.DataFrame(response.data)
        
        if df.empty:
            return {}
        
        perf = df.groupby('signal_type').agg(
            Total=('outcome_label', 'count'),
            Wins=('outcome_label', lambda x: (x == 'Win').sum())
        )
        perf['Win_Rate'] = (perf['Wins'] / perf['Total']) * 100
        return perf.to_dict('index')
    except Exception as e:
        print(f"Stats Error: {e}")
        return {}

def calculate_conviction_score(ticker, signal, similarity, mtf_score, regime, perf_stats, rsi=50, rel_vol=1.0, score_velocity=0, sector_rs=0, price_change=0, hist_scores=None):
    """
    Unified Conviction Score Engine.
    Calculate the conviction score for a stock based on various metrics.
    Supports Strategy Classification and Sector Relative Strength (SRS).
    """
    persistence = "Flat"
    if hist_scores is not None and not hist_scores.empty and len(hist_scores) >= 2:
        latest = hist_scores['bull_score'].iloc[0]
        prev = hist_scores['bull_score'].iloc[-1]
        if latest > prev: persistence = "Rising 📈"
        elif latest < prev: persistence = "Falling 📉"
    
    sig_stats = perf_stats.get(signal, {})
    sig_win_rate = sig_stats.get('Win_Rate', 0)
    sig_total = sig_stats.get('Total', 0)
    
    formula_score = 0
    reasons = []
    warnings = []
    strategy = "SWING" # Default strategy
    
    is_spike_potential = (score_velocity > 10) or (rel_vol > 1.8) or (signal in ['PRE-FLY', 'GOLDEN BUY'])
    
    if is_spike_potential:
        strategy = "DAY TRADE (SPIKE)"
        if rsi > 65:
            warnings.append("⚠️ DAY TRADE Risk: High RSI (Spike & Drop potential)")
        if persistence == "Falling 📉":
            warnings.append("⚠️ DAY TRADE Risk: Weakening Momentum")
    
    if regime == 'BULL': 
        formula_score += 20
        reasons.append("Market BULL (+20)")
        
    if signal in ['PRE-FLY', 'GOLDEN BUY']: 
        formula_score += 30
        reasons.append(f"Signal {signal} (+30)")
    elif signal == 'SILENT ACCUM':
        formula_score += 15
        reasons.append("Signal Accumulation (+15)")
        
    if price_change > 0 and (price_change - sector_rs) < 0:
        formula_score += 20
        reasons.append("Leading Star 🌟 (Up while Sector Down) (+20)")
    elif sector_rs > 0.5:
        formula_score += 10
        reasons.append(f"Outperformer (RS: {sector_rs:+.1f}%) (+10)")
    elif sector_rs < -1.0:
        formula_score -= 25
        warnings.append(f"Underperformer ⚠️ (RS: {sector_rs:+.1f}%) (-25)")
        strategy = "WAIT (Weak Sector Flow)"
    elif sector_rs < -0.5:
        formula_score -= 10
        warnings.append(f"Underperformer (RS: {sector_rs:+.1f}%) (-10)")
    elif price_change < 0 and (price_change - sector_rs) > 0:
        formula_score -= 20
        warnings.append("Laggard ⚠️ (Down while Sector Up) (-20)")

    if sig_total >= 5:
        if sig_win_rate >= 60:
            formula_score += 15
            reasons.append(f"High Reliability Signal (+15)")
        elif sig_win_rate < 40:
            formula_score -= 25
            warnings.append(f"Low Reliability Signal ({sig_win_rate:.1f}% Win Rate) (-25)")
        
    if persistence == "Rising 📈": 
        formula_score += 20
        reasons.append("Score Rising (+20)")
        
    if similarity >= 90: 
        formula_score += 30
        reasons.append("Pattern Match > 90% (+30)")
    elif similarity >= 80:
        formula_score += 15
        reasons.append("Pattern Match > 80% (+15)")
        
    if mtf_score >= 80:
        formula_score += 20
        reasons.append("Strong MTF Conf (+20)")
    elif mtf_score >= 60:
        formula_score += 10
        reasons.append("Moderate MTF Conf (+10)")

    if 40 <= rsi <= 55:
        formula_score += 15
        reasons.append("Audit: RSI Sweet Spot (+15)")

    if rsi > 70:
        formula_score -= 20
        warnings.append("Overbought RSI (>70) (-20)")
    if rsi > 80:
        formula_score -= 30
        warnings.append("Extreme Overbought RSI (>80) (-30)")

    if signal in ['WAIT', 'WAIT (DOWNTREND)', 'WAIT (BEARISH TRAP)', 'FADING MOMENTUM', 'CONFLICT (HIGH RISK)']:
        formula_score -= 50
        warnings.append(f"Neutral/Bearish Signal: {signal} (-50)")
    
    if persistence == "Falling 📉":
        formula_score -= 15
        warnings.append("Momentum is Weakening (-15)")

    if signal == 'REJECTION WICK':
        penalty = -60
        if regime == 'BULL': penalty += 10
        if rsi < 50: penalty += 25
        if rsi > 70: penalty -= 40
        if rel_vol > 1.8: penalty -= 20
        formula_score += penalty
        if penalty < 0:
            warnings.append(f"REJECTION WICK ({penalty})")
        else:
            reasons.append(f"Wick at Support/Neutral RSI ({penalty})")
            
    elif signal == 'PIN BAR (SUPPORT)':
        formula_score += 25
        reasons.append("Bullish Pin Bar at Support (+25)")
            
    elif signal == 'CONFLICT (HIGH RISK)':
        formula_score -= 20
        warnings.append("CONFLICT SIGNAL (-20)")
        
    return formula_score, strategy, reasons, warnings

def get_market_regime():
    """Determine if SET Index is in Bull or Bear market."""
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        t = yq.Ticker("^SET.BK", session=session)
        set_idx = t.history(period='2y').reset_index()
        if not set_idx.empty:
            set_idx['date'] = pd.to_datetime(set_idx['date'])
            set_idx = set_idx.set_index("date")[["close"]].rename(columns={"close": "Close"})
            set_idx['EMA200'] = set_idx['Close'].ewm(span=200, adjust=False).mean()
            curr_price = set_idx['Close'].iloc[-1]
            ema200 = set_idx['EMA200'].iloc[-1]
            return "BULL" if curr_price > ema200 else "BEAR", curr_price
    except:
        pass
    return "UNKNOWN", 0

def get_mtf_confluence(ticker):
    """Multi-Timeframe analysis (1H and 15M)."""
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        t = yq.Ticker(ticker, session=session)
        
        df_1h = t.history(period='7d', interval='1h').reset_index()
        if df_1h.empty: return "N/A", 0
        
        close_1h = df_1h['close']
        ema10_1h = close_1h.ewm(span=10, adjust=False).mean()
        ema20_1h = close_1h.ewm(span=20, adjust=False).mean()
        is_1h_bull = ema10_1h.iloc[-1] > ema20_1h.iloc[-1]
        is_1h_trending = close_1h.iloc[-1] > ema10_1h.iloc[-1]
        
        df_15m = t.history(period='2d', interval='15m').reset_index()
        if df_15m.empty: return "1H Only", 50 if is_1h_bull else 0
        
        close_15m = df_15m['close']
        delta = close_15m.diff()
        gain = delta.clip(lower=0)
        loss = delta.clip(upper=0).abs()
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rsi_15m = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))
        
        is_15m_recovering = rsi_15m.iloc[-1] > rsi_15m.iloc[-2] and rsi_15m.iloc[-1] > 45
        is_15m_bull = close_15m.iloc[-1] > close_15m.ewm(span=10, adjust=False).mean().iloc[-1]
        
        score = 0
        status = []
        if is_1h_bull: score += 30; status.append("1H Bull")
        if is_1h_trending: score += 20; status.append("1H Trend")
        if is_15m_recovering: score += 30; status.append("15M RSI Up")
        if is_15m_bull: score += 20; status.append("15M Bull")
            
        return " + ".join(status) if status else "No Confluence", score
    except:
        return "Error", 0

def validate_scanner_accuracy(df, winning_patterns, price_threshold=80.0, vol_threshold=70.0, lookback=5):
    """Unified Validator for Pattern Consensus."""
    if len(df) < 100 or not winning_patterns: return None
    scaler = StandardScaler()
    df_scan = df.copy()
    df_scan['Volatility'] = df_scan['Close'].rolling(5).std()
    df_scan['EMA_Cross_Up'] = (df_scan['EMA_Fast'] > df_scan['EMA_Slow']) & (df_scan['EMA_Fast'].shift(1) <= df_scan['EMA_Slow'].shift(1))
    df_scan['EMA_Slow_Slope'] = df_scan['EMA_Slow'].diff(3)
    df_scan['Strong_Trend'] = (df_scan['Close'] > df_scan['EMA_Slow']) & (df_scan['EMA_Fast'] > df_scan['EMA_Slow']) & (df_scan['EMA_Slow_Slope'] > 0)
    df_scan['Next_3D_Max_Return'] = df_scan['High'].rolling(3).max().shift(-3) / df_scan['Close'] - 1
    
    regime, _ = get_market_regime()
    results = []
    scan_start = max(lookback, len(df_scan) - 500) 
    
    for i in range(scan_start, len(df_scan) - 3):
        has_cross = df_scan['EMA_Cross_Up'].iloc[i-lookback : i].any()
        is_trending = df_scan['Strong_Trend'].iloc[i-1]
        if not (has_cross or is_trending): continue
            
        adj_p_threshold = price_threshold if regime == "BULL" else price_threshold + 5.0
        adj_v_threshold = vol_threshold if regime == "BULL" else vol_threshold + 5.0
            
        window = df_scan.iloc[i-lookback : i]
        if (window['Close'].iloc[-1] - window['Close'].iloc[-3]) / window['Close'].iloc[-3] < -0.01: continue

        win_p_norm = scaler.fit_transform(window[['Close']]).flatten()
        win_v_norm = scaler.fit_transform(window[['Volume']]).flatten()
        
        best_sim_p = 0
        best_sim_v = 0
        for p in winning_patterns:
            sim_p = max(0, 100 - (dtw.distance(win_p_norm, p['price_pattern'], window=2) * 20))
            sim_v = max(0, 100 - (dtw.distance(win_v_norm, p['vol_pattern'], window=2) * 20))
            if sim_p > best_sim_p: 
                best_sim_p = sim_p
                best_sim_v = sim_v
        
        if best_sim_p >= adj_p_threshold and best_sim_v >= adj_v_threshold:
            next_ret = df_scan['Next_3D_Max_Return'].iloc[i-1]
            results.append({'Hit': 1 if next_ret > 0.01 else 0})
            
    if not results: return None
    hit_rate = (sum([r['Hit'] for r in results]) / len(results)) * 100
    return {'summary': {'Hit Rate': hit_rate}}

def core_strategy_scanner(ticker, df_raw, target_date=None, mtf_check=True):
    """
    Unified Core Strategy Scanner (V7). 
    Used by both Manual and Auto scans for 100% consistency.
    """
    if df_raw is None or len(df_raw) < 100: return None

    # 1. Indicators
    df = calculate_quant_indicators(df_raw)
    
    # 2. Bullish/Bearish Scan
    bullish = get_pre_breakout_scanner(df_raw, mode='bullish')
    bearish = get_pre_breakout_scanner(df_raw, mode='bearish')
    
    bull_score = 0; bull_jump = 0
    if bullish and bullish['matches']:
        best_bull = bullish['matches'][0]
        bull_score = max(0, 100 - (best_bull['dist'] * 20))
        bull_jump = best_bull['jump']
    
    bear_score = 0; bear_jump = 0
    if bearish and bearish['matches']:
        best_bear = bearish['matches'][0]
        bear_score = max(0, 100 - (best_bear['dist'] * 20))
        bear_jump = best_bear['jump']
    
    # 3. Pattern Consensus
    consensus = 0
    try:
        if bullish and bullish['patterns']:
            val = validate_scanner_accuracy(df, bullish['patterns'])
            hit_rate = val['summary']['Hit Rate'] if val else 0
            
            ema_f_curr = df['EMA_Fast'].iloc[-1]
            ema_s_curr = df['EMA_Slow'].iloc[-1]
            close_curr = df['Close'].iloc[-1]
            ema_s_slope = (df['EMA_Slow'].iloc[-1] - df['EMA_Slow'].iloc[-5]) / df['EMA_Slow'].iloc[-5] * 100
            
            trend_score = 0
            if close_curr > ema_s_curr: trend_score += 40
            if ema_f_curr > ema_s_curr: trend_score += 40
            if ema_s_slope > 0: trend_score += 20
            
            consensus = (hit_rate * 0.6) + (trend_score * 0.4) if hit_rate > 0 else trend_score
    except:
        pass
    
    # 4. Score Velocity
    score_velocity = 0
    if len(df_raw) > 2:
        df_prev = df_raw.iloc[:-1]
        bull_prev = get_pre_breakout_scanner(df_prev, mode='bullish')
        if bull_prev and bull_prev['matches']:
            prev_bull_score = max(0, 100 - (bull_prev['matches'][0]['dist'] * 20))
            score_velocity = bull_score - prev_bull_score
            
    # 5. Price Action & Volume
    c_open, c_high, c_low, c_close = df_raw['Open'].iloc[-1], df_raw['High'].iloc[-1], df_raw['Low'].iloc[-1], df_raw['Close'].iloc[-1]
    body_size = abs(c_close - c_open)
    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low
    
    rel_vol = df['RV'].iloc[-1]
    vol_avg5 = df_raw['Volume'].rolling(5).mean().iloc[-1]
    curr_vol = df_raw['Volume'].iloc[-1]
    is_vol_compressed = curr_vol < vol_avg5 and 0.6 <= rel_vol <= 1.0
    
    pct_change = ((c_close / df_raw['Close'].iloc[-2]) - 1) * 100 if len(df_raw) > 1 else 0
    atc_risk = ((c_high - c_close) / c_high) * 100 if c_high > 0 else 0
    
    # 6. Detection Logic
    is_rejection_wick = upper_wick > (body_size * 1.5) and (c_high > c_close)
    is_pinbar = lower_wick > (body_size * 1.5) and (c_close > c_open)
    
    rsi_curr = df['RSI'].iloc[-1]
    ema_fast = df['EMA_Fast'].iloc[-1]
    ema_slow = df['EMA_Slow'].iloc[-1]
    is_downtrend = c_close < ema_slow
    
    # 7. MTF Confirmation
    mtf_status = "N/A"
    mtf_score = 0
    if mtf_check and target_date is None and bull_score > 40:
        mtf_status, mtf_score = get_mtf_confluence(ticker)
    
    # 8. Recovery Signal
    recovery_data = get_recovery_signals(ticker, df)
    is_recovery = True if recovery_data else False
    
    atr_now = df['ATR'].iloc[-1]
    atr_avg5 = df['ATR_Avg_5'].iloc[-1]
    high_vol = atr_now > (atr_avg5 * 1.2)
    
    # 9. Signal Logic V7
    score_diff = bull_score - bear_score
    is_pre_fly = 12.0 <= score_diff <= 22.0
    is_caution = score_diff > 25.0
    is_momentum_strong = rel_vol > 1.5
    is_silent_accum = (pct_change > 0 and (is_vol_compressed or 0.8 <= rel_vol <= 1.2) and atc_risk < 0.5)
    is_rr_good = bull_jump > abs(bear_jump)
    is_blow_off = (rsi_curr > 72 and rel_vol > 1.8 and pct_change > 4.5) or (rsi_curr > 82)
    is_conflict_zone = (bull_score > 70 and bear_score > 70)
    is_bearish_trap = (bear_score > bull_score) or (bear_score > 80.0)
    is_overextended = (c_close > ema_fast * 1.06)
    is_fading_momentum = score_velocity < -5.0 and score_diff < 15
    
    signal = 'WAIT'
    strategy = 'SWING'
    
    if is_rejection_wick:
        signal = 'REJECTION WICK'; strategy = 'FADING'
    elif is_pinbar and is_downtrend:
        signal = 'PIN BAR (SUPPORT)'; strategy = 'REVERSAL'
    elif is_fading_momentum:
        signal = 'FADING MOMENTUM'; strategy = 'CAUTION'
    elif is_blow_off or is_overextended:
        signal = 'TAKE PROFIT / WAIT'; strategy = 'OVEREXTENDED'
    elif is_bearish_trap and score_diff < 5:
        signal = 'WAIT (BEARISH TRAP)'; strategy = 'CAUTION'
    elif is_conflict_zone:
        signal = 'CONFLICT (HIGH RISK)'; strategy = 'INDECISION'
    elif pct_change > 0:
        if is_pre_fly and is_momentum_strong and is_rr_good:
            signal = 'PRE-FLY'; strategy = 'MOMENTUM'
        elif is_silent_accum and score_diff > 5:
            signal = 'SILENT ACCUM'; strategy = 'ACCUMULATION'
        elif is_caution:
            signal = 'CAUTION'; strategy = 'SWING'
        elif 2.0 <= score_diff <= 15.0 and is_momentum_strong and is_rr_good:
            if is_downtrend: signal = 'BOTTOM PLAY'; strategy = 'REVERSAL'
            else: signal = 'GOLDEN BUY'; strategy = 'BREAKOUT'
        elif bull_score > 80 and score_diff > 15 and is_momentum_strong:
            signal = 'BUY'; strategy = 'BREAKOUT'
        elif is_downtrend:
            signal = 'WAIT (DOWNTREND)'; strategy = 'SWING'
    elif is_recovery:
        signal = 'RECOVERY'; strategy = 'BOTTOM FISHING'
    
    return {
        'ticker': ticker,
        'score': bull_score,
        'bull_score': bull_score,
        'bear_score': bear_score,
        'score_diff': score_diff,
        'signal': signal,
        'strategy': strategy,
        'close_price': c_close,
        'change_percent': pct_change,
        'rsi': rsi_curr,
        'volume': c_close * df_raw['Volume'].iloc[-1],
        'is_recovery': is_recovery,
        'is_pinbar': is_pinbar,
        'is_silent_accum': is_silent_accum,
        'atc_risk': atc_risk,
        'consensus': consensus,
        'mtf_status': mtf_status,
        'mtf_score': mtf_score,
        'recovery_data': recovery_data,
        'rel_vol': rel_vol,
        'atr_now': atr_now,
        'high_vol': high_vol,
        'bull_jump': bull_jump,
        'bear_jump': bear_jump,
        'score_velocity': score_velocity
    }
