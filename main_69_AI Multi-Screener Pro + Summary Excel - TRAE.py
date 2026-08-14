import pandas as pd
import numpy as np
import yfinance as yf
import yahooquery as yq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw
from tkinter import Tk, Label, Entry, Button, filedialog, Text, Frame, END
from tkinter.messagebox import showinfo
from datetime import datetime
import gc
import time


# --- 1. ฟังก์ชันคำนวณ (ประสิทธิภาพสูงสุดเหมือนเดิม) ---
def calculate_all_indicators(data):
    if len(data) < 60: return None
    df = data.copy()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0);
    loss = delta.clip(upper=0).abs()
    avg_gain = gain.ewm(span=7, adjust=False).mean()
    avg_loss = loss.ewm(span=7, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_Gap'] = (df['EMA5'] - df['EMA20']) / (df['EMA20'] + 1e-9)
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (std20 * 4) / (ma20 + 1e-9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()
    df['Anomaly_Signal'] = np.where((df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9) > 2.0), 1.0, 0.0)
    return df.fillna(0)


def calculate_realistic_stats(sim_scores, matched_futures, matched_past_ends):
    changes, ups, downs = [], [], []
    for i in range(len(matched_futures)):
        pct_change = ((matched_futures[i][-1] - matched_past_ends[i]) / matched_past_ends[i]) * 100
        changes.append(pct_change);
        weight = sim_scores[i]
        if pct_change > 0:
            ups.append(weight)
        else:
            downs.append(weight)
    total_weight = sum(ups) + sum(downs)
    prob_up = (sum(ups) / total_weight * 100) if total_weight > 0 else 50
    return prob_up, 100 - prob_up, np.mean(changes)


# --- 2. ฟังก์ชันวิเคราะห์พร้อมกราฟแบบละเอียด (Full Visuals) ---
def process_full_analysis(ticker, end_date, save_path, log_widget):
    try:
        log_widget.insert(END, f"> Analyzing {ticker}...")
        log_widget.update_idletasks()

        data = yq.Ticker(ticker).history(start="2018-01-01", end=end_date).reset_index()
        if "symbol" in data.columns: data = data[data["symbol"] == ticker]
        df = data.set_index("date")[["close", "volume"]].rename(columns={"close": "Close", "volume": "Volume"})
        df = calculate_all_indicators(df)

        lookback, future_days = 45, 30
        features = ['RSI', 'Volume', 'Close', 'Anomaly_Signal', 'EMA_Gap', 'BB_Width', 'MACD_Hist']

        # Hybrid Weights Logic
        curr_vol_ratio = df['Volume'].iloc[-1] / (df['Volume'].rolling(20).mean().iloc[-1] + 1e-9)
        curr_ema_gap = abs(df['EMA_Gap'].iloc[-1])
        if curr_vol_ratio > 1.8 or curr_ema_gap > 0.03:
            weights = np.array([0.15, 0.20, 0.25, 0.10, 0.15, 0.05, 0.10])
            mode_text = "HYBRID ANALYSIS"
        else:
            weights = np.array([0.10, 0.05, 0.50, 0.05, 0.10, 0.10, 0.10])
            mode_text = "STANDARD (Shape-Heavy)"

        patterns, future_patterns, past_end_prices = [], [], []
        scaler = MinMaxScaler()
        valid_range = len(df) - lookback - future_days

        for i in range(valid_range):
            slice_data = df[features].iloc[i:i + lookback].values
            if np.std(slice_data[:, 2]) < (slice_data[:, 2].mean() * 0.005): continue
            patterns.append(slice_data)
            future_patterns.append(df['Close'].iloc[i + lookback: i + lookback + future_days].values)
            past_end_prices.append(df['Close'].iloc[i + lookback - 1])

        current_dna = df[features].iloc[-lookback:].values
        current_scaled = (scaler.fit_transform(current_dna) * weights).flatten()
        distances = [dtw.distance(current_scaled, (scaler.fit_transform(p) * weights).flatten()) for p in patterns]
        best_indices = np.argsort(distances)[:3]
        max_dist = max(distances) if distances else 1
        sim_scores = [100 * (1 - (distances[idx] / (max_dist + 1e-9))) for idx in best_indices]

        p_up, p_down, exp_move = calculate_realistic_stats(sim_scores, [future_patterns[idx] for idx in best_indices],
                                                           [past_end_prices[idx] for idx in best_indices])

        # --- ส่วนสร้างกราฟแบบละเอียด (คืนค่าความสวยงาม) ---
        pdf_file = f"{save_path}/AI_{ticker}_{end_date.replace('-', '')}.pdf"
        with PdfPages(pdf_file) as pdf:
            fig, ax = plt.subplots(figsize=(11, 7))
            curr_prices = df['Close'].iloc[-lookback:].values
            ax.plot(range(-lookback, 0), curr_prices, color='red', lw=3, label="Current Price")

            for i, idx in enumerate(best_indices):
                ratio = curr_prices[-1] / patterns[idx][-1, 2]
                combined = np.concatenate([patterns[idx][:, 2] * ratio, future_patterns[idx] * ratio])
                ax.plot(range(-lookback, future_days), combined, alpha=0.6, ls='--',
                        label=f"Match {i + 1} ({sim_scores[i]:.1f}%)")

            # กล่องข้อความแบบเดิมที่ตัวเลขครบ
            sim_alert = " *Low Similarity Alert*" if np.mean(sim_scores) < 70 else ""
            stats_box = f"AI PREDICTION: {mode_text}{sim_alert}\n" \
                        f"----------------------------------------\n" \
                        f"Chance of Going UP: {p_up:.1f}%\n" \
                        f"Chance of Going DOWN: {p_down:.1f}%\n" \
                        f"Average Expected Move: {exp_move:+.2f}%"

            ax.text(0.02, 0.85, stats_box, transform=ax.transAxes, fontsize=10, family='monospace',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#3498db', boxstyle='round,pad=1'))

            ax.axvline(x=0, color='black', lw=2)
            ax.set_title(f"SCENARIO ANALYSIS: {ticker} (As of {end_date})", fontsize=14, fontweight='bold')
            ax.set_xlabel("Days from Today");
            ax.set_ylabel("Price Level");
            ax.legend(loc='lower left');
            ax.grid(True, alpha=0.2)
            pdf.savefig(fig);
            plt.close(fig)

        log_widget.insert(END, " [DONE]\n");
        log_widget.see(END)
        return {"Ticker": ticker, "Prob Up (%)": p_up, "Prob Down (%)": p_down, "Exp. Move (%)": exp_move,
                "Similarity": np.mean(sim_scores), "Mode": mode_text}
    except Exception as e:
        log_widget.insert(END, f" [ERROR: {str(e)}]\n");
        return None
    finally:
        gc.collect()


# --- 3. UI (รองรับ Multi-Ticker + Excel) ---
def show_form():
    root = Tk()
    root.withdraw()
    root.title("AI Full Visual Screener + Excel Summary")
    f = Frame(root, padx=20, pady=20);
    f.pack()

    Label(f, text="Stock Tickers (e.g., KTC.BK, RCL.BK):", font=('Arial', 9, 'bold')).pack(anchor='w')
    t_entry = Entry(f, width=65);
    t_entry.insert(0, "KTC.BK, AMATA.BK, RCL.BK, CPF.BK");
    t_entry.pack(pady=5)

    Label(f, text="Analysis Date (YYYY-MM-DD):").pack(anchor='w')
    d_entry = Entry(f, width=65);
    d_entry.insert(0, datetime.now().strftime("%Y-%m-%d"));
    d_entry.pack(pady=5)

    log_txt = Text(f, height=12, width=75, font=('Consolas', 9));
    log_txt.pack(pady=10)

    def run_screener():
        tickers = [t.strip().upper() for t in t_entry.get().split(",")]
        save_path = filedialog.askdirectory()
        if not save_path: return

        results = []
        log_txt.delete(1.0, END)
        log_txt.insert(END, f"🚀 Analyzing {len(tickers)} stocks with Full Visuals...\n" + "-" * 60 + "\n")

        for i, ticker in enumerate(tickers):
            res = process_full_analysis(ticker, d_entry.get(), save_path, log_txt)
            if res: results.append(res)
            if i < len(tickers) - 1: time.sleep(2)

        if results:
            try:
                pd.DataFrame(results).to_excel(
                    f"{save_path}/Full_Summary_{datetime.now().strftime('%H%M%S')}.xlsx",
                    index=False
                )
                log_txt.insert(END, f"\n✅ Excel Summary and Detailed PDFs are ready!")
                showinfo("Complete", "วิเคราะห์เสร็จสิ้น! ไฟล์ PDF และ Excel ถูกบันทึกแล้ว")
            except Exception as e:
                log_txt.insert(END, f"\n[ERROR] Save Excel failed: {str(e)}\n")
                showinfo("Error", f"บันทึกไฟล์ Excel ไม่สำเร็จ: {str(e)}")

    Button(f, text="📊 RUN DETAILED SCREENER", command=run_screener, bg="#2c3e50", fg="white",
           font=('Arial', 10, 'bold'), height=2).pack(pady=10, fill='x')
    root.update_idletasks()
    try:
        w = root.winfo_width()
        h = root.winfo_height()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)
        root.geometry(f"+{x}+{y}")
    except Exception:
        pass
    root.deiconify()
    try:
        root.lift()
        root.focus_force()
        root.attributes("-topmost", True)
        root.after(250, lambda: root.attributes("-topmost", False))
    except Exception:
        pass
    root.mainloop()


if __name__ == "__main__":
    show_form()
