# Quant Strategy Station (SET100 Analysis)

เครื่องมือวิเคราะห์หุ้นในดัชนี SET100 ด้วยเทคนิคอลคอนเทนต์และระบบ Sector Relative Strength (SRS) พัฒนาด้วย Python และ Streamlit

## ฟีเจอร์หลัก
- **SET100 Multi-Scanner:** สแกนหุ้นทั้งดัชนีเพื่อหาจังหวะซื้อขาย
- **Unified Report:** ระบบคัดกรองอัจฉริยะ (Volume Compression, SRS Filter, Dynamic Stop Loss)
- **SILENT ACCUM Insight:** วิเคราะห์การเก็บของของ Smart Money
- **AI Trading Plan:** วางแผนการเทรดอัตโนมัติด้วย Google Gemini AI

## วิธีการติดตั้งและใช้งาน

1. **Clone โปรเจกต์:**
   ```bash
   git clone <your-repository-url>
   cd "SET Project TRAE AI"
   ```

2. **สร้าง Virtual Environment และติดตั้ง Dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # สำหรับ Windows ใช้: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **ตั้งค่า Environment Variables:**
   - Copy ไฟล์ `.env.example` เป็น `.env`
   - ใส่ค่า `GEMINI_API_KEY` และการตั้งค่า Supabase (ถ้ามี)

4. **รันโปรแกรม:**
   ```bash
   streamlit run stock_dashboard.py
   ```

## การ Deploy บน Streamlit Cloud
- นำโค้ดขึ้น GitHub (ยกเว้นไฟล์ที่ระบุใน `.gitignore`)
- เชื่อมต่อ Repository กับ Streamlit Cloud
- ตั้งค่า **Secrets** ใน Dashboard ของ Streamlit Cloud โดยใช้ค่าจากไฟล์ `.env`

## โครงสร้างฐานข้อมูล
โปรเจกต์นี้ใช้ SQLite (`quant_scanner.db` และ `trading_log.db`) สำหรับเก็บข้อมูลภายในเครื่อง และมีแผนที่จะย้ายไปใช้ Supabase ในอนาคต
