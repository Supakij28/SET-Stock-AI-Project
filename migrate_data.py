import sqlite3
import os
import pandas as pd
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import math

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: SUPABASE_URL or SUPABASE_KEY not found in .env file")
    exit(1)

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def clean_record(record):
    """Clean a single record to be JSON compliant."""
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                cleaned[k] = None
            else:
                cleaned[k] = v
        else:
            cleaned[k] = v
    return cleaned

def migrate_table(db_name, table_name, supabase_table):
    """Migrate data from a specific SQLite table to Supabase."""
    if not os.path.exists(db_name):
        print(f"⚠️ Skip: {db_name} not found.")
        return

    print(f"📦 Starting migration: {db_name} -> {supabase_table}...")
    
    try:
        conn = sqlite3.connect(db_name)
        # Read all data from SQLite
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()

        if df.empty:
            print(f"ℹ️ Table {table_name} is empty. Skipping.")
            return

        # Convert to records and clean each one
        raw_records = df.to_dict('records')
        records = [clean_record(r) for r in raw_records]
        
        total = len(records)
        chunk_size = 50  # Supabase handles chunks better
        
        for i in range(0, total, chunk_size):
            chunk = records[i:i + chunk_size]
            supabase.table(supabase_table).upsert(chunk).execute()
            
            progress = min(i + chunk_size, total)
            print(f"   🚀 Progress: {progress}/{total} records migrated...", end='\r')
            
        print(f"\n✅ Successfully migrated {total} records to {supabase_table}!")

    except Exception as e:
        print(f"\n❌ Error migrating {table_name}: {e}")

if __name__ == "__main__":
    print("--- 🔄 SET Project Data Migration Tool ---")
    
    # 1. Migrate Scan Results
    migrate_table("quant_scanner.db", "scan_results", "scan_results")
    
    # 2. Migrate Trading Logs
    migrate_table("trading_log.db", "trading_log", "trading_log")
    
    print("\n🎉 All migrations completed!")
