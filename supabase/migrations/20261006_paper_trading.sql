-- Migration to add Paper Trading fields to trading_log
ALTER TABLE trading_log ADD COLUMN IF NOT EXISTS action TEXT;
ALTER TABLE trading_log ADD COLUMN IF NOT EXISTS entry_price DOUBLE PRECISION;
ALTER TABLE trading_log ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;
ALTER TABLE trading_log ADD COLUMN IF NOT EXISTS quantity INTEGER;
ALTER TABLE trading_log ADD COLUMN IF NOT EXISTS entry_date TEXT;
ALTER TABLE trading_log ADD COLUMN IF NOT EXISTS exit_date TEXT;
ALTER TABLE trading_log ADD COLUMN IF NOT EXISTS pnl_percent DOUBLE PRECISION;
ALTER TABLE trading_log ADD COLUMN IF NOT EXISTS exit_reason TEXT;

-- Update existing records to have a default action if needed, though mostly for new records
UPDATE trading_log SET action = 'BUY' WHERE action IS NULL;
