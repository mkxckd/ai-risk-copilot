CREATE TABLE IF NOT EXISTS instruments (
  symbol TEXT PRIMARY KEY, sector TEXT NOT NULL, beta REAL NOT NULL, ann_vol REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS portfolios (
  portfolio_id TEXT PRIMARY KEY, name TEXT NOT NULL, strategy TEXT NOT NULL,
  capital REAL NOT NULL, cash REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prices (
  tick INTEGER NOT NULL, ts TEXT NOT NULL, symbol TEXT NOT NULL, price REAL NOT NULL,
  PRIMARY KEY (tick, symbol)
);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tick INTEGER NOT NULL, ts TEXT NOT NULL,
  portfolio_id TEXT NOT NULL, symbol TEXT NOT NULL, quantity REAL NOT NULL,
  price REAL NOT NULL, reason TEXT
);
CREATE INDEX IF NOT EXISTS ix_trades_pf ON trades (portfolio_id, tick);
CREATE TABLE IF NOT EXISTS risk_snapshots (
  portfolio_id TEXT NOT NULL, tick INTEGER NOT NULL, ts TEXT NOT NULL,
  nav REAL, gross REAL, net REAL, leverage REAL, beta REAL,
  var_95 REAL, var_99 REAL, var_95_pct REAL, vol_ann REAL, drawdown REAL,
  top_weight REAL, top_symbol TEXT, hhi REAL, sector_max REAL, sector_max_name TEXT,
  anomaly_score REAL, is_anomaly INTEGER,
  PRIMARY KEY (portfolio_id, tick)
);
CREATE TABLE IF NOT EXISTS sector_exposures (
  portfolio_id TEXT NOT NULL, tick INTEGER NOT NULL, sector TEXT NOT NULL, net_weight REAL NOT NULL,
  PRIMARY KEY (portfolio_id, tick, sector)
);
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tick INTEGER NOT NULL, ts TEXT NOT NULL,
  portfolio_id TEXT NOT NULL, severity TEXT NOT NULL, anomaly_score REAL,
  drivers TEXT, summary TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_pf ON alerts (portfolio_id, tick);
CREATE VIEW IF NOT EXISTS v_positions AS
  SELECT portfolio_id, symbol, SUM(quantity) AS quantity
  FROM trades GROUP BY portfolio_id, symbol HAVING ABS(SUM(quantity)) > 1e-9;
