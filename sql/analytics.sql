-- name: latest_snapshots
SELECT s.*, RANK() OVER (ORDER BY s.var_95_pct DESC) AS var_rank
FROM risk_snapshots s
JOIN (SELECT portfolio_id, MAX(tick) AS tick FROM risk_snapshots GROUP BY portfolio_id) m
  ON m.portfolio_id = s.portfolio_id AND m.tick = s.tick
ORDER BY var_rank;

-- name: risk_change
WITH latest AS (SELECT portfolio_id, MAX(tick) AS tick FROM risk_snapshots GROUP BY portfolio_id)
SELECT n.portfolio_id, n.tick AS tick_now, p.tick AS tick_then,
       n.var_95 AS var_95_now, p.var_95 AS var_95_then,
       ROUND(100.0 * (n.var_95 / p.var_95 - 1), 2) AS var_95_change_pct,
       n.leverage AS leverage_now, p.leverage AS leverage_then,
       n.vol_ann AS vol_now, p.vol_ann AS vol_then
FROM latest l
JOIN risk_snapshots n ON n.portfolio_id = l.portfolio_id AND n.tick = l.tick
LEFT JOIN risk_snapshots p ON p.portfolio_id = n.portfolio_id AND p.tick = n.tick - :lookback
ORDER BY var_95_change_pct DESC;

-- name: risk_regime
SELECT * FROM (
  SELECT portfolio_id, tick, ts, var_95, var_95_pct, vol_ann, leverage, top_weight, anomaly_score, is_anomaly,
         AVG(var_95) OVER (PARTITION BY portfolio_id ORDER BY tick ROWS BETWEEN 39 PRECEDING AND CURRENT ROW) AS var_95_ma40,
         AVG(vol_ann) OVER (PARTITION BY portfolio_id ORDER BY tick ROWS BETWEEN 39 PRECEDING AND CURRENT ROW) AS vol_ma40
  FROM risk_snapshots WHERE portfolio_id = :portfolio_id
) ORDER BY tick DESC LIMIT :n;

-- name: alert_summary
SELECT portfolio_id, severity, COUNT(*) AS n_alerts, MIN(tick) AS first_tick, MAX(tick) AS last_tick,
       ROUND(AVG(anomaly_score), 4) AS avg_score
FROM alerts GROUP BY portfolio_id, severity ORDER BY portfolio_id, severity;

-- name: top_movers
WITH bounds AS (SELECT MAX(tick) AS t1 FROM prices)
SELECT a.symbol, i.sector, p0.price AS price_then, a.price AS price_now,
       a.price / p0.price - 1 AS ret
FROM prices a
JOIN bounds b ON a.tick = b.t1
JOIN prices p0 ON p0.symbol = a.symbol AND p0.tick = b.t1 - :lookback
JOIN instruments i ON i.symbol = a.symbol
ORDER BY ABS(a.price / p0.price - 1) DESC;
