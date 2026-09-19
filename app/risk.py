"""Portfolio risk metrics: exposures, historical-simulation VaR, concentration, component risk."""
import numpy as np
import pandas as pd

from .config import TICKS_PER_DAY, TICKS_PER_YEAR


def compute_risk(quantities: dict, prices: dict, cash: float, returns: pd.DataFrame,
                 instruments: dict) -> dict:
    """returns: DataFrame of per-tick simple returns (rows) by symbol (columns).
    VaR is historical simulation on the portfolio's *current* weights, scaled to 1 day by sqrt-time."""
    syms = [s for s, q in quantities.items() if abs(q) > 1e-9]
    mv = pd.Series({s: quantities[s] * prices[s] for s in syms})
    nav = float(cash + mv.sum())
    w = mv / nav
    gross, net = float(mv.abs().sum()), float(mv.sum())
    beta = float(sum(w[s] * instruments[s]["beta"] for s in syms))

    pr = returns[syms].to_numpy() @ w.to_numpy()
    day = np.sqrt(TICKS_PER_DAY)
    var95 = max(0.0, -float(np.quantile(pr, 0.05))) * day * nav
    var99 = max(0.0, -float(np.quantile(pr, 0.01))) * day * nav
    vol_ann = float(pr.std(ddof=1) * np.sqrt(TICKS_PER_YEAR))
    cum = np.concatenate([[1.0], np.cumprod(1 + pr)])
    drawdown = float(1 - cum[-1] / cum.max())

    absw = w.abs()
    a = absw / absw.sum()
    sector_w = w.groupby({s: instruments[s]["sector"] for s in syms}).sum()
    top_sector = sector_w.abs().idxmax()
    return {
        "nav": nav, "gross": gross, "net": net, "leverage": gross / nav, "beta": beta,
        "var_95": var95, "var_99": var99, "var_95_pct": var95 / nav,
        "vol_ann": vol_ann, "drawdown": drawdown,
        "top_weight": float(absw.max()), "top_symbol": str(absw.idxmax()),
        "hhi": float((a**2).sum()),
        "sector_max": float(abs(sector_w[top_sector])), "sector_max_name": str(top_sector),
        "sector_weights": {k: float(v) for k, v in sector_w.items()},
    }


def component_risk(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance attributable to each position (sums to 1)."""
    cov = returns[weights.index].cov()
    marginal = cov @ weights
    var = float(weights @ marginal)
    contrib = weights * marginal / var if var > 0 else weights * 0.0
    return contrib.sort_values(key=lambda s: s.abs(), ascending=False)
