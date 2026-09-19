"""Factor-based market simulator: r_i = beta_i * market + sector + idiosyncratic."""
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .config import TICKS_PER_YEAR

MKT_VOL, SEC_VOL = 0.16, 0.10
START = datetime(2026, 1, 5, 9, 30)
TICK_MINUTES = 5


@dataclass(frozen=True)
class Instrument:
    symbol: str
    sector: str
    beta: float
    ann_vol: float
    price0: float


UNIVERSE = [
    Instrument("AURX", "Tech", 1.25, 0.32, 180), Instrument("NEXO", "Tech", 1.35, 0.38, 95),
    Instrument("QBIT", "Tech", 1.50, 0.45, 60), Instrument("CLDR", "Tech", 1.10, 0.28, 240),
    Instrument("SEMI", "Tech", 1.40, 0.40, 130),
    Instrument("BNKA", "Financials", 1.10, 0.24, 48), Instrument("INSR", "Financials", 0.90, 0.20, 72),
    Instrument("PAYX", "Financials", 1.20, 0.30, 110), Instrument("CAPM", "Financials", 1.30, 0.28, 85),
    Instrument("OILX", "Energy", 1.00, 0.34, 70), Instrument("GASN", "Energy", 0.85, 0.30, 38),
    Instrument("SOLR", "Energy", 1.30, 0.42, 55), Instrument("PIPE", "Energy", 0.70, 0.22, 44),
    Instrument("MEDI", "Health", 0.70, 0.20, 150), Instrument("BIOG", "Health", 1.00, 0.38, 88),
    Instrument("PHRM", "Health", 0.60, 0.18, 120), Instrument("DIAG", "Health", 0.90, 0.26, 65),
    Instrument("RETL", "Consumer", 1.00, 0.26, 95), Instrument("FOOD", "Consumer", 0.50, 0.16, 68),
    Instrument("LUXE", "Consumer", 1.15, 0.31, 210),
]
SECTORS = sorted({i.sector for i in UNIVERSE})


def tick_ts(tick: int) -> str:
    return (START + timedelta(minutes=TICK_MINUTES * tick)).isoformat()


class MarketSimulator:
    def __init__(self, seed: int = 7, prices: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self.prices = dict(prices) if prices else {i.symbol: float(i.price0) for i in UNIVERSE}
        self.vol_mult, self._vol_left = 1.0, 0
        self.sector_shock: dict[str, list] = {}  # sector -> [ticks_left, drift_per_tick]

    def spike_vol(self, mult: float, duration: int):
        self.vol_mult, self._vol_left = mult, duration

    def shock_sector(self, sector: str, drift: float, duration: int):
        self.sector_shock[sector] = [duration, drift]

    def step(self) -> dict:
        k = 1 / np.sqrt(TICKS_PER_YEAR)
        vm = self.vol_mult
        mkt = self.rng.normal() * MKT_VOL * k * vm
        sec = {s: self.rng.normal() * SEC_VOL * k * vm for s in SECTORS}
        for s, st in list(self.sector_shock.items()):
            sec[s] += st[1]
            st[0] -= 1
            if st[0] <= 0:
                del self.sector_shock[s]
        rets = {}
        for i in UNIVERSE:
            idio = np.sqrt(max(i.ann_vol**2 - (i.beta * MKT_VOL) ** 2 - SEC_VOL**2, 0.02**2))
            r = i.beta * mkt + sec[i.sector] + self.rng.normal() * idio * k * vm
            self.prices[i.symbol] *= float(np.exp(r))
            rets[i.symbol] = float(r)
        if self._vol_left > 0:
            self._vol_left -= 1
            if self._vol_left == 0:
                self.vol_mult = 1.0
        return rets


def initial_portfolios(prices: dict, capital: float = 10_000_000.0) -> list[dict]:
    syms = [i.symbol for i in UNIVERSE]
    beta = {i.symbol: i.beta for i in UNIVERSE}

    def shares(weights):
        return {s: capital * w / prices[s] for s, w in weights.items()}

    bal = {s: 0.05 for s in syms}
    by_beta = sorted(syms, key=lambda s: beta[s])
    longs, shorts = by_beta[10:], by_beta[:10]          # long high-beta, short low-beta, beta-balanced
    ratio = sum(beta[s] for s in longs) / sum(beta[s] for s in shorts)
    neut = {**{s: 0.10 for s in longs}, **{s: -0.10 * ratio for s in shorts}}
    tech_syms = [i.symbol for i in UNIVERSE if i.sector == "Tech"]
    tech = {s: (0.12 if s in tech_syms else 0.40 / 15) for s in syms}
    macro = {s: (0.15 if idx % 2 == 0 else -0.15) for idx, s in enumerate(syms)}
    defs = [
        ("BAL", "Balanced Long", "Long-only, equal weight", bal),
        ("NEUT", "Beta-Neutral L/S", "Long/short, beta balanced", neut),
        ("TECH", "Tech Growth", "Long-only, 60% tech", tech),
        ("MACRO", "Levered Macro", "3x gross long/short", macro),
    ]
    out = []
    for pid, name, strat, w in defs:
        pos = shares(w)
        cash = capital - sum(pos[s] * prices[s] for s in pos)
        out.append({"portfolio_id": pid, "name": name, "strategy": strat,
                    "capital": capital, "cash": cash, "positions": pos})
    return out
