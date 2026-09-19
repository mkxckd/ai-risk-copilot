"""FastAPI inference service.  Run: uvicorn app.main:app --port 8000"""
import asyncio
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .bootstrap import build_system
from .config import SIM_INTERVAL
from .service import RiskService


class ScoreRequest(BaseModel):
    portfolio_id: str
    leverage: float
    beta: float
    var_95_pct: float
    top_weight: float
    hhi: float
    sector_max: float
    vol_ann: float
    drawdown: float


class ExplainRequest(BaseModel):
    question: Optional[str] = Field(None, max_length=500)


class InjectRequest(BaseModel):
    kind: Literal["concentrate", "vol_spike", "sector_crash"]
    delay: int = Field(1, ge=1, le=100)
    portfolio_id: Optional[str] = None
    symbol: Optional[str] = None
    target_weight: Optional[float] = Field(None, gt=0, le=1)
    sector: Optional[str] = None
    drift: Optional[float] = None
    duration: Optional[int] = Field(None, ge=1, le=500)
    mult: Optional[float] = Field(None, gt=1, le=20)


def create_app(system=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = RiskService(system or await asyncio.to_thread(build_system))
        task = None
        if SIM_INTERVAL > 0:
            async def live():
                while True:
                    await asyncio.sleep(SIM_INTERVAL)
                    await asyncio.to_thread(app.state.svc.step, 1)
            task = asyncio.create_task(live())
        yield
        if task:
            task.cancel()

    app = FastAPI(title="AI Trading Risk Copilot", version="1.0", lifespan=lifespan)

    def svc() -> RiskService:
        return app.state.svc

    def guard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except KeyError as e:
            raise HTTPException(404, f"unknown portfolio {e}")
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/health")
    def health():
        e = svc().s.engine
        return {"status": "ok", "tick": e.tick, "portfolios": e.portfolio_ids,
                "model_loaded": svc().s.detector is not None, "agent_mode": "llm" if svc().s.agent.client else "rule_based"}

    @app.get("/portfolios")
    def portfolios():
        return svc().portfolios()

    @app.get("/portfolios/{pid}/risk")
    def risk(pid: str):
        return guard(svc().risk, pid)

    @app.get("/portfolios/{pid}/history")
    def history(pid: str, n: int = Query(60, ge=1, le=1000)):
        return guard(svc().history, pid, n)

    @app.post("/portfolios/{pid}/explain")
    def explain(pid: str, body: ExplainRequest = ExplainRequest()):
        return guard(svc().explain, pid, body.question)

    @app.get("/alerts")
    def alerts(portfolio_id: Optional[str] = None, severity: Optional[Literal["warning", "critical"]] = None,
               limit: int = Query(50, ge=1, le=500)):
        return svc().alerts(portfolio_id, severity, limit)

    @app.post("/score")
    def score(req: ScoreRequest):
        body = req.model_dump()
        return guard(svc().score, body.pop("portfolio_id"), body)

    @app.get("/analytics/summary")
    def analytics(lookback: int = Query(24, ge=1, le=100)):
        return svc().analytics(lookback)

    @app.post("/simulation/step")
    def step(n: int = Query(1, ge=1, le=500)):
        return svc().step(n)

    @app.post("/simulation/inject")
    def inject(req: InjectRequest):
        params = req.model_dump(exclude_none=True)
        kind, delay = params.pop("kind"), params.pop("delay")
        return guard(svc().inject, kind, delay, **params)

    return app


app = create_app()
