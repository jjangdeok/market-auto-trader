"""
AutoTrader API 엔드포인트

자동매매 엔진의 스캔, 실행, 설정 조회/변경 API를 제공합니다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import get_kis_client
from src.broker.kis_client import KISClient
from src.strategy.auto_trader import AutoTrader, AutoTraderConfig, RiskLimits
from src.strategy.auto_trader_scheduler import AutoTraderScheduler

from asyncio import AbstractEventLoop

router = APIRouter(
    prefix="/api/v1/auto-trader",
    tags=["AutoTrader"],
)

# 모듈 수준 설정 (PUT으로 변경 가능)
_current_config = AutoTraderConfig()

# 모듈 수준 스케줄러 인스턴스 (싱글턴)
_scheduler: AutoTraderScheduler | None = None
# FastAPI 애플리케이션 메인 이벤트 루프 (lifespan에서 주입)
_scheduler_loop: AbstractEventLoop | None = None


def set_scheduler_event_loop(loop: AbstractEventLoop) -> None:
    """AutoTraderScheduler가 사용할 이벤트 루프를 설정합니다.

    FastAPI lifespan 컨텍스트(메인 이벤트 루프가 활성화된 곳)에서 호출되어,
    이후 동기 엔드포인트(쓰레드풀)에서도 APScheduler가 올바른 루프에 붙도록 합니다.
    """
    global _scheduler_loop
    _scheduler_loop = loop


def _get_scheduler(client: KISClient) -> AutoTraderScheduler:
    """스케줄러 싱글턴 반환"""
    global _scheduler
    if _scheduler is None:
        trader = AutoTrader(client, _current_config)
        kwargs: dict[str, Any] = {}
        if _scheduler_loop is not None:
            kwargs["event_loop"] = _scheduler_loop
        _scheduler = AutoTraderScheduler(trader, **kwargs)
    return _scheduler


# ───────────────── Schemas ─────────────────


class RiskLimitsSchema(BaseModel):
    max_daily_trades: int = 10
    max_position_pct: float = 0.2
    max_total_position_pct: float = 0.8
    max_daily_loss_pct: float = 0.03
    min_signal_score_buy: float = 40.0
    max_signal_score_sell: float = -30.0


class AutoTraderConfigSchema(BaseModel):
    universe_name: str = "kospi_top30"
    risk_limits: RiskLimitsSchema = Field(default_factory=RiskLimitsSchema)
    dry_run: bool = True
    max_notional_krw: int = 5_000_000


class TradeSignalSchema(BaseModel):
    stock_code: str
    stock_name: str
    signal_type: str
    score: float
    sentiment_score: float
    quality_score: float
    technical_score: float
    reason: str
    recommended_action: str


class ScanResponse(BaseModel):
    signals: list[TradeSignalSchema]
    total: int


class RunCycleResponse(BaseModel):
    timestamp: str
    sentiment: dict[str, Any]
    scanned: int
    buy_signals: list[dict[str, Any]]
    sell_signals: list[dict[str, Any]]
    executed_buys: list[dict[str, Any]]
    executed_sells: list[dict[str, Any]]
    dry_run: bool


# ───────────────── Endpoints ─────────────────


@router.post(
    "/scan",
    response_model=ScanResponse,
    summary="유니버스 스캔",
    description="유니버스 전체를 스캔하여 매매 시그널만 반환합니다 (주문 실행 없음).",
)
def scan_universe(
    client: KISClient = Depends(get_kis_client),
) -> ScanResponse:
    trader = AutoTrader(client, _current_config)
    signals = trader.scan_universe()
    return ScanResponse(
        signals=[
            TradeSignalSchema(
                stock_code=s.stock_code,
                stock_name=s.stock_name,
                signal_type=s.signal_type.value,
                score=s.score,
                sentiment_score=s.sentiment_score,
                quality_score=s.quality_score,
                technical_score=s.technical_score,
                reason=s.reason,
                recommended_action=s.recommended_action,
            )
            for s in signals
        ],
        total=len(signals),
    )


@router.post(
    "/run",
    response_model=RunCycleResponse,
    summary="한 사이클 실행",
    description="유니버스 스캔 + 매매 시그널 기반 주문 실행 + 보유종목 매도 체크를 수행합니다.",
)
def run_cycle(
    client: KISClient = Depends(get_kis_client),
) -> RunCycleResponse:
    trader = AutoTrader(client, _current_config)
    result = trader.run_cycle()
    return RunCycleResponse(**result)


@router.get(
    "/config",
    response_model=AutoTraderConfigSchema,
    summary="현재 설정 조회",
    description="자동매매 엔진의 현재 설정을 조회합니다.",
)
def get_config() -> AutoTraderConfigSchema:
    c = _current_config
    return AutoTraderConfigSchema(
        universe_name=c.universe_name,
        risk_limits=RiskLimitsSchema(
            max_daily_trades=c.risk_limits.max_daily_trades,
            max_position_pct=c.risk_limits.max_position_pct,
            max_total_position_pct=c.risk_limits.max_total_position_pct,
            max_daily_loss_pct=c.risk_limits.max_daily_loss_pct,
            min_signal_score_buy=c.risk_limits.min_signal_score_buy,
            max_signal_score_sell=c.risk_limits.max_signal_score_sell,
        ),
        dry_run=c.dry_run,
        max_notional_krw=c.max_notional_krw,
    )


@router.put(
    "/config",
    response_model=AutoTraderConfigSchema,
    summary="설정 변경",
    description="자동매매 엔진의 설정을 변경합니다.",
)
def update_config(
    config: AutoTraderConfigSchema,
) -> AutoTraderConfigSchema:
    global _current_config
    _current_config = AutoTraderConfig(
        universe_name=config.universe_name,
        risk_limits=RiskLimits(
            max_daily_trades=config.risk_limits.max_daily_trades,
            max_position_pct=config.risk_limits.max_position_pct,
            max_total_position_pct=config.risk_limits.max_total_position_pct,
            max_daily_loss_pct=config.risk_limits.max_daily_loss_pct,
            min_signal_score_buy=config.risk_limits.min_signal_score_buy,
            max_signal_score_sell=config.risk_limits.max_signal_score_sell,
        ),
        dry_run=config.dry_run,
        max_notional_krw=config.max_notional_krw,
    )
    return config


# ───────────────── Scheduler Schemas ─────────────────


class SchedulerStartRequest(BaseModel):
    interval_minutes: int = Field(default=30, ge=1, le=480)
    kr_market_only: bool = True
    us_market: bool = False


class SchedulerStatusResponse(BaseModel):
    is_running: bool
    interval_minutes: int
    next_run_time: str | None = None
    total_cycles: int
    last_cycle_result: dict[str, Any] | None = None
    kr_market_hours: str
    us_market_hours: str | None = None


# ───────────────── Scheduler Endpoints ─────────────────


@router.post(
    "/scheduler/start",
    response_model=SchedulerStatusResponse,
    summary="스케줄러 시작",
    description="자동매매 스케줄러를 시작합니다.",
)
def scheduler_start(
    req: SchedulerStartRequest,
    client: KISClient = Depends(get_kis_client),
) -> SchedulerStatusResponse:
    scheduler = _get_scheduler(client)
    scheduler.start(
        interval_minutes=req.interval_minutes,
        kr_market_only=req.kr_market_only,
        us_market=req.us_market,
    )
    return SchedulerStatusResponse(**scheduler.get_status())


@router.post(
    "/scheduler/stop",
    response_model=SchedulerStatusResponse,
    summary="스케줄러 중지",
    description="자동매매 스케줄러를 중지합니다.",
)
def scheduler_stop(
    client: KISClient = Depends(get_kis_client),
) -> SchedulerStatusResponse:
    scheduler = _get_scheduler(client)
    scheduler.stop()
    return SchedulerStatusResponse(**scheduler.get_status())


@router.get(
    "/scheduler/status",
    response_model=SchedulerStatusResponse,
    summary="스케줄러 상태",
    description="자동매매 스케줄러의 현재 상태를 조회합니다.",
)
def scheduler_status(
    client: KISClient = Depends(get_kis_client),
) -> SchedulerStatusResponse:
    scheduler = _get_scheduler(client)
    return SchedulerStatusResponse(**scheduler.get_status())


@router.get(
    "/scheduler/history",
    summary="사이클 히스토리",
    description="최근 사이클 실행 히스토리를 조회합니다.",
)
def scheduler_history(
    limit: int = 10,
    client: KISClient = Depends(get_kis_client),
) -> list[dict[str, Any]]:
    scheduler = _get_scheduler(client)
    return scheduler.get_cycle_history(limit=limit)
