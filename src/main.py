"""
FastAPI 애플리케이션 엔트리포인트
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.settings import settings
from src.api.alerts import router as alerts_router
from src.api.auto_trader import set_scheduler_event_loop
import asyncio
from src.api.health import router as health_router
from src.api.orders import router as orders_router
from src.api.policies import router as policies_router
from src.api.portfolio import router as portfolio_router
from src.api.rebalancing import router as rebalancing_router
from src.api.routes import router as base_router
from src.api.signals import router as signals_router
from src.api.strategy_manager import router as strategy_manager_router
from src.api.data_pipeline import router as data_pipeline_router
from src.api.trade_report import router as trade_report_router
from src.api.streaming import router as streaming_router
from src.api.sentiment import router as sentiment_router
from src.api.analysis import router as analysis_router
from src.api.auto_trader import router as auto_trader_router
from src.api.dashboard import router as dashboard_router
from src.db import engine
from src.exceptions import register_exception_handlers
from src.utils.logger import get_logger

logger = get_logger(__name__)


OPENAPI_TAGS = [
    {
        "name": "System",
        "description": "시스템 상태 확인",
    },
    {
        "name": "Portfolio",
        "description": "포트폴리오 조회 (보유종목, 계좌요약)",
    },
    {
        "name": "Orders",
        "description": "매매 주문 실행 및 주문 내역 조회",
    },
    {
        "name": "Signals",
        "description": "이동평균 교차 전략 기반 매매 신호 생성 및 조회",
    },
    {
        "name": "policies",
        "description": "원샷 매매 정책 실행 (국내/해외)",
    },
    {
        "name": "Strategies",
        "description": "복합 전략 매니저 — 다중 전략 신호 종합, 투표, 성과 비교",
    },
    {
        "name": "Rebalancing",
        "description": "포트폴리오 리밸런싱 실행, 내역 조회, 스케줄 관리",
    },
    {
        "name": "Alerts",
        "description": "알림 규칙 관리 — 손절/목표가 알림, 가격 등락 감지, Discord 연동",
    },
    {
        "name": "DataPipeline",
        "description": "시세 데이터 수집, 캐시, 품질 검증 파이프라인",
    },
    {
        "name": "Reports",
        "description": "거래 리포트 — 일일 거래 요약, 포트폴리오 스냅샷, 실현 손익 조회",
    },
    {
        "name": "Streaming",
        "description": "실시간 시세 스트리밍 — WebSocket 기반 실시간 체결가 구독/수신",
    },
    {
        "name": "Dashboard",
        "description": "실시간 대시보드 — 포트폴리오 PnL, 수익률 추이, 종합 요약",
    },
    {
        "name": "Analysis",
        "description": "시장 분석 — 공포탐욕지수, 센티멘트 분석, 매수 강도 배율",
    },
    {
        "name": "AutoTrader",
        "description": "자동매매 엔진 — 센티멘트 + 스크리너 + 기술적 분석 기반 자동 매매",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """애플리케이션 시작/종료 시 실행되는 로직"""
    # Startup
    logger.info("🚀 Market Auto Trader 시작 (환경: %s)", settings.app_env)
    db_host = settings.database_url.split("@")[-1] if "@" in settings.database_url else "unknown"
    logger.info("📊 데이터베이스: %s", db_host)

    # APScheduler가 FastAPI 메인 이벤트 루프에 붙도록 루프 객체를 주입
    loop = asyncio.get_running_loop()
    set_scheduler_event_loop(loop)

    yield

    # Shutdown
    await engine.dispose()
    logger.info("👋 Market Auto Trader 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="Market Auto Trader",
    description=(
        "한국 주식 시장 자동매매 프로그램 🐈‍⬛\n\n"
        "한국투자증권 OpenAPI를 활용한 자동 매매 시스템입니다.\n"
        "이동평균 교차 전략으로 매매 신호를 생성하고, "
        "포트폴리오를 관리합니다."
    ),
    version="0.3.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)

# 예외 핸들러 등록
register_exception_handlers(app)

# 라우터 등록
app.include_router(base_router)
app.include_router(health_router)
app.include_router(portfolio_router)
app.include_router(orders_router)
app.include_router(signals_router)
app.include_router(policies_router)
app.include_router(strategy_manager_router)
app.include_router(rebalancing_router)
app.include_router(alerts_router)
app.include_router(data_pipeline_router)
app.include_router(trade_report_router)
app.include_router(streaming_router)
app.include_router(dashboard_router)
app.include_router(sentiment_router)
app.include_router(analysis_router)
app.include_router(auto_trader_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
