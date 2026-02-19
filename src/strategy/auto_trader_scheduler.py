"""
AutoTrader 스케줄러 — 장 시간에 맞춰 AutoTrader 사이클 자동 실행

국내 장(09:00~15:30 KST) 및 미국 장(23:30~06:00 KST)에 맞춰
주기적으로 AutoTrader.run_cycle()을 실행합니다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.strategy.auto_trader import AutoTrader
from src.utils.logger import get_logger

logger = get_logger(__name__)

KST = ZoneInfo("Asia/Seoul")


class AutoTraderScheduler:
    """자동매매 스케줄러 — 장 시간에 맞춰 AutoTrader 사이클 자동 실행"""

    MAX_HISTORY = 100

    def __init__(self, auto_trader: AutoTrader, event_loop: Any | None = None) -> None:
        """스케줄러 초기화

        Parameters
        ----------
        auto_trader:
            자동매매 엔진 인스턴스
        event_loop:
            APScheduler가 붙을 asyncio 이벤트 루프. FastAPI lifespan 등에서
            애플리케이션 메인 이벤트 루프를 주입하는 용도로 사용합니다.
            None 인 경우 기본 이벤트 루프 정책을 사용합니다.
        """
        self._trader = auto_trader
        # event_loop가 지정되면 AsyncIOScheduler가 get_running_loop()를 호출하지 않고
        # 주어진 루프에 바로 붙기 때문에, FastAPI sync 엔드포인트의 쓰레드풀 컨텍스트에서도
        # "no running event loop" 오류 없이 동작합니다.
        scheduler_kwargs: dict[str, Any] = {"timezone": KST}
        if event_loop is not None:
            scheduler_kwargs["event_loop"] = event_loop
        self._scheduler = AsyncIOScheduler(**scheduler_kwargs)
        self._is_running = False
        self._interval_minutes: int = 30
        self._kr_market_only: bool = True
        self._us_market: bool = False
        self._cycle_history: list[dict[str, Any]] = []

    # ───────────────── 시작 / 중지 ─────────────────

    def start(
        self,
        interval_minutes: int = 30,
        kr_market_only: bool = True,
        us_market: bool = False,
    ) -> None:
        """스케줄러 시작"""
        if self._is_running:
            logger.warning("스케줄러가 이미 실행 중입니다")
            return

        self._interval_minutes = interval_minutes
        self._kr_market_only = kr_market_only
        self._us_market = us_market

        self._scheduler.add_job(
            self.run_scheduled_cycle,
            trigger=IntervalTrigger(minutes=interval_minutes, timezone=KST),
            id="auto_trader_cycle",
            replace_existing=True,
        )
        self._scheduler.start()
        self._is_running = True
        logger.info(
            "AutoTrader 스케줄러 시작: %d분 간격, KR=%s, US=%s",
            interval_minutes,
            kr_market_only,
            us_market,
        )

    def stop(self) -> None:
        """스케줄러 중지"""
        if not self._is_running:
            logger.warning("스케줄러가 실행 중이 아닙니다")
            return

        self._scheduler.shutdown(wait=False)
        # 새 스케줄러 인스턴스 준비 (재시작 가능하도록)
        self._scheduler = AsyncIOScheduler(timezone=KST)
        self._is_running = False
        logger.info("AutoTrader 스케줄러 중지")

    # ───────────────── 사이클 실행 ─────────────────

    async def run_scheduled_cycle(self) -> dict[str, Any]:
        """스케줄된 사이클 실행"""
        now = datetime.now(tz=KST)

        # 장 시간 확인
        kr_open = self.is_kr_market_open()
        us_open = self.is_us_market_open()

        should_run = False
        if self._kr_market_only and kr_open:
            should_run = True
        if self._us_market and us_open:
            should_run = True
        # kr_market_only=False, us_market=False → 항상 실행
        if not self._kr_market_only and not self._us_market:
            should_run = True

        if not should_run:
            result: dict[str, Any] = {
                "timestamp": now.isoformat(),
                "status": "skipped",
                "reason": "장 시간 외",
                "kr_market_open": kr_open,
                "us_market_open": us_open,
            }
            self._append_history(result)
            logger.info("장 시간 외 — 사이클 스킵")
            return result

        try:
            cycle_result = self._trader.run_cycle()
            result = {
                "timestamp": now.isoformat(),
                "status": "completed",
                "kr_market_open": kr_open,
                "us_market_open": us_open,
                "cycle_result": cycle_result,
            }
            logger.info(
                "사이클 완료: 스캔 %d, 매수시그널 %d, 매도시그널 %d",
                cycle_result.get("scanned", 0),
                len(cycle_result.get("buy_signals", [])),
                len(cycle_result.get("sell_signals", [])),
            )
        except Exception:
            logger.exception("사이클 실행 실패")
            result = {
                "timestamp": now.isoformat(),
                "status": "error",
                "error": "사이클 실행 중 오류 발생",
                "kr_market_open": kr_open,
                "us_market_open": us_open,
            }

        self._append_history(result)
        return result

    # ───────────────── 상태 조회 ─────────────────

    def get_status(self) -> dict[str, Any]:
        """스케줄러 상태 조회"""
        next_run_time = None
        if self._is_running:
            job = self._scheduler.get_job("auto_trader_cycle")
            if job and job.next_run_time:
                next_run_time = job.next_run_time.isoformat()

        status: dict[str, Any] = {
            "is_running": self._is_running,
            "interval_minutes": self._interval_minutes,
            "next_run_time": next_run_time,
            "total_cycles": len(self._cycle_history),
            "last_cycle_result": self._cycle_history[-1] if self._cycle_history else None,
            "kr_market_hours": "09:00-15:30 KST",
        }
        if self._us_market:
            status["us_market_hours"] = "23:30-06:00 KST"
        return status

    def get_cycle_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """최근 사이클 히스토리 (최신순)"""
        return list(reversed(self._cycle_history[-limit:]))

    # ───────────────── 장 시간 판단 ─────────────────

    @staticmethod
    def is_kr_market_open() -> bool:
        """국내 장 시간 여부 (09:00~15:30 KST, 평일만)"""
        now = datetime.now(tz=KST)
        # 평일만 (월=0 ~ 금=4)
        if now.weekday() >= 5:
            return False
        t = now.time()
        from datetime import time as _time

        return _time(9, 0) <= t <= _time(15, 30)

    @staticmethod
    def is_us_market_open() -> bool:
        """미국 장 시간 여부 (23:30~06:00 KST, 평일만)

        23:30~24:00 → 해당일이 평일이어야 함
        00:00~06:00 → 전날이 평일이어야 함 (실제로는 당일 weekday-1 체크)
        초기에는 간소화: 현재 시간이 23:30~06:00이고 평일이면 True
        """
        now = datetime.now(tz=KST)
        t = now.time()
        from datetime import time as _time

        if t >= _time(23, 30):
            # 23:30 이후 — 당일이 평일이어야 함 (월~금)
            return now.weekday() < 5
        if t <= _time(6, 0):
            # 06:00 이전 — 전날이 평일이어야 함
            # 월요일(0) 새벽 → 전날 일요일(6) → False
            # 화~토 새벽 → 전날 월~금 → True
            return now.weekday() not in (0, 6)
        return False

    # ───────────────── 내부 ─────────────────

    def _append_history(self, result: dict[str, Any]) -> None:
        self._cycle_history.append(result)
        if len(self._cycle_history) > self.MAX_HISTORY:
            self._cycle_history = self._cycle_history[-self.MAX_HISTORY:]
