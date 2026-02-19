"""AutoTraderScheduler 테스트"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio  # noqa: F401

from src.strategy.auto_trader_scheduler import AutoTraderScheduler

KST = ZoneInfo("Asia/Seoul")


# ───────────────── Fixtures ─────────────────


@pytest.fixture()
def mock_trader() -> MagicMock:
    trader = MagicMock()
    trader.run_cycle.return_value = {
        "timestamp": "2026-02-19T10:00:00+09:00",
        "sentiment": {"score": 45, "classification": "fear", "buy_multiplier": 1.2, "recommendation": "매수 확대"},
        "scanned": 30,
        "buy_signals": [{"stock_code": "005930", "stock_name": "삼성전자", "signal_type": "buy", "score": 55.0, "reason": "test"}],
        "sell_signals": [],
        "executed_buys": [],
        "executed_sells": [],
        "dry_run": True,
    }
    return trader


@pytest.fixture()
def scheduler(mock_trader: MagicMock) -> AutoTraderScheduler:
    return AutoTraderScheduler(mock_trader)


# ───────────────── 장 시간 판단 테스트 ─────────────────


class TestIsKrMarketOpen:
    """국내 장 시간 판단 테스트"""

    def test_weekday_market_hours(self) -> None:
        # 수요일 10시 → True
        dt = datetime(2026, 2, 18, 10, 0, tzinfo=KST)  # 수요일
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_kr_market_open() is True

    def test_weekday_before_open(self) -> None:
        dt = datetime(2026, 2, 18, 8, 30, tzinfo=KST)
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_kr_market_open() is False

    def test_weekday_after_close(self) -> None:
        dt = datetime(2026, 2, 18, 16, 0, tzinfo=KST)
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_kr_market_open() is False

    def test_weekend(self) -> None:
        dt = datetime(2026, 2, 21, 10, 0, tzinfo=KST)  # 토요일
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_kr_market_open() is False

    def test_market_open_boundary(self) -> None:
        dt = datetime(2026, 2, 18, 9, 0, tzinfo=KST)
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_kr_market_open() is True

    def test_market_close_boundary(self) -> None:
        dt = datetime(2026, 2, 18, 15, 30, tzinfo=KST)
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_kr_market_open() is True


class TestIsUsMarketOpen:
    """미국 장 시간 판단 테스트"""

    def test_late_night_weekday(self) -> None:
        # 수요일 23:45 → True
        dt = datetime(2026, 2, 18, 23, 45, tzinfo=KST)
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_us_market_open() is True

    def test_early_morning_weekday(self) -> None:
        # 목요일 03:00 (전날 수요일이 평일) → True
        dt = datetime(2026, 2, 19, 3, 0, tzinfo=KST)  # 목요일
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_us_market_open() is True

    def test_daytime(self) -> None:
        # 수요일 14:00 → False
        dt = datetime(2026, 2, 18, 14, 0, tzinfo=KST)
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_us_market_open() is False

    def test_saturday_night(self) -> None:
        # 토요일 23:45 → False (주말)
        dt = datetime(2026, 2, 21, 23, 45, tzinfo=KST)
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_us_market_open() is False

    def test_monday_early_morning(self) -> None:
        # 월요일 03:00 (전날 일요일) → False
        dt = datetime(2026, 2, 16, 3, 0, tzinfo=KST)  # 월요일
        with patch("src.strategy.auto_trader_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert AutoTraderScheduler.is_us_market_open() is False


# ───────────────── 스케줄러 시작/중지 테스트 ─────────────────


class TestStartStop:
    def test_start(self, scheduler: AutoTraderScheduler) -> None:
        with patch.object(scheduler._scheduler, "start"), \
             patch.object(scheduler._scheduler, "add_job"):
            scheduler.start(interval_minutes=15)
            assert scheduler._is_running is True
            assert scheduler._interval_minutes == 15

    def test_start_already_running(self, scheduler: AutoTraderScheduler) -> None:
        with patch.object(scheduler._scheduler, "start"), \
             patch.object(scheduler._scheduler, "add_job"):
            scheduler.start()
            # 두 번째 호출은 무시
            scheduler.start()
            assert scheduler._is_running is True

    def test_stop(self, scheduler: AutoTraderScheduler) -> None:
        with patch.object(scheduler._scheduler, "start"), \
             patch.object(scheduler._scheduler, "add_job"):
            scheduler.start()

        with patch.object(scheduler._scheduler, "shutdown"):
            scheduler.stop()
            assert scheduler._is_running is False

    def test_stop_not_running(self, scheduler: AutoTraderScheduler) -> None:
        # 실행 중이 아닐 때 stop → 경고만
        scheduler.stop()
        assert scheduler._is_running is False


# ───────────────── 사이클 실행 테스트 ─────────────────


class TestRunScheduledCycle:
    @pytest.mark.asyncio()
    async def test_run_during_kr_market(self, scheduler: AutoTraderScheduler, mock_trader: MagicMock) -> None:
        """장 시간 내 실행"""
        with patch.object(AutoTraderScheduler, "is_kr_market_open", return_value=True), \
             patch.object(AutoTraderScheduler, "is_us_market_open", return_value=False):
            scheduler._kr_market_only = True
            result = await scheduler.run_scheduled_cycle()
            assert result["status"] == "completed"
            mock_trader.run_cycle.assert_called_once()

    @pytest.mark.asyncio()
    async def test_skip_outside_market(self, scheduler: AutoTraderScheduler, mock_trader: MagicMock) -> None:
        """장 마감 시 스킵"""
        with patch.object(AutoTraderScheduler, "is_kr_market_open", return_value=False), \
             patch.object(AutoTraderScheduler, "is_us_market_open", return_value=False):
            scheduler._kr_market_only = True
            result = await scheduler.run_scheduled_cycle()
            assert result["status"] == "skipped"
            mock_trader.run_cycle.assert_not_called()

    @pytest.mark.asyncio()
    async def test_run_during_us_market(self, scheduler: AutoTraderScheduler, mock_trader: MagicMock) -> None:
        """미장 시간 실행"""
        with patch.object(AutoTraderScheduler, "is_kr_market_open", return_value=False), \
             patch.object(AutoTraderScheduler, "is_us_market_open", return_value=True):
            scheduler._kr_market_only = False
            scheduler._us_market = True
            result = await scheduler.run_scheduled_cycle()
            assert result["status"] == "completed"

    @pytest.mark.asyncio()
    async def test_run_always_when_no_market_filter(self, scheduler: AutoTraderScheduler, mock_trader: MagicMock) -> None:
        """필터 없으면 항상 실행"""
        with patch.object(AutoTraderScheduler, "is_kr_market_open", return_value=False), \
             patch.object(AutoTraderScheduler, "is_us_market_open", return_value=False):
            scheduler._kr_market_only = False
            scheduler._us_market = False
            result = await scheduler.run_scheduled_cycle()
            assert result["status"] == "completed"

    @pytest.mark.asyncio()
    async def test_run_cycle_error(self, scheduler: AutoTraderScheduler, mock_trader: MagicMock) -> None:
        """사이클 실행 실패 시 에러 기록"""
        mock_trader.run_cycle.side_effect = RuntimeError("API 오류")
        with patch.object(AutoTraderScheduler, "is_kr_market_open", return_value=True), \
             patch.object(AutoTraderScheduler, "is_us_market_open", return_value=False):
            scheduler._kr_market_only = True
            result = await scheduler.run_scheduled_cycle()
            assert result["status"] == "error"


# ───────────────── 상태 조회 테스트 ─────────────────


class TestGetStatus:
    def test_status_not_running(self, scheduler: AutoTraderScheduler) -> None:
        status = scheduler.get_status()
        assert status["is_running"] is False
        assert status["total_cycles"] == 0
        assert status["last_cycle_result"] is None
        assert status["kr_market_hours"] == "09:00-15:30 KST"

    def test_status_with_us_market(self, scheduler: AutoTraderScheduler) -> None:
        scheduler._us_market = True
        status = scheduler.get_status()
        assert "us_market_hours" in status


# ───────────────── 히스토리 테스트 ─────────────────


class TestGetCycleHistory:
    def test_empty_history(self, scheduler: AutoTraderScheduler) -> None:
        assert scheduler.get_cycle_history() == []

    def test_history_limit(self, scheduler: AutoTraderScheduler) -> None:
        for i in range(5):
            scheduler._cycle_history.append({"index": i})
        history = scheduler.get_cycle_history(limit=3)
        assert len(history) == 3
        # 최신순
        assert history[0]["index"] == 4
        assert history[2]["index"] == 2

    def test_history_max_cap(self, scheduler: AutoTraderScheduler) -> None:
        for i in range(150):
            scheduler._append_history({"index": i})
        assert len(scheduler._cycle_history) == AutoTraderScheduler.MAX_HISTORY


# ───────────────── API 엔드포인트 테스트 ─────────────────


class TestSchedulerAPI:
    @pytest.fixture()
    def client(self) -> Any:
        """FastAPI TestClient (API 라우터 단위, 기존 동작 유지)"""
        from unittest.mock import patch as _patch

        # Mock dependencies
        mock_kis = MagicMock()
        mock_kis.get_balance.return_value = {"holdings": [], "summary": [{"tot_evlu_amt": 100000000}]}
        mock_kis.get_price.return_value = {"stck_prpr": "50000", "prdy_ctrt": "1.5", "stck_hgpr": "51000", "stck_lwpr": "49000"}

        with _patch("src.api.auto_trader._get_scheduler") as mock_get_sched:
            mock_sched = MagicMock(spec=AutoTraderScheduler)
            mock_sched.get_status.return_value = {
                "is_running": False,
                "interval_minutes": 30,
                "next_run_time": None,
                "total_cycles": 0,
                "last_cycle_result": None,
                "kr_market_hours": "09:00-15:30 KST",
            }
            mock_sched.get_cycle_history.return_value = []
            mock_get_sched.return_value = mock_sched

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from src.api.auto_trader import router

            app = FastAPI()
            app.include_router(router)

            with _patch("src.api.auto_trader.get_kis_client", return_value=mock_kis):
                yield TestClient(app), mock_sched

    def test_scheduler_start(self, client: Any) -> None:
        test_client, mock_sched = client
        mock_sched.get_status.return_value["is_running"] = True
        resp = test_client.post("/api/v1/auto-trader/scheduler/start", json={"interval_minutes": 15})
        assert resp.status_code == 200
        mock_sched.start.assert_called_once_with(interval_minutes=15, kr_market_only=True, us_market=False)

    def test_scheduler_stop(self, client: Any) -> None:
        test_client, mock_sched = client
        resp = test_client.post("/api/v1/auto-trader/scheduler/stop")
        assert resp.status_code == 200
        mock_sched.stop.assert_called_once()

    def test_scheduler_status(self, client: Any) -> None:
        test_client, mock_sched = client
        resp = test_client.get("/api/v1/auto-trader/scheduler/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_running" in data

    def test_scheduler_history(self, client: Any) -> None:
        test_client, mock_sched = client
        resp = test_client.get("/api/v1/auto-trader/scheduler/history")
        assert resp.status_code == 200
        assert resp.json() == []


class TestSchedulerLifecycleIntegration:
    """실제 FastAPI 앱 + lifespan 기반 스케줄러 통합 테스트.

    /scheduler/start 호출 시 500이 발생하지 않고, 스케줄러 상태에서
    is_running=True 가 반환되는지 확인합니다.
    """

    @pytest.fixture()
    def client(self) -> Any:
        from unittest.mock import patch as _patch

        from fastapi.testclient import TestClient

        from src.api import auto_trader as auto_trader_api
        from src.main import app

        # 기존 싱글턴 스케줄러 초기화
        auto_trader_api._scheduler = None  # type: ignore[attr-defined]

        # KISClient 및 AutoTrader.run_cycle 모킹 (실제 외부 호출 방지)
        mock_kis = MagicMock()
        mock_kis.get_balance.return_value = {"holdings": [], "summary": [{"tot_evlu_amt": 100000000}]}
        mock_kis.get_price.return_value = {"stck_prpr": "50000", "prdy_ctrt": "1.5", "stck_hgpr": "51000", "stck_lwpr": "49000"}

        with _patch("src.api.auto_trader.get_kis_client", return_value=mock_kis), \
             _patch("src.strategy.auto_trader.AutoTrader.run_cycle", return_value={}):
            # TestClient 는 lifespan 컨텍스트를 실행하므로, 내부에서
            # set_scheduler_event_loop() 가 호출되어 APScheduler가 FastAPI 메인
            # 이벤트 루프에 바인딩됩니다.
            with TestClient(app) as test_client:
                yield test_client

    def test_scheduler_start_and_status_running(self, client: Any) -> None:
        # /scheduler/start 호출 시 500이 아닌 200이 반환되어야 함
        resp = client.post("/api/v1/auto-trader/scheduler/start", json={"interval_minutes": 15})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_running"] is True

        # /scheduler/status 에서도 is_running=True 유지
        resp_status = client.get("/api/v1/auto-trader/scheduler/status")
        assert resp_status.status_code == 200
        status = resp_status.json()
        assert status["is_running"] is True
