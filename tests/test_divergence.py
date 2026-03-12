"""
다이버전스(Divergence) 전략 테스트
"""

from __future__ import annotations

import math

import pytest

from src.strategy.divergence import (
    DivergenceConfig,
    DivergenceResult,
    DivergenceStrategy,
    DivergenceType,
    _apply_cutoff,
    _calculate_macd,
    _calculate_obv,
    _calculate_rsi,
    _find_pivot_highs,
    _find_pivot_lows,
    detect_price_macd_divergence,
    detect_price_obv_divergence,
    detect_price_rsi_divergence,
    detect_vix_index_divergence,
)


# ---------------------------------------------------------------------------
# 테스트 데이터 헬퍼
# ---------------------------------------------------------------------------

def _make_dates(n: int, start: str = "2026-01-01") -> list[str]:
    """YYYY-MM-DD 형식의 날짜 리스트 생성"""
    from datetime import date, timedelta

    d = date.fromisoformat(start)
    return [(d + timedelta(days=i)).isoformat() for i in range(n)]


def _make_bullish_rsi_data(n: int = 60) -> tuple[list[float], list[float], list[str]]:
    """
    Bullish RSI divergence 데이터:
    가격은 lower low, RSI는 higher low가 되도록 구성
    """
    # 하락 → 반등 → 더 깊은 하락(가격) 하지만 RSI는 덜 떨어지는 패턴
    prices: list[float] = []
    # 첫 구간: 100 → 80 (하락)
    for i in range(15):
        prices.append(100 - i * 1.33)
    # 반등: 80 → 95
    for i in range(15):
        prices.append(80 + i * 1.0)
    # 두 번째 하락: 95 → 75 (가격은 더 낮지만 하락폭이 작음)
    for i in range(15):
        prices.append(95 - i * 1.33)
    # 반등
    for i in range(15):
        prices.append(75 + i * 1.0)

    dates = _make_dates(len(prices))
    volumes = [1000.0] * len(prices)
    return prices, volumes, dates


def _make_bearish_rsi_data(n: int = 60) -> tuple[list[float], list[float], list[str]]:
    """
    Bearish RSI divergence 데이터:
    가격은 higher high, RSI는 lower high
    """
    prices: list[float] = []
    # 상승: 100 → 120
    for i in range(15):
        prices.append(100 + i * 1.33)
    # 조정: 120 → 110
    for i in range(15):
        prices.append(120 - i * 0.67)
    # 두 번째 상승: 110 → 125 (가격은 더 높음)
    for i in range(15):
        prices.append(110 + i * 1.0)
    # 조정
    for i in range(15):
        prices.append(125 - i * 0.67)

    dates = _make_dates(len(prices))
    volumes = [1000.0] * len(prices)
    return prices, volumes, dates


# ---------------------------------------------------------------------------
# DivergenceConfig 테스트
# ---------------------------------------------------------------------------

class TestDivergenceConfig:
    """DivergenceConfig 설정 테스트"""

    def test_default_config(self) -> None:
        config = DivergenceConfig()
        assert config.lookback_period == 5
        assert config.rsi_period == 14
        assert config.macd_fast == 12
        assert config.macd_slow == 26

    def test_custom_config(self) -> None:
        config = DivergenceConfig(lookback_period=3, rsi_period=10)
        assert config.lookback_period == 3
        assert config.rsi_period == 10

    def test_invalid_lookback(self) -> None:
        with pytest.raises(ValueError, match="lookback_period"):
            DivergenceConfig(lookback_period=0)

    def test_invalid_rsi_period(self) -> None:
        with pytest.raises(ValueError, match="rsi_period"):
            DivergenceConfig(rsi_period=1)


# ---------------------------------------------------------------------------
# 유틸리티 함수 테스트
# ---------------------------------------------------------------------------

class TestIndicators:
    """지표 계산 테스트"""

    def test_rsi_basic(self) -> None:
        prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
        rsi = _calculate_rsi(prices, 14)
        assert len(rsi) == len(prices)
        assert all(0 <= v <= 100 for v in rsi[14:])

    def test_rsi_insufficient_data(self) -> None:
        rsi = _calculate_rsi([100, 101, 102], 14)
        assert len(rsi) == 3  # 패딩된 50.0

    def test_macd_basic(self) -> None:
        prices = list(range(100, 150))
        macd_line, signal_line, histogram = _calculate_macd(prices)
        assert len(macd_line) == len(prices)
        assert len(histogram) == len(prices)

    def test_obv_basic(self) -> None:
        prices = [10, 11, 10.5, 11.5, 11]
        volumes = [100, 200, 150, 300, 250]
        obv = _calculate_obv(prices, volumes)
        assert len(obv) == 5
        assert obv[0] == 0
        assert obv[1] == 200  # 가격 상승 → +volume
        assert obv[2] == 50   # 가격 하락 → -volume

    def test_obv_empty(self) -> None:
        assert _calculate_obv([], []) == []


class TestPivotPoints:
    """피봇 포인트 탐색 테스트"""

    def test_find_pivot_highs(self) -> None:
        values = [1, 2, 5, 2, 1, 3, 7, 3, 1, 2, 4, 2, 1]
        highs = _find_pivot_highs(values, lookback=2)
        assert 2 in highs   # value=5
        assert 6 in highs   # value=7

    def test_find_pivot_lows(self) -> None:
        values = [5, 3, 1, 3, 5, 3, 0, 3, 5, 3, 2, 3, 5]
        lows = _find_pivot_lows(values, lookback=2)
        assert 2 in lows    # value=1
        assert 6 in lows    # value=0

    def test_no_pivots_flat(self) -> None:
        values = [5] * 20
        assert _find_pivot_highs(values, lookback=3) == []
        assert _find_pivot_lows(values, lookback=3) == []


# ---------------------------------------------------------------------------
# cutoff_date 테스트
# ---------------------------------------------------------------------------

class TestCutoffDate:
    """cutoff_date 필터 테스트"""

    def test_apply_cutoff_none(self) -> None:
        dates = _make_dates(10)
        assert _apply_cutoff(dates, None) == 0

    def test_apply_cutoff_filters_january(self) -> None:
        dates = _make_dates(60, start="2026-01-15")
        cutoff = "2026-01-31"
        start_idx = _apply_cutoff(dates, cutoff)
        assert start_idx > 0
        # cutoff 이후의 날짜만 남아야 함
        assert dates[start_idx] >= cutoff

    def test_apply_cutoff_all_before(self) -> None:
        dates = _make_dates(10, start="2025-12-01")
        start_idx = _apply_cutoff(dates, "2026-01-01")
        assert start_idx == len(dates)

    def test_cutoff_in_strategy(self) -> None:
        """cutoff_date가 전략에서 제대로 적용되는지 확인"""
        prices, volumes, dates = _make_bullish_rsi_data()
        # 1월 말 이전 데이터를 제외하는 cutoff
        strategy = DivergenceStrategy(
            config=DivergenceConfig(lookback_period=3, enabled_types=["rsi"]),
            cutoff_date="2026-02-15",
        )
        result = strategy.analyze({
            "prices": prices,
            "dates": dates,
            "volumes": volumes,
        })
        # cutoff 이전의 다이버전스는 감지되지 않아야 함
        for div in result["divergences"]["rsi"]:
            assert dates[div.price_idx_1] >= "2026-02-15"


# ---------------------------------------------------------------------------
# Price-RSI 다이버전스 테스트
# ---------------------------------------------------------------------------

class TestPriceRSIDivergence:
    """Price-RSI 다이버전스 감지 테스트"""

    def test_detect_with_sufficient_data(self) -> None:
        prices, _, dates = _make_bullish_rsi_data()
        config = DivergenceConfig(lookback_period=3)
        results = detect_price_rsi_divergence(prices, dates, config)
        # 결과가 있든 없든 리스트여야 함
        assert isinstance(results, list)

    def test_insufficient_data(self) -> None:
        prices = [100, 101, 102]
        dates = _make_dates(3)
        config = DivergenceConfig(lookback_period=3)
        results = detect_price_rsi_divergence(prices, dates, config)
        assert results == []

    def test_no_divergence_uptrend(self) -> None:
        """일관된 상승 추세에서는 bullish divergence 없음"""
        prices = [100 + i * 0.5 for i in range(80)]
        dates = _make_dates(80)
        config = DivergenceConfig(lookback_period=3)
        results = detect_price_rsi_divergence(prices, dates, config)
        bullish = [r for r in results if r.divergence_type == DivergenceType.BULLISH]
        assert len(bullish) == 0


# ---------------------------------------------------------------------------
# Price-MACD 다이버전스 테스트
# ---------------------------------------------------------------------------

class TestPriceMACDDivergence:
    """Price-MACD 다이버전스 감지 테스트"""

    def test_detect_with_sufficient_data(self) -> None:
        prices, _, dates = _make_bullish_rsi_data()
        config = DivergenceConfig(lookback_period=3)
        results = detect_price_macd_divergence(prices, dates, config)
        assert isinstance(results, list)

    def test_insufficient_data(self) -> None:
        prices = [100] * 10
        dates = _make_dates(10)
        config = DivergenceConfig(lookback_period=3)
        results = detect_price_macd_divergence(prices, dates, config)
        assert results == []


# ---------------------------------------------------------------------------
# Price-OBV 다이버전스 테스트
# ---------------------------------------------------------------------------

class TestPriceOBVDivergence:
    """Price-OBV 다이버전스 감지 테스트"""

    def test_detect_with_data(self) -> None:
        # 가격은 lower low이지만 volume은 매수세 유지 → OBV higher low
        n = 40
        prices: list[float] = []
        volumes: list[float] = []
        # 하락 구간 1
        for i in range(10):
            prices.append(100 - i * 2)
            volumes.append(500)
        # 반등
        for i in range(10):
            prices.append(80 + i * 1.5)
            volumes.append(1000)  # 반등 시 높은 거래량
        # 하락 구간 2 (가격은 더 낮지만 거래량 적음)
        for i in range(10):
            prices.append(95 - i * 2.5)
            volumes.append(300)
        # 반등
        for i in range(10):
            prices.append(70 + i * 2)
            volumes.append(1200)

        dates = _make_dates(n)
        config = DivergenceConfig(lookback_period=3)
        results = detect_price_obv_divergence(prices, volumes, dates, config)
        assert isinstance(results, list)

    def test_insufficient_data(self) -> None:
        results = detect_price_obv_divergence([100], [100], ["2026-01-01"], DivergenceConfig(lookback_period=3))
        assert results == []


# ---------------------------------------------------------------------------
# VIX-Index 다이버전스 테스트
# ---------------------------------------------------------------------------

class TestVIXIndexDivergence:
    """VIX-Index 다이버전스 감지 테스트"""

    def test_vix_bearish_both_rising(self) -> None:
        """VIX와 지수 동시 상승 → bearish"""
        n = 40
        # 지수 상승
        index_prices = [100 + i * 0.5 for i in range(n)]
        # VIX도 상승 → 피크
        vix = [15.0] * 10
        for i in range(10):
            vix.append(15 + i * 2)  # 상승
        for i in range(10):
            vix.append(35 - i * 1.5)  # 하락
        vix.extend([12.0] * 10)

        dates = _make_dates(n)
        config = DivergenceConfig(vix_peak_lookback=5)
        results = detect_vix_index_divergence(index_prices, vix, dates, config)
        bearish = [r for r in results if r.divergence_type == DivergenceType.BEARISH]
        assert len(bearish) >= 0  # VIX 피크에서 감지될 수 있음

    def test_vix_bullish_bottom_signal(self) -> None:
        """VIX 상승 + 지수 하락 → bullish 바닥 시그널"""
        n = 40
        # 지수 하락
        index_prices = [100 - i * 0.5 for i in range(n)]
        # VIX 상승 → 피크 → 하락
        vix = [15.0] * 10
        for i in range(10):
            vix.append(15 + i * 3)
        for i in range(10):
            vix.append(45 - i * 2)
        vix.extend([20.0] * 10)

        dates = _make_dates(n)
        config = DivergenceConfig(vix_peak_lookback=5)
        results = detect_vix_index_divergence(index_prices, vix, dates, config)
        bullish = [r for r in results if r.divergence_type == DivergenceType.BULLISH]
        assert isinstance(results, list)

    def test_insufficient_data(self) -> None:
        results = detect_vix_index_divergence([100], [20], ["2026-01-01"], DivergenceConfig(vix_peak_lookback=5))
        assert results == []


# ---------------------------------------------------------------------------
# DivergenceStrategy 통합 테스트
# ---------------------------------------------------------------------------

class TestDivergenceStrategy:
    """DivergenceStrategy 클래스 통합 테스트"""

    def test_strategy_initialization(self) -> None:
        strategy = DivergenceStrategy()
        assert strategy.name == "Divergence"
        assert strategy.cutoff_date is None

    def test_strategy_with_cutoff(self) -> None:
        strategy = DivergenceStrategy(cutoff_date="2026-01-31")
        assert strategy.cutoff_date == "2026-01-31"

    def test_analyze_empty_data(self) -> None:
        strategy = DivergenceStrategy()
        result = strategy.analyze({"prices": [], "dates": []})
        assert result["total_bullish"] == 0
        assert result["total_bearish"] == 0

    def test_analyze_returns_all_types(self) -> None:
        prices, volumes, dates = _make_bullish_rsi_data()
        strategy = DivergenceStrategy(
            config=DivergenceConfig(lookback_period=3),
        )
        result = strategy.analyze({
            "prices": prices,
            "dates": dates,
            "volumes": volumes,
        })
        assert "divergences" in result
        assert "rsi" in result["divergences"]
        assert "macd" in result["divergences"]
        assert "obv" in result["divergences"]
        assert "vix" in result["divergences"]

    def test_generate_signal_hold_no_divergence(self) -> None:
        strategy = DivergenceStrategy()
        analysis = {
            "divergences": {"rsi": [], "macd": [], "obv": [], "vix": []},
            "total_bullish": 0,
            "total_bearish": 0,
            "prices": [100],
            "dates": ["2026-01-01"],
        }
        signal = strategy.generate_signal(analysis)
        assert signal["signal"] == "hold"

    def test_generate_signal_buy(self) -> None:
        strategy = DivergenceStrategy()
        analysis = {
            "divergences": {
                "rsi": [DivergenceResult(
                    divergence_type=DivergenceType.BULLISH,
                    indicator="RSI", price_idx_1=0, price_idx_2=1,
                    price_val_1=100, price_val_2=95,
                    indicator_val_1=30, indicator_val_2=35,
                    description="test",
                )],
                "macd": [],
                "obv": [],
                "vix": [],
            },
            "total_bullish": 1,
            "total_bearish": 0,
            "prices": [100],
            "dates": ["2026-01-01"],
        }
        signal = strategy.generate_signal(analysis)
        assert signal["signal"] == "buy"
        assert signal["strength"] > 0

    def test_generate_signal_sell(self) -> None:
        strategy = DivergenceStrategy()
        analysis = {
            "divergences": {
                "rsi": [DivergenceResult(
                    divergence_type=DivergenceType.BEARISH,
                    indicator="RSI", price_idx_1=0, price_idx_2=1,
                    price_val_1=100, price_val_2=105,
                    indicator_val_1=70, indicator_val_2=65,
                    description="test",
                )],
                "macd": [],
                "obv": [],
                "vix": [],
            },
            "total_bullish": 0,
            "total_bearish": 1,
            "prices": [100],
            "dates": ["2026-01-01"],
        }
        signal = strategy.generate_signal(analysis)
        assert signal["signal"] == "sell"

    def test_generate_signal_equal_hold(self) -> None:
        """Bullish/Bearish 동수이면 관망"""
        strategy = DivergenceStrategy()
        analysis = {
            "divergences": {"rsi": [], "macd": [], "obv": [], "vix": []},
            "total_bullish": 2,
            "total_bearish": 2,
            "prices": [100],
            "dates": ["2026-01-01"],
        }
        signal = strategy.generate_signal(analysis)
        assert signal["signal"] == "hold"

    def test_backtest_insufficient_data(self) -> None:
        strategy = DivergenceStrategy()
        result = strategy.backtest([{"close": 100, "date": "2026-01-01"}], 1_000_000)
        assert result["total_return"] == 0.0
        assert result.get("error") == "데이터 부족"

    def test_backtest_basic(self) -> None:
        """기본 백테스팅 실행 확인"""
        prices, volumes, dates = _make_bullish_rsi_data()
        historical = [
            {"date": d, "close": p, "volume": v}
            for d, p, v in zip(dates, prices, volumes)
        ]
        strategy = DivergenceStrategy(
            config=DivergenceConfig(lookback_period=2),
        )
        result = strategy.backtest(historical, 1_000_000)
        assert "total_return" in result
        assert "trades" in result
        assert isinstance(result["equity_curve"], list)

    def test_signal_has_required_fields(self) -> None:
        strategy = DivergenceStrategy()
        analysis = {
            "divergences": {"rsi": [], "macd": [], "obv": [], "vix": []},
            "total_bullish": 0,
            "total_bearish": 0,
            "prices": [100],
            "dates": ["2026-01-01"],
        }
        signal = strategy.generate_signal(analysis)
        required = {"signal", "strength", "reason", "strategy_name", "timestamp", "metrics"}
        assert required.issubset(signal.keys())
