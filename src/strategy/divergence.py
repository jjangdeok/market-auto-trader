"""
다이버전스(Divergence) 감지 전략

가격과 기술적 지표 간의 방향 불일치를 감지하여 추세 전환 시그널을 생성합니다.

지원하는 다이버전스 유형:
1. Price-RSI Divergence
2. Price-MACD Divergence
3. Price-OBV (On Balance Volume) Divergence
4. VIX-Index Divergence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.strategy.base import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------

class SignalType(str, Enum):
    """매매 신호 종류"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class DivergenceType(str, Enum):
    """다이버전스 종류"""

    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class DivergenceConfig:
    """다이버전스 전략 설정

    Attributes:
        lookback_period: 피봇 포인트 탐색 기간 (기본 5)
        pivot_threshold: 피봇 포인트 인식 최소 변화율 (기본 0.0, %)
        rsi_period: RSI 계산 기간 (기본 14)
        macd_fast: MACD 빠른 이동평균 기간 (기본 12)
        macd_slow: MACD 느린 이동평균 기간 (기본 26)
        macd_signal: MACD 시그널 기간 (기본 9)
        vix_peak_lookback: VIX 피크 판단 lookback (기본 10)
        enabled_types: 활성화할 다이버전스 유형 목록
    """

    lookback_period: int = 5
    pivot_threshold: float = 0.0
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vix_peak_lookback: int = 10
    enabled_types: list[str] = field(
        default_factory=lambda: ["rsi", "macd", "obv", "vix"],
    )

    def __post_init__(self) -> None:
        if self.lookback_period < 1:
            msg = "lookback_period는 최소 1 이상이어야 합니다"
            raise ValueError(msg)
        if self.rsi_period < 2:
            msg = "rsi_period는 최소 2 이상이어야 합니다"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# 지표 계산 유틸리티
# ---------------------------------------------------------------------------

def _calculate_rsi(prices: list[float], period: int = 14) -> list[float]:
    """RSI 계산 (Wilder 방식)"""
    if len(prices) < period + 1:
        return [50.0] * len(prices)

    result: list[float] = [50.0] * period

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(100.0 - 100.0 / (1.0 + rs))

    for i in range(period + 1, len(prices)):
        change = prices[i] - prices[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - 100.0 / (1.0 + rs))

    return result


def _calculate_ema(values: list[float], period: int) -> list[float]:
    """지수 이동평균(EMA) 계산"""
    if not values:
        return []
    result: list[float] = [values[0]]
    multiplier = 2.0 / (period + 1)
    for i in range(1, len(values)):
        ema = values[i] * multiplier + result[-1] * (1 - multiplier)
        result.append(ema)
    return result


def _calculate_macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[float], list[float], list[float]]:
    """MACD 계산 → (macd_line, signal_line, histogram)"""
    ema_fast = _calculate_ema(prices, fast)
    ema_slow = _calculate_ema(prices, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _calculate_ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


def _calculate_obv(prices: list[float], volumes: list[float]) -> list[float]:
    """OBV (On Balance Volume) 계산"""
    if not prices or not volumes:
        return []
    obv: list[float] = [0.0]
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif prices[i] < prices[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv


# ---------------------------------------------------------------------------
# 피봇 포인트 탐색
# ---------------------------------------------------------------------------

def _find_pivot_highs(
    values: list[float],
    lookback: int,
    threshold: float = 0.0,
) -> list[int]:
    """로컬 고점(피봇 하이) 인덱스 목록 반환"""
    pivots: list[int] = []
    for i in range(lookback, len(values) - lookback):
        is_pivot = True
        for j in range(1, lookback + 1):
            if values[i] <= values[i - j] or values[i] <= values[i + j]:
                is_pivot = False
                break
        if is_pivot:
            if threshold > 0 and pivots:
                change = abs(values[i] - values[pivots[-1]]) / max(abs(values[pivots[-1]]), 1e-10)
                if change < threshold / 100:
                    continue
            pivots.append(i)
    return pivots


def _find_pivot_lows(
    values: list[float],
    lookback: int,
    threshold: float = 0.0,
) -> list[int]:
    """로컬 저점(피봇 로우) 인덱스 목록 반환"""
    pivots: list[int] = []
    for i in range(lookback, len(values) - lookback):
        is_pivot = True
        for j in range(1, lookback + 1):
            if values[i] >= values[i - j] or values[i] >= values[i + j]:
                is_pivot = False
                break
        if is_pivot:
            if threshold > 0 and pivots:
                change = abs(values[i] - values[pivots[-1]]) / max(abs(values[pivots[-1]]), 1e-10)
                if change < threshold / 100:
                    continue
            pivots.append(i)
    return pivots


# ---------------------------------------------------------------------------
# 다이버전스 감지 함수들
# ---------------------------------------------------------------------------

@dataclass
class DivergenceResult:
    """다이버전스 감지 결과"""

    divergence_type: DivergenceType
    indicator: str
    price_idx_1: int
    price_idx_2: int
    price_val_1: float
    price_val_2: float
    indicator_val_1: float
    indicator_val_2: float
    description: str


def _apply_cutoff(
    dates: list[str],
    cutoff_date: str | None,
) -> int:
    """cutoff_date 이전 데이터를 제외할 시작 인덱스 반환"""
    if not cutoff_date or not dates:
        return 0
    for i, d in enumerate(dates):
        if d >= cutoff_date:
            return i
    return len(dates)


def detect_price_rsi_divergence(
    prices: list[float],
    dates: list[str],
    config: DivergenceConfig,
    cutoff_date: str | None = None,
) -> list[DivergenceResult]:
    """Price-RSI 다이버전스 감지"""
    start = _apply_cutoff(dates, cutoff_date)
    if len(prices) - start < config.rsi_period + config.lookback_period * 2 + 1:
        logger.warning("Price-RSI 다이버전스 감지에 데이터 부족")
        return []

    rsi = _calculate_rsi(prices, config.rsi_period)
    lb = config.lookback_period
    th = config.pivot_threshold
    results: list[DivergenceResult] = []

    # Bullish: 가격 lower low + RSI higher low
    price_lows = [i for i in _find_pivot_lows(prices, lb, th) if i >= start]
    rsi_lows = [i for i in _find_pivot_lows(rsi, lb, th) if i >= start]

    if len(price_lows) >= 2:
        for a_idx in range(len(price_lows) - 1):
            i1, i2 = price_lows[a_idx], price_lows[a_idx + 1]
            if prices[i2] < prices[i1]:
                # 가격이 lower low → RSI에서 가까운 피봇 저점 찾기
                r1_candidates = [r for r in rsi_lows if abs(r - i1) <= lb]
                r2_candidates = [r for r in rsi_lows if abs(r - i2) <= lb]
                if not r1_candidates or not r2_candidates:
                    # 피봇이 정확히 안 맞으면 해당 인덱스의 RSI 값 직접 비교
                    if rsi[i2] > rsi[i1]:
                        results.append(DivergenceResult(
                            divergence_type=DivergenceType.BULLISH,
                            indicator="RSI",
                            price_idx_1=i1, price_idx_2=i2,
                            price_val_1=prices[i1], price_val_2=prices[i2],
                            indicator_val_1=rsi[i1], indicator_val_2=rsi[i2],
                            description=f"Bullish RSI 다이버전스: 가격 하락({prices[i1]:.2f}→{prices[i2]:.2f}) vs RSI 상승({rsi[i1]:.2f}→{rsi[i2]:.2f})",
                        ))
                else:
                    r1 = min(r1_candidates, key=lambda r: abs(r - i1))
                    r2 = min(r2_candidates, key=lambda r: abs(r - i2))
                    if rsi[r2] > rsi[r1]:
                        results.append(DivergenceResult(
                            divergence_type=DivergenceType.BULLISH,
                            indicator="RSI",
                            price_idx_1=i1, price_idx_2=i2,
                            price_val_1=prices[i1], price_val_2=prices[i2],
                            indicator_val_1=rsi[r1], indicator_val_2=rsi[r2],
                            description=f"Bullish RSI 다이버전스: 가격 하락({prices[i1]:.2f}→{prices[i2]:.2f}) vs RSI 상승({rsi[r1]:.2f}→{rsi[r2]:.2f})",
                        ))

    # Bearish: 가격 higher high + RSI lower high
    price_highs = [i for i in _find_pivot_highs(prices, lb, th) if i >= start]
    rsi_highs = [i for i in _find_pivot_highs(rsi, lb, th) if i >= start]

    if len(price_highs) >= 2:
        for a_idx in range(len(price_highs) - 1):
            i1, i2 = price_highs[a_idx], price_highs[a_idx + 1]
            if prices[i2] > prices[i1]:
                r1_candidates = [r for r in rsi_highs if abs(r - i1) <= lb]
                r2_candidates = [r for r in rsi_highs if abs(r - i2) <= lb]
                if not r1_candidates or not r2_candidates:
                    if rsi[i2] < rsi[i1]:
                        results.append(DivergenceResult(
                            divergence_type=DivergenceType.BEARISH,
                            indicator="RSI",
                            price_idx_1=i1, price_idx_2=i2,
                            price_val_1=prices[i1], price_val_2=prices[i2],
                            indicator_val_1=rsi[i1], indicator_val_2=rsi[i2],
                            description=f"Bearish RSI 다이버전스: 가격 상승({prices[i1]:.2f}→{prices[i2]:.2f}) vs RSI 하락({rsi[i1]:.2f}→{rsi[i2]:.2f})",
                        ))
                else:
                    r1 = min(r1_candidates, key=lambda r: abs(r - i1))
                    r2 = min(r2_candidates, key=lambda r: abs(r - i2))
                    if rsi[r2] < rsi[r1]:
                        results.append(DivergenceResult(
                            divergence_type=DivergenceType.BEARISH,
                            indicator="RSI",
                            price_idx_1=i1, price_idx_2=i2,
                            price_val_1=prices[i1], price_val_2=prices[i2],
                            indicator_val_1=rsi[r1], indicator_val_2=rsi[r2],
                            description=f"Bearish RSI 다이버전스: 가격 상승({prices[i1]:.2f}→{prices[i2]:.2f}) vs RSI 하락({rsi[r1]:.2f}→{rsi[r2]:.2f})",
                        ))

    logger.info("Price-RSI 다이버전스 %d건 감지", len(results))
    return results


def detect_price_macd_divergence(
    prices: list[float],
    dates: list[str],
    config: DivergenceConfig,
    cutoff_date: str | None = None,
) -> list[DivergenceResult]:
    """Price-MACD Histogram 다이버전스 감지"""
    start = _apply_cutoff(dates, cutoff_date)
    min_len = config.macd_slow + config.macd_signal + config.lookback_period * 2
    if len(prices) - start < min_len:
        logger.warning("Price-MACD 다이버전스 감지에 데이터 부족")
        return []

    _, _, histogram = _calculate_macd(prices, config.macd_fast, config.macd_slow, config.macd_signal)
    lb = config.lookback_period
    th = config.pivot_threshold
    results: list[DivergenceResult] = []

    # Bullish: 가격 lower low + histogram higher low
    price_lows = [i for i in _find_pivot_lows(prices, lb, th) if i >= start]
    hist_lows = _find_pivot_lows(histogram, lb, 0.0)

    if len(price_lows) >= 2:
        for a_idx in range(len(price_lows) - 1):
            i1, i2 = price_lows[a_idx], price_lows[a_idx + 1]
            if prices[i2] < prices[i1] and histogram[i2] > histogram[i1]:
                results.append(DivergenceResult(
                    divergence_type=DivergenceType.BULLISH,
                    indicator="MACD",
                    price_idx_1=i1, price_idx_2=i2,
                    price_val_1=prices[i1], price_val_2=prices[i2],
                    indicator_val_1=histogram[i1], indicator_val_2=histogram[i2],
                    description=f"Bullish MACD 다이버전스: 가격 하락 vs MACD histogram 상승",
                ))

    # Bearish: 가격 higher high + histogram lower high
    price_highs = [i for i in _find_pivot_highs(prices, lb, th) if i >= start]

    if len(price_highs) >= 2:
        for a_idx in range(len(price_highs) - 1):
            i1, i2 = price_highs[a_idx], price_highs[a_idx + 1]
            if prices[i2] > prices[i1] and histogram[i2] < histogram[i1]:
                results.append(DivergenceResult(
                    divergence_type=DivergenceType.BEARISH,
                    indicator="MACD",
                    price_idx_1=i1, price_idx_2=i2,
                    price_val_1=prices[i1], price_val_2=prices[i2],
                    indicator_val_1=histogram[i1], indicator_val_2=histogram[i2],
                    description=f"Bearish MACD 다이버전스: 가격 상승 vs MACD histogram 하락",
                ))

    logger.info("Price-MACD 다이버전스 %d건 감지", len(results))
    return results


def detect_price_obv_divergence(
    prices: list[float],
    volumes: list[float],
    dates: list[str],
    config: DivergenceConfig,
    cutoff_date: str | None = None,
) -> list[DivergenceResult]:
    """Price-OBV 다이버전스 감지"""
    start = _apply_cutoff(dates, cutoff_date)
    if len(prices) - start < config.lookback_period * 2 + 1:
        logger.warning("Price-OBV 다이버전스 감지에 데이터 부족")
        return []

    obv = _calculate_obv(prices, volumes)
    lb = config.lookback_period
    th = config.pivot_threshold
    results: list[DivergenceResult] = []

    # Bullish: 가격 lower low + OBV higher low
    price_lows = [i for i in _find_pivot_lows(prices, lb, th) if i >= start]

    if len(price_lows) >= 2:
        for a_idx in range(len(price_lows) - 1):
            i1, i2 = price_lows[a_idx], price_lows[a_idx + 1]
            if prices[i2] < prices[i1] and obv[i2] > obv[i1]:
                results.append(DivergenceResult(
                    divergence_type=DivergenceType.BULLISH,
                    indicator="OBV",
                    price_idx_1=i1, price_idx_2=i2,
                    price_val_1=prices[i1], price_val_2=prices[i2],
                    indicator_val_1=obv[i1], indicator_val_2=obv[i2],
                    description=f"Bullish OBV 다이버전스: 가격 하락 vs OBV 상승",
                ))

    # Bearish: 가격 higher high + OBV lower high
    price_highs = [i for i in _find_pivot_highs(prices, lb, th) if i >= start]

    if len(price_highs) >= 2:
        for a_idx in range(len(price_highs) - 1):
            i1, i2 = price_highs[a_idx], price_highs[a_idx + 1]
            if prices[i2] > prices[i1] and obv[i2] < obv[i1]:
                results.append(DivergenceResult(
                    divergence_type=DivergenceType.BEARISH,
                    indicator="OBV",
                    price_idx_1=i1, price_idx_2=i2,
                    price_val_1=prices[i1], price_val_2=prices[i2],
                    indicator_val_1=obv[i1], indicator_val_2=obv[i2],
                    description=f"Bearish OBV 다이버전스: 가격 상승 vs OBV 하락",
                ))

    logger.info("Price-OBV 다이버전스 %d건 감지", len(results))
    return results


def detect_vix_index_divergence(
    index_prices: list[float],
    vix_values: list[float],
    dates: list[str],
    config: DivergenceConfig,
    cutoff_date: str | None = None,
) -> list[DivergenceResult]:
    """
    VIX-Index 다이버전스 감지

    - VIX와 지수 동시 상승: 위험 시그널 (bearish)
    - VIX 상승 + 지수 하락 후 VIX 피크: 바닥 시그널 (bullish)
    """
    start = _apply_cutoff(dates, cutoff_date)
    if len(index_prices) - start < config.vix_peak_lookback + 1:
        logger.warning("VIX-Index 다이버전스 감지에 데이터 부족")
        return []

    results: list[DivergenceResult] = []
    lb = config.vix_peak_lookback

    # VIX 피크 찾기
    vix_highs = [i for i in _find_pivot_highs(vix_values, lb, 0.0) if i >= start]

    for peak_idx in vix_highs:
        # peak_idx 근처의 지수 동향 확인
        window_start = max(start, peak_idx - lb)
        idx_change = index_prices[peak_idx] - index_prices[window_start]
        vix_change = vix_values[peak_idx] - vix_values[window_start]

        if vix_change > 0 and idx_change > 0:
            # VIX와 지수 동시 상승 → bearish 위험 시그널
            results.append(DivergenceResult(
                divergence_type=DivergenceType.BEARISH,
                indicator="VIX",
                price_idx_1=window_start, price_idx_2=peak_idx,
                price_val_1=index_prices[window_start], price_val_2=index_prices[peak_idx],
                indicator_val_1=vix_values[window_start], indicator_val_2=vix_values[peak_idx],
                description=f"VIX-Index 위험 시그널: VIX와 지수 동시 상승 (VIX {vix_values[window_start]:.2f}→{vix_values[peak_idx]:.2f})",
            ))
        elif vix_change > 0 and idx_change < 0:
            # VIX 상승 + 지수 하락 → VIX 피크에서 bullish 바닥 시그널
            results.append(DivergenceResult(
                divergence_type=DivergenceType.BULLISH,
                indicator="VIX",
                price_idx_1=window_start, price_idx_2=peak_idx,
                price_val_1=index_prices[window_start], price_val_2=index_prices[peak_idx],
                indicator_val_1=vix_values[window_start], indicator_val_2=vix_values[peak_idx],
                description=f"VIX 바닥 시그널: VIX 피크({vix_values[peak_idx]:.2f}) + 지수 하락 → 반등 가능",
            ))

    logger.info("VIX-Index 다이버전스 %d건 감지", len(results))
    return results


# ---------------------------------------------------------------------------
# 전략 클래스
# ---------------------------------------------------------------------------

class DivergenceStrategy(BaseStrategy):
    """
    다이버전스 기반 매매 전략

    가격과 기술적 지표 사이의 방향 불일치를 감지하여
    추세 전환 시점을 포착합니다.

    Usage::

        config = DivergenceConfig(lookback_period=5)
        strategy = DivergenceStrategy(config)

        analysis = strategy.analyze({
            "prices": [...],
            "dates": [...],
            "volumes": [...],
        })
        signal = strategy.generate_signal(analysis)
    """

    def __init__(
        self,
        config: DivergenceConfig | None = None,
        cutoff_date: str | None = None,
    ) -> None:
        self.config = config or DivergenceConfig()
        self.cutoff_date = cutoff_date
        name = "Divergence"
        super().__init__(name=name)
        logger.info(
            "다이버전스 전략 설정: lookback=%d, rsi_period=%d, cutoff=%s, 유형=%s",
            self.config.lookback_period,
            self.config.rsi_period,
            self.cutoff_date or "없음",
            self.config.enabled_types,
        )

    def analyze(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """
        시장 데이터 분석 — 모든 다이버전스 유형 감지

        Args:
            market_data: {
                "prices": list[float],          # 종가 리스트 (필수)
                "dates": list[str],             # 날짜 리스트 (필수)
                "volumes": list[float],         # 거래량 (OBV용, 선택)
                "vix": list[float],             # VIX 값 (VIX용, 선택)
                "index_prices": list[float],    # 지수 가격 (VIX용, 선택)
                "stock_code": str,              # 종목 코드 (선택)
            }

        Returns:
            {
                "divergences": {
                    "rsi": list[DivergenceResult],
                    "macd": list[DivergenceResult],
                    "obv": list[DivergenceResult],
                    "vix": list[DivergenceResult],
                },
                "total_bullish": int,
                "total_bearish": int,
                "prices": list[float],
                "dates": list[str],
            }
        """
        prices: list[float] = market_data.get("prices", [])
        dates: list[str] = market_data.get("dates", [])
        volumes: list[float] = market_data.get("volumes", [])
        vix: list[float] = market_data.get("vix", [])
        index_prices: list[float] = market_data.get("index_prices", prices)
        stock_code: str = market_data.get("stock_code", "unknown")

        divergences: dict[str, list[DivergenceResult]] = {
            "rsi": [],
            "macd": [],
            "obv": [],
            "vix": [],
        }

        if "rsi" in self.config.enabled_types:
            divergences["rsi"] = detect_price_rsi_divergence(
                prices, dates, self.config, self.cutoff_date,
            )

        if "macd" in self.config.enabled_types:
            divergences["macd"] = detect_price_macd_divergence(
                prices, dates, self.config, self.cutoff_date,
            )

        if "obv" in self.config.enabled_types and volumes:
            divergences["obv"] = detect_price_obv_divergence(
                prices, volumes, dates, self.config, self.cutoff_date,
            )

        if "vix" in self.config.enabled_types and vix:
            divergences["vix"] = detect_vix_index_divergence(
                index_prices, vix, dates, self.config, self.cutoff_date,
            )

        all_results = [r for results in divergences.values() for r in results]
        total_bullish = sum(1 for r in all_results if r.divergence_type == DivergenceType.BULLISH)
        total_bearish = sum(1 for r in all_results if r.divergence_type == DivergenceType.BEARISH)

        logger.info(
            "[%s] 다이버전스 분석 완료: 총 %d건 (Bullish %d, Bearish %d)",
            stock_code, len(all_results), total_bullish, total_bearish,
        )

        return {
            "divergences": divergences,
            "total_bullish": total_bullish,
            "total_bearish": total_bearish,
            "prices": prices,
            "dates": dates,
        }

    def generate_signal(self, analysis_result: dict[str, Any]) -> dict[str, Any]:
        """
        다이버전스 기반 매매 신호 생성

        Bullish 다이버전스가 우세하면 매수, Bearish가 우세하면 매도.

        Args:
            analysis_result: analyze()의 결과

        Returns:
            매매 신호 딕셔너리
        """
        total_bullish = analysis_result.get("total_bullish", 0)
        total_bearish = analysis_result.get("total_bearish", 0)
        total = total_bullish + total_bearish

        if total == 0:
            return self._build_signal(
                SignalType.HOLD, 0.0,
                "다이버전스 미감지 — 관망",
                analysis_result,
            )

        if total_bullish > total_bearish:
            strength = min(total_bullish / max(total, 1), 1.0)
            indicators = self._get_indicator_summary(analysis_result, DivergenceType.BULLISH)
            return self._build_signal(
                SignalType.BUY, strength,
                f"Bullish 다이버전스 {total_bullish}건 감지 ({indicators})",
                analysis_result,
            )

        if total_bearish > total_bullish:
            strength = min(total_bearish / max(total, 1), 1.0)
            indicators = self._get_indicator_summary(analysis_result, DivergenceType.BEARISH)
            return self._build_signal(
                SignalType.SELL, strength,
                f"Bearish 다이버전스 {total_bearish}건 감지 ({indicators})",
                analysis_result,
            )

        # 동수일 때 관망
        return self._build_signal(
            SignalType.HOLD, 0.0,
            f"Bullish/Bearish 동수 ({total_bullish}건) — 관망",
            analysis_result,
        )

    def _get_indicator_summary(
        self,
        analysis_result: dict[str, Any],
        div_type: DivergenceType,
    ) -> str:
        """해당 타입의 다이버전스 지표 요약"""
        divergences = analysis_result.get("divergences", {})
        indicators: list[str] = []
        for key, results in divergences.items():
            count = sum(1 for r in results if r.divergence_type == div_type)
            if count > 0:
                indicators.append(f"{key.upper()}:{count}")
        return ", ".join(indicators)

    def _build_signal(
        self,
        signal_type: SignalType,
        strength: float,
        reason: str,
        analysis_result: dict[str, Any],
    ) -> dict[str, Any]:
        """매매 신호 딕셔너리 구성"""
        signal = {
            "signal": signal_type.value,
            "strength": round(strength, 4),
            "reason": reason,
            "strategy_name": self.name,
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": {
                "total_bullish": analysis_result.get("total_bullish", 0),
                "total_bearish": analysis_result.get("total_bearish", 0),
                "current_price": analysis_result["prices"][-1] if analysis_result.get("prices") else 0.0,
            },
        }
        logger.info(
            "신호 생성: %s (강도=%.2f) — %s",
            signal_type.value.upper(), strength, reason,
        )
        return signal

    def backtest(
        self,
        historical_data: list[dict[str, Any]],
        initial_capital: float,
    ) -> dict[str, Any]:
        """
        과거 데이터로 다이버전스 전략 백테스팅

        Args:
            historical_data: [{"date": str, "close": float, "volume": float, ...}, ...]
            initial_capital: 초기 자본금

        Returns:
            백테스팅 결과
        """
        if len(historical_data) < self.config.macd_slow + self.config.lookback_period * 2:
            logger.warning("백테스팅 데이터 부족: %d개", len(historical_data))
            return {
                "strategy_name": self.name,
                "initial_capital": initial_capital,
                "final_capital": initial_capital,
                "total_return": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "trades": [],
                "equity_curve": [],
                "error": "데이터 부족",
            }

        prices = [d["close"] for d in historical_data]
        dates = [d.get("date", str(i)) for i, d in enumerate(historical_data)]
        volumes = [d.get("volume", 0.0) for d in historical_data]

        # 슬라이딩 윈도우로 다이버전스 감지 및 매매
        capital = initial_capital
        shares = 0
        position_price = 0.0
        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []

        window_size = self.config.macd_slow + self.config.lookback_period * 2 + 10
        for i in range(window_size, len(prices)):
            equity = capital + shares * prices[i]
            equity_curve.append({"date": dates[i], "equity": round(equity, 2)})

            window_prices = prices[max(0, i - window_size):i + 1]
            window_dates = dates[max(0, i - window_size):i + 1]
            window_volumes = volumes[max(0, i - window_size):i + 1]

            analysis = self.analyze({
                "prices": window_prices,
                "dates": window_dates,
                "volumes": window_volumes,
            })
            signal = self.generate_signal(analysis)

            if signal["signal"] == "buy" and shares == 0:
                shares = int(capital // prices[i])
                if shares > 0:
                    cost = shares * prices[i]
                    capital -= cost
                    position_price = prices[i]
                    trades.append({
                        "date": dates[i],
                        "type": "buy",
                        "price": prices[i],
                        "shares": shares,
                        "reason": signal["reason"],
                    })

            elif signal["signal"] == "sell" and shares > 0:
                revenue = shares * prices[i]
                capital += revenue
                pnl = (prices[i] - position_price) / position_price * 100
                trades.append({
                    "date": dates[i],
                    "type": "sell",
                    "price": prices[i],
                    "shares": shares,
                    "pnl_pct": round(pnl, 2),
                    "reason": signal["reason"],
                })
                shares = 0
                position_price = 0.0

        # 최종 정산
        if shares > 0:
            capital += shares * prices[-1]
            shares = 0

        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100

        sell_trades = [t for t in trades if t["type"] == "sell"]
        winning = sum(1 for t in sell_trades if t.get("pnl_pct", 0) > 0)
        losing = sum(1 for t in sell_trades if t.get("pnl_pct", 0) <= 0)
        win_rate = (winning / len(sell_trades) * 100) if sell_trades else 0.0

        # MDD
        max_dd = 0.0
        peak = initial_capital
        for point in equity_curve:
            eq = point["equity"]
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        result = {
            "strategy_name": self.name,
            "initial_capital": initial_capital,
            "final_capital": round(final_capital, 2),
            "total_return": round(total_return, 2),
            "total_trades": len(trades),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": round(win_rate, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe_ratio": 0.0,
            "trades": trades,
            "equity_curve": equity_curve,
        }

        logger.info(
            "다이버전스 백테스팅 완료: 수익률=%.2f%%, 승률=%.1f%%, MDD=%.2f%%, 거래=%d건",
            total_return, win_rate, max_dd, len(trades),
        )

        return result
