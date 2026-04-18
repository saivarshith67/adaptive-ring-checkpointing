import math
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConvergenceFit:
    theta1: float
    theta2: float


class ConvergenceAwareScheduler:
    """COCI-inspired online scheduler for convergence-aware checkpoint timing.

    The scheduler fits loss(t) = exp(theta1 * t + theta2) over a rolling segment and
    computes an adaptive checkpoint interval from the closed-form progress segment size.
    """

    def __init__(
        self,
        failure_rate_lambda: float,
        checkpoint_cost_seconds: float,
        fit_interval_steps: int = 100,
        min_interval_seconds: float = 5.0,
        max_interval_seconds: float = 1800.0,
        theta1_change_threshold: float = 1e-4,
        theta2_change_threshold: float = 0.1,
    ):
        if failure_rate_lambda <= 0:
            raise ValueError("failure_rate_lambda must be > 0")
        if checkpoint_cost_seconds <= 0:
            raise ValueError("checkpoint_cost_seconds must be > 0")

        self.failure_rate_lambda = failure_rate_lambda
        self.checkpoint_cost_seconds = checkpoint_cost_seconds
        self.fit_interval_steps = max(1, int(fit_interval_steps))
        self.min_interval_seconds = max(0.1, float(min_interval_seconds))
        self.max_interval_seconds = max(
            self.min_interval_seconds, float(max_interval_seconds)
        )
        self.theta1_change_threshold = theta1_change_threshold
        self.theta2_change_threshold = theta2_change_threshold

        self._segment_start_time: Optional[float] = None
        self._last_checkpoint_time: float = time.time()
        self._current_interval_seconds: float = self.min_interval_seconds

        self._times: List[float] = []
        self._losses: List[float] = []
        self._steps_seen: int = 0
        self._last_fit: Optional[ConvergenceFit] = None

    @property
    def current_interval_seconds(self) -> float:
        return self._current_interval_seconds

    def mark_checkpoint(self, now: Optional[float] = None) -> None:
        self._last_checkpoint_time = now if now is not None else time.time()

    def should_checkpoint(self, now: Optional[float] = None) -> bool:
        check_time = now if now is not None else time.time()
        return (check_time - self._last_checkpoint_time) >= self._current_interval_seconds

    def observe(self, loss_value: float, now: Optional[float] = None) -> None:
        obs_time = now if now is not None else time.time()

        if self._segment_start_time is None:
            self._segment_start_time = obs_time

        elapsed = max(0.0, obs_time - self._segment_start_time)
        clamped_loss = max(float(loss_value), 1e-12)

        self._times.append(elapsed)
        self._losses.append(clamped_loss)
        self._steps_seen += 1

        if self._steps_seen % self.fit_interval_steps != 0:
            return

        fit = self._fit_exponential()
        if fit is None:
            return

        if self._last_fit is not None:
            theta1_drift = abs(fit.theta1 - self._last_fit.theta1)
            theta2_drift = abs(fit.theta2 - self._last_fit.theta2)
            if (
                theta1_drift > self.theta1_change_threshold
                or theta2_drift > self.theta2_change_threshold
            ):
                # Piecewise online fitting: restart the fitting segment.
                self._segment_start_time = obs_time
                self._times = [0.0]
                self._losses = [clamped_loss]

        self._last_fit = fit
        self._current_interval_seconds = self._compute_interval_seconds(
            fit=fit,
            current_elapsed=elapsed,
            current_loss=clamped_loss,
        )

    def _fit_exponential(self) -> Optional[ConvergenceFit]:
        if len(self._times) < 2:
            return None

        xs = self._times
        ys = [math.log(v) for v in self._losses]

        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)

        var_x = sum((x - x_mean) ** 2 for x in xs)
        if var_x <= 1e-12:
            return None

        cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        theta1 = cov_xy / var_x
        theta2 = y_mean - theta1 * x_mean

        # We need a decaying curve to apply convergence-aware intervals.
        if theta1 >= -1e-9:
            return None

        return ConvergenceFit(theta1=theta1, theta2=theta2)

    def _compute_interval_seconds(
        self,
        fit: ConvergenceFit,
        current_elapsed: float,
        current_loss: float,
    ) -> float:
        theta1 = fit.theta1
        lam = self.failure_rate_lambda
        ts = self.checkpoint_cost_seconds

        try:
            pe = (1.0 - math.exp(theta1 * ts)) * ((1.0 / (2.0 * lam)) - (1.0 / (2.0 * theta1)))
        except OverflowError:
            pe = 0.0

        if not math.isfinite(pe) or pe <= 0.0:
            return self._current_interval_seconds

        target_loss = current_loss - pe
        if target_loss <= 1e-12:
            return self.max_interval_seconds

        # loss(t) = exp(theta1*t + theta2), solve dt from current loss to target loss.
        try:
            target_elapsed = (math.log(target_loss) - fit.theta2) / theta1
        except (ValueError, ZeroDivisionError):
            return self._current_interval_seconds

        interval = target_elapsed - current_elapsed
        if not math.isfinite(interval) or interval <= 0.0:
            return self.min_interval_seconds

        return max(self.min_interval_seconds, min(self.max_interval_seconds, interval))
