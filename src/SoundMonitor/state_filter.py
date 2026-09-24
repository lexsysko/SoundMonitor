import logging
from collections import deque
from dataclasses import dataclass
import numpy as np

from SoundMonitor.enums import AnalyzeMethods
from SoundMonitor.settings import (
    FILTER_WINDOW_SIZE,
    FILTER_THRESH_STREAK_PERC,
    FILTER_THRESH_ON_DENSITY,
    FILTER_THRESH_OFF_DENSITY,
    FILTER_THRESH_WEIGHTS_ON_DENSITY,
    FILTER_THRESH_WEIGHTS_OFF_DENSITY,
    FILTER_THRESH_MIN_SCORE,
)

logger = logging.getLogger(__name__)


@dataclass
class StateFilterThresh:
    """
    :param on_density: Частка '1' для перемикання OFF -> ON (наприклад, 0.8)
    :param off_density: Частка '1', нижче якої перемикається ON > OFF (наприклад, 0.3)
    :param streak: Кількість послідовних кадрів для CONSECUTIVE_TAIL
    :param weights_on_density: Зважений поріг для увімкнення ON
    :param weights_off_density: Зважений поріг для вимкнення OFF
    :param min_score: Мінімальна схожість k-NN для прийняття кадру в історію
    """

    on_density: float = FILTER_THRESH_ON_DENSITY
    off_density: float = FILTER_THRESH_OFF_DENSITY
    streak: int = 0
    weights_on_density: float = FILTER_THRESH_WEIGHTS_ON_DENSITY
    weights_off_density: float = FILTER_THRESH_WEIGHTS_OFF_DENSITY
    min_score: float = FILTER_THRESH_MIN_SCORE


class StateFilter:
    def __init__(
        self,
        window_size: int = FILTER_WINDOW_SIZE,
        method: str = "stability_state",
        thresh: StateFilterThresh | None = None,
    ):
        self.history = deque(maxlen=window_size)
        self.scores_history = deque(maxlen=window_size)
        self.weights_history = np.arange(1, window_size + 1)
        self.weights_history_sum = self.weights_history.sum()
        self.thresh: StateFilterThresh = thresh or StateFilterThresh()
        if not self.thresh.streak:
            self.thresh.streak = int((window_size / 100) * int(FILTER_THRESH_STREAK_PERC))
        self.current_state = 0  # 0 = OFF, 1 = ON
        self.method: AnalyzeMethods = AnalyzeMethods(method.lower())
        logger.debug(self.method)

    def is_history_ready(self) -> bool:
        return len(self.history) == self.history.maxlen

    def get_weighted_density(self) -> float:
        """Зважене заповнення: нові кадри мають більшу вагу."""
        n = len(self.history)
        if n == 0:
            return 0.0

        if n < self.history.maxlen:
            weights = np.arange(1, n + 1)
            return float(np.dot(self.history, weights) / weights.sum())

        return float(np.dot(self.history, self.weights_history) / self.weights_history_sum)

    def get_scored_weighted_density(self) -> float:
        """
        Рахує зважене заповнення, ураховуючи як свіжість кадру (часовий вектор),
        так і впевненість k-NN (scores_history).
        """
        n = len(self.history)
        if n == 0:
            return 0.0

        # 1. Створюємо комбінований масив: raw_state * score для кожного кадру
        # Якщо raw_state=1, значення буде score; якщо raw_state=0, значення буде 0.0
        effective_values = np.array(self.history) * np.array(self.scores_history)

        # 2. Формуємо часові ваги (новіші кадри мають більший коефіцієнт)
        time_weights = np.arange(1, n + 1)

        # 3. Скалярний добуток, нормований на суму часових ваг
        weighted_score = float(np.dot(effective_values, time_weights) / time_weights.sum())

        return weighted_score

    def get_consecutive_tail(self) -> tuple[int, int]:
        """Повертає останнє значення та довжину безперервної серії."""
        if not self.history:
            return 0, 0

        last_val = self.history[-1]
        streak = 0

        for val in reversed(self.history):
            if val == last_val:
                streak += 1
            else:
                break

        return last_val, streak

    def get_stability_state(self) -> tuple[float, float]:
        """Повертає щільність одиниць та метрику стабільності."""
        if not self.history:
            return 0.0, 1.0

        density_ones = sum(self.history) / len(self.history)
        stability = abs(density_ones - 0.5) * 2.0
        return density_ones, stability

    def update_weighted_density_state(self, weighted_density):
        if self.current_state == 0:
            if weighted_density >= self.thresh.weights_on_density:
                self.current_state = 1
        else:
            if weighted_density <= self.thresh.weights_off_density:
                self.current_state = 0

    def update(self, raw_state: int, score: float) -> int:
        """
        Оновлює історію та повертає відфільтрований стан.
        """
        # if score < self.thresh.min_score:
        #     # logger.debug(f"Ignored low-confidence frame: {raw_state=} with {score=:.2f} < {self.thresh.min_score}")
        #     # return self.current_state
        #     raw_state = int(not raw_state)

        self.history.append(raw_state)

        match self.method:
            case AnalyzeMethods.WEIGHTED_DENSITY:
                weighted_density = self.get_weighted_density()
                self.update_weighted_density_state(weighted_density)
                logger.debug(
                    f"{self.method.name} : {score=:.2f}, state={self.current_state}, {weighted_density=:.2f}."
                    f" ON/OFF: [ {self.thresh.weights_on_density:.2f} | {self.thresh.weights_off_density:.2f} ]"
                )
                return self.current_state

            case AnalyzeMethods.SCORED_WEIGHTED_DENSITY:
                self.scores_history.append(score)

                weighted_density = self.get_scored_weighted_density()
                self.update_weighted_density_state(weighted_density)
                logger.debug(
                    f"{self.method.name} : {score=:.2f}, state={self.current_state}, {weighted_density=:.2f}."
                    f" ON/OFF: [ {self.thresh.weights_on_density:.2f} | {self.thresh.weights_off_density:.2f} ]"
                )
                return self.current_state

            case AnalyzeMethods.CONSECUTIVE_TAIL:
                last_val, streak = self.get_consecutive_tail()

                if self.current_state == 0 and last_val == 1 and streak >= self.thresh.streak:
                    self.current_state = 1
                elif self.current_state == 1 and last_val == 0 and streak >= self.thresh.streak:
                    self.current_state = 0
                logger.debug(
                    f"{self.method.name} : {score=:.2f}, state={self.current_state}, {last_val=}, {streak=}. "
                    f"ON/OFF: [ {self.thresh.streak} ]"
                )
                return self.current_state

            case AnalyzeMethods.STABILITY_STATE:
                density_ones, stability = self.get_stability_state()

                if self.current_state == 0:
                    if density_ones >= self.thresh.on_density:
                        self.current_state = 1
                else:
                    if density_ones <= self.thresh.off_density:
                        self.current_state = 0

                logger.debug(
                    f"{self.method.name} : {score=:.2f}, state={self.current_state}, {density_ones=:.2f}, {stability=:.2f}"
                    f" ON/OFF: [ {self.thresh.on_density:.2f} | {self.thresh.off_density:.2f} ]"
                )
                return self.current_state

        raise NotImplementedError(f"Method {self.method} is not implemented")
