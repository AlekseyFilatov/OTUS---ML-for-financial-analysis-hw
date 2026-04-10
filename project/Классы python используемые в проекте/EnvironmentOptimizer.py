import logging
import gc
import platform
import numpy as np
import pandas as pd
import torch
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class EnvironmentOptimizer(BaseEstimator, TransformerMixin):
    """
    Трансформер для оптимизации памяти перед обучением RL-моделей.
    """

    def __init__(
        self,
        downcast_float: bool = True,
        clear_cuda: bool = True,
        verbose: bool = True,
        exclude_columns: Optional[List[str]] = None,
        gc_timeout: Optional[int] = None
    ):
        self.downcast_float = downcast_float
        self.clear_cuda = clear_cuda
        self.verbose = verbose
        self.exclude_columns = exclude_columns or []
        self.gc_timeout = gc_timeout

    def fit(self, X, y=None):
        return self

    def _safe_gc_collect(self) -> None:
        """Безопасный вызов сборщика мусора с проверкой ОС."""
        # signal.alarm работает только на UNIX (Linux/Mac) и в Main Thread
        if self.gc_timeout is not None and platform.system() != 'Windows':
            try:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"GC превысил таймаут {self.gc_timeout}с")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.gc_timeout)
                gc.collect()
                signal.alarm(0)
            except Exception as e:
                logger.warning(f"Ошибка таймера GC: {e}")
                gc.collect()
        else:
            gc.collect()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if X is None or X.empty:
            raise ValueError("Входные данные X пусты")

        # Создаем копию, чтобы не менять исходный датафрейм (стандарт sklearn)
        X_res = X.copy()

        # 1. Сжатие данных
        if self.downcast_float:
            cols_float = X_res.select_dtypes(include=['float64']).columns
            cols_to_compress = [c for c in cols_float if c not in self.exclude_columns]

            if cols_to_compress:
                X_res[cols_to_compress] = X_res[cols_to_compress].astype(np.float32)
                if self.verbose:
                    logger.info(f"📉 Сжато {len(cols_to_compress)} колонок float64")
            elif self.verbose:
                logger.info("🚫 Нет колонок для сжатия")

        # 2. Очистка системного мусора
        self._safe_gc_collect()

        # 3. Очистка видеопамяти
        if self.clear_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.verbose:
                logger.info("🧹 CUDA кеш очищен")

        if self.verbose:
            logger.info("🚀 Среда готова к обучению")

        return X_res

