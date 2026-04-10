import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
import logging

# Настройка логирования (tracer)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InverseETFPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, fee_rate: float = 0.005,
                 prob_cols: List[str] = ['cb_prob_up', 'tsmixer_prob'],
                 ticker_name: str = "SBER_USD"):
        self.fee_rate = fee_rate
        self.prob_cols = prob_cols
        self.ticker_name = ticker_name


    def fit(self, X, y=None):
        return self

    def transform(self, df_proc: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
        if df_proc.empty or df_orig.empty:
            return pd.DataFrame()

        # 1. Позиционная синхронизация (Исправляет KeyError)
        # Берем хвост оригинала, равный длине обработанных данных (учет лагов)
        df_orig_tail = df_orig.iloc[-len(df_proc):].copy()
        df_long = df_proc.copy().reset_index(drop=True)
        df_orig_tail = df_orig_tail.reset_index(drop=True)

        # 2. Интеллектуальное извлечение даты (Поддержка MultiIndex и именованных уровней)
        if isinstance(df_orig.index, pd.MultiIndex):
            # Ищем уровень 'date' или берем первый (0)
            level = 'date' if 'date' in df_orig.index.names else 0
            original_dates = df_orig.index.get_level_values(level)[-len(df_proc):]
        else:
            original_dates = df_orig.index[-len(df_proc):]

        df_long['date'] = pd.to_datetime(original_dates)

        # 3. Перенос OHLCV (Исправлено: убран ошибочный break)
        ohlcv_map = {
            'open': ['Open', 'open', 'OPEN'],
            'high': ['High', 'high', 'HIGH'],
            'low': ['Low', 'low', 'LOW'],
            'close': ['Close', 'close', 'CLOSE'],
            'volume': ['Volume', 'volume', 'VOLUME']
        }
        for target, sources in ohlcv_map.items():
            for src in sources:
                if src in df_orig_tail.columns:
                    df_long[target] = df_orig_tail[src].values
                    break  # Переходим к следующей целевой колонке

        # 4. Формирование тикеров
        current_tic = df_orig_tail['tic'].iloc[0] if 'tic' in df_orig_tail.columns else self.ticker_name
        base_name = str(current_tic).split('_')[0]
        df_long['tic'] = str(current_tic)

        df_inv = df_long.copy()
        df_inv['tic'] = f"{base_name}_INV"

        # 5. Математическая инверсия (Корректный учёт "Жирных хвостов" и комиссий)
        returns = df_long['close'].pct_change().fillna(0)
        inv_returns = -returns
        daily_fee = self.fee_rate / 365
        start_price = df_long['close'].iloc[0]

        # Накопительный расчёт цены и комиссий
        cumulative_factor = (1 + inv_returns).cumprod()
        # Геометрический учёт ежедневной комиссии
        cumulative_fee = (1 - daily_fee) ** np.arange(len(df_long))

        df_inv['close'] = (start_price * cumulative_factor * cumulative_fee).astype('float32')
        df_inv['open'] = df_inv['close'].shift(1).fillna(df_inv['close'].iloc[0])

        # Инверсия волатильности: когда лонг на High, инверс на Low
        # Используем соотношение High/Low к Close для корректного переноса хвостов
        df_inv['high'] = (df_inv['close'] * (df_long['close'] / df_long['low'].replace(0, 1e-6))).astype('float32')
        df_inv['low'] = (df_inv['close'] * (df_long['close'] / df_long['high'].replace(0, 1e-6))).astype('float32')

        # 6. Зеркалирование вероятностей (Нормализация для Ensemble)
        for col in self.prob_cols:
            if col in df_long.columns:
                # Превращаем в число и зеркалим: 0.8 up -> 0.2 up для инверсного актива
                prob_clean = pd.to_numeric(df_long[col], errors='coerce').fillna(0.5)
                df_inv[col] = (1.0 - prob_clean).clip(0, 1)

        # 7. Сборка и финальная чистка (Защита от NaN для FinRL)
        df_final = pd.concat([df_long, df_inv], ignore_index=True)
        # Категории отключены для стабильности типов в RL-среде
        df_final['tic'] = df_final['tic'].astype(str)
        df_final = df_final.sort_values(['date', 'tic']).reset_index(drop=True)
        # Добавление шума для предотвращения корреляции
        df_final = self.inject_market_noise(df_final, noise_level=0.001)

        # Групповое заполнение пропусков (чтобы bfill не перепрыгнул между тикерами)
        num_cols = df_final.select_dtypes(include=[np.number]).columns
        df_final[num_cols] = df_final.groupby('tic')[num_cols].transform(lambda x: x.ffill().bfill())

        return self._optimize_memory(df_final)

    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизация типов данных: Float32 достаточно для финансовых расчётов."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        return df

    def inject_market_noise(
            self,
            df: pd.DataFrame,
            noise_level: float = 0.0005,
            invariant_suffix: str = 'INV',
            price_columns: list = None,
            time_lag_prob: float = 0.1,
            lag_steps: int = 1,
            random_seed: int = None
    ) -> pd.DataFrame:
        try:
            logger.info("Запуск inject_market_noise...")
            rng = np.random.default_rng(random_seed)
            df_noisy = df.copy()

            # Настройка колонок
            if price_columns is None:
                price_columns = ['open', 'high', 'low', 'close']
            available_price_cols = [col for col in price_columns if col in df_noisy.columns]

            if not available_price_cols:
                raise ValueError("В DataFrame отсутствуют указанные колонки с ценами.")

            # Фильтруем инварианты
            mask = df_noisy['tic'].str.contains(invariant_suffix, na=False)
            if not mask.any():
                logger.warning(f"Тикеры с суффиксом '{invariant_suffix}' не найдены.")
                return df_noisy

            n_invariant_rows = mask.sum()
            invariant_indices = df_noisy.index[mask]

            # 1. Добавление относительного шума (среднее = 0)
            relative_noise = rng.normal(0.0, noise_level, size=n_invariant_rows)
            for col in available_price_cols:
                original_prices = df_noisy.loc[mask, col].values
                noisy_prices = original_prices * (1 + relative_noise)
                df_noisy.loc[mask, col] = noisy_prices
            logger.info(f"Шум добавлен в {n_invariant_rows} строк.")

            # 2. Векторизованный временной лаг (с учётом изменённых цен)
            if time_lag_prob > 0:
                lag_mask = rng.random(n_invariant_rows) < time_lag_prob
                lagged_indices = invariant_indices[lag_mask]

                if len(lagged_indices) > 0:
                    all_indices = df_noisy.index
                    pos = all_indices.get_indexer(lagged_indices)
                    prev_pos = np.maximum(0, pos - lag_steps)
                    # Используем df_noisy (с шумом), а не оригинальный df
                    df_noisy.loc[lagged_indices, available_price_cols] = \
                        df_noisy.iloc[prev_pos][available_price_cols].values
                    logger.info(f"Применён временной лаг для {len(lagged_indices)} строк.")

            # 3. Логическая коррекция OHLC
            required_ohlc = ['open', 'high', 'low', 'close']
            has_ohlc = all(col in df_noisy.columns for col in required_ohlc)
            if has_ohlc:
                df_noisy = self._correct_ohlcv_prices(df_noisy, mask, required_ohlc)
                logger.info("Коррекция OHLC и проверка на положительные цены завершена.")

            return df_noisy

        except Exception as e:
            logger.error(f"Ошибка в работе inject_market_noise: {str(e)}", exc_info=True)
            return df.copy()

    def _correct_ohlcv_prices(
            self,
            df: pd.DataFrame,
            mask: pd.Series,
            ohlcv_columns: list
    ) -> pd.DataFrame:
        """Гарантирует положительность цен и логическую связь High >= Open/Close >= Low."""
        # 1. Чистим цены от отрицательных значений и нулей
        df.loc[mask, ohlcv_columns] = df.loc[mask, ohlcv_columns].clip(lower=1e-8)

        # 2. Временный срез для расчётов
        subset = df.loc[mask, ohlcv_columns]

        # 3. Корректируем High (максимум среди всех 4-х цен)
        df.loc[mask, 'high'] = subset.max(axis=1)

        # 4. Корректируем Low (минимум среди всех 4-х цен)
        df.loc[mask, 'low'] = subset.min(axis=1)

        return df