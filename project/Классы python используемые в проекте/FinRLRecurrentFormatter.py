import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew
from sklearn.pipeline import Pipeline
import time
import logging


logger = logging.getLogger(__name__)

class FinRLFormatter(BaseEstimator, TransformerMixin):
    def __init__(self, adf_test=True, log_fix=True, macro_indicators=None):
        self.adf_test = adf_test
        self.log_fix = log_fix
        self.macro_indicators = macro_indicators if macro_indicators is not None else []
        self.tech_indicators = []
        self.macro_ids = []
        self.removed_features = {}

    def fit(self, X, y=None):
        if X is None or (isinstance(X, pd.DataFrame) and X.empty):
            raise ValueError("X пустой в FinRLFormatter.fit")
        return self

    def transform(self, X):
        try:
            # 1. ПРИВЕДЕНИЕ ТИПА (Уже добавлено вами)
            if isinstance(X, pd.Series):
                df = X.to_frame()
            else:
                df = pd.DataFrame(X).copy()

            # 1. ПЕРВИЧНАЯ ОЧИСТКА
            # Сбрасываем индекс, если дата или тикер там, и удаляем мусор
            if df.index.names[0] is not None:
                df = df.reset_index()

            cols_to_drop = ['level_0', 'index', 'Unnamed: 0']
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

            # Чистим суффиксы после merge
            cols_suffixes = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
            if cols_suffixes:
                df = df.drop(columns=cols_suffixes)

            # 2. LOG-FIX (С защитой от RuntimeWarning)
            if self.log_fix:
                for col in ['volume', 'amihud']:
                    if col in df.columns:
                        # Принудительно в float и отсекаем отрицательные
                        vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        df[col] = np.log1p(vals.clip(lower=0))

            # 3. ADF TEST (Исправленная передача аргумента)
            if self.adf_test:
                # Исключаем служебные колонки
                ignored = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'day', 'label', 'target']
                numeric_cols = df.select_dtypes(include=[np.number]).columns

                for col in numeric_cols:
                    if col in ignored: continue
                    try:
                        # КЛЮЧЕВОЙ ФИКС: Очищаем данные и приводим к 1D-массиву numpy
                        clean_data = df[col].replace([np.inf, -np.inf], np.nan).dropna().values

                        # Проверка, что данных достаточно и это массив
                        if len(clean_data) > 20:
                            # Извлекаем p-value: adfuller возвращает кортеж, p-value — это второй элемент [1]
                            p_val = adfuller(clean_data)[1]

                            if p_val > 0.05:
                                df[col] = df[col].diff().fillna(0)
                    except Exception as e:
                        # Если один признак "сломан", идем дальше
                        continue

            # 4. ФИНАЛЬНАЯ ПОДГОТОВКА СПИСКОВ (Для MoexAgentTrainer)
            blacklist = ['date', 'tic', 'target', 'label', 'open', 'high', 'low', 'close', 'day']
            all_features = [c for c in df.columns if c not in blacklist]

            self.tech_indicators = [f for f in all_features if f not in self.macro_indicators]
            self.macro_ids = [f for f in all_features if f in self.macro_indicators]

            # Сохраняем метаданные в атрибуты для Pipeline
            df.attrs['tech_ids'] = self.tech_indicators
            df.attrs['macro_ids'] = self.macro_ids

            return df

        except Exception as e:
            print(f"❌ Критическая ошибка в transform: {e}")
            logger.debug(f"Трассировка ошибки:\n{traceback.format_exc()}")
            # Возвращаем исходный DF, чтобы не ломать весь TickerParallel
            return X


class FinRLRecurrentFormatter(BaseEstimator, TransformerMixin):
    def __init__(self, original_df=None, adf_test=True, log_fix=True):
        self.original_df = original_df
        self.adf_test = adf_test  # Проверка на стационарность
        self.log_fix = log_fix  # Исправление асимметрии (Log-scaling)
        self.feature_list = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.original_df is None:
            raise ValueError("original_df не установлен!")

        final_df, feature_list = self._prepare_and_check(X, self.original_df)
        self.feature_list = feature_list
        return final_df

    def _prepare_and_check(self, df_transformed, original_df):
        # 1. Подготовка оригинальных OHLCV
        df_ohlcv = original_df.copy()

        # Если date в индексе — выносим в колонку
        if 'date' not in df_ohlcv.columns and 'Date' not in df_ohlcv.columns:
            df_ohlcv = df_ohlcv.reset_index()

        # Приводим всё к нижнему регистру
        df_ohlcv.columns = [col.lower() for col in df_ohlcv.columns]

        # Проверка: если после всех манипуляций даты нет
        if 'date' not in df_ohlcv.columns:
            raise KeyError(f"Колонка 'date' не найдена! Доступные колонки: {list(df_ohlcv.columns)}")

        # 2. Подготовка признаков (df_transformed)
        df_features = df_transformed.copy()
        if 'tic' not in df_features.columns:
            # Если потеряли тикер, пытаемся восстановить (обычно он там есть)
            pass

        # Создаем временный индекс для точной склейки
        df_ohlcv['temp_idx'] = df_ohlcv.groupby('tic').cumcount()
        df_features['temp_idx'] = df_features.groupby('tic').cumcount()

        # Выбираем только нужные колонки из оригинала, чтобы не плодить дубликаты
        ohlcv_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'temp_idx']
        ohlcv_cols = [c for c in ohlcv_cols if c in df_ohlcv.columns]

        # Удаляем 'date' из признаков перед мерджем, если она там есть
        if 'date' in df_features.columns:
            df_features = df_features.drop(columns=['date'])

        # 3. Склейка
        final_df = pd.merge(
            df_ohlcv[ohlcv_cols],
            df_features,
            on=['tic', 'temp_idx'],
            how='inner'
        )

        # Если мердж прошел странно и дата стала date_x
        if 'date' not in final_df.columns and 'date_x' in final_df.columns:
            final_df = final_df.rename(columns={'date_x': 'date'})

        # 4. Финальная сортировка и очистка
        final_df = final_df.drop(columns=['temp_idx']).sort_values(['date', 'tic']).reset_index(drop=True)

        # 3. АНАЛИЗ И ТРАНСФОРМАЦИЯ ПРИЗНАКОВ
        ignored = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']
        all_features = [col for col in final_df.columns if col not in ignored]
        valid_features = []

        print(f"\n{'=' * 40}\nГЛУБОКИЙ АНАЛИЗ ПРИЗНАКОВ ДЛЯ RECURRENT RL\n{'=' * 40}")

        for col in all_features:
            series = final_df[col]

            # А. Проверка на "живучесть"
            if series.nunique() <= 1:
                print(f"[-] {col:.<20} УДАЛЕН (константа)")
                continue

            # Б. ИСПРАВЛЕНИЕ АСИММЕТРИИ (Log-scaling)
            # Если коэффициент асимметрии > 2.0, применяем логарифм
            if self.log_fix:
                current_skew = skew(series.dropna())
                if abs(current_skew) > 2.0:
                    # Используем sign(x) * log(|x|+1) для работы с отрицательными числами
                    final_df[col] = np.sign(series) * np.log1p(np.abs(series))
                    new_skew = skew(final_df[col].dropna())
                    print(f"[*] {col:.<20} LOG-FIX (Skew: {current_skew:.1f} -> {new_skew:.1f})")

            # В. ADF ТЕСТ (Стационарность)
            if self.adf_test:
                try:
                    # Берем срез данных для скорости
                    p_val = adfuller(final_df[col].iloc[:500].values)[1]
                    if p_val > 0.05:
                        print(f"[!] {col:.<20} НЕСТАЦИОНАРЕН (p={p_val:.3f}) - Опасно для RNN")
                except:
                    pass

            valid_features.append(col)

        # 4. Удаление мультиколлинеарности (дубликатов)
        corr_matrix = final_df[valid_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]

        if to_drop:
            print(f"[-] Удалено (Correlation > 0.98): {to_drop}")
            valid_features = [f for f in valid_features if f not in to_drop]

        print(f"{'=' * 40}")
        print(f"ИТОГО: {len(valid_features)} признаков готовы к подаче в LSTM.")

        return final_df, valid_features