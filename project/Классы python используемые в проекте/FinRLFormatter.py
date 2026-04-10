import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew
import os, logging, warnings
import traceback

logger = logging.getLogger(__name__)

class FinRLFormatter(BaseEstimator, TransformerMixin):
    def __init__(self, adf_test=True, log_fix=True, macro_indicators=None):
        self.adf_test = adf_test
        self.log_fix = log_fix
        self.macro_indicators = macro_indicators if macro_indicators is not None else []
        self.tech_indicators = []
        self.macro_ids = []
        self.removed_features = {}
        self.corr_threshold = 0.9
        self.blacklist = ['date', 'tic', 'target', 'label', 'open', 'high', 'low', 'close', 'day']

    def _apply_log_fix(self, df):
        """Метод 1: Исправление асимметрии (Skewness) по Талебу"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in self.blacklist: continue
            try:
                # Считаем асимметрию только для чистых данных
                clean_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean_data) > 30 and abs(skew(clean_data)) > 2.0:
                    # Симметричный логарифм: сохраняет знак, сжимает масштаб
                    df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
            except:
                continue
        return df

    def _apply_adf_test(self, df):
        """Метод 2: Проверка на стационарность и дифференцирование"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in self.blacklist: continue
            try:
                clean_data = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
                if len(clean_data) > 30:
                    p_val = adfuller(clean_data)[1]
                    if p_val > 0.05:
                        # Если не стационарен — берем первую разность (Delta)
                        df[col] = df[col].diff().fillna(0)
            except:
                continue
        return df

    def _remove_multicollinearity(self, df, features):
        """Метод 3: Удаление дублирующих признаков (Correlation Filter)"""
        if not features: return features

        corr_matrix = df[features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]

        if to_drop:
            print(f"[-] Удалено по корреляции > {self.corr_threshold}: {len(to_drop)} признаков")
            features = [f for f in features if f not in to_drop]
        return features

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
            print("1. ПЕРВИЧНАЯ ОЧИСТКА")
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
            print("2. LOG-FIX")
            if self.log_fix:
                for col in ['volume', 'amihud']:
                    if col in df.columns:
                        # Принудительно в float и отсекаем отрицательные
                        vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        df[col] = np.log1p(vals.clip(lower=0))

            if self.log_fix:
                df = self._apply_log_fix(df)

            # 3. ADF TEST (Исправленная передача аргумента)
            if self.adf_test:
                print("3. ADF TEST")
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

            if self.adf_test:
                df = self._apply_adf_test(df)

            # 4. ФИНАЛЬНАЯ ПОДГОТОВКА СПИСКОВ (Для MoexAgentTrainer)
            #blacklist = ['date', 'tic', 'target', 'label', 'open', 'high', 'low', 'close', 'day']
            all_features = [c for c in df.columns if c not in self.blacklist]
            valid_features = self._remove_multicollinearity(df, all_features)

            self.tech_indicators = [f for f in valid_features if f not in self.macro_indicators]
            self.macro_ids = [f for f in valid_features if f in self.macro_indicators]

            # Сохраняем метаданные в атрибуты для Pipeline
            df.attrs['tech_ids'] = self.tech_indicators
            df.attrs['macro_ids'] = self.macro_ids

            print(f"✅ Formatter: Tech={len(self.tech_indicators)}, Macro={len(self.macro_ids)}")
            return df

        except Exception as e:
            print(f"❌ Критическая ошибка в transform: {e}")
            logger.debug(f"Трассировка ошибки:\n{traceback.format_exc()}")
            # Возвращаем исходный DF, чтобы не ломать весь TickerParallel
            return X