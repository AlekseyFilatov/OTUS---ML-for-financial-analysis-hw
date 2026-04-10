import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re

class DataFrameReconstructor(BaseEstimator, TransformerMixin):
    """
    Элитный восстановитель структуры для мультитикерных данных:
    1. Лечит 'tic' и 'date' после ColumnTransformer (убирает префиксы).
    2. Восстанавливает стандартные имена OHLCV и типы данных.
    3. Выполняет безопасную очистку NaN внутри каждого тикера (защита от утечек).
    """

    def __init__(self, date_col_names=None, default_tic=None, strict_tic_check=True, group_by='tic', verbose=False):
        self.date_col_names = date_col_names or ['date', 'datetime', 'timestamp']
        self.default_tic = default_tic
        self.group_by = group_by
        self.strict_tic_check = strict_tic_check
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 1. Гарантируем DataFrame
        if isinstance(X, np.ndarray):
            X_res = pd.DataFrame(X)
            print(f"DEBUG: Колонки после Scaler: {X.columns.tolist()}")
        elif not isinstance(X, pd.DataFrame):
            raise ValueError(f"Ожидался DataFrame, получено {type(X)}")
        else:
            X_res = X.copy()

        # 2. Умный поиск и переименование колонок (Regex-подобный поиск)
        new_cols = {}
        for col in X_res.columns:
            col_str = str(col).lower()
            if 'tic' in col_str:
                new_cols[col] = 'tic'
            elif any(d_name in col_str for d_name in self.date_col_names):
                new_cols[col] = 'date'
            elif 'close' in col_str: new_cols[col] = 'Close'
            elif 'open' in col_str: new_cols[col] = 'Open'
            elif 'high' in col_str: new_cols[col] = 'High'
            elif 'low' in col_str: new_cols[col] = 'Low'
            elif 'volume' in col_str: new_cols[col] = 'Volume'

        X_res = X_res.rename(columns=new_cols)

        # Обработка дубликатов (если passthrough вернул оригинал + трансформированную колонку)
        if X_res.columns.duplicated().any():
            if self.verbose:
                print(f"⚠️ Удаление дубликатов колонок: {X_res.columns[X_res.columns.duplicated()].tolist()}")
            X_res = X_res.loc[:, ~X_res.columns.duplicated()]

        # 3. Восстановление 'date' из индекса
        if 'date' not in X_res.columns:
            if isinstance(X_res.index, pd.DatetimeIndex):
                X_res['date'] = X_res.index
            elif X_res.index.name and any(name in str(X_res.index.name).lower() for name in self.date_col_names):
                X_res = X_res.reset_index().rename(columns={X_res.index.name: 'date'})

        # Приводим дату к формату string (YYYY-MM-DD) для FinRL
        if 'date' in X_res.columns:
            X_res['date'] = pd.to_datetime(X_res['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        if 'tic' not in X_res.columns:
            # А) Поиск по маске (учитываем префиксы sklearn)
            potential_tic_cols = [c for c in X_res.columns if 'tic' in str(c).lower()]

            if potential_tic_cols:
                # Берем первый найденный (обычно 'remainder__tic')
                X_res = X_res.rename(columns={potential_tic_cols[0]: 'tic'})

            # Б) Если в колонках нет, проверяем индекс (если мы его туда "спрятали")
            elif self.group_by in X_res.index.names:
                X_res = X_res.reset_index(level=self.group_by)
                if 'tic' not in X_res.columns: # Если имя индекса было другим
                     X_res = X_res.rename(columns={self.group_by: 'tic'})

            elif self.default_tic:
                X_res['tic'] = self.default_tic
            elif self.strict_tic_check:
                raise KeyError(f"Колонка 'tic' не найдена! Колонки: {X_res.columns.tolist()}")

        # 5. Приведение типов и очистка числовых данных
        numeric_targets = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_targets:
            if col in X_res.columns:
                X_res[col] = pd.to_numeric(X_res[col], errors='coerce')
                if col == 'Volume':
                    # Сначала во float, чтобы не упасть на NaN, потом в int
                    X_res[col] = X_res[col].fillna(0).astype(np.float64).astype(np.int64)

        # 6. Групповая очистка (Ключевое для мультитикерности)
        # Заполняем NaN только внутри окон одного тикера, чтобы данные Сбера не попали в Газпром
        num_cols = X_res.select_dtypes(include=[np.number]).columns
        X_res[num_cols] = X_res.groupby('tic')[num_cols].transform(lambda x: x.ffill().bfill().fillna(0))

        return X_res