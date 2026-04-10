import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FinRLDataFormatter(BaseEstimator, TransformerMixin):
    """Класс для финальной 'прически' данных перед FinRL"""

    def __init__(self, val_window_days=10):
        self.val_window_days = val_window_days

    def _clean_and_sort(self, df):
        df = df.copy()
        # Исправляем конфликт имен индекса и колонки 'date'
        if df.index.name == 'date':
            df.index.name = None
        if 'date' in df.columns:
            df.reset_index(drop=True, inplace=True)

        # Сортировка критически важна для FinRL
        return df.sort_values(['date', 'tic']).reset_index(drop=True)

    def transform(self, df_processed):
        df_full = self._clean_and_sort(df_processed)
        unique_dates = sorted(df_full['date'].unique())

        if len(unique_dates) > self.val_window_days:
            train_split_date = unique_dates[-self.val_window_days]
            train_finrl = df_full[df_full['date'] < train_split_date].copy()
            val_finrl = df_full[df_full['date'] >= train_split_date].copy()

            print(
                f"📈 Обучение RL: {train_finrl['date'].min()} >>> {train_finrl['date'].max()} ({train_finrl['date'].nunique()} дн.)")
            print(
                f"🎯 Валидация RL: {val_finrl['date'].min()} >>> {val_finrl['date'].max()} ({val_finrl['date'].nunique()} дн.)")
        else:
            train_finrl = df_full.copy()
            val_finrl = df_full.copy()
            print("⚠️ Слишком мало данных для разделения, Train и Val совпадают!")

        return train_finrl, val_finrl, df_full