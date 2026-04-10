import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from FinRLDataFormatter import FinRLDataFormatter


class FinRLDataProcessor:
    """Оркестратор подготовки: ML Pipeline -> Inverse Transform -> FinRL Formatting"""

    def __init__(self, ml_pipeline, inv_transformer, val_window_days=45):
        self.ml_pipeline = ml_pipeline
        self.inv_transformer = inv_transformer
        self.val_window_days = val_window_days

    def _align_tickers(self, df):
        """Гарантирует, что для каждой даты есть строка для каждого тикера."""
        # 1. Получаем все уникальные даты и тикеры в этом наборе
        unique_dates = df['date'].unique()
        unique_tics = df['tic'].unique()

        # 2. Создаем эталонную сетку (Cartesian Product)
        grid = pd.MultiIndex.from_product([unique_dates, unique_tics], names=['date', 'tic']).to_frame(index=False)

        # 3. Джойним реальные данные к сетке
        df_aligned = pd.merge(grid, df, on=['date', 'tic'], how='left')

        # 4. Заполняем пропуски по каждому тикеру отдельно (цены тянем вперед)
        # Сначала сортируем, чтобы ffill/bfill работали корректно по времени
        df_aligned = df_aligned.sort_values(['tic', 'date'])
        df_aligned = df_aligned.groupby('tic', group_keys=False).apply(
            lambda x: x.ffill().bfill()
        )
        return df_aligned.sort_values(['date', 'tic']).reset_index(drop=True)

    def _validate_triple_stack(self, X_raw, X_test_raw, processed_test, pipeline, fold_num):
        print(f"\n🔍 Валидация стекинга (Фолд №{fold_num}):")

        # 1. Проверка на пропуски (NaN)
        nan_counts = processed_test.isna().sum()
        if nan_counts.any():
            print(f"⚠️ ВНИМАНИЕ: Обнаружены NaN в колонках:\n{nan_counts[nan_counts > 0]}")
        else:
            print("✅ Пропусков не обнаружено.")

        # 2. Проверка новых признаков
        expert_cols = [c for c in processed_test.columns if any(word in c for word in ['prob', 'cb_', 'tsmixer'])]
        if not expert_cols:
            print("❌ ОШИБКА: Экспертные признаки не найдены.")
        else:
            print(f"✅ Добавлены признаки ({len(expert_cols)} шт.): {expert_cols[:3]}...")

        # 3. Сравнение размерностей (с учетом инверсного тикера)
        n_tickers = processed_test['tic'].nunique()
        n_dates = X_test_f['date'].nunique()
        expected_len = n_dates * n_tickers
        actual_len = len(processed_test)

        print(f"📈 Тикеров: {n_tickers} | Строк в X_test_raw: {len(X_test_raw)}")
        if actual_len != expected_len:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Размерность {actual_len} != ожидаемой {expected_len}")
        else:
            print(f"✅ Размерность корректна: {actual_len} строк.")

        # 4. Проверка уникальности пар (date, tic) — критично для FinRL
        duplicates = processed_test.duplicated(subset=['date', 'tic']).sum()
        if duplicates > 0:
            print(f"❌ ОШИБКА: Найдено {duplicates} дубликатов (date, tic)!")
        return None

    def _fix_finrl_format(self, df):
        """Полная логика исправления конфликтов имен из вашего оригинала"""
        df = df.copy()

        # Исправляем конфликт имен перед подачей в FinRL (из вашего фрагмента)
        if 'date' in df.columns and df.index.name == 'date':
            df = df.reset_index(drop=True)

        if df.index.name == 'date':
            df.index.name = None

        if 'date' in df.columns:
            df.reset_index(drop=True, inplace=True)

        # Сортировка (критически важна для FinRL)
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)

        # ЗАЩИТА ОТ NaN (после лагов или трансформаций)
        nan_count = df[['open', 'high', 'low', 'close']].isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️ ВНИМАНИЕ: Обнаружено {nan_count} NaN в OHLC. Применяю bfill/ffill.")
            # Сначала заполняем назад (первые строки), потом вперед
            df = df.groupby('tic', group_keys=False).apply(lambda x: x.bfill().ffill())

        return df

    def process_fold(self, X_train_f, y_train_f, X_test_f, fold_idx):
        print(f"\n{'=' * 30}\n>>> ПОДГОТОВКА ПОРТФЕЛЯ (Фолд №{fold_idx})\n{'=' * 30}")

        # А) ML Pipeline (SBER, MOEX, SNGSP и т.д. обучаются вместе)
        self.ml_pipeline.fit(X_train_f, y_train_f)
        processed_train = self.ml_pipeline.transform(X_train_f)
        processed_test = self.ml_pipeline.transform(X_test_f)

        # Б) Инверсные тикеры (SBER_INV, MOEX_INV, SNGSP_INV)
        # Убедитесь, что ваш inv_transformer создает их для ВСЕХ входящих тикеров
        processed_train = self.inv_transformer.transform(processed_train, X_train_f)
        processed_test = self.inv_transformer.transform(processed_test, X_test_f)

        # В) ВАЖНО: Выравнивание сетки (для FinRL)
        processed_train = self._align_tickers(processed_train)
        processed_test = self._align_tickers(processed_test)

        # Г) Валидация (исправленная логика)
        n_tickers = processed_test['tic'].nunique()
        n_dates = processed_test['date'].nunique()
        print(f"📈 Портфель: {n_tickers} тикеров | Период: {n_dates} дат")

        # Проверка на дубликаты и NaN в критических полях
        if processed_test.duplicated(subset=['date', 'tic']).any():
            print("❌ ОШИБКА: Обнаружены дубликаты (date, tic)!")
            processed_test = processed_test.drop_duplicates(subset=['date', 'tic'])

        # Д) Финальное форматирование и Split
        train_full = self._fix_finrl_format(processed_train)
        test_finrl = self._fix_finrl_format(processed_test)

        unique_train_dates = sorted(train_full['date'].unique())

        if len(unique_train_dates) > self.val_window_days:
            train_split_date = unique_train_dates[-self.val_window_days]
            train_finrl = train_full[train_full['date'] < train_split_date].copy()
            val_finrl = train_full[train_full['date'] >= train_split_date].copy()

            print(
                f"📈 Обучение RL: {train_finrl['date'].min()} >>> {train_finrl['date'].max()} ({train_finrl['date'].nunique()} дн.)")
            print(
                f"🎯 Валидация RL: {val_finrl['date'].min()} >>> {val_finrl['date'].max()} ({val_finrl['date'].nunique()} дн.)")
        else:
            train_finrl = train_full.copy()
            val_finrl = train_full.copy()
            print("⚠️ Слишком мало данных для разделения, Train и Val совпадают!")

        print(f"✅ Данные готовы: Train {train_finrl.shape}, Test {test_finrl.shape}")

        return {"train": train_finrl, "val": val_finrl, "test": test_finrl}