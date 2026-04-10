import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TalebAuditor(BaseEstimator, TransformerMixin):
    def __init__(self, macro_indicators=None):
        self.macro_indicators = macro_indicators or []
        # Расширенный список для аудита "хвостов"
        self.risk_cols = ['tail_alpha', 'kurtosis', 'hurst_z', 'amihud', 'taleb_kappa', 'entropy_risk', 'fat_tail_risk']
        self.tech_indicators = []

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return df

        # 1. Определяем признаки (исключаем служебные и макро)
        service_cols = ['date', 'tic', 'close', 'open', 'high', 'low', 'volume', 'target', 'label', 'day', 'index',
                        'level_0']
        all_features = [c for c in df.columns if c not in service_cols]

        self.tech_indicators = [f for f in all_features if f not in self.macro_indicators]
        # Важно: используем только те макро, что реально есть в данных
        actual_macro = [f for f in self.macro_indicators if f in df.columns]

        print("\n" + "=" * 60)
        print(f" 🧪 ТАЛЕБ-АУДИТ (Фолд {time.strftime('%H:%M:%S')}) ")
        print("=" * 60)

        # 2. Анализ риск-метрик (Жирные хвосты)
        available_risk = [c for c in self.risk_cols if c in df.columns]
        if available_risk:
            print(f"Анализ 'хвостов' по {len(available_risk)} параметрам:")
            # describe() считаем один раз
            stats = df[available_risk].describe().loc[['mean', 'std']]

            # Расчет эксцесса только для риск-колонок (быстрее)
            kurt_values = df[available_risk].kurt()
            stats.loc['kurtosis'] = kurt_values

            # Определяем уровень риска для каждой колонки индивидуально
            risk_levels = ['FAT_TAIL' if v > 3 else 'NORMAL' for v in kurt_values]
            stats.loc['tail_risk_level'] = risk_levels

            print(stats.T)
        else:
            print("⚠️ Риск-метрики (tail_alpha/kappa) не найдены.")

        # 3. Проверка State Space для RL-агента
        stock_dim = len(df.tic.unique())
        # 1 (cash) + 2*stock_dim (shares/price) + (features * stock_dim)
        expected_state = 1 + (2 * stock_dim) + (len(all_features) * stock_dim)

        print(f"\n[INFO] Конфигурация для RL-агента:")
        print(f" - Тикеров (Stock Dim): {stock_dim}")
        print(f" - Тех. индикаторы (LSTM): {len(self.tech_indicators)}")
        print(f" - Макро-индикаторы (Gate): {len(actual_macro)}")
        print(f" - Итоговый State Space: {expected_state}")

        # 4. Финальная проверка данных
        nan_count = df.isnull().sum().sum()
        if nan_count == 0:
            print(f"✅ [SUCCESS] Данные чисты. Готов запуск 'train_taleb_agent'.")
        else:
            print(f"❌ [WARNING] Найдено {nan_count} пропусков! Модель может упасть.")

        # Сохраняем списки признаков в атрибуты датафрейма (для MoexAgentTrainer)
        df.attrs['tech_ids'] = self.tech_indicators
        df.attrs['macro_ids'] = actual_macro

        return df
