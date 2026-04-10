import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class CatBoostFilter(BaseEstimator, TransformerMixin):
    def __init__(self, iterations=1000, depth=6, learning_rate=0.03, horizon=5, fillna_strategy='ffill'):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.fillna_strategy = fillna_strategy
        self.model = None
        self.feature_names_ = None

    def _prepare_features(self, df):
        if 'tic' in df.index.names:
            df = df.reset_index()
        elif df.index.name == 'tic':
            df = df.reset_index()
        df = df.copy()

        # --- ФИКС РЕГИСТРА ---
        # Находим реальные имена колонок в датафрейме, как бы они ни назывались
        cols = {c.lower(): c for c in df.columns}
        c_close = cols.get('close', 'Close')
        c_volume = cols.get('volume', 'Volume')
        c_high = cols.get('high', 'High')
        c_low = cols.get('low', 'Low')

        # 1. Сила объёма (Relative Volume)
        df['rel_vol'] = df.groupby('tic')[c_volume].transform(
            lambda x: x / (x.rolling(window=20).mean() + 1e-8)
        )
        # 2. Относительная волатильность
        price_range = (df[c_high] - df[c_low]) / (df[c_close] + 1e-8)
        volatility_std = df.groupby('tic')[c_close].transform(lambda x: x.pct_change().rolling(20).std())
        df['volatility_ratio'] = price_range / (volatility_std + 1e-8)

        # 3. Безопасное заполнение
        if self.fillna_strategy == 'ffill':
            df = df.ffill().fillna(0)
        else:
            df = df.fillna(0)
        return df

    def fit(self, X, y=None):
        # 1. Приводим названия к нижнему регистру для поиска
        cols_lower = [c.lower() for c in X.columns]

        # 'tic' теперь необязателен для валидации, если мы работаем с одним инструментом
        required_base = ['volume', 'high', 'low', 'close']
        missing = [f for f in required_base if f not in cols_lower]

        if missing:
            raise ValueError(f"CatBoostFilter: пропущены {missing}. Доступны: {list(X.columns)}")

        # 2. Если 'tic' нет в данных, создаем временную заглушку 'dummy_tic'
        # Это нужно, чтобы groupby('tic') не падал ниже по коду
        X_mapped = X.copy()
        if 'tic' in X_mapped.index.names or X_mapped.index.name == 'tic':
            X_mapped = X_mapped.reset_index()
        if 'tic' not in X_mapped.columns and 'TIC' not in X_mapped.columns:
            X_mapped['tic'] = 'single_ticker'
            print("⚠️ CatBoostFilter: колонка 'tic' не найдена, использую 'single_ticker'")
        elif 'TIC' in X_mapped.columns:
            X_mapped['tic'] = X_mapped['TIC']

        # 3. Обогащаем признаки (теперь groupby('tic') сработает всегда)
        df_enriched = self._prepare_features(X_mapped)

        # 4. Жесткий отбор признаков (только числа)
        exclude = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'target', 'future_ret']
        exclude += [c.upper() for c in exclude] + [c.capitalize() for c in exclude]

        numeric_df = df_enriched.select_dtypes(include=[np.number])
        self.feature_names_ = [c for c in numeric_df.columns if c not in exclude]
        self.feature_names_ = [c for c in numeric_df.columns if c not in exclude]

        # Генерация ТАРГЕТА (используем найденное имя Close)
        c_close = next((c for c in df_enriched.columns if c.lower() == 'close'), 'Close')
        future_ret = df_enriched.groupby('tic')[c_close].shift(-self.horizon) / df_enriched[c_close] - 1.0
        vol = df_enriched.groupby('tic')[c_close].transform(lambda x: x.pct_change().rolling(20).std())
        threshold = (vol * 1.5).bfill()

        conditions = [
            (future_ret > threshold),
            (future_ret < -threshold)
        ]
        target = np.select(conditions, [2, 0], default=1)
        train_mask = (future_ret.notna()) & (df_enriched[c_close] > 0)

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function='MultiClass',
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0
        )

        self.model.fit(
            df_enriched.loc[train_mask, self.feature_names_],
            target[train_mask]
        )
        return self

    def transform(self, X):
        if self.model is None: raise RuntimeError("Model not fitted!")
        df_enriched = self._prepare_features(X)

        # Гарантируем наличие всех колонок (заполняем нулями если Strategy4 что-то не выдала)
        for col in self.feature_names_:
            if col not in df_enriched.columns:
                df_enriched[col] = 0.0

        probs = self.model.predict_proba(df_enriched[self.feature_names_])
        res = df_enriched.copy()

        res['cb_prob_down'] = probs[:, 0].astype(np.float32)
        res['cb_prob_up'] = probs[:, 2].astype(np.float32)
        res['cb_confidence'] = (res['cb_prob_up'] - res['cb_prob_down']).astype(np.float32)

        # Перегрев
        conf_roll = res.groupby('tic')['cb_confidence']
        res['cb_overheat'] = (res['cb_confidence'] - conf_roll.transform(lambda x: x.rolling(50).mean())) / \
                             (conf_roll.transform(lambda x: x.rolling(50).std()) + 1e-9)

        res['cb_signal_clean'] = res['cb_confidence'].copy()
        res.loc[res['cb_overheat'].abs() > 2.5, 'cb_signal_clean'] *= 0.5

        return res