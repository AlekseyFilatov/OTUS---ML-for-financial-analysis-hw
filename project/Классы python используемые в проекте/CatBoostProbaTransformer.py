import joblib
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import os
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier


class CatBoostProbaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None, model_path=None, iterations=1200, depth=6,
                 fillna_strategy='zero', adaptive_sigma=1.5):
        self.estimator = estimator
        self.model_path = model_path
        self.iterations = iterations
        self.depth = depth
        self.fillna_strategy = fillna_strategy
        self.adaptive_sigma = adaptive_sigma  # Коэффициент для порога разметки
        self.model_ = None
        self.feature_names_ = None
        self.cat_features_indices_ = []

    def _add_volume_features(self, X):
        """Расчет относительного объема и динамики (глаза модели)"""
        if isinstance(X, pd.DataFrame) and 'volume' in X.columns and 'tic' in X.columns:
            X = X.copy()
            # Относительный объем (Relative Volume) к 20-дневной средней
            X['rel_vol'] = X.groupby('tic')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + 1e-8)
            )
            # Изменение объема (Momentum объема)
            X['vol_delta'] = X.groupby('tic')['volume'].pct_change()

            if self.fillna_strategy == 'zero':
                X = X.fillna(0)
            elif self.fillna_strategy == 'mean':
                X = X.fillna(X.mean(numeric_only=True))
            return X
        return X

    def _create_adaptive_target(self, X):
        """Автоматическая разметка 3-х классов на основе волатильности (Талеб-style)"""
        if not isinstance(X, pd.DataFrame):
            return None

        # Считаем будущую доходность на 5 дней вперед
        future_ret = X.groupby('tic')['close'].shift(-5) / X['close'] - 1.0

        # Считаем историческую волатильность (стандартное отклонение за 20 дней)
        volatility = X.groupby('tic')['close'].pct_change().rolling(window=20).std()

        # Адаптивный порог: 1.5 сигмы (отсекаем шум)
        threshold = volatility.mean() * self.adaptive_sigma

        conditions = [
            (future_ret > threshold),  # Класс 2: Рост выше нормы
            (future_ret < -threshold)  # Класс 0: Падение выше нормы
        ]
        # 2 - UP, 0 - DOWN, 1 - NEUTRAL
        return np.select(conditions, [2, 0], default=1)

    def fit(self, X, y=None):
        try:
            X_enriched = self._add_volume_features(X)

            X_numeric = X_enriched.select_dtypes(include=[np.number])
            # 1. Сначала находим ВСЕ числовые колонки
            numeric_cols = X_numeric.columns.tolist()

            # 2. Исключаем из них те, что точно не должны быть фичами (таргеты, лейблы)
            blacklist = ['tic', 'date', 'target', 'label', 'open', 'high', 'low', 'close']
            self.feature_names_ = [c for c in numeric_cols if c not in blacklist]

            # Печатаем для проверки (вы увидите это в логах)
            print(f"--- CatBoost fit: {len(self.feature_names_)} features selected ---")
            if 'tic' in self.feature_names_:
                print("!!! ВНИМАНИЕ: 'tic' все еще в списке фич!")


            # Готовим чистую матрицу для всех типов обучения
            X_values = X_numeric.values.astype(np.float32)

            if y is None:
                y = self._create_adaptive_target(X_enriched)
                if y is None:
                    raise ValueError("Не удалось создать таргет.")

            # 2. Логика обучения (без смешивания категорий и чисел)
            if self.model_path and os.path.exists(self.model_path):
                self.model_ = joblib.load(self.model_path)
                print(f"✅ CatBoost загружен: {self.model_path}")

            elif self.estimator is not None:
                self.model_ = self.estimator
                # Учим на чистых числах. Большинство внешних Estimator (Calibrated)
                # упадут, если им передать текст.
                self.model_.fit(X_values, y)
                print("🚀 Внешний Estimator обучен на числовых данных")

            else:
                # Для стандартного CatBoost убираем cat_features,
                # так как мы подаем матрицу float32
                self.model_ = CatBoostClassifier(
                    iterations=self.iterations,
                    depth=self.depth,
                    loss_function='MultiClass',
                    eval_metric='Accuracy',
                    auto_class_weights='Balanced',
                    random_seed=42,
                    verbose=False
                )

                valid_mask = pd.Series(y).notna()
                self.model_.fit(X_values[valid_mask], np.array(y)[valid_mask])
                print(f"🚀 CatBoost MultiClass обучен (Фич: {len(self.feature_names_)})")

            return self
        except Exception as e:
            print(f"❌ Критическая ошибка в CatBoostProbaTransformer.fit: {e}")
            return self

    def transform(self, X):
        try:
            if self.model_ is None:
                raise RuntimeError("Трансформер не обучен!")

            X_enriched = self._add_volume_features(X)

            # Выбираем строго те колонки, на которых учились
            try:
                X_input = X_enriched[self.feature_names_].values.astype(np.float32)
            except Exception as e:
                # Логируем конкретную колонку, которая мешает конвертации
                for col in self.feature_names_:
                    try:
                        X_enriched[col].values.astype(np.float32)
                    except:
                        print(f"❌ Критическая ошибка в колонке: {col} (Тип: {X_enriched[col].dtype})")
                raise e

            # Предсказание вероятностей
            probs = self.model_.predict_proba(X_input)

            X_res = X_enriched.copy()
            num_classes = probs.shape[1]

            # Записываем вероятности
            X_res['cb_prob_down'] = probs[:, 0].astype(np.float32)
            # Если классов 3 [Down, Neutral, Up], Up — это индекс 2. Если 2 — индекс 1.
            X_res['cb_prob_up'] = probs[:, 2 if num_classes >= 3 else 1].astype(np.float32)

            return X_res
        except Exception as e:
            print(f"❌ Критическая ошибка в CatBoostProbaTransformer.transform: {e}")
            return X

    def save(self, path):
        """Сохранение весов для продакшена"""
        if self.model_:
            joblib.dump(self.model_, path)
            print(f"📦 Модель сохранена в {path}")
