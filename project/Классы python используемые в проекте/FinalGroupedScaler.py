from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np

class GroupedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_scale, group_by='tic', fillna_strategy='zero', verbose=False):
        self.features_to_scale = features_to_scale
        self.group_by = group_by
        self.fillna_strategy = fillna_strategy
        self.verbose = verbose
        self.scalers = {}
        self.global_scaler = RobustScaler()
        # Для отладки: статистика по скалерам
        self.scaler_stats_ = {}
        self.mean_ = None
        self.scale_ = None

    def _validate_features(self, X):
        """Проверка наличия всех требуемых фич в датафрейме"""
        missing = [f for f in self.features_to_scale if f not in X.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные фичи: {missing}")
        return True

    def _safe_fit_scaler(self, scaler, data):
        """Безопасное обучение скалера"""
        if data.empty:
            return scaler

        # RobustScaler устойчив к выбросам, но константы делают scale_ = 0
        scaler.fit(data)

        # Исправляем нулевой масштаб, чтобы избежать деления на 0 в будущем
        if hasattr(scaler, 'scale_'):
            scaler.scale_ = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        return scaler

    def fit(self, X, y=None):
        try:
            # Работаем с копией, чтобы не портить исходный X
            X_df = X.reset_index() if self.group_by in X.index.names else X.copy()

            self._validate_features(X_df)

            # 1. Глобальный скалер
            self.global_scaler = self._safe_fit_scaler(
                RobustScaler(),  # Создаем новый экземпляр
                X_df[self.features_to_scale]
            )

            # Сохраняем статистику (center_ вместо mean_)
            self.scaler_stats_['global'] = {
                'center': self.global_scaler.center_,
                'scale': self.global_scaler.scale_
            }

            # 2. Персональные скалеры
            for tic, group in X_df.groupby(self.group_by):
                scaler = self._safe_fit_scaler(RobustScaler(), group[self.features_to_scale])
                self.scalers[tic] = scaler

                self.scaler_stats_[tic] = {
                    'center': scaler.center_,
                    'scale': scaler.scale_,
                    'count': len(group)
                }
            return self
        except Exception as e:
            # В проде лучше рейзить ошибку, иначе Pipeline "проглотит" пустой скалер
            raise RuntimeError(f"Ошибка в GroupedScaler.fit: {e}")

    def transform(self, X):
        try:
            # Сохраняем оригинальный индекс для восстановления
            original_index = X.index
            X_res = X.copy()

            # Приводим к колонкам, если тикер в индексе
            if self.group_by in X_res.index.names:
                X_res = X_res.reset_index()

            if not self.scalers and not hasattr(self.global_scaler, 'scale_'):
                raise RuntimeError("Трансформер не обучен!")

            self._validate_features(X_res)

            # Группируем по тикерам
            for tic, group in X_res.groupby(self.group_by, sort=False):
                # Берем данные только нужных колонок
                group_data = group[self.features_to_scale]

                # Выбираем скалер (персональный или глобальный)
                scaler = self.scalers.get(tic, self.global_scaler)

                if tic not in self.scalers and self.verbose:
                    print(f"⚠️ Тикер {tic} не найден. Используется глобальный скалер.")

                # Трансформируем и записываем обратно по индексам группы
                transformed = scaler.transform(group_data)
                X_res.loc[group.index, self.features_to_scale] = transformed

            # Заполнение пропусков (NaN)
            if self.fillna_strategy == 'zero':
                X_res[self.features_to_scale] = X_res[self.features_to_scale].fillna(0)
            elif self.fillna_strategy == 'mean':
                X_res[self.features_to_scale] = X_res[self.features_to_scale].fillna(
                    X_res[self.features_to_scale].mean())

            # Восстанавливаем индекс
            X_res.index = X_res[self.group_by].index  # Гарантируем соответствие
            # Если мы делали reset_index, возвращаем как было
            if self.group_by not in X.columns:
                X_res = X_res.set_index(original_index.names)
            else:
                X_res.index = original_index

            return X_res

        except Exception as e:
            # В Pipeline лучше пробрасывать ошибку дальше
            raise RuntimeError(f"Ошибка в GroupedScaler.transform: {e}")
