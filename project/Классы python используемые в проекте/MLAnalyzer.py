import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class MLAnalyzer:
    def __init__(self, target_tic='SBER_USD'):
        self.target_tic = target_tic

    def analyze(self, ml_pipeline, processed_train, processed_test, y_test_f, fold_idx):
        print(f"\n🧪 Анализ важности драйверов и AUC (Фолд {fold_idx})")

        # 0. Извлекаем внутренний пайплайн для конкретного тикера
        actual_pipe = ml_pipeline
        if hasattr(ml_pipeline, 'models') and self.target_tic in ml_pipeline.models:
            actual_pipe = ml_pipeline.models[self.target_tic]
            print(f"✅ Извлечена модель для: {self.target_tic}")

        try:
            # 1. Находим ваш трансформер
            if 'catboost_expert' not in actual_pipe.named_steps:
                print("❌ Ошибка: Шаг 'catboost_expert' не найден.")
                return

            cat_transformer = actual_pipe.named_steps['catboost_expert']

            # 2. Достаем реальную модель ИЗ трансформера
            # В вашем коде это self.model_
            inner_model = getattr(cat_transformer, 'model_', None)

            # Достаем имена признаков ИЗ трансформера
            actual_features = getattr(cat_transformer, 'feature_names_', [])

            importances = []

            if inner_model is not None:
                # 3. Извлечение важности (с учетом возможной калибровки внутри трансформера)
                if hasattr(inner_model, 'calibrated_classifiers_'):
                    all_imps = []
                    for clf in inner_model.calibrated_classifiers_:
                        sub = getattr(clf, 'estimator', getattr(clf, 'base_estimator', None))
                        if sub and hasattr(sub, 'get_feature_importance'):
                            all_imps.append(sub.get_feature_importance())
                    if all_imps:
                        importances = np.mean(all_imps, axis=0)

                elif hasattr(inner_model, 'get_feature_importance'):
                    importances = inner_model.get_feature_importance()
                elif hasattr(inner_model, 'feature_importances_'):
                    importances = inner_model.feature_importances_

            # 4. Вывод ТОП-5
            if len(importances) > 0 and len(actual_features) > 0:
                # Синхронизируем длины (иногда CatBoost добавляет служебные фичи)
                min_len = min(len(actual_features), len(importances))
                importance_df = pd.Series(importances[:min_len], index=actual_features[:min_len])

                print(f"📊 Модель использует {len(importances)} признаков.")
                print("\n🔝 ТОП-5 ДРАЙВЕРОВ (CatBoost):")
                print(importance_df.sort_values(ascending=False).head(5))
            else:
                print("⚠️ Важность признаков не найдена внутри CatBoostProbaTransformer.model_")

        except Exception as e:
            print(f"⚠️ Ошибка анализа важности: {e}")

        # 5. Итоговый AUC фолда (с исправлением длины 1024/512)
        try:
            # Выравниваем окно данных
            test_subset = processed_test.iloc[-len(y_test_f):].copy()
            mask = (test_subset['tic'] == self.target_tic).values

            y_true = y_test_f.iloc[mask].values if hasattr(y_test_f, 'iloc') else y_test_f[mask]

            if 'cb_prob_up' in test_subset.columns:
                y_prob = test_subset.iloc[mask]['cb_prob_up'].values
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_prob)
                    print(f"\n🏁 ИТОГ ФОЛДА {fold_idx}: ML AUC-ROC ({self.target_tic}) = {auc:.4f}")
                else:
                    print(f"⚠️ AUC пропущен: один класс в выборке.")
        except Exception as e:
            print(f"⚠️ Ошибка расчёта AUC: {e}")

        return None