import os
import gc
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit


class FinRLCrossValidationOrchestrator(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            ml_pipeline,
            env_class,
            base_env_params,
            analyzer=None,
            backtester=None,
            n_splits=3,
            save_path="./",
            models_to_train=['ppo'],
            val_size=0.2
    ):
        self.ml_pipeline = ml_pipeline
        self.env_class = env_class
        self.base_env_params = base_env_params
        self.analyzer = analyzer
        self.backtester = backtester
        self.n_splits = n_splits
        self.save_path = save_path
        self.models_to_train = models_to_train
        self.val_size = val_size

        # Результаты будут храниться здесь для доступа после обучения
        self.all_fold_results_ = []
        self.final_models_ = {}

    def fit(self, X, y):
        """Запуск полного цикла обучения по фолдам."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            fold_idx = fold + 1
            print(f"\n{'=' * 60}\n🚀 ОБРАБОТКА ФОЛДА №{fold_idx}\n{'=' * 60}")

            # Разделение данных
            X_train_f, X_test_f = X.iloc[train_idx], X.iloc[test_idx]
            y_train_f, y_test_f = y.iloc[train_idx], y.iloc[test_idx]

            # --- ШАГ 1: ML ПОДГОТОВКА ---
            X_train_processed = self.ml_pipeline.fit_transform(X_train_f, y_train_f)
            X_test_processed = self.ml_pipeline.transform(X_test_f)

            # Валидация (80/20)
            split_val = int(len(X_train_processed) * (1 - self.val_size))
            df_train = X_train_processed.iloc[:split_val]
            df_val = X_train_processed.iloc[split_val:]

            # --- ШАГ 2: АНАЛИТИКА (если передана) ---
            if self.analyzer:
                self.analyzer.analyze(
                    ml_pipeline=self.ml_pipeline,
                    processed_train=df_train,
                    processed_test=X_test_processed,
                    y_test_f=y_test_f,
                    fold_idx=fold_idx
                )

            # --- ШАГ 3: RL ОРКЕСТРАЦИЯ ---
            orchestrator = FinRLOrchestrator(
                df_train=df_train,
                df_test=X_test_processed,
                env_class=self.env_class,
                base_env_kwargs={
                    **self.base_env_params,
                    "stock_dim": len(df_train['tic'].unique()),
                    "make_plots": False
                },
                save_path=self.save_path
            )

            results, models, report, env_kwargs = orchestrator.train_ensemble_parallel(
                df_train, df_val, fold=fold_idx, models_to_train=self.models_to_train
            )

            # Сохраняем модели последнего фолда или обновляем карту
            self.final_models_.update(models)

            # --- ШАГ 4: ИНФЕРЕНС И БЭКТЕСТ ---
            if models and self.backtester:
                raw_signals = orchestrator.get_ensemble_predictions(
                    X_test_processed, models_to_use=self.models_to_train
                )
                self.backtester.run_comparison(X_test_processed, models, env_kwargs, fold_idx)

                # Сохраняем мета-данные фолда
                self.all_fold_results_.append({
                    "fold": fold_idx,
                    "report": report,
                    "signals": raw_signals
                })

            gc.collect()

        return self

    def transform(self, X):
        """Инференс на новых данных с использованием лучших моделей."""
        # Прогоняем новые данные через ML Pipeline (последнее состояние после fit)
        X_processed = self.ml_pipeline.transform(X)
        return X_processed


'''
# 1. Параметры снаружи
base_env_params = {
    "hmax": 100, 
    "initial_amount": 1000000, 
    "buy_cost_pct": 0.001, 
    "sell_cost_pct": 0.001,
    "reward_scaling": 1e-4
}

# 2. Инициализация инструментов
analyzer = MLAnalyzer(target_tic='SBER_USD')
backtester = FinRLBacktester(ModifiedStockTradingEnv, "./models/")

# 3. Инициализация супер-трансформера
cv_orchestrator = FinRLCrossValidationOrchestrator(
    ml_pipeline=ml_feature_pipeline, # Ваш Pipeline с CatBoost, Scalers и т.д.
    env_class=ModifiedStockTradingEnv,
    base_env_params=base_env_params,
    analyzer=analyzer,
    backtester=backtester,
    n_splits=3,
    models_to_train=['ppo', 'sac'] # Теперь можно выбрать несколько
)

# 4. ЗАПУСК ВСЕГО ЦИКЛА ОДНОЙ КОМАНДОЙ
cv_orchestrator.fit(X, y)
'''