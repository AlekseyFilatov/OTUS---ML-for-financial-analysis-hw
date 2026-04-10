import torch
import numpy as np
import os
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from chronos import BaseChronosPipeline
from torch.utils.data import DataLoader, TensorDataset


class ChronosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="amazon/chronos-bolt-small", lookback=128,
                 thresh=0.002, ema_span=5, batch_size=64, device=None,
                 output_dir="./chronos_tuned", patience=4):
        self.model_name = model_name
        self.lookback = lookback
        self.thresh = thresh
        self.ema_span = ema_span
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

    def fit(self, X, y=None, epochs=10, lr=5e-5):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        path = self.output_dir if os.path.isdir(self.output_dir) else self.model_name

        self.pipeline = BaseChronosPipeline.from_pretrained(
            path, device_map=self.device, torch_dtype=dtype
        )

        if epochs > 0:
            self._perform_fine_tuning(X, epochs, lr)
        return self

    def _perform_fine_tuning(self, X, epochs, lr):
        """Внутренний цикл дообучения с защитой от NaN и взрыва градиентов"""
        print(f"--- Starting Fine-tuning (Device: {self.device}) ---")
        model = self.pipeline.model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # 1. Жесткая чистка данных
        # Убираем нули (чтобы log не выдал -inf) и NaN
        # close_values = X["Close"].replace(0, 1e-6).fillna(method='ffill').values
        # smoothed = pd.Series(close_values).ewm(span=self.ema_span, adjust=False).mean()
        # Переход в log-пространство
        # data = np.log(smoothed.values.astype(np.float32))

        # Гарантируем отсутствие нулей и пропусков
        # Сначала ffill, потом bfill (на случай если NaN в самом начале)
        close_clean = X["Close"].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').values

        # Добавляем маленькое смещение 1e-6, чтобы log(1e-6) был числом, а не -inf
        data = np.log(close_clean.astype(np.float32) + 1e-6)

        # Финальная проверка: если что-то осталось, зануляем
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Проверка на наличие мусора после логарифма
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print("Warning: Found NaNs or Infs in log-prices. Cleaning...")
            data = np.nan_to_num(data, nan=0.0, posinf=data[np.isfinite(data)].max(),
                                 neginf=data[np.isfinite(data)].min())

        # 2. Подготовка окон
        windows, targets = [], []
        for i in range(self.lookback, len(data)):
            windows.append(data[i - self.lookback: i])
            targets.append(data[i])

        if len(windows) == 0:
            print("Error: Not enough data for the given lookback!")
            return

        dataset = TensorDataset(torch.tensor(np.array(windows)), torch.tensor(np.array(targets)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        no_improve = 0
        target_dtype = next(model.parameters()).dtype
        # 3. Цикл обучения
        for epoch in range(epochs):
            total_loss = 0
            model.train()  # Убеждаемся, что модель в режиме обучения

            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)

                # Приводим данные к девайсу И ТИПУ модели
                batch_x = batch_x.to(device=self.device, dtype=target_dtype)
                batch_y = batch_y.to(device=self.device, dtype=target_dtype)

                outputs = model(batch_x)

                # 1. Извлекаем тензор (логика доступа)
                raw_tensor = None
                if isinstance(outputs, dict):
                    raw_tensor = outputs.get("prediction_logits") or outputs.get("forecast")
                elif hasattr(outputs, "prediction_logits"):
                    raw_tensor = outputs.prediction_logits
                elif hasattr(outputs, "forecast"):
                    raw_tensor = outputs.forecast
                elif torch.is_tensor(outputs):
                    raw_tensor = outputs

                if raw_tensor is None:
                    try:
                        raw_tensor = outputs[0]
                    except:
                        raise ValueError(f"Could not extract tensor from {type(outputs)}")

                # 2. Обработка размерностей тензора
                if len(raw_tensor.shape) == 3:
                    # [Batch, Horizon, Dim] -> берем первый шаг
                    predicted = raw_tensor[:, 0, 0]
                elif len(raw_tensor.shape) == 2:
                    predicted = raw_tensor[:, -1]
                else:
                    predicted = raw_tensor.flatten()

                # 3. Расчет Loss (в float32 для стабильности)
                loss = torch.nn.functional.huber_loss(predicted.float(), batch_y.float())

                if torch.isnan(loss):
                    continue

                    # --- ЕДИНЫЙ ЦИКЛ ОБРАТНОГО ПРОХОДА ---
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

                # Очистка ссылок для экономии памяти
                del outputs, raw_tensor, predicted, loss

            # Конец батчей в эпохе
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs}, Huber Loss: {avg_loss:.6f}")

            # Сохранение лучшей модели и Early Stopping
            if avg_loss < best_loss and not np.isnan(avg_loss):
                best_loss = avg_loss
                no_improve = 0
                self.save()
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def transform(self, X):
        if self.pipeline is None:
            raise RuntimeError("Chronos Pipeline не инициализирован!")

        self.pipeline.model.eval()
        df_res = X.copy()

        # 1. Безопасный поиск колонки Close (после Scaler-а)
        close_cols = [c for c in df_res.columns if c.endswith('Close')]
        if not close_cols:
            raise KeyError(f"Chronos не нашел Close. Колонки: {df_res.columns.tolist()}")
        close_col = close_cols[0]

        # 2. Подготовка цен (чистка от нулей и NaN)
        raw_prices = df_res[close_col].replace(0, np.nan).ffill().bfill().values.astype(np.float32)
        smoothed = pd.Series(raw_prices).ewm(span=self.ema_span, adjust=False).mean()
        log_prices = np.log(smoothed.values + 1e-6)

        n = len(df_res)
        preds = np.full(n, np.nan, dtype=np.float32)
        signals = np.zeros(n, dtype=np.int8)

        # Индексы для прогноза
        indices = list(range(self.lookback, n))

        # 3. Пакетный инференс
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i: i + self.batch_size]
            # Создаем батч тензоров
            batch_ctx = [torch.tensor(log_prices[idx - self.lookback: idx]) for idx in batch_idx]
            ctx_tensor = torch.stack(batch_ctx).to(self.device)

            with torch.no_grad():
                # Прогноз Chronos
                outputs = self.pipeline.predict(ctx_tensor, prediction_length=1)

            # Берем среднее по квантилям (mean по dim=1)
            batch_preds_log = outputs.mean(dim=1).cpu().numpy()[:, 0]

            for j, idx in enumerate(batch_idx):
                pred_p = np.exp(batch_preds_log[j])
                curr_p = raw_prices[idx - 1]
                preds[idx] = pred_p

                # Логика сигналов
                diff = (pred_p - curr_p) / (curr_p + 1e-9)
                if diff > self.thresh:
                    signals[idx] = 1
                elif diff < -self.thresh:
                    signals[idx] = -1

        # 4. Генерация признаков для CatBoost/RL
        # Используем относительную доходность (Expected Return)
        df_res["chronos_return"] = (preds / (raw_prices + 1e-9)) - 1

        # Индикатор уверенности (размах волатильности)
        rolling_std = pd.Series(raw_prices).rolling(window=20).std().values
        df_res["chronos_conf"] = (preds - raw_prices) / (rolling_std + 1e-9)
        df_res["chronos_signal"] = signals

        # Заполняем NaN (для первых lookback строк)
        return df_res.ffill().fillna(0)

    def save(self):
        if self.pipeline:
            if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
            self.pipeline.model.save_pretrained(self.output_dir)

    # --- НОВЫЙ БЛОК: ОПТИМИЗАЦИЯ ---
    def optimize(self, X_val, n_trials=20):
        """Подбор оптимальных параметров под конкретный рынок"""

        def objective(trial):
            # Предлагаем параметры
            self.ema_span = trial.suggest_int("ema_span", 2, 20)
            self.thresh = trial.suggest_float("thresh", 0.0005, 0.01)

            # Запускаем предсказание
            res = self.transform(X_val)

            # Целевая метрика: Sharpe Ratio или Точность направления
            actual_dir = np.sign(X_val["Close"].diff().shift(-1).dropna())
            pred_dir = res["chronos_signal"].iloc[:len(actual_dir)]

            # Считаем "доходность" по сигналам
            returns = actual_dir * pred_dir
            score = returns.mean() / (returns.std() + 1e-9)  # Simple Sharpe-like
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print(f"Best Params: {study.best_params}")
        self.ema_span = study.best_params["ema_span"]
        self.thresh = study.best_params["thresh"]
        return study.best_params