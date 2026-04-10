import torch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from chronos import BaseChronosPipeline, Chronos2Pipeline
from sklearn.metrics import roc_auc_score


class Chronos2MultivariateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="amazon/chronos-bolt-tiny", lookback=128,
                 thresh=0.0015, device=None, output_dir="./chronos2_model"):
        self.model_name = model_name
        self.lookback = lookback
        self.thresh = thresh
        self.output_dir = output_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

    def fit(self, X, y=None):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        # Загружаем корректную версию v2-bolt
        self.pipeline =  BaseChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=dtype
        )
        return self

    def transform(self, X):
        """Прогноз с использованием всех колонок X как признаков (Covariates)"""
        df_res = X.copy()

        # Преобразуем весь DataFrame в 3D тензор: [1, seq_len, num_features]
        # Chronos-2 ожидает, что Close — это первый канал (index 0)
        data_values = df_res.values.astype(np.float32)

        preds = np.full(len(df_res), np.nan, dtype=np.float32)
        signals = np.zeros(len(df_res), dtype=np.int8)

        self.pipeline.model.eval()

        # Для ускорения в Colab: делаем предсказание не каждый шаг, а батчами
        # или только для последних данных. Но для обучения FinRL нужен полный ряд.
        for t in range(self.lookback, len(df_res)):
            # Берем окно всех признаков (Close + RSI + MACD и т.д.)
            # Shape: [1, lookback, num_features]
            ctx = torch.tensor(data_values[t - self.lookback: t + 1], device=self.device).unsqueeze(0)

            if self.device == "cuda":
                ctx = ctx.to(torch.bfloat16)

            with torch.no_grad():
                # Chronos-2 анализирует зависимости между всеми каналами ctx
                fc = self.pipeline.predict(ctx, prediction_length=1)

            # Извлекаем медиану (квантиль 0.5) для целевого ряда (Close)
            yhat = self._get_median_forecast(fc)
            preds[t] = yhat

            # Генерация сигнала на основе порога THRESH
            today_close = data_values[t, 0]  # Close всегда первая колонка после Strategy4
            if yhat > today_close * (1.0 + self.thresh):
                signals[t] = 1
            elif yhat < today_close * (1.0 - self.thresh):
                signals[t] = -1

        df_res["chronos2_pred"] = preds
        df_res["chronos2_signal"] = signals

        # Логируем точность эксперта
        self._log_performance(df_res)

        return df_res.ffill().fillna(0)

    def _get_median_forecast(self, forecast_tensor):
        # В Chronos-2 структура квантилей может быть в атрибуте .quantiles
        q_list = getattr(self.pipeline, "quantiles", [0.1, 0.5, 0.9])
        qi = q_list.index(0.5) if 0.5 in q_list else len(q_list) // 2
        return float(forecast_tensor[0, qi, 0].detach().cpu().float().numpy())

    def _log_performance(self, df):
        valid = df.dropna(subset=["chronos2_pred"]).iloc[:-1]
        if len(valid) > 50:
            actual = (valid["Close"].shift(-1) > valid["Close"]).astype(int)[:-1]
            pred_change = (valid["chronos2_pred"] - valid["Close"])[:-1]
            auc = roc_auc_score(actual, pred_change)
            print(f"| Chronos-2 Multivariate AUC: {auc:.4f} |")
