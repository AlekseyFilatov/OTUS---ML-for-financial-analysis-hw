import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics.classification import BinaryAUROC
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction

class TSMixerBinaryClassifier(nn.Module):
    """
    Бинарный классификатор на основе PatchTSMixer.

    Архитектура:
    1. Backbone: стандартный PatchTSMixerForPrediction из HuggingFace
    2. Feature extraction: используем prediction_outputs как признаки
    3. Classification head: полносвязные слои с dropout

    Преимущества:
    - Используем проверенную архитектуру из transformers
    - Backbone уже оптимизирован для работы с временными рядами
    - Простота и прозрачность кода
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Backbone: стандартная модель PatchTSMixer для prediction
        # Она обучается предсказывать следующий шаг, и мы используем её выходы как признаки
        self.backbone = PatchTSMixerForPrediction(config)

        # Classification head
        # Входные признаки: (batch, prediction_length=1, num_channels)
        # После flatten: (batch, num_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (batch, 1, num_channels) -> (batch, num_channels)
            nn.Linear(config.num_input_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # 2 класса: падение (0) и рост (1)
        )

    def forward(self, past_values):
        """
        Forward pass.

        Параметры:
        - past_values: (batch, context_length, num_channels)

        Возвращает:
        - logits: (batch, 2) - логиты для двух классов
        """
        # Получаем prediction outputs из backbone
        outputs = self.backbone(past_values=past_values)

        # prediction_outputs имеет размер (batch, prediction_length=1, num_channels)
        # Эти выходы содержат закодированную информацию о временном ряде
        features = outputs.prediction_outputs

        # Пропускаем через classification head
        logits = self.classifier(features)

        return logits

class PatchTSMixerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lookback=128, epochs=50, batch_size=64, lr=3e-4,
                 weight_decay=1e-2, output_dir="tsmixer_best.pt", verbose=True):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def fit(self, X, y):
        # 1. Подготовка весов классов
        y_int = y.astype(int)
        class_counts = np.bincount(y_int)
        weights = len(y) / (2 * class_counts) if len(class_counts) > 1 else [1.0, 1.0]
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # 2. Split для внутренней валидации (последние 20% - OOS)
        split_idx = int(len(X) * 0.8)
        X_train_raw, X_val_raw = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_raw, y_val_raw = y[:split_idx], y[split_idx:]

        # Создаем последовательности
        train_X, train_y = self._create_sequences(X_train_raw, y_train_raw)
        val_X, val_y = self._create_sequences(X_val_raw, y_val_raw)

        # ДИНАМИЧЕСКОЕ ОПРЕДЕЛЕНИЕ КАНАЛОВ
        # train_X имеет форму [Batch, Lookback, Channels]
        # Берем именно из тензора, так как там уже только ЧИСЛА
        actual_num_channels = train_X.shape[2]

        if self.verbose:
            print(f"📊 PatchTSMixer: входных каналов (числовых): {actual_num_channels}")

        train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=self.batch_size)

        # 3. Инициализация модели с правильным num_input_channels
        config = PatchTSMixerConfig(
            context_length=self.lookback,
            prediction_length=1,
            num_input_channels=actual_num_channels,  # ИСПОЛЬЗУЕМ РЕАЛЬНОЕ ЧИСЛО
            patch_length=16,  # Рекомендуемое значение для стабильности
            patch_stride=8,
            d_model=96,
            num_layers=6,
            dropout=0.1
        )

        self.model = TSMixerBinaryClassifier(config).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        # 4. Обучение с сохранением лучшего Checkpoint
        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            # Валидация в конце каждой эпохи
            val_acc, _, _ = self._evaluate(val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, self.output_dir)

        # 5. Финальная загрузка лучших весов и Бэктест
        self._load_best_and_report(val_loader, X_val_raw)
        return self

    def _load_best_and_report(self, loader, X_val_raw):
        """Загрузка лучших весов и безопасная печать отчета"""
        if not os.path.exists(self.output_dir):
            return  # Если модель не обучилась из-за нехватки данных

        checkpoint = torch.load(self.output_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        acc, y_pred, y_true = self._evaluate(loader)

        # ПРОВЕРКА: Если данных нет или класс только один, пропускаем отчет
        unique_classes = np.unique(y_true)
        if len(y_true) == 0 or len(unique_classes) < 2:
            if self.verbose:
                print("! Недостаточно данных или классов в валидационном сете для отчета.")
            return

        if self.verbose:
            print(f"\nФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (Internal Validation): Acc: {acc:.4f}")
            # Безопасный вызов отчета
            print(classification_report(y_true, y_pred,
                                        target_names=['Down', 'Up'],
                                        labels=[0, 1],  # Явно указываем ожидаемые метки
                                        zero_division=0))

            # Расчет Sharpe (только если есть данные)
            returns = X_val_raw['Close'].pct_change().iloc[self.lookback:].values
            if len(returns) > 1 and len(y_pred) > 1:
                # Синхронизируем длины
                min_len = min(len(y_pred), len(returns))
                strat_returns = np.where(y_pred[:min_len - 1] == 1, returns[1:min_len], -returns[1:min_len])
                sharpe = (strat_returns.mean() / (strat_returns.std() + 1e-8)) * np.sqrt(252)
                print(f"Sharpe Ratio (Internal Val): {sharpe:.3f}")

    def _evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.model(xb.to(self.device))
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(yb.numpy())
        return (np.array(all_preds) == np.array(all_labels)).mean(), np.array(all_preds), np.array(all_labels)

    def _create_sequences(self, X, y=None):
        # 1. Оставляем ТОЛЬКО числа (игнорируем 'tic', 'date' и прочие строки)
        if isinstance(X, pd.DataFrame):
            # Выбираем только числовые колонки
            X_numeric = X.select_dtypes(include=[np.number])
            # Сохраняем количество каналов для автоматической настройки модели
            self.num_input_channels = X_numeric.shape[1]
            X_values = X_numeric.values.astype(np.float32)
        else:
            # Если уже массив, принудительно переводим в float32
            X_values = np.asanyarray(X, dtype=np.float32)

        X_win, y_win = [], []

        # 2. Формируем окна (lookback)
        for i in range(self.lookback, len(X_values)):
            # Берем срез окна по числовым значениям
            X_win.append(X_values[i - self.lookback: i, :])

            # Обработка таргета
            if y is not None:
                # Берем значение y по индексу (y обычно уже числовой)
                y_val = y.iloc[i] if hasattr(y, 'iloc') else y[i]
                y_win.append(y_val)

        # 3. Преобразование в тензоры
        # np.array(X_win) теперь будет иметь тип float32, а не object_
        x_tensor = torch.tensor(np.array(X_win), dtype=torch.float32)

        if y is not None:
            y_tensor = torch.tensor(np.array(y_win), dtype=torch.long)
            return x_tensor, y_tensor

        return x_tensor

    def transform(self, X):
        """Добавляет колонку вероятностей в общий Pipeline"""
        if self.model is None:
            raise RuntimeError("Модель PatchTSMixer не обучена!")

        self.model.eval()

        # 1. ОТСЕКАЕМ ТЕКСТ (tic, date), ОСТАВЛЯЕМ ТОЛЬКО ЧИСЛА
        # Это лечит ошибку numpy.object_
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=[np.number])
            X_values = X_numeric.values.astype(np.float32)
        else:
            X_values = np.asanyarray(X, dtype=np.float32)

        n_samples = len(X_values)
        num_features = X_values.shape[1]

        # 2. СОЗДАЕМ ОКНА (Инференс)
        windows = []
        for i in range(n_samples):
            if i < self.lookback:
                # Паддинг для начала ряда (чтобы длина выхода совпала с входом)
                pad_len = self.lookback - (i + 1)
                padding = np.zeros((pad_len, num_features), dtype=np.float32)
                seq = np.vstack([padding, X_values[:i + 1]])
            else:
                # Стандартное окно [t-lookback : t]
                seq = X_values[i - self.lookback: i]
            windows.append(seq)

        # 3. ПРЕВРАЩАЕМ В ТЕНЗОР (теперь здесь чистый float32)
        X_tensor = torch.tensor(np.array(windows), dtype=torch.float32)

        probs = []
        # 4. ПАКЕТНЫЙ ИНФЕРЕНС (Batch Inference)
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i: i + self.batch_size].to(self.device)
                logits = self.model(batch)
                # Берем вероятность класса Up (индекс 1)
                p_up = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(p_up)

        # 5. ВОЗВРАЩАЕМ DATAFRAME
        X_res = X.copy()
        X_res['tsmixer_prob'] = np.array(probs, dtype=np.float32)

        return X_res


