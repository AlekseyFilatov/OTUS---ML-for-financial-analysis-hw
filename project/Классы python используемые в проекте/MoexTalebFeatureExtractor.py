import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MoexTalebExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, stock_dim, window_size=60,
                 features_dim=128, dropout=0.2, tail_sensitivity=2.5,
                 macro_indices=None):
        # Увеличиваем выходной дим на 2 (для tail_score и confidence), чтобы Actor их видел
        super().__init__(observation_space, features_dim + (2 * stock_dim))

        full_size = observation_space.shape[0]
        self.stock_dim = stock_dim
        self.tail_sensitivity = tail_sensitivity

        # 1. Точный маппинг
        self.static_size = 1 + (2 * stock_dim)
        # Ваши макро: kurtosis, alfa_tail, brent_ret, rv_ret, rgbi_ret, hurst_z
        self.macro_indices = macro_indices if macro_indices is not None else list(range(full_size - 13, full_size))
        self.macro_size = len(self.macro_indices)
        self.dynamic_total = full_size - self.static_size - self.macro_size

        # 2. Окно (динамическая подстройка под остаток)
        if self.dynamic_total % window_size != 0:
            divs = [w for w in range(2, 21) if self.dynamic_total % w == 0]
            self.window_size = min(divs, key=lambda x: abs(x - window_size)) if divs else 1
        else:
            self.window_size = window_size
        self.features_per_step = self.dynamic_total // self.window_size

        # 3. Сети (Убрали LayerNorm на входе, чтобы не резать "хвосты")

        self.lstm = nn.LSTM(self.features_per_step, features_dim, num_layers=2,
                            batch_first=True, dropout=dropout)

        self.attention = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 4. Анализатор Хвостов (Tail Analyzer) # Теперь видит И динамику рынка (context), И ваши стат. метрики (kurtosis, alfa_tail)
        self.tail_analyzer = nn.Sequential(
            nn.Linear(features_dim + self.macro_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, stock_dim),
            nn.Sigmoid()
        )

        # 5. Risk Gate (Оценка уверенности)
        combined_dim = (features_dim * 2) + self.static_size + self.macro_size
        self.risk_gate = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.fc_final = nn.Linear(combined_dim, features_dim)
        self.final_norm = nn.LayerNorm(features_dim)
        self.diagnostics = {}

    def forward(self, observations):
        if th.isnan(observations).any():
            observations = th.nan_to_num(observations, 0.0)

        batch_size = observations.shape[0]

        # Срезы данных
        dynamic_x = observations[:, :self.dynamic_total]
        #static_x = observations[:, self.dynamic_total : self.dynamic_total + self.static_size]
        static_x = observations[:, -(self.macro_size + self.static_size) : -self.macro_size]
        #macro_x = observations[:, self.macro_indices] # Без макро-нормы! Сохраняем масштаб аномалий. # LSTM (извлекаем временные паттерны)
        macro_x = observations[:, -self.macro_size:]

        x = dynamic_x.reshape(batch_size, self.window_size, self.features_per_step)
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_w = F.softmax(self.attention(lstm_out), dim=1)
        context = th.sum(attn_w * lstm_out, dim=1)
        last_step = lstm_out[:, -1, :]

        # Логика Талеба: Анализ риска на основе скрытых паттернов + явных метрик (куртозис и т.д.)
        risk_input = th.cat([context, macro_x], dim=1)
        tail_score = self.tail_analyzer(risk_input)


        combined = th.cat([last_step, context, static_x, macro_x], dim=1)

        # Экспоненциальное затухание уверенности при росте риска (Fat Tails)
        base_conf = self.risk_gate(combined)

        confidence = base_conf * th.exp(-self.tail_sensitivity * tail_score)
        confidence = th.clamp(confidence, 0.01, 1.0) # Защита от полного обнуления

        self.diagnostics = {
            'tail_risk': tail_score.detach().mean(dim=1).mean().item(), # Средний риск по портфелю
            'tail_map': tail_score.detach().mean(dim=0).tolist(),      # Карта рисков для Riskfolio
            'confidence': confidence.detach().mean().item(),
            'kurtosis_avg': macro_x[:, 0].mean().item()
        }

        # Финальный вектор: основные фичи + ЯВНЫЕ сигналы риска для Actor-а
        main_features = self.final_norm(self.fc_final(combined))

        # Вместо умножения out * confidence, конкатенируем.  # Агент сам научится снижать объем позиции (Position Sizing), видя эти числа.
        return th.cat([main_features, tail_score, confidence], dim=1)