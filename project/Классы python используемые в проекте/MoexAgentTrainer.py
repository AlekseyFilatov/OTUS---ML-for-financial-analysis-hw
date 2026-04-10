from sklearn.base import BaseEstimator, TransformerMixin
import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C, SAC, DDPG, TD3
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
#from TalebRiskWrapper import TalebRiskWrapper

class MoexAgentTrainer(BaseEstimator, TransformerMixin):
    def __init__(self, stock_dim=4, total_timesteps=100000,
                 window_size=60, tail_sensitivity=2.5, learning_rate=1e-4):
        self.stock_dim = stock_dim
        self.total_timesteps = total_timesteps
        self.window_size = window_size
        self.tail_sensitivity = tail_sensitivity
        self.learning_rate = learning_rate

        # Атрибуты, которые заполнятся во время fit
        self.model = None
        self.env_kwargs = None
        self.tech_indicators = None
        self.macro_indicators = None

    @contextmanager
    def suppress_everything(self):
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
    def fit(self, X, y=None, tech_indicators=None, macro_indicators=None):
        X_sorted = X.copy()

        # 1. Приведение имен и удаление дубликатов (защита от AttributeError 'close')
        X_sorted.columns = [str(c).lower() for c in X_sorted.columns]
        X_sorted = X_sorted.loc[:, ~X_sorted.columns.duplicated()].copy()

        # 2. Очистка данных от аномалий (защита от OverflowError)
        # Заменяем бесконечности и ограничиваем экстремальные значения
        X_sorted = X_sorted.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        num_cols = X_sorted.select_dtypes(include=[np.number]).columns
        X_sorted[num_cols] = np.clip(X_sorted[num_cols], -1e6, 1e6)

        # 3. Синхронизация списков признаков
        if self.tech_indicators is None or self.macro_indicators is None:
             self.tech_indicators = getattr(X_sorted, 'attrs', {}).get('tech_ids', [])
             self.macro_indicators = getattr(X_sorted, 'attrs', {}).get('macro_ids', [])

        if tech_indicators is not None: self.tech_indicators = [c.lower() for c in tech_indicators]
        if macro_indicators is not None: self.macro_indicators = [c.lower() for c in macro_indicators]

        all_features_req = list(dict.fromkeys(self.tech_indicators + self.macro_indicators))
        base_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        valid_indicators = [c for c in all_features_req if c in X_sorted.columns and c not in base_cols]

        # 4. Выравнивание сетки (day) и сортировка (Date -> Tic)
        actual_tics = sorted(X_sorted['tic'].unique())
        self.stock_dim = len(actual_tics)

        unique_dates = sorted(X_sorted['date'].unique())
        date_map = {date: i for i, date in enumerate(unique_dates)}
        #X_sorted['day'] = X_sorted['date'].map(date_map).astype(int)
        X_sorted['day'] = X_sorted.groupby('date').ngroup()

        X_sorted = X_sorted.sort_values(['day', 'tic']).reset_index(drop=True)

        # 5. Подготовка параметров среды
        # Предварительный расчет State Space для инициализации
        init_state_space = 1 + 2 * self.stock_dim + (self.stock_dim * len(valid_indicators))

        self.env_kwargs = {
            "stock_dim": self.stock_dim,
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": [0.0] * self.stock_dim,
            "buy_cost_pct": [0.001] * self.stock_dim,
            "sell_cost_pct": [0.001] * self.stock_dim,
            "state_space": init_state_space,
            "tech_indicator_list": valid_indicators,
            "action_space": self.stock_dim,
            "reward_scaling": 1e-4,
            "print_verbosity": 1
        }

        # 6. Создание среды и СИНХРОНИЗАЦИЯ РАЗМЕРНОСТИ
        # Создаем экземпляр вашей новой модифицированной среды
        e_train_gym = ModifiedTalebStockTradingEnv(df=X_sorted, **self.env_kwargs)

        # Получаем реальный размер вектора через reset()
        res = e_train_gym.reset()
        obs = res[0] if isinstance(res, tuple) else res
        actual_obs_dim = len(obs)

        # Перезаписываем state_space реальным значением
        self.env_kwargs["state_space"] = actual_obs_dim
        e_train_gym.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(actual_obs_dim,))

        print(f"--- SYNC COMPLETE ---")
        print(f"Тикеров: {self.stock_dim}, Фич: {len(valid_indicators)}, State Space: {actual_obs_dim}")

        # Обертки
        e_train_gym = TalebRiskWrapper(e_train_gym, confidence_threshold=0.3)
        env_train = DummyVecEnv([lambda: e_train_gym])

        # 7. Настройка нейросети
        macro_indices = [valid_indicators.index(m) for m in self.macro_indicators if m in valid_indicators]

        policy_kwargs = dict(
            features_extractor_class=MoexTalebExtractor,
            features_extractor_kwargs=dict(
                stock_dim=self.stock_dim,
                window_size=self.window_size,
                tail_sensitivity=self.tail_sensitivity,
                macro_indices=macro_indices
            ),
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )

        # 8. Обучение
        agent = DRLAgent(env=env_train)
        print(f"[{time.strftime('%H:%M:%S')}] 🚀 Запуск обучения PPO...")

        with self.suppress_everything():
            self.model = agent.get_model(
                "ppo",
                policy_kwargs=policy_kwargs,
                tensorboard_log="./taleb_moex_logs/",
                model_kwargs={"learning_rate": self.learning_rate,
                              "n_steps": 2048, "ent_coef": 0.02, "batch_size": 128}
            )
            self.model.learn(total_timesteps=self.total_timesteps, callback=TalebRiskLoggerCallback())

        print(f"✅ [SUCCESS] Агент обучен. Шагов: {self.total_timesteps}")
        return self

    def transform(self, X):
        # 1. Достаем экстрактор из обученной модели
        extractor = self.model.policy.features_extractor

        # 2. Получаем последние рассчитанные риск-метрики
        # (Они обновляются при каждом forward-проходе модели)
        taleb_stats = getattr(extractor, 'diagnostics', {
            'tail_risk': 0.5,
            'tail_map': [0.5] * self.stock_dim, # Добавили карту рисков по тикерам
            'confidence': 1.0,
            'confidence_map': [1.0] * self.stock_dim, # Уверенность по каждому активу
            'kurtosis_avg': 0.0
        })

        # Возвращаем обученную модель для дальнейшего использования
        return {
            'model': self.model,
            'env_kwargs': self.env_kwargs,
            'df': X,
            'dates': X['date'].unique() if 'date' in X.columns else [],
            'taleb_risk_stats': taleb_stats,  # <-- ЭТОТ КЛЮЧ НУЖЕН ДЛЯ RISK_FOLIO
            'stock_dim': self.stock_dim
        }