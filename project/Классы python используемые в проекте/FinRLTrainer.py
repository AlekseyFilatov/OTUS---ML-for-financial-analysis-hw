import os
import logging
import warnings
import torch as th
import absl.logging
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from TradingMetricsCallback import TradingMetricsCallback


class FinRLTrainer:
    def __init__(self, env_class, base_env_kwargs, save_path="./models/", total_timesteps=50000):
        self.env_class = env_class
        self.base_env_kwargs = base_env_kwargs
        self.save_path = save_path
        self.total_timesteps = total_timesteps
        self.log_path = "./tensorboard_log/"
        # НОВОЕ: Хранилище для динамических параметров
        self.current_env_kwargs = None

    @contextmanager
    def suppress_everything(self):
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")
        absl.logging.set_verbosity(absl.logging.ERROR)
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            yield

    def _build_env(self, df):
        os.makedirs(self.log_path, exist_ok=True)
        exclude_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'day']
        indicators = [c for c in df.columns if c not in exclude_cols]
        stock_dimension = df['tic'].nunique()
        state_space = 1 + (5 + len(indicators)) * stock_dimension

        # Формируем и сохраняем параметры в self.current_env_kwargs
        env_kwargs = self.base_env_kwargs.copy()
        env_kwargs.update({
            "stock_dim": stock_dimension,
            "state_space": state_space,
            "tech_indicator_list": indicators,
            "action_space": stock_dimension,
            "num_stock_shares": [0] * stock_dimension,
            # Извлекаем первый элемент из списка комиссий, если пришел список, иначе берем число
            "buy_cost_pct": [self.get_fee_value('buy_cost_pct')] * stock_dimension,
            "sell_cost_pct": [self.get_fee_value('sell_cost_pct')] * stock_dimension
        })

        # СОХРАНЯЕМ ДЛЯ БЭКТЕСТЕРА
        self.current_env_kwargs = env_kwargs

        print(f"🛠 Параметры среды: Dim={stock_dimension}, State Space={state_space}")
        print(f"📊 Используемые индикаторы: {len(indicators)} шт.")

        gym_obj = self.env_class(df=df, **env_kwargs)
        env_sb, _ = gym_obj.get_sb_env()
        return env_sb

    def get_fee_value(self, key):
            val = self.base_env_kwargs.get(key, 0.001)
            if isinstance(val, list):
                return val[0] # Если список, берем первый элемент
            return val        # Если число, берем как есть

    def train(self, train_finrl, val_finrl, fold):
        print(f"🤖 FinRL: Обучение PPO для фолда {fold}...")
        env_train = self._build_env(train_finrl)
        env_val = self._build_env(val_finrl)

        eval_callback = EvalCallback(env_val, best_model_save_path=self.save_path,
                                     deterministic=True, verbose=1)
        metrics_callback = TradingMetricsCallback()
        callback_list = CallbackList([eval_callback, metrics_callback])

        safe_n_steps = 512 if len(train_finrl) > 1024 else 128
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
        model_params = {"n_steps": safe_n_steps, "batch_size": 128, "learning_rate": 7e-5,
                        "ent_coef": 0.1, "clip_range": 0.2, "gae_lambda": 0.95, "vf_coef": 0.5, "gamma": 0.99}

        with self.suppress_everything():
            agent = DRLAgent(env=env_train)
            model_ppo = agent.get_model("ppo", policy_kwargs=policy_kwargs, model_kwargs=model_params, verbose=1)
            trained_ppo = model_ppo.learn(total_timesteps=self.total_timesteps, callback=callback_list,
                                          tb_log_name=f'ppo_sber_f{fold}', reset_num_timesteps=True)

        print(f"✅ Обучение PPO завершено для фолда {fold}")
        return trained_ppo
