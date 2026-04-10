import os, logging, warnings, pandas as pd, numpy as np, torch as th
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import ray
from ray import tune
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from ray.tune import CLIReporter

import multiprocessing as mp
import os
import logging
import warnings
import absl.logging
from tqdm import tqdm
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# SB3 и FinRL компоненты
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from ray.train import CheckpointConfig

import os
import logging
import warnings
import absl.logging
from tqdm import tqdm
from contextlib import contextmanager, redirect_stdout, redirect_stderr



def _ensemble_mp_worker(args):
    """Автономный воркер для параллельного обучения."""
    (cfg, train_df, val_df, fold, env_class, env_kwargs, save_path,
     TradingMetricsCallbackClass, DRLAgentClass) = args

    name = cfg["name"]
    params = cfg["params"].copy()
    timesteps = cfg["timesteps"]

    # Пути: создаем уникальную подпапку для каждой модели
    model_specific_path = os.path.join(save_path, f"{name}_f{fold}")
    os.makedirs(model_specific_path, exist_ok=True)
    final_path = os.path.join(model_specific_path, f"{name}_model.zip")

    try:
        # Создание сред
        e_train_gym = env_class(df=train_df, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        e_val_gym = env_class(df=val_df, **env_kwargs)
        env_val, _ = e_val_gym.get_sb_env()

        # Callbacks
        eval_cb = EvalCallback(
            eval_env=env_val,
            best_model_save_path=model_specific_path,
            eval_freq=2000,
            deterministic=True,
            verbose=0
        )
        metrics_cb = TradingMetricsCallbackClass()
        cb_list = CallbackList([eval_cb, metrics_cb])

        # Обучение
        agent = DRLAgentClass(env=env_train)
        model = agent.get_model(name, model_kwargs=params)
        model.learn(total_timesteps=timesteps, callback=cb_list)

        # Сохранение финальной версии
        model.save(final_path)

        return {
            "status": "success",
            "name": name,
            "path": final_path,
            "best_reward": float(getattr(eval_cb, 'best_mean_reward', 0.0)),
            "metrics_data": getattr(metrics_cb, 'metrics', {}),
            "fold": fold
        }
    except Exception as e:
        return {"status": "failed", "name": name, "error": str(e)}


class FinRLProductionOrchestratorMultiprocessing:
    def __init__(self, df_train, df_test, env_class, base_env_kwargs,
                 save_path="./models/", total_timesteps=50000, model_configs=None, log_path="./tensorboard_log/"):

        self.env_class = env_class
        self.base_env_kwargs = base_env_kwargs.copy()
        self.save_path = save_path
        self.total_timesteps = total_timesteps
        self.log_path = log_path
        self.train_data = df_train
        self.val_data = df_test

        self.val_data = None
        self.current_env_kwargs = None

        # 1. КОНФИГУРАЦИЯ АНСАМБЛЯ (С включением всех параметров PPO из вашего фрагмента)
        self.model_configs = model_configs or [
            {
                "name": "ppo",
                "timesteps": total_timesteps,
                "params": {
                    "batch_size": 128,
                    "learning_rate": 7e-5,
                    "ent_coef": 0.1,
                    "gamma": 0.99,
                    "clip_range": 0.2,  # Консервативные обновления
                    "gae_lambda": 0.95,  # Сглаживание вознаграждения
                    "vf_coef": 0.5  # Точность оценки выгоды
                }
            },
            {"name": "sac", "timesteps": 30000, "params": {"batch_size": 128, "buffer_size": 100000, "learning_rate": 0.0001, "learning_starts": 100, "ent_coef": "auto_0.1"}},
            {"name": "a2c", "timesteps": 30000, "params": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0001}},
            {"name": "ddpg", "timesteps": 20000, "params": {"batch_size": 128,          # Увеличиваем (стандарт 64), чтобы градиент был стабильнее
                                                            "buffer_size": 50000,       # Размер памяти (соответствует вашему конфигу)
                                                            "learning_rate": 0.00005,   # Снижаем (был 0.001) — это ВАЖНЕЙШИЙ пункт
                                                            "tau": 0.001,               # Скорость обновления целевых сетей (сделайте мягче, стандарт 0.005)
                                                            "gamma": 0.99,              # Фактор дисконтирования (стандарт)
                                                            "action_noise": "ornstein_uhlenbeck", # Добавляет "умный" шум для поиска стратегий
                                                        }},
            {"name": "td3", "timesteps": 30000, "params": {"batch_size": 128, "buffer_size": 100000, "learning_rate": 1e-3}}
        ]

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

    @contextmanager
    def suppress_everything(self):
        """Полная тишина в консоли во время обучения."""
        original_level = logging.getLogger().getEffectiveLevel()
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")
        absl.logging.set_verbosity(absl.logging.ERROR)
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            try:
                yield
            finally:
                logging.disable(original_level)

    def get_fee(self, key):
        """Универсальный помощник для извлечения комиссий (из FinRLTrainer)."""
        val = self.base_env_kwargs.get(key, 0.001)
        if isinstance(val, (list, np.ndarray)) and len(val) > 0:
            return val[0]  # Берем число из списка, если пришел список
        return val  # Возвращаем число, если пришло число

    def _build_env_config(self, df):
        """
        ЕДИНАЯ ЛОГИКА: Расчет State Space, Индикаторов и формирование словаря.
        Эта функция готовит 'чертеж' среды для всех моделей ансамбля.
        """
        os.makedirs(self.log_path, exist_ok=True)

        # 1. Список индикаторов (динамический расчет)
        exclude_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'day']
        actual_indicators = [c for c in df.columns if c not in exclude_cols]

        # 2. Размерности
        stock_dim = df['tic'].nunique()
        # Формула из вашего оригинального фрагмента (1 + 5*dim + ind*dim)
        state_space = 1 + (5 + len(actual_indicators)) * stock_dim

        # 3. Сборка финального словаря env_kwargs
        env_kwargs = self.base_env_kwargs.copy()
        env_kwargs.update({
            "stock_dim": stock_dim,
            "state_space": state_space,
            "tech_indicator_list": actual_indicators,
            "action_space": stock_dim,
            "num_stock_shares": [0] * stock_dim,
            "buy_cost_pct": [self.get_fee('buy_cost_pct')] * stock_dim,
            "sell_cost_pct": [self.get_fee('sell_cost_pct')] * stock_dim,
            "reward_scaling": 1e-4,
            "print_verbosity": 5
        })

        print(f"🛠 СИНХРОНИЗАЦИЯ: Dim={stock_dim}, State Space={state_space}")
        print(f"📊 Индикаторов в модели: {len(actual_indicators)}")

        # Сохраняем в self для доступа из Backtester и воркеров
        self.current_env_kwargs = env_kwargs
        return env_kwargs

    def _ray_train_worker(self, config):
        """Воркер Ray: Обучает модель с использованием ray.train.report."""
        # Локальные импорты (необходимы для изолированных процессов Ray)
        import torch as th
        import numpy as np
        import os
        from ray import train
        from stable_baselines3.common.callbacks import EvalCallback, CallbackList
        from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

        # 1. Распаковка конфигурации
        # Ray Tune может передавать конфиг напрямую или вложенным
        inner_config = config.get("config", config)
        if isinstance(inner_config, dict) and "config" in inner_config:
            inner_config = inner_config["config"]

        name = inner_config["name"]
        params = inner_config["params"].copy()
        timesteps = inner_config["timesteps"]
        fold = inner_config.get("fold", 1)

        # 2. Работа с путями (используем абсолютные пути)
        base_save_dir = os.path.abspath(self.save_path)
        model_specific_path = os.path.join(base_save_dir, f"{name}_f{fold}")
        os.makedirs(model_specific_path, exist_ok=True)
        final_path = os.path.join(model_specific_path, f"{name}_model.zip")

        # 3. Создание сред
        e_train_gym = self.env_class(df=self.train_data, **self.current_env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()

        e_val_gym = self.env_class(df=self.val_data, **self.current_env_kwargs)
        env_val, _ = e_val_gym.get_sb_env()

        # 4. Callbacks
        eval_cb = EvalCallback(
            eval_env=env_val,
            best_model_save_path=model_specific_path,
            log_path=model_specific_path,
            eval_freq=2000,
            deterministic=True,
            verbose=0
        )
        metrics_cb = TradingMetricsCallback()
        cb_list = CallbackList([eval_cb, metrics_cb])

        # 5. Настройка архитектуры и специфических параметров
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
        )

        if name == "ppo":
            safe_n_steps = 512 if len(self.train_data) > 1024 else 128
            params.update({"n_steps": safe_n_steps, "policy_kwargs": policy_kwargs})
        elif name == "ddpg":
            n_actions = self.current_env_kwargs.get("action_space", self.current_env_kwargs.get("stock_dim"))
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.5 * np.ones(n_actions)
            )
            params.update({"action_noise": action_noise, "policy_kwargs": policy_kwargs})

        # 6. Обучение
        with self.suppress_everything():
            try:
                agent = DRLAgent(env=env_train)
                model = agent.get_model(name, model_kwargs=params)

                model.learn(
                    total_timesteps=timesteps,
                    callback=cb_list,
                    tb_log_name=f"{name}_f{fold}",
                    reset_num_timesteps=True
                )

                # Сохранение
                model.save(final_path)

                # Подготовка метрик
                best_reward = getattr(eval_cb, 'best_mean_reward', -np.inf)
                best_reward = float(best_reward) if best_reward != -np.inf else 0.0
                final_metrics = getattr(metrics_cb, 'metrics', {})

                result_data = {
                    "status": "success",
                    "name": name,
                    "path": final_path,
                    "fold": fold,
                    "best_reward": best_reward,
                    "metrics_data": final_metrics,
                }

                # ОТЧЕТ В RAY (Критично для Ray Train v2)
                train.report(result_data)
                return result_data

            except Exception as e:
                error_data = {"status": "failed", "error": str(e), "name": name}
                train.report(error_data)
                return error_data

    def train_ensemble_parallel(self, train_df, val_df, fold=1):
        """Последовательное обучение ансамбля (оптимально для Colab)."""
        self.train_data, self.val_data = train_df, val_df
        self._build_env_config(self.train_data)

        abs_save_path = os.path.abspath(self.save_path)
        trained_models_map = {}
        performance_report = {}
        results_list = []

        print(f"🚀 Начало последовательного обучения ансамбля ({len(self.model_configs)} моделей)...")

        for cfg in self.model_configs:
            name = cfg["name"]
            params = cfg["params"].copy()
            timesteps = cfg["timesteps"]

            print(f"🔹 Обучение {name.upper()} (Fold {fold})...")

            # Уникальный путь для модели
            model_specific_path = os.path.join(abs_save_path, f"{name}_f{fold}")
            os.makedirs(model_specific_path, exist_ok=True)
            final_path = os.path.join(model_specific_path, f"{name}_model.zip")

            try:
                # 1. Создание сред
                e_train_gym = self.env_class(df=train_df, **self.current_env_kwargs)
                env_train, _ = e_train_gym.get_sb_env()
                e_val_gym = self.env_class(df=val_df, **self.current_env_kwargs)
                env_val, _ = e_val_gym.get_sb_env()

                # 2. Настройка Callbacks
                eval_cb = EvalCallback(
                    eval_env=env_val,
                    best_model_save_path=model_specific_path,
                    eval_freq=2000,
                    deterministic=True,
                    verbose=0
                )
                metrics_cb = TradingMetricsCallback()
                cb_list = CallbackList([eval_cb, metrics_cb])

                # 3. Обучение через DRLAgent
                agent = DRLAgent(env=env_train)
                model = agent.get_model(name, model_kwargs=params)
                model.learn(total_timesteps=timesteps, callback=cb_list)

                # 4. Сохранение
                model.save(final_path)

                # 5. Сбор результатов в мапу
                trained_models_map[name] = model
                performance_report[name] = {
                    "best_reward": float(getattr(eval_cb, 'best_mean_reward', 0.0)),
                    "metrics": getattr(metrics_cb, 'metrics', {}),
                    "path": final_path,
                    "fold": fold
                }

                results_list.append({"status": "success", "name": name})
                print(f"✅ {name.upper()} обучена. Best Reward: {performance_report[name]['best_reward']:.2f}")

                # Очистка памяти после каждой модели
                del model, agent, env_train, env_val
                import gc;
                gc.collect();
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Ошибка при обучении {name}: {e}")
                results_list.append({"status": "failed", "name": name, "error": str(e)})

        return results_list, trained_models_map, performance_report

    def get_ensemble_predictions(self, test_df):
        """Инференс: сбор прогнозов для всех моделей ансамбля."""
        print("🔮 Сбор сигналов ансамбля...")
        raw_outputs = {}
        env_test = self.env_class(df=test_df, **self.current_env_kwargs)

        # Словарь классов для загрузки
        from stable_baselines3 import PPO, A2C, SAC, DDPG, TD3
        model_classes = {"ppo": PPO, "a2c": A2C, "sac": SAC, "ddpg": DDPG, "td3": TD3}

        for cfg in self.model_configs:
            name = cfg["name"]
            model = None

            # Рекурсивный поиск файла модели в папке сохранения
            for root, dirs, files in os.walk(self.save_path):
                if f"{name}_model.zip" in files:
                    path = os.path.join(root, f"{name}_model.zip")
                    print(f"🔍 Загрузка {name} из {path}")
                    model = model_classes[name].load(path)
                    break

            if model:
                _, df_actions = DRLAgent.DRL_prediction(model=model, environment=env_test)
                raw_outputs[name] = df_actions
            else:
                print(f"⚠️ Модель {name} не найдена. Прогноз пропущен.")

        return raw_outputs
