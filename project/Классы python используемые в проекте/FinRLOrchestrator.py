import os, logging, warnings, pandas as pd, numpy as np, torch as th
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.agents.stablebaselines3.models import DRLAgent
import os
import logging
import warnings
import absl.logging
from tqdm import tqdm
import gc


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


class FinRLOrchestrator:
    def __init__(self, df_train, df_test, env_class, base_env_kwargs,
                 save_path="./", total_timesteps=50000, model_configs=None, log_path="./tensorboard_log/"):

        self.env_class = env_class
        self.base_env_kwargs = base_env_kwargs.copy()
        self.save_path = save_path
        self.total_timesteps = total_timesteps
        self.log_path = log_path
        self.train_data = df_train
        self.val_data = df_test

        self.val_data = None
        self.current_env_kwargs = None
        self.final_path = None

        # 1. КОНФИГУРАЦИЯ АНСАМБЛЯ (С включением всех параметров PPO из вашего фрагмента)
        self.model_configs = model_configs or [
            {
                "name": "ppo",
                "timesteps": total_timesteps,
                "params": {
                    "batch_size": 128,
                    "learning_rate": 7e-5,
                    "ent_coef": 0.2,
                    "gamma": 0.99,
                    "clip_range": 0.2,  # Консервативные обновления
                    "gae_lambda": 0.95,  # Сглаживание вознаграждения
                    "vf_coef": 0.5  # Точность оценки выгоды
                }
            },
            {"name": "sac", "timesteps": 30000,
             "params": {"batch_size": 128, "buffer_size": 100000, "learning_rate": 0.0001, "learning_starts": 100,
                        "ent_coef": "auto_0.1"}},
            {"name": "a2c", "timesteps": 30000, "params": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0001}},
            {"name": "ddpg", "timesteps": 20000,
             "params": {"batch_size": 128,  # Увеличиваем (стандарт 64), чтобы градиент был стабильнее
                        "buffer_size": 50000,  # Размер памяти (соответствует вашему конфигу)
                        "learning_rate": 0.00005,  # Снижаем (был 0.001) — это ВАЖНЕЙШИЙ пункт
                        "tau": 0.001,  # Скорость обновления целевых сетей (сделайте мягче, стандарт 0.005)
                        "gamma": 0.95,  # Фактор дисконтирования (стандарт)
                        "action_noise": "ornstein_uhlenbeck",  # Добавляет "умный" шум для поиска стратегий
                        }},
            {"name": "td3", "timesteps": 30000,
             "params": {"batch_size": 128, "buffer_size": 100000, "learning_rate": 1e-3}}
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
        self.stock_dim = df['tic'].nunique()

        # Формула: 1(кэш) + 2(Талеб: L-Kurt, Drawdown) + stock_dim(доли) + stock_dim(цены) + (инд * тикеры)
        self.state_space = 1 + 2 + 2 * self.stock_dim + (len(actual_indicators) * self.stock_dim)

        # 3. Сборка финального словаря
        env_kwargs = self.base_env_kwargs.copy()
        env_kwargs.update({
            "stock_dim": self.stock_dim,
            "state_space": self.state_space,
            "tech_indicator_list": actual_indicators,
            "taleb_window": 30,  # ДОБАВИЛИ окно для расчета Талеба
            "drawdown_window": 30,  # ДОБАВИЛИ окно для просадки
            "action_space": self.stock_dim,
            # Исправляем обращение к stock_dim (теперь это self.stock_dim)
            "num_stock_shares": [0] * self.stock_dim,
            "buy_cost_pct": 0.001,  # Можно передать числом, среда сама размножит в __init__
            "sell_cost_pct": 0.0015,
            "reward_scaling": 1e-2,
            "print_verbosity": 5
        })

        print(f"🛠 СИНХРОНИЗАЦИЯ: Dim={self.stock_dim}, State Space={self.state_space}")
        print(f"📊 Индикаторов в модели: {len(actual_indicators)}")

        # Сохраняем в self для доступа из Backtester и воркеров
        self.current_env_kwargs = env_kwargs
        return env_kwargs

    def _ray_train_worker(self, config):
        """Воркер Ray: Обучает модель с использованием ray.train.report."""
        # Локальные импорты (необходимы для изолированных процессов Ray)
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

    def train_ensemble_parallel(self, train_df, val_df, fold=1, models_to_train: list = None):
        """
        Последовательное обучение выбранных моделей с верификацией сохранения.
        """
        self.train_data, self.val_data = train_df, val_df
        self._build_env_config(self.train_data)

        abs_save_path = os.path.abspath(self.save_path)
        trained_models_map = {}
        performance_report = {}
        results_list = []

        # Фильтруем конфиги моделей
        active_configs = self.model_configs
        if models_to_train:
            active_configs = [cfg for cfg in self.model_configs if cfg["name"] in models_to_train]
            if not active_configs:
                print(f"⚠️ Модели {models_to_train} не найдены в конфигах!")
                return [], {}, {}

        print(f"\n🚀 Запуск обучения ансамбля (Fold {fold})")

        for idx, cfg in enumerate(active_configs, 1):
            name = cfg["name"]
            params = cfg["params"].copy()
            timesteps = cfg["timesteps"]

            print(f"\n🔹 [{idx}/{len(active_configs)}] Обучение {name.upper()}...")

            # Путь для сохранения
            model_folder = os.path.join(abs_save_path, f"{name}_f{fold}")
            os.makedirs(model_folder, exist_ok=True)
            final_path = os.path.join(model_folder, f"{name}_model.zip")
            self.final_path = final_path

            try:
                # 1. Инициализация сред
                e_train_gym = self.env_class(df=train_df, **self.current_env_kwargs)
                env_train, _ = e_train_gym.get_sb_env()
                e_val_gym = self.env_class(df=val_df, **self.current_env_kwargs)
                env_val, _ = e_val_gym.get_sb_env()

                # 2. Настройка Callbacks
                eval_cb = EvalCallback(
                    eval_env=env_val,
                    best_model_save_path=model_folder,
                    eval_freq=2000,
                    deterministic=True,
                    verbose=0
                )

                # Поиск TradingMetricsCallback в глобальном пространстве
                metrics_class = globals().get('TradingMetricsCallback')
                metrics_cb = metrics_class() if metrics_class else None
                cb_list = CallbackList([eval_cb, metrics_cb] if metrics_cb else [eval_cb])

                # 3. Создание агента и обучение
                from finrl.agents.stablebaselines3.models import DRLAgent
                agent = DRLAgent(env=env_train)

                # 1. Сначала готовим архитектуру
                p_kwargs = {
                    "activation_fn": th.nn.ReLU,
                    "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128])
                }


                # 2. ОЧИЩАЕМ params от возможных дублей перед обновлением
                params.pop("policy_kwargs", None)

                if name == "ppo":
                    params.update({
                        "n_steps": 512 if len(train_df) > 1024 else 128
                    })
                elif name == "ddpg":
                    n_actions = self.current_env_kwargs.get("stock_dim")
                    params.update({
                        "action_noise": OrnsteinUhlenbeckActionNoise(
                            mean=np.zeros(n_actions),
                            sigma=0.5 * np.ones(n_actions)
                        )
                    })

                model = agent.get_model(
                    model_name=name,  # Имя алгоритма (ppo, sac и т.д.)
                    policy="MlpPolicy",  # Тип политики (явное указание)
                    model_kwargs=params,  # Настройки алгоритма
                    policy_kwargs=p_kwargs  # Настройки нейросети
                )
                with self.suppress_everything():
                    model.learn(total_timesteps=timesteps, callback=cb_list, reset_num_timesteps=True)

                # 4. ВЫЗОВ ВЕРИФИКАЦИИ (Здесь переменная 'model' определена)
                print(f"💾 Сохранение и проверка модели {name}...")
                success = False
                try:
                    model.save(final_path)
                    # Проверяем: файл существует и он не пустой
                    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                        # Финальный тест: пробуем загрузить веса обратно
                        model.__class__.load(final_path)
                        success = True
                        print(f"✅ Модель верифицирована ({os.path.getsize(final_path)} байт)")

                except Exception as save_err:
                    print(f"❌ Ошибка при верификации сохранения: {save_err}")

                if success:
                    # Сохраняем в мапу только если верификация прошла успешно
                    trained_models_map[name] = model

                    # Извлечение метрик
                    best_reward = getattr(eval_cb, 'best_mean_reward', 0.0)
                    if best_reward == -float('inf'): best_reward = 0.0

                    performance_report[name] = {
                        "best_reward": float(best_reward),
                        "metrics": getattr(metrics_cb, 'metrics', {}) if metrics_cb else {},
                        "path": final_path,
                        "fold": fold
                    }
                    results_list.append({"status": "success", "name": name})
                else:
                    results_list.append({"status": "save_error", "name": name})

                # 5. Очистка ресурсов после каждой модели
                # del model, agent, env_train, env_val
                gc.collect()
                if th.cuda.is_available():
                    th.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Критическая ошибка {name}: {e}")
                results_list.append({"status": "failed", "name": name, "error": str(e)})

        print(f"\n✅ Обучение фолда {fold} завершено. Успешно: {len(trained_models_map)}/{len(active_configs)}")

        self.final_path = final_path
        self.save_path = final_path
        self.trained_models_map = trained_models_map

        return results_list, trained_models_map, performance_report, self.current_env_kwargs

    def get_ensemble_predictions(self, test_df, models_to_use: list = None):
        """
        Инференс: сбор прогнозов. Исправленная и стабильная версия.
        """
        import os
        import gc
        import pandas as pd
        from stable_baselines3 import PPO, A2C, SAC, DDPG, TD3
        from finrl.agents.stablebaselines3.models import DRLAgent

        print("\n🔮 Сбор сигналов ансамбля...")
        raw_outputs = {}

        # 1. Подготовка среды
        e_test_gym = self.env_class(df=test_df, **self.current_env_kwargs)
        model_classes = {"ppo": PPO, "a2c": A2C, "sac": SAC, "ddpg": DDPG, "td3": TD3}
        active_names = models_to_use if models_to_use else [cfg["name"] for cfg in self.model_configs]

        for name in active_names:
            model = None
            load_source = "unknown"

            # А) Поиск модели в памяти
            if hasattr(self, 'trained_models_map') and name in self.trained_models_map:
                model = self.trained_models_map[name]
                load_source = "memory"

            # Б) Поиск модели на диске (если не найдена в памяти)
            if model is None:
                model_path = None
                for root, dirs, files in os.walk(self.save_path):
                    if f"{name}_model.zip" in files:
                        model_path = os.path.join(root, f"{name}_model.zip")
                        break  # Нашли путь, выходим из walk

                if model_path:
                    try:
                        model = model_classes[name].load(model_path)
                        load_source = f"disk ({model_path})"
                    except Exception as load_err:
                        print(f"⚠️ Не удалось загрузить {name} из {model_path}: {load_err}")

            # В) Выполнение инференса
            if model is not None:
                try:
                    df_actions = None
                    print(f"🔹 Инференс {name.upper()}...")

                    # 1. ОБЯЗАТЕЛЬНО: Сброс среды перед прогоном, чтобы очистить старую память
                    _ = e_test_gym.reset()
                    print(f"Сброс среды перед прогоном выполнен")
                    # 2. Запуск прогона БЕЗ ПРИСВАИВАНИЯ.
                    # Мы игнорируем то, что возвращает функция, так как она может вернуть None.
                    with self.suppress_everything():
                        try:
                            DRLAgent.DRL_prediction(model=model, environment=e_test_gym)
                        except Exception as e:
                            print(f"❌ Ошибка DRLAgent.DRL_prediction: {e}")
                            print("Трассировка:")
                            traceback.print_exc()
                    # Так как в методе step теперь есть append, данные точно будут здесь.
                    df_actions = e_test_gym.save_action_memory()
                    print(f"ПРЯМОЕ ИЗВЛЕЧЕНИЕ выполнено")
                    # 4. Проверка и сохранение в итоговый словарь
                    if df_actions is not None and not df_actions.empty:
                        print(f"Проверка и сохранение в итоговый словарь")
                        raw_outputs[name] = df_actions
                        print(f"📈 {name.upper()}: сигналы успешно собраны ({len(df_actions)} шагов)")
                    else:
                        print(f"❌ {name.upper()}: Память действий пуста после прогона. Проверьте step!")


                except Exception as e:
                    print(f"❌ Ошибка инференса {name} (источник: {load_source}): {e}")
            else:
                print(f"⚠️ Модель {name} не найдена ни в памяти, ни на диске.")

        # Очистка
        del e_test_gym
        gc.collect()

        if not raw_outputs:
            print("❌ Не удалось получить прогнозы ни от одной модели!")
        else:
            print(f"✅ Успешно получены прогнозы от {len(raw_outputs)} моделей: {list(raw_outputs.keys())}")

        return raw_outputs

    def get_ensemble_alpha_signals(self, trained_models_map, test_df):
        """
        Получает прогнозы от всех обученных моделей для передачи в Riskfolio.
        """
        signals = {}

        # Создаем тестовую среду (без обучения, только для инференса)
        env_test_gym = self.env_class(df=test_df, **self.current_env_kwargs)
        env_test, _ = env_test_gym.get_sb_env()

        for name, model in trained_models_map.items():
            print(f"🔮 Генерируем сигналы для {name.upper()}...")
            # Стандартный метод FinRL для получения экшенов
            # Эти экшены [-1, 1] мы трактуем как "уверенность модели в росте/падении"
            from finrl.agents.stablebaselines3.models import DRLAgent
            _, df_actions = DRLAgent.DRL_prediction(model=model, environment=env_test)

            # Убеждаемся, что индексы - это дата и тикер
            signals[name] = df_actions.set_index(['date', 'tic'])

        # Усредняем сигналы всех моделей в один вектор (Ансамбль)
        ensemble_alpha = pd.concat(signals.values(), axis=1).mean(axis=1)
        return ensemble_alpha

