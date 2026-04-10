import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class TalebRiskInspector(BaseEstimator, TransformerMixin):
    def __init__(self, max_steps=10000, deterministic=True):
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.stock_dim = None  # Инициализируем пустым
        self.env = None
        self.wrapped_env = None

    def fit(self, X, y=None):
        return self

    def transform(self, training_output):
        """
        training_output: словарь {'model': ..., 'env_kwargs': ..., 'df': ..., 'dates': ...}
        """
        try:
            # 1. Извлечение данных
            model = training_output.get('model')
            env_kwargs = training_output.get('env_kwargs')
            df_test_raw = training_output.get('df')

            if model is None or env_kwargs is None or df_test_raw is None:
                print("❌ Ошибка: Входные данные для аудита неполные.")
                return pd.DataFrame()

            # 2. Финализация данных (важно! используем тот же метод, что в трейне)
            # Это гарантирует наличие 'close', 'open' и всех tech_indicator_list
            all_indicators = env_kwargs.get('tech_indicator_list', [])

            # Вызываем глобальную функцию очистки
            X_sorted = finalize_market_data(df_test_raw, all_indicators)

            # 3. Синхронизация stock_dim и выравнивание сетки
            self.stock_dim = env_kwargs.get('stock_dim')
            X_sorted = X_sorted.sort_values(['date', 'tic']).reset_index(drop=True)

            # Проверка: на каждую дату должно быть ровно stock_dim строк
            counts = X_sorted.groupby('date')['tic'].count()
            if not (counts == self.stock_dim).all():
                print(f"⚠️ Несоответствие тикеров (найдено {counts.unique()}). Оставляем полные дни...")
                valid_dates = counts[counts == self.stock_dim].index
                X_sorted = X_sorted[X_sorted['date'].isin(valid_dates)].reset_index(drop=True)

            # Создаем 'day' строго после фильтрации
            X_sorted['day'] = X_sorted.groupby('date', sort=True).ngroup()

            # Получаем финальный список дат для лога
            test_dates_list = sorted(X_sorted['date'].unique())
            unique_dates_count = len(test_dates_list)

            # 4. Создание среды
            try:
                # Используем ваш модифицированный класс среды
                e_test_gym = ModifiedTalebStockTradingEnv(df=X_sorted, **env_kwargs)
                self.env = e_test_gym
                e_test_gym = TalebRiskWrapper(e_test_gym, model,
                                              confidence_threshold=env_kwargs.get('confidence_threshold', 0.25))
                self.wrapped_env = e_test_gym
                env_test = DummyVecEnv([lambda: e_test_gym])
            except Exception as e:
                print(f"❌ Не удалось создать тестовую среду: {e}")
                return pd.DataFrame()

            # 5. Цикл аудита
            records = []
            obs = env_test.reset()

            # Получаем доступ к "живой" среде с расширенной диагностикой
            try:
                print(f"DEBUG: Тип env_test: {type(env_test)}")

                # Вариант 1: Стандартный DummyVecEnv из SB3
                if hasattr(env_test, 'envs'):
                    print(f"DEBUG: Найдено .envs, количество: {len(env_test.envs)}")
                    working_env = env_test.envs[0]
                # Вариант 2: Если это SubprocVecEnv или другая обертка
                elif hasattr(env_test, 'get_attr'):
                    print("DEBUG: Используется метод get_attr для доступа к атрибутам")
                    # Пытаемся вытащить саму среду (это хак для VecEnv)
                    working_env = env_test.get_attr('env')[0] if hasattr(env_test.get_attr('env')[0],
                                                                         'step') else env_test
                else:
                    working_env = env_test
                    print("DEBUG: Прямой доступ к среде (без вектора)")

                # Пробиваемся через стек оберток (Wrappers)
                depth = 0
                while hasattr(working_env, 'env') and depth < 10:
                    print(f"DEBUG: Слой {depth}: {type(working_env)} -> углубляемся...")
                    working_env = working_env.env
                    depth += 1

                print(f"✅ Финальный тип рабочей среды: {type(working_env)}")

                # Проверка наличия памяти
                if not hasattr(working_env, 'account_value_memory'):
                    print(f"❌ ОШИБКА: В среде {type(working_env)} нет атрибута account_value_memory!")
                    print(f"Доступные атрибуты: {dir(working_env)}")

            except Exception as e:
                import traceback
                print(f"⚠️ Критическая ошибка доступа к базовой среде: {e}")
                traceback.print_exc()
                return pd.DataFrame()

            steps = min(unique_dates_count, self.max_steps)
            print(f"[{time.strftime('%H:%M:%S')}] 🔍 Запуск риск-аудита ({steps} шагов)...")

            for i in range(steps):
                try:
                    # 1. Прогноз агента
                    action, _ = model.predict(obs, deterministic=self.deterministic)

                    # 2. Извлечение риск-метрик из экстрактора (MoexTalebExtractor)
                    extractor = model.policy.features_extractor
                    diag = getattr(extractor, 'diagnostics', {})

                    # 3. Шаг в среде
                    obs, reward, done, info = env_test.step(action)

                    # 4. СОХРАНЕНИЕ ДАННЫХ (Берем из working_env в реальном времени)
                    # Извлекаем текущий баланс напрямую из памяти работающей среды
                    current_acc_val = working_env.account_value_memory[-1]

                    record = {
                        'date': pd.to_datetime(test_dates_list[i]),
                        'account_value': current_acc_val,  # ТЕПЕРЬ ТУТ БУДУТ РЕАЛЬНЫЕ ЦИФРЫ
                        'tail_risk': diag.get('tail_risk', 0.0),
                        'confidence': diag.get('confidence', 0.0),
                        'kurtosis': diag.get('kurtosis_avg', 0.0),
                        'reward': reward[0] if isinstance(reward, np.ndarray) else reward,
                        'action_mean': np.mean(action),
                    }

                    # Добавляем индивидуальные риски по каждому тикеру
                    if 'tail_map' in diag:
                        for idx, val in enumerate(diag['tail_map']):
                            record[f'tail_risk_stock_{idx}'] = val

                    records.append(record)

                    if done:
                        print(f"🏁 Среда завершила работу на шаге {i}")
                        break

                except Exception as e:
                    print(f"🛑 Ошибка на шаге {i} ({test_dates_list[i]}): {e}")
                    break

            # --- 6. СИНХРОНИЗАЦИЯ И ФИНАЛИЗАЦИЯ ---
            try:
                # Копируем полную историю в основной объект инспектора
                self.env.account_value_memory = list(working_env.account_value_memory)
                self.env.date_memory = list(working_env.date_memory)
                self.env.rewards_memory = list(working_env.rewards_memory)

                # Создаем финальный DataFrame
                df_results = pd.DataFrame(records)

                # Если список records все же пуст (ошибка цикла), восстанавливаем из памяти
                if df_results.empty and len(self.env.account_value_memory) > 1:
                    print("⚠️ Восстановление данных из памяти среды...")
                    # ... (код восстановления, если нужен)

            except Exception as e:
                print(f"⚠️ Ошибка финальной синхронизации: {e}")
                df_results = pd.DataFrame(records)

            if not df_results.empty:
                print(f"✅ Аудит завершен. Длина истории: {len(self.env.account_value_memory)}")
                # print(f"📈 Финальный баланс: {self.env.account_value_memory[-1]:.2f}")

            return df_results
        except Exception as e:
            print(f"❌ Не удалось создать тестовую среду: {e}")
            return pd.DataFrame()