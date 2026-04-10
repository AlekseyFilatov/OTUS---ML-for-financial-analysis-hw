from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import numpy as np
import pandas as pd
import datetime
import sys
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C, SAC, DDPG, TD3
import traceback
from lmoments3 import stats
import lmoments3 as lm

# Эмуляция старого интерфейса gym для совместимости с FinRL
sys.modules["gym"] = gym

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import numpy as np
import pandas as pd
import datetime
import sys
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C, SAC, DDPG, TD3
import traceback
from lmoments3 import stats
import lmoments3 as lm
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# Эмуляция старого интерфейса gym для совместимости с FinRL
sys.modules["gym"] = gym

class ModifiedTalebStockTradingEnv(StockTradingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sample_obs = self._initiate_state()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(sample_obs),), dtype=np.float32
        )

    def _calculate_reward(self, assets_before, assets_after):
        """Расчёт награды с защитой от крайних случаев и ограничением (clipping)"""
        print(f"DEBUG: Before={assets_before:.2f}, After={assets_after:.2f}, Diff={assets_after - assets_before}")
        if assets_before < 1e-8:
            return 0.0

        # Считаем процентное изменение
        reward = (assets_after - assets_before) / assets_before

        # ВАЖНО: возвращаем значение и ограничиваем его диапазон для стабильности PPO
        # Ограничение [-0.1, 0.1] (10% изменения за день) обычно достаточно для рынков
        return np.clip(float(reward), -0.1, 0.1)

    def _scale_actions(self, actions):
        """Масштабирование действий: выбирайте между осторожным (log1p) и агрессивным (tanh)"""
        signs = np.sign(actions)
        # Оставляем log1p для защиты от выбросов на этапе отладки
        actions_scaled = np.tanh(signs * np.log1p(np.abs(actions))) * self.hmax
        return actions_scaled

    def _get_day_data(self, day):
        """Безопасное получение данных по дню"""
        data = self.df[self.df.day == day].sort_values('tic')
        return data if not data.empty else pd.DataFrame()

    def step(self, actions):
        if self.terminated:
            # Если мы уже закончили, возвращаем 0 и True
            return np.array(self.state, dtype=np.float32), 0.0, True, False, {}
        # 1. ВАЛИДАЦИЯ И ПОДГОТОВКА
        if len(actions) != self.stock_dim:
            raise ValueError(f"Длина actions ({len(actions)}) != stock_dim ({self.stock_dim})")

        max_days = len(self.df['day'].unique()) - 1
        if self.day >= max_days:
            self.terminated = True  # Исправлено: было terminate
            return np.array(self.state, dtype=np.float32), 0.0, True, False, {}

        try:
            # 2. ОПРЕДЕЛЯЕМ СТОИМОСТЬ ПОРТФЕЛЯ ДО СДЕЛОК
            current_day_data = self._get_day_data(self.day)
            if current_day_data.empty:
                self.terminated = True
                return np.array(self.state, dtype=np.float32), 0.0, True, False, {}

            curr_close = current_day_data['close'].values.astype(float)
            self.assets_before = self.amount + np.sum(np.array(self.stocks) * curr_close)
            self.data = current_day_data

            # 3. ОБРАБОТКА ДЕЙСТВИЙ С КОНТРОЛЕМ ОБЪЁМОВ
            actions_scaled = self._scale_actions(actions)
            actions_int = np.round(actions_scaled).astype(int)

            # 4. ИСПОЛНЕНИЕ СДЕЛОК С ПРОВЕРКОЙ ДОСТУПНОСТИ
            tic_names = self.data['tic'].tolist()
            total_cost = 0.0

            for i in range(self.stock_dim):
                action = actions_int[i]
                price = curr_close[i]
                if action < 0:  # ПРОДАЖА
                    sell_num = min(abs(action), self.stocks[i])
                    if sell_num > 0:
                        # Достаем комиссию для продажи
                        cost_pct = self.sell_cost_pct[i] if isinstance(self.sell_cost_pct, list) else self.sell_cost_pct

                        self.stocks[i] -= sell_num
                        # При продаже комиссия ВЫЧИТАЕТСЯ из выручки
                        self.amount += (sell_num * price * (1 - cost_pct))
                        self.trades += 1
                        print(f"  [SELL] {tic_names[i]}: {sell_num} units | Cash Left: {self.amount:.2f}")
                elif action > 0:  # ПОКУПКА
                    # Достаем комиссию для конкретного тикера (если это список)
                    cost_pct = self.buy_cost_pct[i] if isinstance(self.buy_cost_pct, list) else self.buy_cost_pct

                    # Считаем, сколько можем купить на имеющийся кэш с учетом комиссии
                    # Используем (1 + cost_pct)
                    max_buy = int(self.amount / (price * (1 + cost_pct)))
                    buy_num = min(action, max_buy)

                    if buy_num > 0:
                        # Прямое списание с учетом комиссии конкретного тикера
                        self.amount -= (buy_num * price * (1 + cost_pct))
                        self.stocks[i] += buy_num
                        self.trades += 1
                        print(f"  [BUY ] {tic_names[i]}: {buy_num} units | Cash Left: {self.amount:.2f}")

            # for i in range(self.stock_dim):
            #    action = actions_int[i]
            #    tic = tic_names[i]

            #    if action < 0:  # Продажа
            #        available_shares = self.stocks[i]
            #        sell_amount = min(abs(action), available_shares)
            #        if sell_amount > 0:
            #            self._sell_stock(i, sell_amount)
            # print(f"  [SELL] {tic}: {sell_amount} units")
            #    elif action > 0:  # Покупка
            #        cost = action * curr_close[i]
            #        if total_cost + cost <= self.amount:  # Проверка бюджета
            #            self._buy_stock(i, action)
            #            total_cost += cost
            # print(f"  [BUY ] {tic}: {action} units")

            # 5. ПЕРЕХОД К СЛЕДУЮЩЕМУ ДНЮ
            self.day += 1
            next_day_data = self._get_day_data(self.day)
            if next_day_data.empty:
                self.terminated = True
                # Считаем финальную стоимость активов по ЦЕНАМ ТЕКУЩЕГО ДНЯ (curr_close)
                # Так как мы уже сделали шаг вперед, а данных нет.
                final_assets = self.amount + np.sum(np.array(self.stocks) * curr_close)
                # Награда за последний рывок цены
                reward = self._calculate_reward(self.assets_before, final_assets)
                # Лог для отладки (удалите после проверки)
                print(
                    f"DEBUG FINAL: Day {self.day} | Cash: {self.amount:.2f} | Stocks: {sum(self.stocks)} | Reward: {reward:.6f}")
                self.account_value_memory.append(final_assets)
                self.date_memory.append(self._get_date())
                return np.array(self.state, dtype=np.float32), float(reward), True, False, {}

            self.data = next_day_data
            self.state = self._update_state()

            # 6. РАСЧЁТ НАГРАДЫ С ЗАЩИТАМИ
            next_close = self.data['close'].values.astype(float)
            assets_after = self.amount + np.sum(np.array(self.stocks) * next_close)

            reward = self._calculate_reward(self.assets_before, assets_after)

            # 7. СОХРАНЕНИЕ И ВОЗВРАТ
            self.account_value_memory.append(assets_after)
            self.date_memory.append(self._get_date())

            # Защита от NaN/Inf
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0

            self.rewards_memory.append(reward)
            self.actions_memory.append(actions_int)
            if self.terminated:
                print("\n" + "=" * 30)
                print(f"🏁 ЭПИЗОД ЗАВЕРШЕН")
                print(f"📅 Дата окончания: {self._get_date()}")
                print(f"💰 Итоговый капитал: {self.amount + np.sum(np.array(self.stocks) * next_close):.2f}")
                print(f"🔄 Всего сделок: {self.trades}")  # <--- СЮДА
                print(f"📈 Средняя награда: {np.mean(self.rewards_memory):.6f}")  # <--- И СЮДА
                print("=" * 30 + "\n")

            return np.array(self.state, dtype=np.float32), float(reward), False, False, {}

        except Exception as e:
            import traceback
            print(f"Критическая ошибка на шаге {self.day}: {e}")
            traceback.print_exc()  # Добавляем стек вызовов для отладки
            return np.array(self.state, dtype=np.float32), 0.0, True, False, {}

    def save_asset_memory(self):
        """Возвращает историю стоимости портфеля"""
        try:
            # FinRL хранит историю в self.account_value_memory
            assets = getattr(self, 'account_value_memory', [])
            dates = getattr(self, 'date_memory', [])

            # Убираем дубликаты дат и синхронизируем длину
            min_len = min(len(assets), len(dates))
            df_acc_value = pd.DataFrame({
                "date": dates[:min_len],
                "account_value": assets[:min_len]
            })
            return df_acc_value.sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ Ошибка в save_asset_memory: {e}")
            return pd.DataFrame()

    def save_action_memory(self) -> pd.DataFrame:
        """Возвращает историю действий (аллокацию по тикерам)"""
        try:
            actions = getattr(self, 'actions_memory', [])
            dates = getattr(self, 'date_memory', [])

            if not actions:
                return pd.DataFrame()

            # Превращаем список действий в массив (шаги x тикеры)
            actions_array = np.array(actions)

            # Названия колонок берем из уникальных тикеров в данных
            tics = sorted(self.df.tic.unique())

            # Синхронизируем по минимальной длине (даты vs действия)
            min_len = min(len(actions_array), len(dates))

            df_actions = pd.DataFrame(
                actions_array[:min_len],
                columns=tics[:actions_array.shape[1]]
            )
            df_actions.index = dates[:min_len]
            return df_actions
        except Exception as e:
            print(f"⚠️ Ошибка в save_action_memory: {e}")
            return pd.DataFrame()

    def _get_current_step_data(self):
        """
        Вместо свойства создаем метод, который всегда
        возвращает ВСЕ тикеры текущего дня.
        """
        return self.df[self.df.day == self.day]

    def reset(self, seed=None, options=None):
        # 1. Обработка сида для совместимости с новыми версиями Gym/Gymnasium
        if seed is not None:
            super().reset(seed=seed)

        try:
            # 2. АРХИВАЦИЯ: Сохраняем результаты предыдущего прогона (для Шага 3 / Аудита)
            if hasattr(self, 'account_value_memory') and len(self.account_value_memory) > 1:
                self.last_episode_assets = self.account_value_memory.copy()
                self.last_episode_dates = self.date_memory.copy()
                self.last_episode_actions = getattr(self, 'actions_memory', []).copy()

            # 3. СБРОС СОСТОЯНИЯ
            self.day = 0  # Начинаем с самого первого 'day' (0)
            self.amount = self.initial_amount
            # Инициализируем портфель нулями
            self.stocks = [0.0] * self.stock_dim
            self.terminated = False
            self.cost_basis = [0] * self.stock_dim

            # 4. ГЛАВНЫЙ ФИКС ДАННЫХ:
            # Вместо iloc по индексам строк, берем срез по значению колонки 'day'
            self.data = self.df[self.df.day == self.day]

            # Если данных на day=0 нет (ошибка сетки), берем первые stock_dim строк как fallback
            if self.data.empty:
                print(f"⚠️ Warning: Данные на day=0 не найдены, использую iloc fallback")
                self.data = self.df.iloc[0: self.stock_dim, :]

            # 5. ИНИЦИАЛИЗАЦИЯ ВЕКТОРА СОСТОЯНИЯ
            # Используем ваш надежный _safe_get_state_vector через _initiate_state
            self.state = self._initiate_state()

            curr_close = self.data['close'].values.astype(float)
            # На старте акции = 0, поэтому assets_before == initial_amount
            self.assets_before = self.initial_amount

            # 6. ОЧИСТКА ПАМЯТИ
            self.account_value_memory = [self.initial_amount]
            self.actions_memory = []
            self.date_memory = [self._get_date()]
            self.rewards_memory = []

            # Дополнительные метрики FinRL (если используются)
            self.cost = 0
            self.trades = 0
            # print(f"  [CASH LEFT]: {self.amount:.2f} | [TOTAL ASSETS]: {self.assets_before:.2f}")
            # Возвращаем вектор состояния и пустой словарь info (стандарт Gymnasium)
            return np.array(self.state, dtype=np.float32), {}

        except Exception as e:
            print(f"❌ Критическая ошибка в reset: {e}")
            import traceback
            # print(traceback.format_exc())
            # Возвращаем нулевой вектор нужной размерности, чтобы не прерывать пайплайн
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def _get_date(self):
        """Безопасное извлечение даты без вызова .unique() или .values"""
        try:
            d = self.data['date']
            # Если это Series
            if hasattr(d, 'iloc'): return pd.to_datetime(d.iloc[0])
            # Если это массив
            if isinstance(d, (np.ndarray, list)): return pd.to_datetime(d[0])
            # Если это уже строка/Timestamp
            return pd.to_datetime(d)
        except:
            return pd.Timestamp('2000-01-01')

    def _initiate_state(self):
        # Инициализируем через наш безопасный сборщик
        return np.array(self._safe_get_state_vector(is_init=True), dtype=np.float32)

    def _update_state(self):
        # Обновляем через наш безопасный сборщик
        return np.array(self._safe_get_state_vector(is_init=False), dtype=np.float32)

    def _safe_get_state_vector(self, is_init=True):
        """Сборка вектора из АКТУАЛЬНЫХ данных кошелька"""

        # 1. Получаем данные и сортируем их ОДИН РАЗ
        curr_data = self.df[self.df.day == self.day].sort_values('tic')

        if curr_data.empty:
            # Возврат заглушки, если данные кончились
            return [float(self.amount)] + ([0.0] * self.stock_dim * 2) + (
                        [0.0] * len(self.tech_indicator_list) * self.stock_dim)

        # Проверка на полноту данных (все ли тикеры на месте)
        if len(curr_data) != self.stock_dim:
            print(f"⚠️ Ошибка: на шаг {self.day} пришло {len(curr_data)} тикеров вместо {self.stock_dim}")

        # 2. ДЕНЬГИ И АКЦИИ (Берем из self.amount и self.stocks)
        # Это сердце исправления: теперь изменения в step() попадут в нейросеть
        cash = [float(self.amount)] if not is_init else [float(self.initial_amount)]
        shares = [float(s) for s in self.stocks] if not is_init else [0.0] * self.stock_dim

        # 3. ЦЕНЫ (CLOSE)
        closes = curr_data['close'].values.astype(float).tolist()
        # Если тикеров меньше, дополняем нулями (защита от вылета)
        if len(closes) < self.stock_dim:
            closes += [0.0] * (self.stock_dim - len(closes))

        # 4. ТЕХ. ИНДИКАТОРЫ
        techs = []
        for tech in self.tech_indicator_list:
            if tech in curr_data.columns:
                tech_values = curr_data[tech].values.astype(float).tolist()
                # Дополняем до stock_dim, если данных не хватает
                if len(tech_values) < self.stock_dim:
                    tech_values += [0.0] * (self.stock_dim - len(tech_values))
                techs.extend(tech_values)
            else:
                techs.extend([0.0] * self.stock_dim)

        # Сборка финального вектора: [Cash, Closes..., Shares..., Techs...]
        state_vector = cash + closes + shares + techs

        return np.nan_to_num(state_vector, nan=0.0).tolist()