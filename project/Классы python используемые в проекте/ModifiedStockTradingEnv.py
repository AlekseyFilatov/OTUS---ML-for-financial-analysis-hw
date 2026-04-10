import numpy as np
import pandas as pd
import datetime
import sys
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C, SAC, DDPG, TD3
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import traceback
from lmoments3 import stats
import lmoments3 as lm

# Эмуляция старого интерфейса gym для совместимости с FinRL
sys.modules["gym"] = gym


#gym.Env - среда не FinRL
class ModifiedStockTradingEnv(StockTradingEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, **kwargs):
        super().__init__()  # Для Gymnasium/Gym

        # 1. Данные
        self.df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        self.stock_dim = len(self.df.tic.unique())
        self.state_space = kwargs.get("state_space", 100)

        # 2. Параметры торговли
        self.hmax = kwargs.get("hmax", 500)
        self.initial_amount = kwargs.get("initial_amount", 1000000)
        self.commission_pct = kwargs.get("commission_pct", 0.001)
        self.tech_indicator_list = kwargs.get("tech_indicator_list", [])
        self.amount = kwargs.get("initial_amount", 1000000)
        self.env_kwargs = kwargs

        # 3. Пространства (ЯВНО указываем dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(self.state_space,), dtype=np.float32)

        # 4. Память (инициализируем пустыми, reset их заполнит)
        self.account_value_memory = [self.initial_amount]
        self.actions_memory = []
        self.date_memory = []
        self.hedge_fraction_memory = []  # История доли шортов

        # 5. Первый запуск
        self.reset()

    def _initiate_state(self):
        """Сборка динамического вектора состояния."""
        try:
            d = self.data
            if d.empty: return [0.0] * self.state_space

            # 1. ПОРТФЕЛЬ
            curr_close = np.clip(d['close'].values.flatten()[:self.stock_dim], 0.0, None)
            stocks_np = np.array(self.stocks[:self.stock_dim])
            total_assets = self.amount + np.sum(stocks_np * curr_close) + 1e-8

            cash_ratio = [self.amount / total_assets]
            stock_shares = (stocks_np * curr_close) / total_assets

            # 2. ТАЛЕБ
            l_kurt, drawdown = self._get_taleb_metrics()
            taleb_features = [
                np.clip((l_kurt - 0.1226) * 5.0, -1.0, 1.0),
                np.clip(drawdown, 0.0, 1.0)
            ]

            # 3. ИНДИКАТОРЫ
            tech_states = []
            for tech in self.tech_indicator_list:
                if tech in d.columns:
                    vals = d[tech].values.flatten()[:self.stock_dim]
                    if 'rsi' in tech.lower():
                        tech_states.extend(vals / 100.0)
                    else:
                        std_val = np.std(vals)
                        scale = (std_val * 2.0) if std_val > 1e-6 else 10.0
                        tech_states.extend(np.tanh(vals / scale))
                else:
                    tech_states.extend([0.0] * self.stock_dim)

            # 4. СБОРКА (Гарантирует размер self.state_space)
            state_vec = np.concatenate([
                cash_ratio,  # 1
                stock_shares,  # stock_dim
                taleb_features,  # 2
                np.log1p(curr_close) / 5.0,  # stock_dim
                tech_states  # indicators * stock_dim
            ]).astype(np.float32)

            return np.nan_to_num(state_vec, nan=0.0).tolist()
        except Exception as e:
            logger.error(f"State Error: {e}")
            return [0.0] * self.state_space

    def _get_taleb_metrics(self):
        """
        Расчёт метрик «Жирных хвостов» по Насиму Талебу с использованием lmoments3.
        """
        # Константы для нормального распределения
        NORMAL_L_KURTOSIS = 0.122601

        # Если истории мало, возвращаем нейтральные значения
        if len(self.account_value_memory) < 20:
            return NORMAL_L_KURTOSIS, 0.0

        try:
            # Берём историю доходностей портфеля (скользящее окно)
            window = self.env_kwargs.get('taleb_window', 30)
            mem = np.array(self.account_value_memory[-window:])

            if len(mem) < 5:
                return NORMAL_L_KURTOSIS, 0.0

            # 1. ЛОГ-ДОХОДНОСТИ (Без фильтрации Z-score для сохранения "хвостов")
            returns = np.log(mem[1:] / (mem[:-1] + 1e-8))
            returns = returns[np.isfinite(returns)]

            if len(returns) < 5 or np.allclose(returns, 0, atol=1e-12):
                return NORMAL_L_KURTOSIS, 0.0

            # 2. КОРРЕКТНЫЙ РАСЧЕТ ЧЕРЕЗ LMOMENTS3
            # lmom_ratios возвращает массив: [L1, L2, tau_3, tau_4]
            try:
                ratios = lm.lmom_ratios(returns, nmom=4)
                # tau_4 (индекс 3) — это и есть L-Kurtosis
                l_kurtosis = float(ratios[3])

                # Ограничиваем физически возможным диапазоном для устойчивости RL
                l_kurtosis = np.clip(l_kurtosis, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Ошибка lmoments3: {e}. Используем дефолт.")
                l_kurtosis = NORMAL_L_KURTOSIS

            # 3. ТЕКУЩАЯ ПРОСАДКА (Drawdown)
            recent_window = self.env_kwargs.get('drawdown_window', 30)
            recent_peak = np.max(self.account_value_memory[-recent_window:])
            current_value = self.account_value_memory[-1]

            drawdown = (recent_peak - current_value) / (recent_peak + 1e-8)
            drawdown = np.clip(drawdown, 0.0, 1.0)

            return float(l_kurtosis), float(drawdown)

        except Exception as e:
            logger.error(f"Критическая ошибка в _get_taleb_metrics: {e}\n{traceback.format_exc()}")
            return NORMAL_L_KURTOSIS, 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            # Архивация данных перед сбросом
            if len(self.account_value_memory) > 1:
                self.last_episode_assets = self.account_value_memory.copy()
                self.last_episode_dates = self.date_memory.copy()
                self.last_episode_actions = self.actions_memory.copy()

            self.day = 0
            self.amount = self.initial_amount
            self.stocks = [0.0] * self.stock_dim
            self.terminate = False

            # Начальный срез данных
            self.data = self.df.iloc[self.day * self.stock_dim: (self.day + 1) * self.stock_dim, :]
            self.state = self._initiate_state()

            # Сброс памяти
            self.account_value_memory = [self.initial_amount]
            self.actions_memory = []
            self.date_memory = [self._get_date()]

            return np.array(self.state, dtype=np.float32), {}
        except Exception as e:
            print(f"⚠️ Ошибка в reset: {e}")
            return np.zeros(self.state_space, dtype=np.float32), {}

    def save_hedge_memory(self):
        return pd.DataFrame({'date': self.date_memory[1:], 'hedge_fraction': self.hedge_fraction_memory})

    def _get_taleb_risk_params(self, l_kurt):
        # Если Куртозис > 0.1226, мы входим в зону "Черных лебедей"
        # risk_zone = max(0, l_kurt - 0.1226)
        # Даем агенту "дышать", если риск умеренно повышен (до 0.15)
        risk_zone = max(0, l_kurt - 0.15)
        # risk_cap = np.clip(np.exp(-risk_zone * 8), 0.4, 1.0)  # Режем лонги
        risk_cap = np.clip(1.0 - max(0.0, l_kurt - 0.1226) * 2.0, 0.4, 1.0)
        inv_mult = 1.0 + (risk_zone * 5.0)  # Разрешаем больше шортов-страховок
        # margin_req = 1.0 + (risk_zone * 25.0)  # Экспоненциальный резерв кэша под шорт (Margin Call)
        margin_req = 1.0 + (risk_zone * 5.0)  # Снизьте множитель с 25.0 до 5.0
        return risk_cap, inv_mult, margin_req

    def _calculate_taleb_rewards(self, port_ret, market_ret, l_kurt, long_val, short_val):
        bonus = 0.0
        # Награда за "выпуклость" при обвалах
        if market_ret < -0.015:
            protection = port_ret - market_ret
            if protection > 0: bonus += protection * 25.0  # Спасли капитал

        # Плата за шорт-страховку в мирное время (Time Decay)
        if market_ret > 0.005 and short_val > 0:
            bonus -= 0.3  # Небольшой "распад" премии
            self.stats["hedge_penalty"] = 1.0

        return bonus

    def _check_liquidation(self, current_total_assets, l_kurt):
        """Имитация Margin Call при нехватке обеспечения."""
        cash_ratio = self.amount / (current_total_assets + 1e-8)

        # Коэффициент агрессивности брокера растет при "жирных хвостах" (L-Kurtosis)
        # 24 февраля брокеры могли требовать 100% и более обеспечения
        required_maintenance_margin = 0.2 + (max(0, l_kurt - 0.1226) * 10.0)

        if cash_ratio < required_maintenance_margin:
            # Штраф за хрупкость (Margin Call)
            # Это то самое "сгорание" шортиста
            liquidation_penalty = -5.0  # Максимальный штраф
            self.stats["liquidation_risk"] = 1.0
            return liquidation_penalty
        return 0.0

    def _check_cash_fragility(self, current_total_assets, l_kurt):
        """Штраф за нехватку кэша перед 'черным лебедем'."""
        cash_ratio = self.amount / (current_total_assets + 1e-8)
        # Если риск высокий, а кэша < 30% — это хрупкость (смерть от Margin Call)
        if l_kurt > 0.14 and cash_ratio < 0.3:
            return -3.0  # Штраф за хрупкость (нет резерва на Margin Call)
        return 0.0

    def _apply_short_squeeze_risk(self, market_ret, short_val, current_total_assets, l_kurt):
        """
        Имитация Short Squeeze с учётом L‑kurtosis:
        - Высокий L‑kurtosis усиливает риск сквиза (жирные хвосты, каскадные закрытия).
        - Штрафы и проскальзывание зависят от волатильности.
        """
        squeeze_penalty = 0.0
        slippage_loss = 0.0

        if market_ret > 0.02 and short_val > 0:
            short_weight = short_val / (current_total_assets + 1e-8)

            # 1. Динамический множитель сквиза: растёт с L‑kurtosis
            # При L‑kurt > 0.1226 (зона «чёрных лебедей») риск резко возрастает
            risk_zone = max(0, l_kurt - 0.1226)
            squeeze_multiplier = 1.0 + (risk_zone * 3.0)  # Экспоненциальный рост риска

            # 2. Нелинейный штраф: зависит от веса шортов, силы роста и волатильности
            squeeze_penalty = - (short_weight * market_ret * 50.0 * squeeze_multiplier)

            # Ограничение штрафа: не более −5.0 (чтобы не нарушать баланс награды)
            squeeze_penalty = np.clip(squeeze_penalty, -5.0, 0.0)

            # 3. Динамическое проскальзывание: выше при высоком L‑kurtosis (низкая ликвидность)
            # slippage_coef = 0.005 + (risk_zone * 0.2)  # До 25 % при экстремальной волатильности
            # Было slippage_coef  -> это 20% комиссии!
            # реалистичнее:
            slippage_coef = 0.002 + (risk_zone * 0.05)  # Максимум 5-7%
            slippage_loss = short_val * slippage_coef

            # 4. Статистика
            self.stats["squeeze_penalty"] = squeeze_penalty
            self.stats["squeeze_events"] = 1.0
            self.stats["squeeze_risk_multiplier"] = squeeze_multiplier  # Для анализа

        return squeeze_penalty, slippage_loss

    def step(self, actions):
        self.reward = 0.0
        # 0. Инициализация статистики (все поля на месте)
        self.stats = {k: 0.0 for k in ["hedge_penalty", "inactivity_penalty",
                                       "diversification_bonus",
                                       "taleb_risk_reduction", "trade_efficiency",
                                       "position_concentration",
                                       "concentration_penalty", "liquidation_risk",
                                       "squeeze_events",
                                       "squeeze_penalty", "squeeze_risk_multiplier"]}
        self.stats["l_kurtosis"] = 0.1226

        try:
            # ОТЛАДКА: Первые 5 дней
            if self.day < 5:
                print(f"ДЕНЬ {self.day} | КЭШ: {int(self.amount)} | АКТИВЫ: {self.stocks}")

            # 1. ТЕРМИНАЦИЯ (с проверкой размерности)
            total_unique_days = len(self.df) // self.stock_dim
            if self.day >= total_unique_days - 1:
                self.terminate = True
                return np.array(self.state, dtype=np.float32), 0.0, True, False, {}

            # 2. ПОДГОТОВКА ДАННЫХ И ЖЕСТКАЯ ПРОВЕРКА
            current_step_date = self._get_date()
            current_tics = self.data['tic'].values[:self.stock_dim]
            curr_close = self.data['close'].values.flatten()[:self.stock_dim]

            if len(curr_close) < self.stock_dim:
                logger.warning(f"Нехватка данных на день {self.day}: ожидается {self.stock_dim}")
                return np.array(self.state, dtype=np.float32), -1.0, False, False, {}

            min_len = min(len(actions), len(self.stocks), len(curr_close))
            curr_close_valid = curr_close[:min_len]
            tic_strings = [str(tic) for tic in current_tics[:min_len]]

            # 3. ТАЛЕБ-РИСК ПАРАМЕТРЫ
            l_kurt, current_sigma = self._get_taleb_metrics()
            # Экстремальный риск (Черный лебедь > 6 сигм или Эксцесс > 20)
            is_black_swan = (l_kurt > 20.0) or (abs(current_sigma) > 6.0)
            if is_black_swan:
                # При лебеде зарезаем риск до минимума (почти кэш)
                risk_cap = 0.05
                self.stats["taleb_risk_reduction"] = 0.95
                logger.warning(
                    f"BLACK SWAN DETECTED! Kurtosis: {l_kurt:.2f}, Sigma: {current_sigma:.2f}. Emergency exit.")
            else:
                # стандартная формула
                #risk_cap = np.clip(1.0 - max(0.0, l_kurt - 0.1226) * 2.0, 0.4, 1.0)
                risk_cap, inv_mult, margin_req = self._get_taleb_risk_params(l_kurt)

            self.stats["l_kurtosis"] = l_kurt
            #risk_cap, inv_mult, margin_req = self._get_taleb_risk_params(l_kurt)
            self.stats["taleb_risk_reduction"] = 1.0 - risk_cap

            # --- 4. ИСПОЛНЕНИЕ СДЕЛОК (с поправкой на панику) ---
            # Если обнаружен лебедь, принудительно переводим все actions в "SELL" (-1.0)
            if is_black_swan:
                actions = np.ones(min_len) * -1.0
            else:
                actions = np.clip(np.array(actions).flatten(), -1.0, 1.0)[:min_len]

            # 4. ФОРМИРОВАНИЕ ОЧЕРЕДЕЙ (Параллельная логика)
            # actions = np.clip(np.array(actions).flatten(), -1.0, 1.0)[:min_len]
            sell_orders, buy_orders = [], []
            base_comm, inv_comm = 0.001, 0.015

            for i in range(min_len):
                price = float(curr_close_valid[i])
                if price <= 1e-6 or price < 0: continue  # Полная защита данных

                is_inv = "INV" in tic_strings[i]
                c_comm = inv_comm if is_inv else base_comm

                if actions[i] < -0.01 and self.stocks[i] > 0:
                    to_sell = int(abs(actions[i]) * self.stocks[i])
                    if to_sell >= 1:
                        sell_orders.append((i, to_sell, price, c_comm))
                elif actions[i] > 0.01:
                    buy_orders.append((i, price, c_comm, is_inv))

            # --- ИСПОЛНЕНИЕ ПРОДАЖ ---
            total_sale_proceeds = 0.0
            for i, to_sell, price, c_comm in sell_orders:
                sale_amount = to_sell * price * (1 - c_comm)
                total_sale_proceeds += sale_amount
                self.stocks[i] -= to_sell
            self.amount += total_sale_proceeds

            # --- ИСПОЛНЕНИЕ ПОКУПОК ---
            available_cash_base = self.amount
            total_buy_cost = 0.0
            target_allocation_pct = 0.15

            for i, price, c_comm, is_inv in buy_orders:
                # Дифференцированный лимит: шорты растут при волатильности, лонги падают
                eff_cap = target_allocation_pct * (risk_cap if not is_inv else inv_mult)
                # intent_amount = available_cash_base * eff_cap
                intent_amount = available_cash_base * min(eff_cap, 0.2)  # Не более 20% кэша в один тикер за раз

                # Округление с учетом частичных лотов
                to_buy = int(intent_amount // (price * (1 + c_comm)))
                if (intent_amount % (price * (1 + c_comm))) > price * 0.5:
                    to_buy += 1

                cost = to_buy * price * (1 + c_comm)
                # РЕЗЕРВ (Margin Call 24.02): блокируем кэш под Short
                required_cash = cost * (margin_req if is_inv else 1.0)

                if self.amount >= required_cash and to_buy > 0:
                    self.amount -= cost
                    total_buy_cost += cost
                    self.stocks[i] += to_buy

            self.stats["trade_efficiency"] = total_buy_cost / (available_cash_base + 1e-8)

            # 5. ПЕРЕХОД К "ЗАВТРА"
            post_trade_assets = self.amount + np.sum(np.array(self.stocks[:min_len]) * curr_close_valid)
            self.day += 1
            self.data = self.df.iloc[self.day * self.stock_dim: (self.day + 1) * self.stock_dim, :]
            new_close = self.data['close'].values.flatten()[:min_len]

            current_total_assets = self.amount + np.sum(np.array(self.stocks[:min_len]) * new_close)
            self.account_value_memory.append(current_total_assets)
            self.date_memory.append(current_step_date)

            # Считаем долю шортов (INV) в текущем портфеле
            short_val = sum(self.stocks[i] * new_close[i] for i in range(min_len) if "INV" in tic_strings[i])
            self.hedge_fraction_memory.append(short_val / (current_total_assets + 1e-8))

            # 6. РАСЧЁТ НАГРАДЫ (Alpha + Beta)
            port_ret = (current_total_assets - post_trade_assets) / (post_trade_assets + 1e-8)
            valid_rets = [(n - o) / o for o, n in zip(curr_close_valid, new_close) if o > 1e-8]
            market_ret = np.mean(valid_rets) if valid_rets else 0.0

            step_reward = (port_ret - market_ret) * 250
            if port_ret > 0:
                step_reward += port_ret * 100 * min(current_total_assets / 5000.0, 2.0)

            # 7. ШТРАФЫ И БОНУСЫ (с заполнением stats)
            pos_values = [self.stocks[i] * new_close[i] for i in range(min_len)]
            long_val = sum(pos_values[i] for i in range(min_len) if "INV" not in tic_strings[i])
            short_val = sum(pos_values[i] for i in range(min_len) if "INV" in tic_strings[i])

            # Диверсификация
            active_pos_count = sum(1 for s in self.stocks[:min_len] if s > 0)

            # Бонус за "Шкуру в игре" (Skin in the Game)
            if active_pos_count > 0:
                # Даем ощутимый бонус за сам факт владения активами
                step_reward += 1.0
                if active_pos_count >= 3:
                    step_reward += 0.3
                    self.stats["diversification_bonus"] = 1.0

            if active_pos_count > 0 and self.day > 0:
                step_reward += 0.1  # Маленький "пряник" за активность

            # Бездействие (Idle)
            if sum(self.stocks) == 0:
                if is_black_swan:
                    step_reward += 1.0  # Поощряем сохранение капитала
                    self.stats["inactivity_penalty"] = -1.0  # Отрицательный штраф = бонус
                else:
                    if market_ret > 0.002:
                        # step_reward -= 2.0
                        # Штраф пропорционален упущенной выгоде рынка
                        step_reward -= (market_ret * 200.0)
                        self.stats["inactivity_penalty"] = 2.0
                    # Поощрение за сохранение кэша на падении
                    elif market_ret < -0.002:
                        step_reward += 1.0
                    else:
                        step_reward -= 0.1

            # Концентрация
            if current_total_assets > 1e-8:
                max_p = max(pos_values) if pos_values else 0.0
                max_w = max_p / current_total_assets
                self.stats["position_concentration"] = max_w
                if max_w > 0.4:
                    c_penalty = -2.0 * (max_w - 0.4) / 0.6
                    step_reward += c_penalty
                    self.stats["concentration_penalty"] = c_penalty

            # 7.1.5 SHORT SQUEEZE (Ваша версия с фиксом)
            squeeze_p, squeeze_s = self._apply_short_squeeze_risk(
                market_ret, short_val, current_total_assets, l_kurt)
            step_reward += squeeze_p
            self.amount -= squeeze_s  # Реальный убыток кэша

            # 7.1 Бонус за Антихрупкость
            step_reward += self._calculate_taleb_rewards(port_ret, market_ret, l_kurt, long_val, short_val)

            # 7.2 Хеджирование (Улучшенный флаг)
            if long_val > 0 and short_val > 0:
                # Сохраняем более высокий статус риска (например, от сквиза), если он есть
                current_h = self.stats.get("hedge_penalty", 0.0)
                self.stats["hedge_penalty"] = max(current_h, 1.0)

            # 7.4 Ликвидация (Крайняя мера)
            liq_penalty = self._check_liquidation(current_total_assets, l_kurt)
            step_reward += liq_penalty
            if liq_penalty < 0:
                self.amount *= 0.90  # Более жесткий штраф за Margin Call (10% депо)

            # Выживаемость (Margin Call 24.02)
            step_reward += self._check_cash_fragility(current_total_assets, l_kurt)
            self.amount = max(0.0, self.amount)

            # 8. ФИНАЛИЗАЦИЯ
            self.reward = np.clip(step_reward, -5.0, 5.0)
            # ЗАПИСЬ ДЕЙСТВИЙ (Чтобы память не была пустой)
            self.actions_memory.append(actions)

            # ОБНОВЛЕНИЕ СОСТОЯНИЯ
            self.state = self._initiate_state()

            # Проверка на завершение данных
            total_unique_days = len(self.df) // self.stock_dim
            is_done = self.day >= total_unique_days - 1

            if is_done:
                print(f"--- Тест завершен: {current_step_date} | Итого: {int(current_total_assets)} ---")

            return np.array(self.state, dtype=np.float32), float(self.reward), False, False, \
                {"assets": current_total_assets, "date": current_step_date, "stats": self.stats, "cash": self.amount}

        except Exception as e:
            import traceback
            logger.error(f"Error Day {self.day}: {e}\n{traceback.format_exc()}")
            return np.array(self.state, dtype=np.float32), -1.0, True, False, {}

    def save_asset_memory(self):
        try:
            # Проверяем, есть ли данные в текущей памяти.  # Если там только 1 запись (после reset), берем архив последнего эпизода. assets = self.account_value_memory
            assets = self.account_value_memory
            dates = self.date_memory

            if len(assets) <= 1 and hasattr(self, 'last_episode_assets'):
                assets = self.last_episode_assets
                dates = self.last_episode_dates

            min_len = min(len(assets), len(dates))
            df_acc_value = pd.DataFrame({
                "date": dates[:min_len],
                "account_value": assets[:min_len]
            })

            df_acc_value['date'] = pd.to_datetime(df_acc_value['date'])
            df_acc_value = df_acc_value.drop_duplicates(subset=['date'], keep='first')
            df_acc_value = df_acc_value.dropna(subset=['account_value'])
            df_acc_value = df_acc_value.sort_values("date").reset_index(drop=True)

            return df_acc_value
        except Exception as e:
            print(f"⚠️ Ошибка в save_asset_memory: {e}")
            return pd.DataFrame() # Возвращаем пустой DF вместо ошибки

    def save_action_memory(self) -> pd.DataFrame:
        """
        Сохраняет историю действий агента в виде DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame с действиями по активам, где:
            - индекс: даты из date_memory;
            - колонки: тикеры активов;
            - значения: доли распределения капитала (0–1).
            При ошибке или отсутствии данных возвращает пустой DataFrame.
        """
        try:
            # Получаем действия — сначала из основного буфера, затем из запасного
            if hasattr(self, 'actions_memory') and len(self.actions_memory) > 0:
                actions = self.actions_memory
            elif hasattr(self, 'last_episode_actions'):
                actions = getattr(self, 'last_episode_actions', [])
            else:
                logging.info("Нет данных о действиях агента для сохранения")
                return pd.DataFrame()

            if not actions:
                logging.info("Пустой буфер действий")
                return pd.DataFrame()

            # Проверяем соответствие длин действий и дат
            actions_array = np.array(actions)
            num_actions = len(actions_array)
            available_dates = len(self.date_memory)

            if num_actions > available_dates:
                logging.warning(
                    f"Недостаточно дат для {num_actions} действий. Доступно: {available_dates}"
                )
                dates = self.date_memory  # используем все доступные даты
            else:
                dates = self.date_memory[:num_actions]

            # Определяем количество активов и тикеров
            actions_array = np.atleast_2d(actions)
            num_cols = actions_array.shape[1] # Сколько колонок выдала нейросеть по факту

            if hasattr(self, 'df') and self.df is not None:
                available_tics = self.df.tic.unique()
                # Берем столько тикеров, сколько есть колонок в действиях
                if len(available_tics) >= num_cols:
                    tics = available_tics[:num_cols]
                else:
                    # Если тикеров в данных меньше, чем действий (странно, но бывает в VecEnv)
                    tics = [f'ASSET_{i}' for i in range(num_cols)]
            else:
                tics = [f'ASSET_{i}' for i in range(num_cols)]

            # Создаём DataFrame
            #df_actions = pd.DataFrame(actions_array, columns=tics)
            #df_actions.index = dates
            min_len_final = min(len(actions_array), len(dates))
            df_actions = pd.DataFrame(actions_array[:min_len_final], columns=tics[:actions_array.shape[1]])
            df_actions.index = dates[:min_len_final]

            # Валидация: проверяем, что все значения в диапазоне [0, 1]
            if not ((df_actions >= 0) & (df_actions <= 1)).all().all():
                logging.warning("Обнаружены действия вне диапазона [0,1]. Применяем clip.")
                df_actions = df_actions.clip(0, 1)

            logging.info(f"Сохранено {num_actions} действий для {num_cols} активов")
            return df_actions

        except Exception as e:
            logging.error(f"Критическая ошибка в save_action_memory: {type(e).__name__}: {e}")
            # Возвращаем пустой DF с колонками тикеров, чтобы не ломать дальнейший код
            return pd.DataFrame(columns=self.df.tic.unique()[:self.stock_dim])

    def _update_state(self):
        """Метод-мостик для внутренних вызовов FinRL"""
        return self._initiate_state()

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def _get_date(self):
        """Единая точка получения даты. Всегда возвращает pd.Timestamp."""
        try:
            # 1. Пытаемся взять из self.data (текущий срез дня)
            if hasattr(self.data, 'date'):
                # Если это Series (несколько строк для разных тикеров)
                d = self.data['date']
                val = d.iloc[0] if hasattr(d, 'iloc') else d
                return pd.to_datetime(val)

            # 2. Если данных в срезе нет, смотрим на индекс
            return pd.to_datetime(self.data.index[0])

        except Exception:
            # 3. Резервный вариант: если данных нет (конец эпизода),
            # берем последнюю дату из памяти и прибавляем день
            if len(self.date_memory) > 0:
                last_date = self.date_memory[-1]
                if isinstance(last_date, (pd.Timestamp, datetime.datetime)):
                    return last_date + pd.Timedelta(days=1)

            # 4. Самый крайний случай (заглушка)
            return pd.Timestamp('2000-01-01') + pd.Timedelta(days=self.day)