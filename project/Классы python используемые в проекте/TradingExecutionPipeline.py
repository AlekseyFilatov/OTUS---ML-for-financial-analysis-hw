import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import riskfolio as rp


class TradingExecutionPipeline:
    def __init__(self, bl_params=None, lot_sizes=None):
        # Параметры Black-Litterman
        self.bl_params = bl_params or {
            'tau': 0.05,
            'conf_base': 0.1,
            'short_limit': -0.5,
            'min_data_points': 30
        }
        # Инициализируем расчетный модуль BL
        self.bl_engine = BlackLittermanPipeline(**self.bl_params)

        # Справочник лотов
        self.lot_sizes = lot_sizes or {
            'SBER_USD': 10, 'SBRF_USD': 10, 'OFZ_FIX': 1,
            'GOLD_USD': 1, 'GOLD_INV': 1, 'SBER_INV': 1
        }

    def execute_cycle(self, df_returns, audit_row, ppo_signals, current_balances, current_prices, total_account):
        """
        Полный цикл: Данные -> Веса BL -> Ордера -> Лоты
        """
        # 1. Генерация целевых весов через Black-Litterman
        target_weights = self.bl_engine.generate_weights(
            df_returns, audit_row, ppo_signals
        )

        if target_weights.sum() == 0 and not target_weights.empty:
            print("Предупреждение: Получены нулевые веса. Ребалансировка пропущена.")
            return None
        # print("Сырые веса от оптимизатора:", target_weights)
        # 2. Расчет базовых ордеров (с учетом текущих позиций и Long/Short)
        orders_df = self._calculate_raw_orders(
            target_weights, current_balances, current_prices, total_account
        )

        # 3. Применение кратности лотов
        final_orders = self._apply_lot_constraints(orders_df)

        # 4. Формирование финального плана действий для робота
        trading_plan = self._generate_action_plan(final_orders)

        return trading_plan

    def _calculate_raw_orders(self, tw, balances, prices, total_cash):
        """Внутренний метод расчета дельты позиций"""
        # 1. Принудительное приведение к числам (float)
        # errors='coerce' превратит строки, которые нельзя сконвертировать, в NaN
        # Очистка весов: убираем '%' и конвертируем в float
        if tw.dtype == 'object':
            tw = tw.str.replace('%', '', regex=False).astype(float) / 100
        else:
            tw = pd.to_numeric(tw, errors='coerce').fillna(0)

        tw = pd.to_numeric(tw, errors='coerce').fillna(0)

        # 2. Убираем шум (меньше 0.5%), сохраняя знаки для Long/Short
        tw = tw.copy()
        tw[tw.abs() < 0.005] = 0

        orders = []
        for tic in tw.index:
            price = prices.get(tic, 0)
            if price <= 0: continue

            target_val = total_cash * tw[tic]
            current_qty = balances.get(tic, 0)
            current_val = current_qty * price

            delta_val = target_val - current_val
            # Используем отсечение дробной части (trunc) вместо округления
            delta_qty = int(np.trunc(delta_val / price))

            orders.append({
                'ticker': tic,
                'target_weight': tw[tic],
                'current_qty': current_qty,
                'order_qty': delta_qty,
                'order_value': delta_val
            })
        return pd.DataFrame(orders).set_index('ticker')

    def _apply_lot_constraints(self, df):
        """Корректировка с учетом лотов"""
        df = df.copy()
        df['lot_size'] = df.index.map(lambda x: self.lot_sizes.get(x, 1))

        # Округляем количество в сторону нуля до ближайшего целого лота
        df['order_qty_lots'] = (np.trunc(df['order_qty'] / df['lot_size']) * df['lot_size']).astype(int)

        return df[df['order_qty_lots'] != 0]

    def _generate_action_plan(self, df):
        """Создание читаемых команд для робота"""
        if df.empty: return pd.DataFrame()

        plan = df.copy()
        plan['Action'] = plan['order_qty_lots'].apply(
            lambda x: 'BUY/COVER' if x > 0 else 'SELL/SHORT'
        )
        plan['Qty_to_Trade'] = plan['order_qty_lots'].abs()

        # Форматирование для вывода
        plan['target_weight'] = plan['target_weight'].apply(lambda x: f"{x:.2%}")
        plan['order_value'] = plan['order_value'].round(2)

        return plan[['Action', 'target_weight', 'Qty_to_Trade', 'order_value']].sort_values(by='Action')