import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from finrl.agents.stablebaselines3.models import DRLAgent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FinRLBacktester:
    def __init__(self, env_class, save_path="./"):
        self.env_class = env_class
        self.save_path = save_path

    def run_ensemble_comparison(self, test_df, trained_models_map, env_kwargs, fold_idx):
        print(f"\n{'=' * 60}\n🚀 ФИНАЛЬНЫЙ БЭКТЕСТ АНСАМБЛЯ (ФОЛД №{fold_idx})\n{'=' * 60}")

        results = {}
        initial_amount = env_kwargs.get('initial_amount', 1000000)

        # 1. РАСЧЕТ BASELINE (Рыночный индекс без INV-инструментов)
        # Берем только обычные акции для понимания динамики рынка
        market_df = test_df[~test_df['tic'].str.contains('INV', case=False)].copy()

        # Создаем таблицу цен: строки - даты, колонки - тикеры
        pivot_df = market_df.pivot(index='date', columns='tic', values='close').sort_index()

        # Считаем рост: цена сегодня / цена в первый день. Затем среднее по всем акциям.
        market_growth = (pivot_df / pivot_df.iloc[0]).mean(axis=1)

        df_baseline = pd.DataFrame({
            'date': pd.to_datetime(market_growth.index),
            'account_value': market_growth.values * initial_amount
        })

        # Подготовка графика
        fig, ax1 = plt.subplots(figsize=(16, 9))
        ax2 = ax1.twinx()  # Вторая ось для доли INV (хеджа)

        # Рисуем рынок (черный пунктир)
        ax1.plot(df_baseline['date'], df_baseline['account_value'],
                 label='MARKET (Buy & Hold SBER)', color='black', lw=2, linestyle='--')

        # 2. ПРОГОН МОДЕЛЕЙ (Извлечение данных из ModifiedStockTradingEnv)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, (name, model) in enumerate(trained_models_map.items()):
            print(f"📊 Тестируем {name.upper()}...")
            try:
                # Создаем среду
                e_test_gym = self.env_class(df=test_df, **env_kwargs)

                # РУЧНОЙ ЦИКЛ (Гарантирует сбор всех данных из памяти Env)
                obs, _ = e_test_gym.reset() if isinstance(e_test_gym.reset(), tuple) else (e_test_gym.reset(), None)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _, _ = e_test_gym.step(action)

                # ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ ModifiedStockTradingEnv
                df_acc = e_test_gym.save_asset_memory()  # Кастомный метод из Env
                df_hedge = e_test_gym.save_hedge_memory()  # Кастомный метод из Env (процент INV)

                df_acc['date'] = pd.to_datetime(df_acc['date'])
                df_hedge['date'] = pd.to_datetime(df_hedge['date'])

                current_color = colors[idx % len(colors)]

                # Отрисовка Equity (левая ось)
                ax1.plot(df_acc['date'], df_acc['account_value'],
                         label=f'Model: {name.upper()}', color=current_color, lw=2)

                # Отрисовка Доли INV (правая ось - заливка)
                ax2.fill_between(df_hedge['date'], 0, df_hedge['hedge_fraction'],
                                 color=current_color, alpha=0.1, label=f'Hedge % ({name.upper()})')

                results[name] = df_acc

            except Exception as e:
                import traceback
                print(f"❌ Ошибка в модели {name}: {e}")
                print(traceback.format_exc())

        # Оформление графика
        ax1.set_title(f'Performance vs Protection Strategy (Fold {fold_idx})', fontsize=15)
        ax1.set_ylabel('Portfolio Value (RUB)')
        ax2.set_ylabel('INV Assets Share (0.0 - 1.0)', color='gray')
        ax2.set_ylim(0, 1.1)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)
        plt.show()

        # 3. ТАБЛИЦА (Передаем df_baseline явно)
        self._print_summary_table(results, df_baseline, initial_amount)
        return results

    def _print_summary_table(self, results, df_baseline, initial_amount):
        print("\n🏆 СВОДНАЯ ТАБЛИЦА:")
        print("-" * 80)

        m_final = df_baseline['account_value'].iloc[-1]
        m_ret = ((m_final / initial_amount) - 1) * 100

        for name, df in results.items():
            f_val = df['account_value'].iloc[-1]
            t_ret = ((f_val / initial_amount) - 1) * 100
            diff = t_ret - m_ret
            print(f"🔹 {name.upper():<10} | Итог: {f_val:>12,.0f} | Доход: {t_ret:>8.2f}% | vs Market: {diff:>+8.2f}%")

        print("-" * 80)
        print(f"🏁 MARKET    | Итог: {m_final:>12,.0f} | Доход: {m_ret:>8.2f}% | vs Market: 0.00%")

