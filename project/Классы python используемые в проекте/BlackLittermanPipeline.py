import pandas as pd
import numpy as np
import warnings
from sklearn.covariance import LedoitWolf
import riskfolio as rp
from numpy.linalg import LinAlgError


class BlackLittermanPipeline:
    def __init__(self, tau=0.05, conf_base=0.1, short_limit=-0.5, min_data_points=30):
        self.tau = tau
        self.conf_base = conf_base
        self.short_limit = short_limit
        self.min_data_points = min_data_points

    def _validate_data(self, df_returns_window):
        if df_returns_window.empty:
            return False, "Empty returns data"
        if len(df_returns_window) < self.min_data_points:
            return False, f"Insufficient data: {len(df_returns_window)}"

        # Очистка данных
        df_returns_window = df_returns_window.fillna(df_returns_window.mean())
        if np.isinf(df_returns_window).any().any():
            return False, "Infinite values"
        return True, df_returns_window

    def _safe_matrix_inversion(self, matrix, epsilon=1e-8):
        try:
            reg_matrix = matrix + np.eye(matrix.shape[0]) * epsilon
            return np.linalg.inv(reg_matrix)
        except LinAlgError:
            return np.linalg.pinv(matrix)

    def clean_weights(self, weights, threshold=0.01):
        """
        Убирает веса меньше 1%, пересчитывает остальные, чтобы сумма была 100%.
        """
        try:
            # 1. Обнуляем все, что меньше порога (0.01 = 1%)
            clean_w = weights.copy()
            clean_w[clean_w < threshold] = 0

            # 2. Нормализуем, чтобы сумма весов снова была равна 1.0 (100%)
            if clean_w.sum() > 0:
                clean_w = clean_w / clean_w.sum()

            # 3. Форматируем для вывода в %
            return clean_w.apply(lambda x: f"{x:.2%}")
        except Exception as e:
            print(f"Pipeline function clean_weights error: {e}")

    def prepare_input_data(self, df_test, df_audit, window_size=50):
        """
        Исправленная подготовка данных: корректная обработка индекса даты и тикеров.
        """
        try:
            # 1. Подготовка доходностей
            prices = df_test.pivot(index='date', columns='tic', values='close')
            prices.index = pd.to_datetime(prices.index)
            returns = prices.pct_change().dropna()
            rets_50 = returns.tail(window_size)

            # 2. Подготовка audit_row (учитываем, что date может быть в индексе)
            audit_df = df_audit.copy()
            if 'date' in audit_df.columns:
                audit_df['date'] = pd.to_datetime(audit_df['date'])
                audit_df = audit_df.set_index('date')
            else:
                audit_df.index = pd.to_datetime(audit_df.index)

            audit_df = audit_df.sort_index()
            audit_last_row = audit_df.iloc[-1]

            # 3. Генерация сигналов PPO (строго под тикеры из rets_50)
            current_tickers = rets_50.columns
            current_date = rets_50.index[-1]

            # Генерируем разные mock-сигналы для каждого инструмента (от 0.2 до 0.8)
            np.random.seed(42)  # Для воспроизводимости mock-данных
            mock_values = np.random.uniform(0.2, 0.8, size=len(current_tickers))
            ppo_row = pd.Series(mock_values, index=current_tickers, name=current_date)

            return rets_50, audit_last_row, ppo_row
        except Exception as e:
            print(f"Pipeline function prepare_input_data error: {e}")
            return pd.Series(0, index=df_test.columns)

    def generate_weights(self, df_returns_window, audit_row, ppo_action_row):
        try:
            df_test = df_returns_window.copy()
            df_audit = audit_row.copy()
            if len(ppo_action_row) < 10:
                df_returns_window, audit_row, ppo_action_row = self.prepare_input_data(df_test, df_audit)
            is_valid, result = self._validate_data(df_returns_window)
            if not is_valid:
                return pd.Series(0, index=ppo_action_row.index)
            df_returns_window = result

            # Оценка параметров
            lw = LedoitWolf().fit(df_returns_window)
            cov_pd = pd.DataFrame(lw.covariance_, index=df_returns_window.columns, columns=df_returns_window.columns)
            mu_eq = df_returns_window.mean()

            # Views (PPO)
            views_mu = ((ppo_action_row - 0.5) * 0.05).clip(-0.2, 0.2)
            P = np.eye(len(df_returns_window.columns))
            Q = views_mu.values

            # Omega (Уверенность)
            diag_omega = []
            conf = audit_row.get('confidence', self.conf_base) + 1e-6
            kurt = audit_row.get('kurtosis', 3)
            for tic in df_returns_window.columns:
                r_val = audit_row.get(f'risk_{tic}', audit_row.get('tail_risk', 0.5))
                unc = (r_val / conf) * 0.02 + 1e-8
                if kurt > 1.15: unc *= 2
                diag_omega.append(unc)
            Omega = np.diag(diag_omega)

            # BL Math
            M_inv = self._safe_matrix_inversion(self.tau * cov_pd.values)
            O_inv = self._safe_matrix_inversion(Omega)
            mu_bl = np.linalg.solve(M_inv + P.T @ O_inv @ P, M_inv @ mu_eq.values + P.T @ O_inv @ Q)
            mu_bl_series = pd.Series(mu_bl, index=df_returns_window.columns)

            # Оптимизация
            port = rp.Portfolio(returns=df_returns_window)
            port.mu, port.cov = mu_bl_series, cov_pd

            # Настройка режима кризиса
            is_crisis = (kurt > 1.15 or audit_row.get('tail_risk', 0.5) > 0.6 or conf < 0.05)

            port.allow_shorts = True
            port.lower_short = self.short_limit
            port.upper_short = 0.5
            port.budget = 1.0
            port.upperbound = 0.3 if is_crisis else 0.7

            w = port.optimization(model='Classic', rm='CVaR' if is_crisis else 'MV',
                                  obj='MinRisk' if is_crisis else 'Sharpe', rf=0, hist=True)

            if w is None or w.empty:
                return pd.Series(1 / len(df_returns_window.columns), index=df_returns_window.columns)

            w['weights'] = self.clean_weights(w['weights'])
            # Возвращаем веса как есть (Riskfolio уже применил budget=1.0)
            return w['weights']

        except Exception as e:
            print(f"Pipeline Error: {e}")
            return pd.Series(0, index=df_returns_window.columns)