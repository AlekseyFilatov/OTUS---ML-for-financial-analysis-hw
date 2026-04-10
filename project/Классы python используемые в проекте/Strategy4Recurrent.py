import pandas as pd
import numpy as np
import talib as ta
import traceback
import statsmodels.formula.api as smf
from scipy.stats import genpareto
from scipy.stats.mstats import winsorize
from arch import arch_model
from lmoments3 import stats
import lmoments3 as lm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import RobustScaler
from scipy.stats import entropy
import logging

# В lmoments3 основные расчеты лежат здесь:

from sklearn.base import BaseEstimator, TransformerMixin
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn import set_config

set_config(transform_output="pandas")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Класс для базовой стратегии
class BaseStrategy:
    def __init__(self):
        self.positions = []
        self.equity = []
        self.ticker = None

    def calculate_metrics(self, data):
        """Описание используемых метрик производительности:
        final_equity: Конечный капитал после завершения торгового периода. Показывает, насколько вырос или уменьшился начальный капитал в результате торговли.
        Используется для оценки общей эффективности стратегии. Чем выше конечный капитал, тем успешнее стратегия.
        n_trades: Количество сделок, совершенных за период тестирования. Показывает активность стратегии. Большое количество сделок может указывать на высокую частоту торговли, что может быть как преимуществом, так и недостатком в зависимости от контекста.
        returns: Общая доходность стратегии в процентах за период тестирования. Основной показатель прибыльности стратегии. Положительная доходность указывает на успешность стратегии, отрицательная — на убыточность.
        max_drawdown: Максимальная просадка капитала за период тестирования. Показывает наибольшее падение капитала от пика до минимума. Используется для оценки риска стратегии. Чем меньше просадка, тем стабильнее стратегия.
        win_rate: Процент прибыльных сделок от общего количества сделок. Показывает, насколько часто стратегия приносит прибыль. Высокий процент выигрышных сделок может указывать на стабильность стратегии.
        sharpe_ratio: Коэффициент Шарпа измеряет доходность с поправкой на риск. Чем выше значение, тем лучше стратегия с точки зрения доходности на единицу риска. Используется для сравнения стратегий с учетом риска. Высокий коэффициент Шарпа указывает на эффективное использование риска.
        sortino_ratio: Коэффициент Сортино, который измеряет доходность с поправкой на риск, учитывая только негативную волатильность. Аналогичен коэффициенту Шарпа, но фокусируется на риске убытков, что делает его более подходящим для оценки стратегий с асимметричным распределением доходности.
        profit_factor: Отношение общей прибыли к общему убытку. Показывает, насколько прибыльна стратегия по сравнению с убытками. Используется для оценки соотношения прибыли и убытков. Значение больше 1 указывает на прибыльность стратегии.
        """
        if not self.positions:
            return {
                "final_equity": 1.0,
                "n_trades": 0,
                "returns": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "profit_factor": 0,
            }

        # Расчет базовых метрик
        final_equity = self.equity[-1]
        n_trades = len(self.positions)
        returns = (final_equity - 1.0) * 100

        # Расчет максимальной просадки
        peak = 1.0
        drawdowns = []
        for eq in self.equity:
            if eq > peak:
                peak = eq
            drawdown = (peak - eq) / peak * 100
            drawdowns.append(drawdown)
        max_drawdown = max(drawdowns)

        # Расчет win rate
        profitable_trades = sum(1 for pos in self.positions if pos["profit"] > 0)
        win_rate = profitable_trades / n_trades * 100 if n_trades > 0 else 0

        # Расчет Sharpe и Sortino ratio
        daily_returns = pd.Series(self.equity).pct_change().dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            downside_returns = daily_returns[daily_returns < 0]
            sortino_ratio = (
                np.sqrt(252) * daily_returns.mean() / downside_returns.std()
                if len(downside_returns) > 0
                else 0
            )
        else:
            sharpe_ratio = sortino_ratio = 0

        # Расчет profit factor
        gross_profit = sum(pos["profit"] for pos in self.positions if pos["profit"] > 0)
        gross_loss = abs(
            sum(pos["profit"] for pos in self.positions if pos["profit"] < 0)
        )
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

        return {
            "final_equity": final_equity,
            "n_trades": n_trades,
            "returns": returns,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "profit_factor": profit_factor,
        }


class Strategy4Recurrent(BaseEstimator, TransformerMixin, BaseStrategy):
    def __init__(self, optimize=True, n_trials=15, min_trades_per_fold=2, lags=3, **params):
        # Инициализируем BaseStrategy для доступа к calculate_metrics
        super().__init__()
        self.optimize = optimize
        self.n_trials = n_trials
        self.lags = lags  # Количество прошлых значений для каждого индикатора
        self.min_trades_per_fold = min_trades_per_fold
        self.quantile = 0.5
        self.garch_window = 252
        self.min_periods_for_stats = 5

        # Полный список параметров для оптимизации и совместимости
        self.param_list = [
            "bb_period",
            "bb_std",
            "rsi_period",
            "sma",
            "ema_period",
            "ma_period",
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "stoch_k",
            "stoch_d",
            "stoch_slow",
            "hurst_window",
            "kurtosis_window",
            "alpha_window"
        ]

        defaults = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "sma": 10,
            "ema_period": 4,
            "ma_period": 10,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "stoch_k": 14,
            "stoch_d": 3,
            "stoch_slow": 3,
            "hurst_window": 100,  # Нужно длинное окно
            "kurtosis_window": 30,
            "alpha_window": 60
        }

        # Установка параметров (обязательно через setattr для sklearn)
        for name in self.param_list:
            setattr(self, name, params.get(name, defaults[name]))

        self.best_params_ = None

    def fit(self, X, y=None):
        if not self.optimize:
            return self

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")

        # Оптимизируем, передавая данные X внутрь
        study.optimize(lambda trial: self._objective(trial, X), n_trials=self.n_trials)

        self.best_params_ = study.best_params
        for name, value in self.best_params_.items():
            setattr(self, name, value)

        print(f"Оптимизация завершена. Лучший Sharpe: {study.best_value:.4f}")
        return self

    def predict_tail_boundary(self, df, window=100):
        """Прогноз левой границы хвоста (5% квантиль) через регрессию."""
        if len(df) < window:
            return 0.0

        subset = df.tail(window).copy()
        # Убеждаемся, что работаем с признаками, которые влияют на риск
        required_cols = ['log_ret', 'rvi_ret', 'atr']
        if not all(col in subset.columns for col in required_cols):
            return 0.0

        try:
            # Предсказываем именно нижнюю границу (убытки)
            model = smf.quantreg('log_ret ~ rvi_ret + atr', subset)
            res = model.fit(q=self.quantile)
            prediction = res.predict(subset.tail(1)).values[0]

            # Нас интересует только отрицательная граница (риск падения)
            # Если прогноз -0.05, возвращаем 0.05. Если прогноз > 0, риска нет.
            return max(0.0, -prediction)
        except:
            return 0.0

    def calculate_evt_risk(self, returns, quantile=None):
        """EVT-анализ: Метод превышения порога (POT) для оценки краха."""
        q = quantile if quantile is not None else self.quantile

        # Работаем с убытками (положительные числа)
        losses = -returns.values
        threshold = np.quantile(losses, 1 - q)
        exceedances = losses[losses > threshold] - threshold

        # Минимум 10 точек для стабильной оценки GPD
        if len(exceedances) < 10:
            return 0.0, 0.0

        try:
            # floc=0 фиксирует начало распределения на пороге
            shape, loc, scale = genpareto.fit(exceedances, floc=0)

            # Защита Талеба: если shape >= 1, матожидание убытка бесконечно.
            # Ограничиваем для численной стабильности.
            shape = min(max(shape, 1e-4), 0.99)

            n = len(returns)
            nu = len(exceedances)

            # VaR по методу Peaks-Over-Threshold
            # Рассчитываем для доверительного интервала 99%
            term = (n / nu) * (1 - 0.99)
            evt_var = threshold + (scale / shape) * (term ** (-shape) - 1)

            # CVaR (Expected Shortfall) для GPD распределения
            evt_cvar = (evt_var + scale - shape * threshold) / (1 - shape)

            return float(max(evt_var, 0)), float(max(evt_cvar, 0))
        except:
            return 0.0, 0.0

    def calculate_garch_metrics(self, returns):
        """GARCH(1,1) с t-распределением для улавливания кластеров волатильности."""
        if len(returns) < self.garch_window:
            return 0.0, 0.0

        try:
            # rescale=100 критичен для сходимости оптимизатора arch
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', rescale=False)
            # Ускоряем расчет для RL-цикла через метод 'SLSQP'
            res = model.fit(disp='off', show_warning=False, method='SLSQP')

            # Прогноз волатильности на 1 шаг вперед
            forecast = res.forecast(horizon=1)
            cond_vol = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100

            # nu — степени свободы. Если nu мало (напр. 3), хвосты очень жирные.
            nu = res.params.get('nu', 10.0)
            # Инвертируем: чем меньше nu, тем больше tail_fatness
            tail_fatness = 4.0 / nu if nu > 2.0 else 1.0

            return float(cond_vol), float(tail_fatness)
        except:
            return 0.0, 0.0

    def calculate_rolling_tail_risk(self, dfall, window=252):
        """
        Рассчитывает tail risk для каждой строки со скользящим окном.
        Использует локальные методы класса для расчёта компонентов риска.
        """
        df = dfall.copy()
        risk_scores = []

        for i in range(len(df)):
            if i < window:
                risk_scores.append(0.0)  # Недостаточно данных для расчёта
            else:
                # Берём окно данных до текущей строки
                window_df = df.iloc[i - window:i + 1].copy()
                returns = window_df['log_ret']

                # 1. Считаем компоненты риска, используя локальные методы
                risk_boundary = self.predict_tail_boundary(window_df)
                evt_var, evt_cvar = self.calculate_evt_risk(returns)
                cond_vol, tail_fatness = self.calculate_garch_metrics(returns)

                # 2. Нормализация под MOEX
                risk_components = [
                    min(risk_boundary * 5, 1.0),  # Было 15. Снижаем: 20% падения = макс риск
                    min(evt_cvar * 4, 1.0),  # Было 10. Снижаем: 25% убытка в хвосте = макс риск
                    min(cond_vol * 10, 1.0),  # Было 20. Снижаем: 10% волатильности = макс риск
                    min(tail_fatness, 1.0)  # Жирность хвостов остается
                ]

                # 3. Взвешивание компонентов
                weights = [0.2, 0.3, 0.2, 0.3]
                composite_risk = np.average(risk_components, weights=weights)

                # 4. Нелинейная коррекция (эффект паники при высоком риске)
                if composite_risk > 0.8:
                    composite_risk = composite_risk ** 1.2

                # Гарантируем результат в диапазоне [0, 1] и тип float
                final_risk = float(min(composite_risk, 1.0))
                risk_scores.append(final_risk)

        return risk_scores

    def get_composite_tail_risk(self, dftail):
        """
        Агрегатор хвостового риска.
        Сводит волатильность, экстремальные убытки и прогноз хвостов в 0-1 скор.
        """
        # Очистка данных
        df = dftail.copy()
        clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_ret'])
        returns = clean_df['log_ret']

        # 1. Считаем компоненты
        risk_boundary = self.predict_tail_boundary(clean_df)
        evt_var, evt_cvar = self.calculate_evt_risk(returns)
        cond_vol, tail_fatness = self.calculate_garch_metrics(returns)

        # 2. Нормализация под MOEX (коэффициенты адаптированы под дневную волатильность ~1-3%)
        # Мы масштабируем значения так, чтобы 1.0 означало "катастрофу"
        risk_components = [
            min(risk_boundary * 15, 1.0),  # 7% прогнозируемого падения = макс риск
            min(evt_cvar * 10, 1.0),  # 10% ожидаемого убытка в хвосте = макс риск
            min(cond_vol * 20, 1.0),  # 5% условной волатильности = макс риск
            min(tail_fatness, 1.0)  # Мера жирности хвостов напрямую
        ]

        # 3. Взвешивание (Приоритет на CVaR и Garch)
        # CVaR (0.4) - самый надежный индикатор глубины падения
        # Garch (0.3) - индикатор текущей паники
        # Boundary (0.2) - прогнозная граница
        # Fatness (0.1) - структурный параметр
        weights = [0.2, 0.4, 0.3, 0.1]
        composite_risk = np.average(risk_components, weights=weights)

        # Добавляем нелинейность: если риск выше 0.5, он растет быстрее (эффект паники)
        if composite_risk > 0.5:
            composite_risk = composite_risk ** 1.5

        return float(min(composite_risk, 1.0))

    def _calculate_hurst_base(self, series, window):
        """Показатель Херста: трендовость против шума"""

        def hurst_logic(ts):
            if len(ts) < 20: return 0.5
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0

        return series.rolling(window=int(window)).apply(hurst_logic)

    def _calculate_hurst(self, series, window, scale_factor=5):
        def get_h_fast(x):
            if len(x) < 30: return 0.5
            y = np.diff(np.log(x + 1e-9))
            y = y[~np.isnan(y)]
            if len(y) < 20: return 0.5

            std_short = np.std(y) + 1e-9
            # Считаем сумму доходностей за период (агрегированное движение)
            y_long = np.convolve(y, np.ones(int(scale_factor)), mode='valid')
            std_long = np.std(y_long) + 1e-9

            # Математически верный расчет Hurst через масштабирование волатильности
            # Для случайного блуждания std_long = std_short * sqrt(scale_factor)
            # H = log(std_long / std_short) / log(scale_factor)
            h = np.log(std_long / std_short) / np.log(scale_factor)

            # Ограничиваем и возвращаем
            return np.clip(h, 0.1, 0.9)

        return pd.Series(series).rolling(window=int(window), min_periods=30).apply(get_h_fast).ffill().bfill()

    def _calculate_tail_alpha_slow(self, returns, window):
        """Hill Estimator: Оценка толщины хвостов (Alpha < 2 = высокий риск)"""

        def hill_estimator(x):
            # Берем только отрицательные доходности (левый хвост риска)
            neg_ret = x[x < 0].abs()
            if len(neg_ret) < 10: return 2.0  # Значение для нормального распределения

            # Берем 10% самых экстремальных убытков
            tail = np.sort(neg_ret)[-max(len(neg_ret) // 10, 2):]
            # Hill formula
            alpha = 1 / (np.mean(np.log(tail / (np.min(tail) + 1e-9))) + 1e-9)
            return alpha

        return returns.rolling(window=int(window)).apply(hill_estimator)

    def _calculate_tail_alpha(self, returns, window):
        """Устойчивый Hill Estimator"""

        def hill_simple(x):
            neg = np.abs(x[x < 0])
            if len(neg) < 5: return 2.0
            # Берем 20% самых больших убытков
            threshold = np.percentile(neg, 80)
            tail = neg[neg >= threshold]
            alpha = 1.0 / (np.mean(np.log(tail / (threshold + 1e-9))) + 1e-9)
            return np.clip(alpha, 0, 10)

        res = pd.Series(returns).rolling(window=int(window), min_periods=10).apply(hill_simple)
        return res.ffill().bfill()

    def _calculate_tail_alpha_eazy(self, returns, window):
        def hill_simple(x):
            # Берем модули убытков
            losses = np.abs(x[x < 0])
            if len(losses) < 10: return 2.0
            # Вместо сложной сортировки — берем значения выше среднего убытка
            threshold = np.mean(losses)
            extreme_losses = losses[losses > threshold]
            if len(extreme_losses) < 2: return 2.0
            return 1.0 / np.mean(np.log(extreme_losses / (threshold + 1e-9)))

        return pd.Series(returns).rolling(window=int(window), min_periods=20).apply(hill_simple)

    def add_market_correlation(self, df, window=20, fill_strategy='ffill'):
        """
        Независимая функция для расчёта корреляции RGBI.
        """
        df = df.copy()

        # Нормализуем имена колонок (чтобы не зависеть от регистра Close/close)
        cols = {c.lower(): c for c in df.columns}
        close_col = cols.get('Close')
        rgbi_col = cols.get('rgbi')

        if close_col and rgbi_col:
            # Расчёт доходностей
            asset_ret = df[close_col].pct_change()
            rgbi_ret = df[rgbi_col].pct_change()

            if asset_ret.notna().sum() < window:
                df['corr_close_rgbi'] = 0.0
                df['market_divergence'] = 0.0
                return df

            # Корреляция
            df['corr_close_rgbi'] = asset_ret.rolling(window=window).corr(rgbi_ret).clip(-1, 1)

            # Дивергенция
            diff_ret = asset_ret - rgbi_ret
            rolling_mean = diff_ret.rolling(window).mean()
            rolling_std = diff_ret.rolling(window).std().replace(0, 1e-8)
            df['market_divergence'] = (diff_ret - rolling_mean) / rolling_std

            # Заполнение NaN
            if fill_strategy == 'ffill':
                df[['corr_close_rgbi', 'market_divergence']] = df[
                    ['corr_close_rgbi', 'market_divergence']].ffill().fillna(0)
            elif fill_strategy == 'zero':
                df[['corr_close_rgbi', 'market_divergence']] = df[['corr_close_rgbi', 'market_divergence']].fillna(0)

            return df
        else:
            df['corr_close_rgbi'] = 0.0
            df['market_divergence'] = 0.0
            return df

    def _objective(self, trial, X):
        # 1. Определение пространства поиска
        current_params = {
            "bb_period": trial.suggest_int("bb_period", 10, 50),
            "bb_std": trial.suggest_float("bb_std", 1.0, 3.0),
            "rsi_period": trial.suggest_int("rsi_period", 5, 30),
            "sma": trial.suggest_int("sma", 5, 40),
            "ma_period": trial.suggest_int("ma_period", 10, 50),
            "macd_fast": trial.suggest_int("macd_fast", 8, 20),
            "macd_slow": trial.suggest_int("macd_slow", 20, 40),
            "macd_signal": trial.suggest_int("macd_signal", 5, 12),
            "stoch_k": trial.suggest_int("stoch_k", 10, 20),
            "stoch_d": trial.suggest_int("stoch_d", 2, 5),
            "stoch_slow": trial.suggest_int("stoch_slow", 2, 5),
            # "hurst_window": trial.suggest_int("hurst_window", 50, 200),
            "kurtosis_window": trial.suggest_int("kurtosis_window", 10, 100),
            # "alpha_window": trial.suggest_int("alpha_window", 30, 150),
            "hurst_window": trial.suggest_int("hurst_window", 30, 100),  # Короткое окно для быстрой реакции
            "alpha_window": trial.suggest_int("alpha_window", 30, 100),
            # Попробуйте также оптимизировать сам lags для Recurrent сети
            "lags": trial.suggest_int("lags", 2, 5)
        }

        # Временно применяем параметры
        for name, value in current_params.items():
            setattr(self, name, value)

        # 2. Динамическая настройка TimeSeriesSplit для защиты от ValueError
        n_samples = len(X)
        n_splits = 3
        # Ограничиваем gap, чтобы он не "съел" все данные (не более 20% выборки)
        safe_gap = min(int(self.ma_period), int(n_samples * 0.2))

        # Проверка: достаточно ли данных для (n_splits * test_size + gap)
        # Если данных мало, уменьшаем n_splits до 2 или убираем gap
        if n_samples < (n_splits + 1) * 10 + safe_gap:
            n_splits = 2
            if n_samples < 50:  # Совсем мало данных
                safe_gap = 0

        try:
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=safe_gap)
            fold_scores = []

            for _, val_idx in tscv.split(X):
                # На маленьких выборках берем небольшой "хвост" из train для индикаторов
                # Но для простоты здесь используем только X_val
                X_val = X.iloc[val_idx]

                if len(X_val) < 10:  # Слишком маленькое окно для расчета
                    continue

                signals, _ = self.generate_signals(X_val)
                if signals is None or signals.nunique() <= 1:
                    fold_scores.append(-1.0)
                    continue

                signals = self.taleb_normalize(signals)

                res = self._internal_backtest(signals, X_val)

                if res["n_trades"] < self.min_trades_per_fold:
                    fold_scores.append(-0.5)
                else:
                    fold_scores.append(res["sharpe_ratio"])

            if not fold_scores:
                return -1.0

            return np.nan_to_num(np.mean(fold_scores))

        except ValueError:
            # Последний рубеж защиты, если расчеты выше все равно не прошли
            return -1.0

    def _internal_backtest(self, signals, data):
        """Эмуляция торгового процесса для наполнения BaseStrategy.equity"""
        self.positions = []

        # Расчет доходности: вход на СЛЕДУЮЩЕМ баре после сигнала
        price_returns = data['Close'].pct_change().fillna(0)
        # Shift(1) критически важен: сигнал на закрытии бара t реализуется в доходность бара t+1
        strat_returns = signals.shift(1).fillna(0) * price_returns

        # Наполнение self.equity для метода calculate_metrics
        self.equity = (1 + strat_returns).cumprod().tolist()

        # Имитация сделок для корректного расчета win_rate и n_trades
        trades = signals.diff().fillna(0)
        for i, change in enumerate(trades):
            if change != 0:  # Смена позиции или вход/выход
                self.positions.append({"profit": strat_returns.iloc[i]})

        return self.calculate_metrics(data)

    def transform(self, X):
        """Финальная обработка данных с сохранением служебных колонок"""
        # 1. Генерация сигналов
        conditions, df_enriched = self.generate_signals(X)

        if conditions is None or (isinstance(conditions, pd.DataFrame) and conditions.empty):
            return X

        # 2. Нормализация риск-метрик (Талеб-style)
        df_res = conditions.copy()
        df_res = self.taleb_normalize(df_res)

        # 3. Список колонок для лагов
        cols_to_lag = ['RSI', 'MACD', 'dist_sma', 'hurst_z', 'tail_alpha', 'lkurtosis', 'amihud']
        cols_to_lag = [c for c in cols_to_lag if c in df_res.columns]

        # 4. Генерируем лаги
        lagged_data = [df_res]
        for col in cols_to_lag:
            for i in range(1, self.lags + 1):
                lagged_col = df_res[col].shift(i).rename(f"{col}_lag_{i}")
                lagged_data.append(lagged_col)

        # 5. Собираем признаки
        features_df = pd.concat(lagged_data, axis=1)

        # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ВОЗВРАТ СЛУЖЕБНЫХ КОЛОНОК ---
        # Из исходного X берем то, что нужно для FinRL и анализа
        # Если 'date' была в индексе после reset_index, она может называться 'index'
        service_cols = ['date', 'tic', 'close', 'open', 'high', 'low', 'volume']
        X.columns = [c.lower() for c in X.columns]

        # Проверяем наличие колонок в X (с учетом возможного переименования индекса)
        available_service = [c for c in service_cols if c in X.columns]
        if 'index' in X.columns and 'date' not in available_service:
            X = X.rename(columns={'index': 'date'})
            available_service.append('date')

        # Склеиваем: Служебные колонки + Новые признаки
        # Используем индексы X, чтобы гарантировать совпадение строк
        df_final = pd.concat([X[available_service].reset_index(drop=True),
                              features_df.reset_index(drop=True)], axis=1)

        # Принудительно чистим мусорные колонки индексов
        cols_to_drop = ['index', 'level_0', 'Unnamed: 0']
        df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])

        return df_final.ffill().fillna(0)

    def add_l_kurtosis_to_df(self, df, window=30):
        temp_df = df.copy()

        # 1. Определяем тикер из MultiIndex
        if 'tic' in temp_df.index.names:
            grouper = temp_df.index.get_level_values('tic')
        else:
            grouper = temp_df['tic']

        # 2. УМНЫЙ ВЫБОР КОЛОНКИ ЦЕНЫ
        # Ищем Close, если нет - rgbi, если нет - любую числовую кроме доходностей
        if 'Close' in temp_df.columns:
            price_col = 'Close'
        elif 'rgbi' in temp_df.columns:
            price_col = 'rgbi'
        else:
            # Берем первую колонку, которая не похожа на уже посчитанную доходность
            cols = [c for c in temp_df.columns if 'ret' not in c.lower()]
            price_col = cols[0] if cols else temp_df.columns[0]

        # print(f"✅ Используем колонку '{price_col}' для расчета L-Kurtosis")

        # 3. РАСЧЕТ ДОХОДНОСТЕЙ
        # Если данных мало (меньше window), transform вернет NaN, это нормально
        returns = temp_df.groupby(grouper)[price_col].transform(lambda x: np.log(x).diff())
        temp_df['returns'] = returns

        # Проверка: если после diff все NaN, значит в колонке цен были проблемы
        if temp_df['returns'].isna().all():
            print(f"❌ ОШИБКА: Все доходности NaN в колонке {price_col}. Проверьте данные!")

        def calc_l_kurt(x):
            x_clean = x[~np.isnan(x)]

            # По Талебу и логике lmoments3 нужно минимум 5 точек для стабильного tau_4
            if len(x_clean) < 5:
                return 0.122601

            try:
                # Прямой расчет коэффициентов (L-ratios)
                ratios = lm.lmom_ratios(x_clean, nmom=4)
                # Берем именно tau_4 (индекс 3) и ограничиваем его [0, 1]
                return float(np.clip(ratios[3], 0.0, 1.0))
            except:
                return 0.122601

        # 4. СКОЛЬЗЯЩЕЕ ОКНО И ЗАПОЛНЕНИЕ
        lk = temp_df.groupby(grouper)['returns'].transform(
            lambda x: x.rolling(window=window, min_periods=5).apply(calc_l_kurt, raw=True)
        )

        lk_final = lk.groupby(grouper).ffill(limit=3).fillna(1.0)

        # 5. ВОЗВРАТ DATAFRAME
        return pd.DataFrame({'lkurtosis': lk_final}, index=df.index)

    def calculate_max_entropy_risk(self, returns, bins=50):
        """
        Оценка риска через информационную энтропию (мера «размытости» распределения).

        Высокая энтропия → распределение близко к равномерному (высокий неопределённость/риск).
        Низкая энтропия → концентрация в нескольких пиках (более предсказуемо).

        Параметры:
            returns (array-like): временные ряды доходностей
            bins (int): количество бинов для гистограммы (по умолчанию 50)

        Возвращает:
            float: нормализованная энтропия в диапазоне [0, 1]
                  0 — минимальная энтропия (детерминированное распределение)
                  1 — максимальная энтропия (равномерное распределение)
        """
        eps = 1e-10
        try:
            # Преобразуем в numpy array и удаляем NaN
            returns = np.array(returns)
            returns = returns[~np.isnan(returns)]

            # Проверка достаточности данных
            if len(returns) < 2:
                logger.warning("Недостаточно данных: менее 2 наблюдений после удаления NaN")
                return 0.0

            # Валидация параметра bins
            if bins < 2:
                logger.error(f"Некорректное количество бинов: bins={bins}. Должно быть >= 2")
                return 0.0

            if bins > len(returns):
                logger.warning(
                    f"Количество бинов ({bins}) превышает число наблюдений ({len(returns)}). "
                    f"Устанавливаем bins = {len(returns) // 2}"
                )
                bins = max(2, len(returns) // 2)

            # Создаём гистограмму распределения (плотность вероятности)
            hist, bin_edges = np.histogram(returns, bins=bins, density=True)

            hist = hist + eps

            # Исключаем нули для логарифма (важно для расчёта энтропии)
            hist_positive = hist[hist > 0]

            # Если все значения нулевые после фильтрации
            if len(hist_positive) == 0:
                logger.debug("Все значения гистограммы нулевые, возвращаем 0.0")
                return 0.0

            if len(hist_positive) <= 1:
                logger.debug("Энтропия не определена для одного значения")
                return 0.0


            # Рассчитываем энтропию Шеннона
            shannon_ent = entropy(hist_positive, base=np.e)  # Используем натуральный логарифм

            # Максимальная возможная энтропия для данного числа бинов (равномерное распределение)
            max_ent = np.log(len(hist_positive))

            # Защита от нулевого знаменателя
            if max_ent <= 1e-10:
                logger.debug("Максимальная энтропия близка к нулю, возвращаем 0.0")
                return 0.0

            # Нормализация
            normalized_ent = shannon_ent / max_ent

            # Ограничение диапазона [0, 1] (из‑за численных погрешностей может быть > 1)
            normalized_ent_clipped = float(np.clip(normalized_ent, 0.0, 1.0))

            return normalized_ent_clipped

        except Exception as e:
            logger.error(f"Критическая ошибка в calculate_max_entropy_risk: {type(e).__name__}: {e}")
            logger.debug(f"Трассировка ошибки:\n{traceback.format_exc()}")
            return 0.0

    def generate_signals(self, data):
        try:
            df = data.copy()
            # Простая доходность
            # Простая доходность (через встроенный метод)
            df['returns'] = df['Close'].pct_change().fillna(0)
            # Логарифмическая доходность (USD)
            # Добавляем обработку inf и замену на 0
            df['log_ret_usd'] = np.log(df['Close'] / df['Close'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)

            # Логарифмическая доходность (RUB)
            df['log_ret_rub'] = np.log(df['Close_rub'] / df['Close_rub'].shift(1)).replace([np.inf, -np.inf], 0).fillna(
                0)

            # Проверка на константность (спасет от ошибки CatBoost)
            for col in ['returns', 'log_ret_usd', 'log_ret_rub']:
                if df[col].nunique() <= 1:
                    print(f"Внимание: колонка {col} содержит только одно значение!")

            df['rolling_kurtosis'] = df['returns'].rolling(self.kurtosis_window).kurt()
            df['rolling_skew'] = df['returns'].rolling(self.kurtosis_window).skew()
            # 1. Группа Риска (Hurst & Alpha)
            # Z-score для Hurst (крайне важно для Recurrent моделей)
            df['hurst'] = self._calculate_hurst(df['Close'], self.hurst_window)
            # df['hurst_z'] = (df['hurst'] - df['hurst'].rolling(100).mean()) / (df['hurst'].rolling(100).std() + 1e-6)
            df['hurst_z'] = (df['hurst'] - df['hurst'].rolling(window=50, min_periods=10).mean()) / (
                        df['hurst'].rolling(window=50, min_periods=10).std() + 1e-6)
            df['tail_alpha'] = self._calculate_tail_alpha(df['returns'], self.alpha_window)
            #df['tail_alpha_eazy'] = self._calculate_tail_alpha_eazy(df['returns'], self.alpha_window)
            df['kurtosis'] = df['returns'].rolling(self.kurtosis_window).kurt()

            # 4. Детекторы Антихрупкости (Специфика РФ)
            # Panic Index на базе RVI (Российский индекс страха)
            rvi_cols = [c for c in df.columns if 'rvi_ret' in c.lower()]
            if rvi_cols:
                col = rvi_cols[0]
                r_mean = df[col].rolling(window=60, min_periods=self.min_periods_for_stats).mean()
                r_std = df[col].rolling(window=60, min_periods=self.min_periods_for_stats).std()
                df['panic_index'] = (df[col] - r_mean) / (r_std + 1e-9)

            close_prices = df['Close'].values.astype(float).flatten()
            open_prices = df['Open'].values.astype(float).flatten()
            high_prices = df['High'].values.astype(float).flatten()
            low_prices = df['Low'].values.astype(float).flatten()
            volume = df['Volume'].values.astype(float).flatten()
            returns = df['returns'].values

            # 2. Группа Риска (Tail Risk)

            df["lkurtosis"] = self.add_l_kurtosis_to_df(df, window=self.kurtosis_window)

            df['log_ret'] = df['log_ret_usd']
            df['taleb_kappa'] = df['log_ret'].rolling(252).apply(lambda x: self.calculate_taleb_kappa(x))
            df['entropy_risk'] = df['log_ret'].rolling(100).apply(self.calculate_max_entropy_risk)

            # Ключевой признак: Валютный стресс (разрыв между рублевой и долларовой ценой)
            # Если рубль падает, а акция в рублях стоит — в долларах это крах.
            df['currency_stress'] = df['log_ret_rub'] - df['log_ret_usd']
            try:
              df['fat_tail_risk'] = self.calculate_rolling_tail_risk(df)
            except Exception as e:
              print(f"{traceback.format_exc()}")
              print(f"fat_tail_risk: {e}")

            rolling_std = df['log_ret_rub'].rolling(window=252, min_periods=20).std()
            df['z_gap'] = df['log_ret_rub'].abs() / (rolling_std + 1e-9)

            # 3. Группа "Межрыночный анализ" (Нужны колонки RGBI и USDRUB в исходном df)
            #if 'rgbi' in df.columns:
            #    df['rgbi_ret'] = df['rgbi'].pct_change().ffill().bfill()
            #if 'usdrub' in df.columns:
            #    df['usd_basis'] = df['usdrub'].pct_change().ffill().bfill()

            # Amihud: нормализация логарифмом для нейросети
            money_volume = df['Volume'] * df['Close']
            amihud_raw = df['Close'].pct_change().abs() / (money_volume + 1e-10)
            df['amihud'] = np.log1p(amihud_raw * 1e6).replace([np.inf, -np.inf], 0).ffill().fillna(0)

            df['sigma_ratio'] = abs(returns) / (df['log_ret_rub'].rolling(252).std() + 1e-9)

            # 4. Классические индикаторы (TA-Lib)
            df["BB_middle"] = ta.SMA(close_prices, timeperiod=self.bb_period)
            std_dev = ta.STDDEV(close_prices, timeperiod=self.bb_period)
            df["BB_upper"] = df["BB_middle"] + self.bb_std * std_dev
            df["BB_lower"] = df["BB_middle"] - self.bb_std * std_dev

            df["sma"] = ta.SMA(close_prices, timeperiod=self.sma)
            df["ma_period"] = ta.MA(close_prices, timeperiod=self.ma_period)
            df["RSI"] = ta.RSI(close_prices, timeperiod=self.rsi_period)
            # df['obv'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)).cumsum()
            df["obv"] = ta.OBV(close_prices, volume)
            df["adosc"] = ta.ADOSC(high_prices, low_prices, close_prices, volume)
            slowk, slowd = ta.STOCH(
                high_prices,
                low_prices,
                close_prices,
                fastk_period=self.stoch_k,
                slowk_period=self.stoch_d,
                slowd_period=self.stoch_slow,
            )
            df["slowk"] = slowk
            df["slowd"] = slowd

            df["atr"] = ta.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            df["sar"] = ta.SAR(high_prices, low_prices, acceleration=0, maximum=0)
            df["cci"] = ta.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            df["adx"] = ta.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            df["mfi"] = ta.MFI(
                high_prices, low_prices, close_prices, volume, timeperiod=14
            )
            df["dist_sma"] = (df["Close"] - df["sma"]) / df["sma"]
            # Добавим относительную дистанцию до полос Боллинджера (лучше чем сырые цены)
            df['bb_upper_dist'] = (df['BB_upper'] - df['Close']) / df['Close']

            macd, signal_macd, _ = ta.MACD(
                close_prices,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal,
            )

            df["MACD"] = macd
            df["signal_macd"] = signal_macd


            # Краш-детектор облигаций (RGBI)
            rgbi_cols = [c for c in df.columns if 'rgbi_ret' in c.lower()]
            if rgbi_cols:
                # Резкое падение за 3 дня — признак системного шока
                df['rgbi_crash'] = df[rgbi_cols[0]].diff(3)

            # Относительная волатильность (Scale-invariant)
            atr_cols = [c for c in df.columns if 'atr' in c.lower()]
            if atr_cols:
                df['rel_volatility'] = df[atr_cols[0]] / (df['Close'] + 1e-9)

            df = self.add_market_correlation(df)

            df = df.bfill().ffill().fillna(0)

            conditions = df.copy()
            conditions = conditions.reset_index()
            conditions = conditions.filter(
                [
                    "rgbi_crash",
                    "panic_index",
                    "rel_volatility",
                    "hurst_z",
                    "tail_alpha",
                    "amihud",
                    "rolling_kurtosis",
                    "rolling_skew",
                    "bb_upper_dist",
                    "returns",
                    "rgbi_ret",
                    "bz_ret",
                    "usd_basis",
                    "rvi_ret",
                    "Volume",
                    "taleb_kappa",
                    "entropy_risk",
                    #"ma_period",
                    #"RSI",
                    "obv",
                    #"slowk",
                    #"slowd",
                    #"atr",
                    "adosc",
                    #"cci",
                    #"adx",
                    #"mfi",
                    #"dist_sma",
                    "signal_macd",
                    "corr_close_rgbi",
                    "market_divergence",
                    "lkurtosis",
                    "sigma_ratio",
                    "currency_stress",
                    "z_gap",
                    "fat_tail_risk"
                ],
                axis=1,
            )
            conditions = conditions.dropna()

            # Нам важен df со всеми колонками для Pipeline
            return conditions, df
        except Exception as e:
            print(f"Ошибка в Strategy4: {e}")
            return None, None, None

    def calculate_taleb_kappa(self, returns, n_small=1, n_large=30):
        """
        Рассчитывает индекс Каппа Талеба (мера жирности хвостов распределения).

        Формула из книги 'Statistical Consequences of Fat Tails'.

        Параметры:
            returns (array-like): временные ряды доходностей
            n_small (int): короткий период агрегирования (обычно 1 день)
            n_large (int): длинный период агрегирования (например, 20–30 дней)

        Возвращает:
            float: значение индекса Каппа в диапазоне [0, 1]
        """
        eps = 1e-12
        try:
            # Преобразуем в numpy array и удаляем NaN
            returns = np.array(returns)
            returns = returns[~np.isnan(returns)]

            # Проверка достаточности данных
            if len(returns) < n_large:
                logger.warning(
                    f"Недостаточно данных для расчёта: {len(returns)} наблюдений при n_large={n_large}"
                )
                return 0.0

            if len(returns) < 2:
                logger.warning("Недостаточно данных: менее 2 наблюдений после удаления NaN")
                return 0.0

            def get_mad(x):
                """Mean Absolute Deviation — робастная мера разброса"""
                if len(x) == 0:
                    return 0.0
                mean_val = np.mean(x)
                return np.mean(np.abs(x - mean_val))

            # MAD для одиночных периодов (дневных доходностей)
            mad_1 = get_mad(returns)

            # MAD для агрегированных периодов (имитация укрупнения выборки)
            rolling_sum_returns = pd.Series(returns).rolling(window=n_large).sum().dropna()
            mad_n = get_mad(rolling_sum_returns)

            mad_1 = max(mad_1, eps)
            mad_n = max(mad_n, eps)

            # ЗАЩИТА: Если нет движения (MAD=0), Каппа не определена
            if mad_1 < 1e-12 or mad_n < 1e-12: return 0.0

            # Защита от деления на ноль и крайних случаев
            if mad_1 <= 1e-10 or mad_n <= 1e-10:
                logger.debug("MAD близок к нулю, возвращаем 0.0")
                return 0.0

            # Расчёт логарифмических отношений
            log_ratio_n = np.log(n_large + eps) - np.log(n_small + eps)
            log_ratio_mad = np.log(mad_n + eps) - np.log(mad_1 + eps)

            if abs(log_ratio_mad) < eps:
                logger.debug("Знаменатель близок к нулю, возвращаем 0.0")
                return 0.0

            # Основная формула Каппа
            kappa = 2 - log_ratio_n / log_ratio_mad

            # Нормализация и ограничение диапазона
            kappa_clipped = float(np.clip(kappa, 0.0, 1.0))

            return kappa_clipped

        except Exception as e:
            logger.error(f"Критическая ошибка в calculate_taleb_kappa: {type(e).__name__}: {e}")
            logger.debug(f"Трассировка ошибки:\n{traceback.format_exc()}")
            return 0.0

    def taleb_normalize(self, conditions, window=60, tail_limit=6.0, min_periods_for_stats=5):
        """
        Финальная версия нормализации по Талебу для MOEX.
        Сохраняет масштаб 'Черных лебедей' (хвосты), приводя их к локальным сигмам.
        """
        try:
            if conditions is None or conditions.empty:
                return None

            # Работаем с копией, чтобы не портить исходник
            df = conditions.copy()

            # 1. Интеллектуальный поиск Close
            close_candidates = [c for c in df.columns if 'close' in c.lower()]
            close_col = close_candidates[0] if close_candidates else df.columns[0]
            current_close = df[close_col].copy()

            # 2. Лог-доходности для волатильных активов (переводим мультипликативный рост в аддитивный)
            # Ищем: объемы, цены, макро-индикаторы
            volatile_keywords = ['close', 'volume', 'rgbi_ret', 'rvi_ret', 'bz_ret', 'usd_basis']
            for col in df.columns:
                if any(kw in col.lower() for kw in volatile_keywords):
                    # Используем np.log1p от процентного изменения - это безопаснее
                    pct_change = df[col].pct_change().fillna(0)
                    # Клипаем значения, чтобы логарифм не брался от отрицательных величин ниже -1
                    df[f'{col}_delta'] = np.log1p(pct_change.clip(lower=-0.99))

            # 3. Дистанция до ценовых уровней (BB, SMA, SAR) в процентах
            price_level_keywords = ['bb_', 'sma', 'ma', 'sar', 'middle', 'upper', 'lower']
            for col in df.columns:
                if any(kw in col.lower() for kw in price_level_keywords) and '_delta' not in col:
                    df[col] = (df[col] - current_close) / (current_close.clip(lower=1e-6))


            # 5. Сжатие взрывных величин (Volume/OBV)
            volume_cols = [c for c in df.columns if 'volume' in c.lower() or c.lower() in ['obv', 'adosc']]
            for col in volume_cols:
                # Используем sign * log1p для сохранения направления и масштаба
                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))


            # 6. Rolling Z-Score (Локальная нормализация)
            # Ключевой этап: переводим всё в "сигмы" относительно последних N дней
            exclude_cols = ['date', 'tic', 'timestamp', close_col]
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if col not in exclude_cols:
                    rolling_mean = df[col].rolling(window=window, min_periods=min_periods_for_stats).mean()
                    rolling_std = df[col].rolling(window=window, min_periods=min_periods_for_stats).std()
                    # Защита от деления на сверхмалые std
                    df[col] = (df[col] - rolling_mean) / (rolling_std.clip(lower=1e-4))
            try:
                scaler = RobustScaler(quantile_range=(5, 95))
                cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]
                if cols_to_scale:
                    # Заполняем пустоты перед скалированием, чтобы не было ошибок fit
                    df[cols_to_scale] = df[cols_to_scale].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
                    # Обучаем скейлер ОДИН раз на всей матрице
                    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
            except Exception as e:
                print(f"{traceback.format_exc()}")
                print(f"⚠️ Ошибка в колонке {col}: {e}")

            # 8. Финальная очистка и удаление утечек (Close, Open, High, Low)
            # Удаляем не только Close, но и все базовые ценовые колонки,
            # так как мы уже перевели их в дельты и относительные величины.
            drop_patterns = ['open', 'high', 'low', 'close']
            cols_to_drop = [c for c in df.columns if any(p in c.lower() for p in drop_patterns) and '_delta' not in c]
            df = df.drop(columns=cols_to_drop, errors='ignore')

            # Обработка пустот:
            # bfill для начала (где не накопилось окно), ffill для дырок, 0 для гарантии.
            df = df.bfill().ffill().fillna(0)
            return df

        except Exception as e:
            print(f"{traceback.format_exc()}")
            print(f"taleb_normalize: {e}")
            return conditions


    def normalize_signals(self, conditions):
        """
        Принимает DataFrame признаков (conditions) и возвращает
        нормализованные данные для Recurrent модели.
        """
        if conditions is None or conditions.empty:
            return None
        # А) Детектор паники (не нормализовать жестко!)

        if 'vix' in conditions.columns or 'rvi_ret' in conditions.columns:
            # Оставляем "хвост" - всё что выше 2-х стандартных отклонений
            conditions['panic_index'] = (conditions['rvi_ret'] - conditions['rvi_ret'].rolling(50).mean()) / conditions[
                'rvi_ret'].rolling(50).std()

        # Б) Скорость падения RGBI (главный опережающий индикатор РФ)
        if 'rgbi_ret' in conditions.columns:
            conditions['rgbi_crash'] = conditions['rgbi_ret'].diff().rolling(3).sum()  # Суммарное падение за 3 дня

        # 1. Список колонок, которые НЕЛЬЗЯ подавать "как есть" (цены/уровни)
        # Эти колонки нужно превратить в относительные (дистанция до цены)
        price_columns = ["BB_middle", "BB_upper", "BB_lower", "sma", "ma_period", "sar"]

        # Предполагаем, что у нас есть доступ к 'Close' из исходного df (сохраняем его до фильтрации)
        # Если 'Close' нет в conditions, его нужно передать или извлечь
        if "Close" in conditions.columns:
            current_close = conditions["Close"]
            for col in price_columns:
                if col in conditions.columns:
                    # Превращаем цену в % отклонения: (Уровень - Цена) / Цена
                    conditions[col] = (conditions[col] - current_close) / (current_close + 1e-9)
            # Удаляем Close, чтобы не "подглядывать" в будущее и не сбивать масштаб
            conditions = conditions.drop(columns=["Close"])

        # 2. Логарифмирование "взрывных" величин (Volume-based)
        # OBV и ADOSC могут быть огромными, RobustScaler-у будет тяжело
        for col in ["obv", "adosc"]:
            if col in conditions.columns:
                conditions[col] = np.sign(conditions[col]) * np.log1p(np.abs(conditions[col]))

        # 3. Применение RobustScaler
        scaler = RobustScaler()

        # Обучаем и трансформируем
        scaled_values = scaler.fit_transform(conditions)

        # Возвращаем DataFrame с теми же именами колонок
        normalized_df = pd.DataFrame(scaled_values, columns=conditions.columns, index=conditions.index)

        return normalized_df

    def prepare_df_for_finrl(df_transformed, original_df):
        """
        Приводит результат Pipeline к стандарту FinRL (OHLCV + Features).
        Подходит для Recurrent RL (LSTM/GRU).
        """
        # 1. Подготовка оригинала (OHLCV)
        df_ohlcv = original_df.copy()
        df_ohlcv.columns = [col.lower() for col in df_ohlcv.columns]

        # 2. Подготовка признаков
        df_features = df_transformed.copy()

        # Если в df_features нет колонки 'date', восстанавливаем её из оригинала по индексам
        # Это критично, если трансформер сбросил индекс даты
        if 'date' not in df_features.columns:
            # Пытаемся сопоставить по тикеру и позиции (так как порядок в TickerParallelWrapper сохранен)
            df_features = df_features.reset_index(drop=True)
            # Добавляем дату из оригинала, убедившись в совпадении тикеров
            # Самый надежный способ: merge по тикеру и порядковому номеру внутри тикера
            df_ohlcv['temp_idx'] = df_ohlcv.groupby('tic').cumcount()
            df_features['temp_idx'] = df_features.groupby('tic').cumcount()

            merge_keys = ['tic', 'temp_idx']
        else:
            merge_keys = ['tic', 'date']

        # 3. Объединение
        # Берем базовые колонки FinRL из оригинала
        required_ohlcv = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
        ohlcv_subset = df_ohlcv[[c for c in required_ohlcv if c in df_ohlcv.columns] + (
            ['temp_idx'] if 'temp_idx' in df_ohlcv.columns else [])]

        final_df = pd.merge(
            ohlcv_subset,
            df_features,
            on=merge_keys,
            how='inner'
        )

        # 4. Очистка временных ключей
        if 'temp_idx' in final_df.columns:
            final_df = final_df.drop(columns=['temp_idx'])

        # 5. Сортировка для RECURRENT моделей (Строго: Дата -> Тикер)
        # FinRL StockTradingEnv требует именно такой порядок для правильного формирования батчей
        final_df['date'] = pd.to_datetime(final_df['date'])
        final_df = final_df.sort_values(['date', 'tic']).reset_index(drop=True)

        # 6. Финальная обработка пропусков (RobustScaler может давать NaN на краях)
        # Сначала заполняем внутри каждого тикера, чтобы не подмешивать данные других компаний
        final_df = final_df.groupby('tic', group_keys=False).apply(lambda x: x.ffill().bfill())

        # Если остались NaN (например, вся колонка пуста), заполняем 0
        final_df = final_df.fillna(0)

        # 7. Генерация списка признаков для агента (Technical Indicator List)
        ignored = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']
        feature_list = [col for col in final_df.columns if col not in ignored]

        print(f"--- Подготовка FinRL завершена ---")
        print(f"Итого строк: {len(final_df)}")
        print(f"Признаков для модели: {len(feature_list)}")

        return final_df, feature_list


class TailRiskAnalyzer:
    """
    Комплексный анализатор хвостовых рисков с тремя методами оценки.
    """
    def __init__(self, quantile=0.05, garch_window=252, evt_window=500):
        self.quantile = quantile
        self.garch_window = garch_window
        self.evt_window = evt_window

    def predict_tail_boundary(self, df, window=100):
        """Прогноз границы хвоста через квантильную регрессию."""
        if len(df) < window:
            return 0.0

        # Берём последние window наблюдений
        subset = df.tail(window).copy()
        # Проверяем наличие нужных колонок
        required_cols = ['log_ret', 'rvi_ret', 'atr']
        if not all(col in subset.columns for col in required_cols):
            return 0.0

        model = smf.quantreg('log_ret ~ rvi_ret + atr', subset)
        try:
            res = model.fit(q=self.quantile)
            prediction = res.predict(subset.tail(1))
            return prediction.values[0]
        except:
            return 0.0

    def calculate_evt_risk(self, returns, quantile=None):
        """EVT-анализ экстремальных убытков."""
        if quantile is None:
            quantile = self.quantile

        losses = -returns
        threshold = np.quantile(losses, 1 - quantile)
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 5:
            return 0.0, 0.0  # Возвращаем нулевые риски при недостатке данных

        try:
            shape, loc, scale = genpareto.fit(exceedances, floc=0)
            n = len(returns)
            nu = len(exceedances)
            # VaR по EVT
            evt_var = threshold + (scale / shape) * (((n / nu) * (1 - 0.99)) ** (-shape) - 1)
            # CVaR (Expected Shortfall)
            evt_cvar = (evt_var + scale - shape * threshold) / (1 - shape)
            return max(evt_var, 0), max(evt_cvar, 0)  # Гарантируем неотрицательность
        except:
            return 0.0, 0.0

    def calculate_garch_metrics(self, returns):
        """GARCH-анализ волатильности и «жирности» хвостов."""
        if len(returns) < self.garch_window:
            return 0.0, 0.0

        try:
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t')
            res = model.fit(disp='off', last_obs=len(returns))
            forecast = res.forecast(horizon=1)
            cond_vol = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100
            nu = res.params['nu']
            tail_fatness = 1 / nu if nu > 0 else 0.0
            return cond_vol, tail_fatness
        except:
            return 0.0, 0.0

    def get_composite_tail_risk(self, df):
        """
        Комплексная оценка хвостового риска: объединяет все три метода.
        Возвращает: tail_risk_score (0–1), где 1 — максимальный риск.
        """
        returns = df['log_ret'].dropna()

        # Метод 1: граница хвоста
        tail_boundary = self.predict_tail_boundary(df)

        # Метод 2: EVT-риски
        evt_var, evt_cvar = self.calculate_evt_risk(returns)

        # Метод 3: GARCH-метрики
        cond_vol, tail_fatness = self.calculate_garch_metrics(returns)

        # Нормализация и взвешивание
        # Веса подобраны эмпирически: EVT (40 %), GARCH (40 %), Quantile (20 %)
        risk_components = [
            min(abs(tail_boundary) * 2, 1.0),  # нормализация
            min(evt_var * 5, 1.0),
            min(cond_vol * 10, 1.0),
            min(tail_fatness * 5, 1.0)
        ]

        # Взвешенное среднее
        weights = [0.2, 0.4, 0.3, 0.1]
        composite_risk = np.average(risk_components, weights=weights)

        return min(composite_risk, 1.0)  # финальный скоринг 0–1