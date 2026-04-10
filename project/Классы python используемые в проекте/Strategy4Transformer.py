import pandas as pd
import numpy as np
import talib as ta
import traceback
from lmoments3 import stats
import lmoments3 as lm

# В lmoments3 основные расчеты лежат здесь:

from sklearn.base import BaseEstimator, TransformerMixin
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn import set_config
set_config(transform_output="pandas")


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


class Strategy4Transformer(BaseEstimator, TransformerMixin, BaseStrategy):
    def __init__(self, optimize=True, n_trials=50, min_trades_per_fold=5, lags=3, **params):
        # Инициализируем BaseStrategy для доступа к calculate_metrics
        super().__init__()
        self.optimize = optimize
        self.n_trials = n_trials
        self.lags = lags  # Количество прошлых значений для каждого индикатора
        self.min_trades_per_fold = min_trades_per_fold

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
                df[['corr_close_rgbi', 'market_divergence']] = df[['corr_close_rgbi', 'market_divergence']].ffill().fillna(0)
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
            "rsi_period": trial.suggest_int("rsi_period", 7, 21),
            "sma": trial.suggest_int("sma", 5, 40),
            "ma_period": trial.suggest_int("ma_period", 10, 50),
            "macd_fast": trial.suggest_int("macd_fast", 8, 18),
            "macd_slow": trial.suggest_int("macd_slow", 20, 40),
            "macd_signal": trial.suggest_int("macd_signal", 5, 12),
            "stoch_k": trial.suggest_int("stoch_k", 10, 20),
            "stoch_d": trial.suggest_int("stoch_d", 2, 5),
            "stoch_slow": trial.suggest_int("stoch_slow", 2, 5),
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

                signals, _, _ = self.generate_signals(X_val)
                if signals is None or signals.nunique() <= 1:
                    fold_scores.append(-1.0)
                    continue

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
        price_returns = data["Close"].pct_change().fillna(0)
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
        """Финальная обработка данных с лучшими параметрами"""
        signals, _, df_enriched = self.generate_signals(X)
        df_res = df_enriched.copy()
        df_res['strategy_signal'] = signals.fillna(0)

        # Список колонок, для которых нужны лаги (динамика важна здесь)
        cols_to_lag = ['RSI', 'MACD', 'signal_macd', 'strategy_signal']

        lagged_dfs = [df_res]
        for col in cols_to_lag:
            for i in range(1, self.lags + 1):
                lagged_col = df_res[col].shift(i).rename(f"{col}_lag_{i}")
                lagged_dfs.append(lagged_col)

        df_res = pd.concat(lagged_dfs, axis=1)

        # Возвращаем именно DataFrame
        return pd.DataFrame(df_res, columns=df_res.columns, index=df_res.index).ffill().fillna(0)

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

        #print(f"✅ Используем колонку '{price_col}' для расчета L-Kurtosis")

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

    def generate_signals(self, data):
        try:
            df = data.copy()

            close_prices = df["Close"].values.astype(float).flatten()
            open_prices = df["Open"].values.astype(float).flatten()
            high_prices = df["High"].values.astype(float).flatten()
            low_prices = df["Low"].values.astype(float).flatten()
            volume = df["Volume"].values.astype(float).flatten()

            # L-Kurtosis
            df["lkurtosis"] = self.add_l_kurtosis_to_df(df, window=30)

            # Полосы Боллинджера - Уровни поддержки и сопротивления
            df["BB_middle"] = ta.SMA(close_prices, timeperiod=self.bb_period)
            std_dev = ta.STDDEV(close_prices, timeperiod=self.bb_period)
            df["BB_upper"] = df["BB_middle"] + self.bb_std * std_dev
            df["BB_lower"] = df["BB_middle"] - self.bb_std * std_dev

            df["sma"] = ta.SMA(close_prices, timeperiod=self.sma)
            df["ma_period"] = ta.MA(close_prices, timeperiod=self.ma_period)
            df["RSI"] = ta.RSI(close_prices, timeperiod=self.rsi_period)
            # df['obv'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)).cumsum()
            df["obv"] = ta.OBV(close_prices, volume)
            df["engulfing"] = ta.CDLENGULFING(
                open_prices, high_prices, low_prices, close_prices
            )
            df["morningstar"] = ta.CDLMORNINGSTAR(
                open_prices, high_prices, low_prices, close_prices
            )
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
            df["doji"] = ta.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            df["hammer"] = ta.CDLHAMMER(
                open_prices, high_prices, low_prices, close_prices
            )
            df["atr14"] = ta.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            df["sar"] = ta.SAR(high_prices, low_prices, acceleration=0, maximum=0)
            df["cci"] = ta.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            df["adx"] = ta.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            df["mfi"] = ta.MFI(
                high_prices, low_prices, close_prices, volume, timeperiod=14
            )
            df["linearreg_angle"] = ta.LINEARREG_ANGLE(close_prices, timeperiod=14)
            df["dist_sma"] = (df["Close"] - df["sma"]) / df["sma"]
            macd, signal_macd, _ = ta.MACD(
                close_prices,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal,
            )

            df["MACD"] = macd
            df["signal_macd"] = signal_macd

            df = self.add_market_correlation(df)

            df = df.ffill().fillna(0)
            signals = pd.Series(index=df.index, data=0)

            # 1. Основной бычий фильтр (Тренд)
            # Цена выше обеих средних и быстрая средняя выше медленной
            bullish_trend = (df["Close"] > df["sma"]) & (df["sma"] > df["ma_period"])
            # 2. Подтверждение притока денег (Volume & Momentum)
            # ADOSC > 0 (накопление), MFI > 50 (индекс денежного потока в зоне роста)
            money_flow = (df["adosc"] > 0) | (df["mfi"] > 50)
            # 3. Свечные триггеры (Ускорение)
            # Паттерны поглощения или утренняя звезда на фоне бычьего тренда
            bullish_candles = (
                (df["engulfing"] > 0) | (df["morningstar"] > 0) | (df["hammer"] > 0)
            )
            long_condition = (
                (df["Close"] > df["sma"])
                & (
                    df["Close"].shift(1) <= df["sma"].shift(1)
                )  # Пересечение снизу вверх
                & (df["adx"] > 20)  # Только когда есть сильный тренд
            )

            # Выход должен быть менее чувствительным, чтобы не выбило из тренда
            short_condition = (
                df["Close"] < df["BB_middle"]
            ) | (  # Пробой средней линии Боллинджера вниз
                df["RSI"] > 80
            )  # Только экстремальная перекупленность

            signals[long_condition] = 1
            signals[short_condition] = -1
            signals = signals.reindex_like(data)

            conditions = df.copy()
            conditions = conditions.reset_index()
            conditions = conditions.filter(
                [
                    "BB_middle",
                    "BB_upper",
                    "BB_lower",
                    "sma",
                    "ma_period",
                    "RSI",
                    "obv",
                    "engulfing",
                    "morningstar",
                    "MACD",
                    "slowk",
                    "slowd",
                    "doji",
                    "hammer",
                    "atr14",
                    "sar",
                    "adosc",
                    "linearreg_angle",
                    "cci",
                    "adx",
                    "mfi",
                    "dist_sma",
                    "signal_macd",
                    'corr_close_rgbi',
                    'market_divergence',
                    'lkurtosis'
                ],
                axis=1,
            )
            conditions = conditions.dropna()

            signals[long_condition] = 1
            signals[short_condition] = -1

            # Нам важен df со всеми колонками для Pipeline
            return signals, conditions, df
        except Exception as e:
            print(f"Ошибка в Strategy4: {e}")
            return None, None, None
