import pandas as pd
import time
import logging
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

import pandas as pd
import time
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted


class TickerParallelWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, verbose=True):
        self.transformer = transformer
        self.verbose = verbose
        self.models = {}
        self.unique_tickers_ = None

    def _log(self, message):
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] [TickerParallel] {message}")

    def fit(self, X, y=None):
        # 1. Определяем, где лежит 'tic' (в индексе или в колонках)
        # Используем getattr для безопасности, если индекс вдруг не имеет атрибута names
        index_names = getattr(X.index, 'names', [])
        is_multiindex = isinstance(X.index, pd.MultiIndex) and 'tic' in index_names

        if is_multiindex:
            unique_tickers = X.index.get_level_values('tic').unique()
        elif 'tic' in X.columns:
            unique_tickers = X['tic'].unique()
        else:
            # Расширенная ошибка для отладки
            raise ValueError(f"Тикер 'tic' не найден! Индекс: {index_names}, Колонки: {X.columns.tolist()}")

        self._log(f"🚀 Начало параллельного обучения для {len(unique_tickers)} тикеров...")

        for i, tic in enumerate(unique_tickers, 1):
            start_time = time.time()
            try:
                # 2. Извлекаем данные одного тикера
                if is_multiindex:
                    # xs(tic, level='tic') возвращает данные, где уровень 'tic' удален из индекса
                    tic_data = X.xs(tic, level='tic').copy()
                else:
                    tic_data = X[X['tic'] == tic].copy()

                if tic_data.empty:
                    self._log(f"⚠️ {tic}: Данные пусты, пропуск.")
                    continue

                # 3. ВАЖНО: Strategy4 требует колонку 'tic' явно для внутренних расчетов
                if 'tic' not in tic_data.columns:
                    tic_data['tic'] = tic

                # 4. Сопоставление целевой переменной y
                tic_y = None
                if y is not None:
                    try:
                        if is_multiindex and isinstance(y, (pd.Series, pd.DataFrame)):
                            # Если y имеет такой же MultiIndex, извлекаем через xs
                            tic_y = y.xs(tic, level='tic')
                        elif not is_multiindex and isinstance(y, (pd.Series, pd.DataFrame)):
                            # Если y обычный, сопоставляем по индексу отфильтрованного X
                            tic_y = y.loc[tic_data.index]
                        else:
                            # Если y — это массив numpy, сопоставление по тикерам невозможно без индекса
                            tic_y = y
                    except Exception as e:
                        self._log(f"⚠️ Не удалось сопоставить y для {tic}: {e}")

                # 5. Обучаем изолированный клон трансформера/модели
                # Используем sklearn.base.clone для чистоты обучения
                self.models[tic] = clone(self.transformer).fit(tic_data, tic_y)

                duration = time.time() - start_time
                self._log(f"({i}/{len(unique_tickers)}) ✅ {tic} обучен за {duration:.1f}с")

            except Exception as e:
                self._log(f"❌ Ошибка при обучении {tic}: {str(e)}")
                continue

        return self

    def transform(self, X):
        check_is_fitted(self)  # Проверка на наличие обученных моделей
        results = []

        # 1. Безопасное определение тикеров (как в fit)
        index_names = getattr(X.index, 'names', [])
        is_multiindex = isinstance(X.index, pd.MultiIndex) and 'tic' in index_names

        if is_multiindex:
            unique_tickers = X.index.get_level_values('tic').unique()
        elif 'tic' in X.columns:
            unique_tickers = X['tic'].unique()
        else:
            raise ValueError("Тикер 'tic' не найден в данных для трансформации")

        for tic in unique_tickers:
            if tic not in self.models:
                continue

            try:
                # 2. Извлекаем данные
                if is_multiindex:
                    tic_data = X.xs(tic, level='tic').copy()
                else:
                    tic_data = X[X['tic'] == tic].copy()

                if tic_data.empty:
                    continue

                # Гарантируем наличие колонки 'tic' для трансформера
                tic_data['tic'] = tic

                # Сортировка по дате (важна для временных рядов)
                if 'date' in tic_data.index.names:
                    tic_data = tic_data.sort_index(level='date')
                elif 'date' in tic_data.columns:
                    tic_data = tic_data.sort_values('date')

                # ВЫЗОВ ТРАНСФОРМАЦИИ
                res = self.models[tic].transform(tic_data)

                if isinstance(res, pd.DataFrame):
                    # Важно: res может потерять индекс date, если трансформер его сбросил
                    res = res.copy()
                    res['tic'] = tic
                    results.append(res)

            except Exception as e:
                self._log(f"❌ Ошибка при трансформации {tic}: {e}")
                continue

        if not results:
            raise ValueError("Нет данных для трансформации")

        # 3. Склеиваем результаты
        final_df = pd.concat(results, axis=0, ignore_index=False)

        # 4. Безопасная обработка индекса 'date'
        # Если 'date' в индексе, переносим в колонку, только если её там еще нет
        if 'date' in getattr(final_df.index, 'names', []):
            if 'date' not in final_df.columns:
                final_df = final_df.reset_index(level='date')
            else:
                final_df = final_df.reset_index(level='date', drop=True)

        # 5. Очистка от дубликатов колонок (бывает, если трансформер вернул 'tic')
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]

        # 6. Финальная подготовка для FinRL
        sort_cols = []
        if 'date' in final_df.columns: sort_cols.append('date')
        if 'tic' in final_df.columns: sort_cols.append('tic')

        if sort_cols:
            final_df = final_df.sort_values(sort_cols).reset_index(drop=True)
        else:
            final_df = final_df.reset_index(drop=True)

        return final_df

