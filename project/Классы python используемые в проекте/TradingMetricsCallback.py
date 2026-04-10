from stable_baselines3.common.callbacks import BaseCallback


class TradingMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TradingMetricsCallback, self).__init__(verbose)
        self.sharpe_history = []

    def _on_step(self) -> bool:
        # Раз в 1000 шагов вытягиваем инфо из среды FinRL
        if self.n_calls % 1000 == 0:
            # FinRL хранит историю наград в env_train.env_method('get_sb_env')
            # Но проще всего взять инфо через locals (внутренние переменные модели)
            if "infos" in self.locals:
                info = self.locals["infos"][0]
                # Проверяем, записала ли среда FinRL текущий профит/шарп
                if 'sharpe_ratio' in info:
                    sharpe = info['sharpe_ratio']
                    self.logger.record("train/sharpe_ratio", sharpe)

                # Также полезно писать общую доходность
                if 'total_assets' in info:
                    self.logger.record("train/total_assets", info['total_assets'])

        return True
