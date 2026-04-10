from stable_baselines3.common.callbacks import BaseCallback


class TalebRiskLoggerCallback(BaseCallback):
    """
    Callback для мониторинга 'хвостов' и уверенности модели в TensorBoard.
    """

    def __init__(self, verbose=0):
        super(TalebRiskLoggerCallback, self).__init__(verbose)
        self.last_tail_risk = 0
        self.last_confidence = 0

    def _on_step(self) -> bool:
        # Достаем экстрактор из policy сети
        # В SB3 путь: model.policy.features_extractor
        extractor = self.model.policy.features_extractor

        # Получаем данные из словаря diagnostics, который мы создали в forward
        if hasattr(extractor, 'diagnostics'):
            risk = extractor.diagnostics.get('tail_risk', 0)
            conf = extractor.diagnostics.get('confidence', 1)

            # Логируем в TensorBoard
            self.logger.record("risk/tail_score", risk)
            self.logger.record("risk/model_confidence", conf)

            # Раз в 1000 шагов выводим в консоль для контроля
            if self.n_calls % 1000 == 0:
                print(f"Step {self.n_calls}: Risk={risk:.2f}, Confidence={conf:.2f}")

        return True
'''
Как запустить обучение с этим мониторингом
Теперь при вызове model.learn() просто передай этот коллбэк.
python
Создаем среду (FinRL/Gym)
# env = ... 

Настраиваем модель с экстрактором
policy_kwargs = dict(
    features_extractor_class=UltimateMoexTalebExtractor,
    features_extractor_kwargs=dict(
        stock_dim=len(train_tickers),
        window_size=10,
        tail_sensitivity=1.8,
        macro_indices=[-4, -3, -2, -1] # Индексы RGBI, USD/RUB, RVI, Brent
    ),
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="./taleb_logs/", verbose=1)

Запускаем обучение с коллбэком
risk_callback = TalebRiskLoggerCallback()

model.learn(
    total_timesteps=100000, 
    callback=risk_callback,
    tb_log_name="PPO_Taleb_MOEX"
)
Используйте код с осторожностью.

Как анализировать результат в TensorBoard
Запусти в терминале:
tensorboard --logdir ./taleb_logs/
В браузере появятся две критически важные вкладки:
risk/tail_score: Если этот график ползет вверх — значит, LSTM видит в динамике индикаторов (Hurst, Kurtosis) признаки «черного лебедя».
risk/model_confidence: Это «вентиль» твоего агента. Если он падает до 0.1 или 0.05 — агент практически перестает открывать новые позиции, фиксирует что может и «замирает», спасая депозит.
Что это дает на MOEX?
На российском рынке часто случаются резкие фазовые переходы (геополитика). Обычный RL-агент на таких скачках пытается «выкупить просадку» и сливает. Твой агент с этим коллбэком покажет: «Я увидел рост Tail Risk на 0.8, снизил Confidence до 0.1 и просто переждал обвал в кэше».
'''