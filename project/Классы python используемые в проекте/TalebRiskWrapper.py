import gymnasium as gym
import numpy as np

class TalebRiskWrapper(gym.Wrapper):
    def __init__(self, env, model=None, confidence_threshold=0.3):
        super().__init__(env)
        self.confidence_threshold = confidence_threshold
        self.model = model
        # Сохраняем историю для диагностики
        self.risk_history = []

    def step(self, action):
        # 1. Извлекаем диагностику напрямую из экстрактора модели,
        # так как в самом векторе состояния (obs) данные могут быть нормализованы
        # Если модель еще не передана, используем fallback
        current_confidence = 1.0

        # ХАК: Если мы внутри сессии Stable Baselines, можем дотянуться до экстрактора
        #if hasattr(self.env, 'model') and hasattr(self.env.model.policy, 'features_extractor'):
        #    diag = getattr(self.env.model.policy.features_extractor, 'diagnostics', {})
        #    current_confidence = diag.get('confidence', 1.0)

        if self.model is not None and hasattr(self.model, 'policy'):
            try:
                extractor = self.model.policy.features_extractor
                # Достаем диагностику, которую обновил экстрактор при последнем obs
                diag = getattr(extractor, 'diagnostics', {})
                current_confidence = diag.get('confidence', 1.0)
            except Exception:
                current_confidence = 1.0
        # 2. Логика Талеба: масштабируем действие на уверенность
        # Если уверенность 0.1, а порог 0.3 — агент "замирает"
        if current_confidence < self.confidence_threshold:
            # В FinRL действия [-1, 1]. 0 означает "держать текущую позицию"
            adjusted_action = np.zeros_like(action)
            taleb_blocked = True
        else:
            # Мягкое масштабирование: чем выше уверенность, тем полнее исполняем приказ
            adjusted_action = action * current_confidence
            taleb_blocked = False

        # 3. Вызываем шаг среды
        # ВАЖНО: передаем только конечное, безопасное число
        obs, reward, terminated, truncated, info = self.env.step(adjusted_action)

        # 4. Обогащаем инфо для Шага 3 (Аудит)
        info['taleb_confidence'] = current_confidence
        info['taleb_blocked'] = taleb_blocked

        # Если это режим теста, логируем для графиков
        if not terminated:
            self.risk_history.append(current_confidence)

        return obs, reward, terminated, truncated, info