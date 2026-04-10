from sklearn.base import BaseEstimator, TransformerMixin
import riskfolio as rp


class RiskfolioEnsembleManager(BaseEstimator, TransformerMixin):
    def __init__(self, risk_free_rate=0.0, risk_measure='CVaR'):
        self.rf = risk_free_rate
        self.rm = risk_measure

    def transform(self, data_package):
        """
        data_package: словарь, содержащий 'ensemble_raw' (от get_ensemble_predictions)
        и 'risk_context' (результат MoexAgentTrainer.transform)
        """
        raw_outputs = data_package['ensemble_raw']
        risk_context = data_package['risk_context']
        taleb_stats = risk_context['taleb_risk_stats']

        # 1. Строим таблицу доходностей каждой модели для Riskfolio
        # (Нам нужно понять ковариацию моделей, чтобы выбрать веса)
        returns_df = self._build_returns_matrix(raw_outputs, risk_context['df'])

        # 2. Настройка портфеля в Riskfolio
        port = rp.Portfolio(returns=returns_df)
        port.assets_stats(method_mu='hist', method_cov='hist')

        # 3. АДАПТИВНАЯ ЦЕЛЬ:
        # Если tail_risk > 0.6, Riskfolio переходит в режим минимизации CVaR
        obj = 'MinRisk' if taleb_stats['tail_risk'] > 0.6 else 'Sharpe'

        # 4. Расчет весов моделей (кто из агентов сейчас "антихрупкий")
        weights = port.optimization(model='Classic', rm=self.rm, obj=obj, rf=self.rf)

        # 5. ФОРМИРОВАНИЕ ЕДИНОГО СИГНАЛА (-1...+1)
        final_action = 0
        for name in raw_outputs.keys():
            w = weights.loc[name, 'weights']
            last_act = raw_outputs[name].iloc[-1, 0]  # Последний сигнал модели
            final_action += last_act * w

        # 6. ТАЛЕБ-МАСШТАБИРОВАНИЕ
        # Режем итоговый сигнал на уровень уверенности (confidence)
        final_action *= taleb_stats['confidence']

        # 7. Финальное количество лотов (Position Sizing)
        # Если confidence < 0.25 — принудительно -1.0 (Close All)
        if taleb_stats['confidence'] < 0.25:
            final_action = -1.0

        return {
            'action': final_action,
            'lots': int(final_action * 100),  # Допустим hmax=100
            'weights': weights,
            'confidence': taleb_stats['confidence']
        }

    def _build_returns_matrix(self, raw_outputs, df):
        rets = {}
        price_change = df['close'].pct_change().fillna(0)
        for name, df_act in raw_outputs.items():
            # Находим доходность стратегии: сигнал * изменение цены
            rets[name] = df_act.iloc[:, 0].shift(1).fillna(0) * price_change.values[-len(df_act):]
        return pd.DataFrame(rets)
    
# 1. Получаем контекст от Трейнера (включая tail_risk)
#risk_context = full_pipeline.named_steps['agent_trainer'].transform(df_test_prepared)
# 2. Получаем сигналы от всех моделей (Ансамбль)
#ensemble_raw = trainer.get_ensemble_predictions(df_test_prepared, models_to_use=['ppo', 'sac', 'a2c'])
# 3. Объединяем и отдаем в Riskfolio
#final_result = RiskfolioEnsembleManager().transform({
#    'ensemble_raw': ensemble_raw,
#    'risk_context': risk_context
#})
#print(f"✅ Итоговый лот SBER: {final_result['lots']}")