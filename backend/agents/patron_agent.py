import time
from .base_agent import BaseAgent


class PatronAgent(BaseAgent):
    """
    Patron (Boss) Ajan - Tum ajanlarin kararlarini degerlendirir,
    son onayi verir. Veto yetkisi vardir.
    Her karar icin hangi ajanin ne oy verdigini gosterr.
    """

    def __init__(self):
        super().__init__("Patron")
        self.decisions: list[dict] = []
        self.market_regime = "neutral"
        self.overall_confidence = 0.0
        self.last_market_assessment = {}
        self.trade_log: list[dict] = []
        self.min_confidence = 0.15

    async def analyze(self, data: dict) -> dict:
        try:
            self.update_status("running")

            scanner = data.get("scanner", {})
            technical = data.get("technical", {})
            sentiment = data.get("sentiment", {})
            risk = data.get("risk", {})
            portfolio = data.get("portfolio", {})

            self.market_regime = self._assess_market_regime(sentiment)
            self.last_market_assessment = {
                "regime": self.market_regime,
                "sentiment": sentiment.get("overall_sentiment", 0),
                "fear_greed": sentiment.get("fear_greed_index", 50),
                "timestamp": time.time(),
            }

            signals = technical.get("signals", [])
            self.decisions = []

            for signal in signals:
                decision = self._evaluate_signal(signal, sentiment, risk, portfolio, scanner)
                self.decisions.append(decision)

            self.decisions.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            approved = [d for d in self.decisions if d["approved"]]
            self.overall_confidence = (
                sum(d["composite_score"] for d in approved) / len(approved)
                if approved else 0
            )

            self.signals = self.decisions
            self.update_status("ready")

            return {
                "decisions": self.decisions,
                "approved_count": len(approved),
                "rejected_count": len(self.decisions) - len(approved),
                "market_regime": self.market_regime,
                "overall_confidence": round(self.overall_confidence, 3),
                "market_assessment": self.last_market_assessment,
                "top_picks": [d for d in self.decisions[:5] if d["approved"]],
                "thinking": self._build_thinking(),
                "summary": self._build_summary(),
            }

        except Exception as e:
            self.update_status("error", str(e))
            return {"decisions": [], "error": str(e)}

    def _assess_market_regime(self, sentiment: dict) -> str:
        fear_greed = sentiment.get("fear_greed_index", 50)
        overall = sentiment.get("overall_sentiment", 0)
        if fear_greed <= 20 or overall < -0.5:
            return "extreme_fear"
        elif fear_greed <= 40 or overall < -0.2:
            return "fear"
        elif fear_greed >= 80 or overall > 0.5:
            return "extreme_greed"
        elif fear_greed >= 60 or overall > 0.2:
            return "greed"
        return "neutral"

    def _evaluate_signal(
        self, signal: dict, sentiment: dict, risk: dict,
        portfolio: dict, scanner: dict
    ) -> dict:
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "hold")
        technical_confidence = signal.get("confidence", 0)

        sentiment_score = self._score_sentiment(sentiment, direction)
        volume_score = self._score_volume(scanner, symbol)
        risk_score = self._score_risk(risk, symbol)
        correlation_penalty = self._score_correlation(portfolio, symbol, direction)
        regime_modifier = self._get_regime_modifier(direction)

        composite = (
            technical_confidence * 0.40
            + sentiment_score * 0.15
            + volume_score * 0.15
            + risk_score * 0.20
            + correlation_penalty * 0.10
        ) * regime_modifier
        composite = max(0.0, min(1.0, composite))

        approved = (
            composite >= self.min_confidence
            and direction != "hold"
            and risk.get("decision", {}).get("approved", False)
        )

        veto_reason = None
        if not approved:
            reasons = []
            if composite < self.min_confidence:
                reasons.append(f"Composite skor dusuk ({composite:.2f} < {self.min_confidence})")
            if direction == "hold":
                reasons.append("Sinyal yonu belirsiz")
            if not risk.get("decision", {}).get("approved", False):
                reasons.append(f"Risk onayi yok: {risk.get('decision', {}).get('reason', '')}")
            if regime_modifier < 1.0:
                reasons.append(f"Piyasa rejimi baskilayici ({self.market_regime})")
            veto_reason = " | ".join(reasons) if reasons else "Bilinmeyen sebep"

        risk_decision = risk.get("decision", {})

        agent_votes = self._build_agent_votes(
            symbol, direction, technical_confidence, sentiment_score,
            volume_score, risk_score, correlation_penalty, regime_modifier,
            signal, sentiment, risk_decision, approved
        )

        return {
            "symbol": symbol,
            "direction": direction,
            "composite_score": round(composite, 3),
            "approved": approved,
            "veto_reason": veto_reason,
            "confidence_level": self._confidence_level(composite),
            "breakdown": {
                "technical": round(technical_confidence, 3),
                "sentiment": round(sentiment_score, 3),
                "volume": round(volume_score, 3),
                "risk_reward": round(risk_score, 3),
                "correlation_penalty": round(correlation_penalty, 3),
                "regime_modifier": round(regime_modifier, 3),
            },
            "risk_params": {
                "position_size_usd": risk_decision.get("position_size_usd", 0),
                "leverage": risk_decision.get("leverage", 1),
                "stop_loss": risk_decision.get("stop_loss", 0),
                "take_profit": risk_decision.get("take_profit", 0),
                "risk_reward_ratio": risk_decision.get("risk_reward_ratio", 0),
            } if approved else {},
            "agent_votes": agent_votes,
            "reasons": signal.get("reasons", []),
            "indicators": signal.get("indicators", {}),
            "price": signal.get("price", 0),
            "timestamp": time.time(),
        }

    def _build_agent_votes(
        self, symbol, direction, tech_score, sent_score, vol_score,
        risk_score, corr_score, regime_mod, signal, sentiment, risk_decision, approved
    ) -> dict:
        tech_vote = "APPROVE" if tech_score >= 0.3 else "REJECT"
        tech_detail = f"Teknik analiz guven: %{tech_score*100:.0f}"
        if tech_score >= 0.5:
            tech_detail += " - Guclu sinyal tespit edildi"
        elif tech_score >= 0.3:
            tech_detail += " - Orta guclu sinyal"
        else:
            tech_detail += " - Zayif sinyal, net yok"

        sent_vote = "APPROVE" if sent_score >= 0.4 else "REJECT"
        sent_detail = f"Sentiment skoru: %{sent_score*100:.0f}"
        mood = sentiment.get("market_mood", "neutral")
        if mood == "extreme_fear":
            sent_detail += " - Piyasa asiri korkuda, short icin firsat"
        elif mood == "fear":
            sent_detail += " - Korku hakim, dikkatli giris"
        elif mood == "extreme_greed":
            sent_detail += " - Piyasa asiri aclgozlu模nda, duzeltme riski"
        else:
            sent_detail += " - Notr piyasa"

        risk_vote = "APPROVE" if risk_decision.get("approved") else "REJECT"
        risk_detail = ""
        if risk_decision.get("approved"):
            rr = risk_decision.get("risk_reward_ratio", 0)
            lev = risk_decision.get("leverage", 1)
            sz = risk_decision.get("position_size_usd", 0)
            risk_detail = f"Risk onaylandi - RR: {rr:.1f}, Kaldirac: {lev}x, Boyut: ${sz:.0f}"
        else:
            risk_detail = f"Risk reddetti: {risk_decision.get('reason', 'N/A')}"

        vol_vote = "APPROVE" if vol_score >= 0.3 else "NEUTRAL"
        vol_detail = f"Hacim skoru: %{vol_score*100:.0f}"

        corr_vote = "APPROVE" if corr_score >= 0.5 else "REJECT"
        corr_detail = f"Korelasyon cezasi: %{corr_score*100:.0f}"

        regime_vote = "APPROVE" if regime_mod >= 0.8 else "REJECT"
        regime_detail = f"Rejim etkisi: %{regime_mod*100:.0f} - {self.market_regime}"

        final_vote = "APPROVE" if approved else "REJECT"
        final_detail = ""
        if approved:
            final_detail = f"Patron ONAYLADI - Composite: %{tech_score*0.4+sent_score*0.15+vol_score*0.15+risk_score*0.2+corr_score*0.1:.0f} x {regime_mod:.2f} = %{tech_score*0.4+sent_score*0.15+vol_score*0.15+risk_score*0.2+corr_score*0.1*regime_mod:.0f}"
        else:
            final_detail = "Patron REDDETTI - Guven esigi asilmadi"

        return {
            "technical": {"vote": tech_vote, "detail": tech_detail, "score": round(tech_score, 3)},
            "sentiment": {"vote": sent_vote, "detail": sent_detail, "score": round(sent_score, 3)},
            "risk": {"vote": risk_vote, "detail": risk_detail, "score": round(risk_score, 3)},
            "volume": {"vote": vol_vote, "detail": vol_detail, "score": round(vol_score, 3)},
            "correlation": {"vote": corr_vote, "detail": corr_detail, "score": round(corr_score, 3)},
            "regime": {"vote": regime_vote, "detail": regime_detail, "score": round(regime_mod, 3)},
            "patron": {"vote": final_vote, "detail": final_detail},
        }

    def _score_sentiment(self, sentiment: dict, direction: str) -> float:
        overall = sentiment.get("overall_sentiment", 0)
        if direction == "long":
            return max(0.0, (overall + 1) / 2)
        elif direction == "short":
            return max(0.0, (1 - overall) / 2)
        return 0.5

    def _score_volume(self, scanner: dict, symbol: str) -> float:
        hot_pairs = scanner.get("hot_pairs", [])
        for pair in hot_pairs:
            if pair.get("symbol") == symbol:
                return min(pair.get("volume_score", 0) / 10.0, 1.0)
        return 0.3

    def _score_risk(self, risk: dict, symbol: str) -> float:
        decision = risk.get("decision", {})
        if not decision.get("approved"):
            return 0.0
        rr = decision.get("risk_reward_ratio", 0)
        return min(rr / 3.0, 1.0)

    def _score_correlation(self, portfolio: dict, symbol: str, direction: str) -> float:
        positions = portfolio.get("open_positions", [])
        same_dir = sum(
            1 for p in positions
            if p.get("direction") == direction and p.get("symbol") != symbol
        )
        total = len(positions)
        if total == 0:
            return 1.0
        correlation = same_dir / total
        return max(0.0, 1.0 - correlation)

    def _get_regime_modifier(self, direction: str) -> float:
        if self.market_regime == "extreme_fear":
            return 1.2 if direction == "long" else 0.5
        elif self.market_regime == "fear":
            return 1.1 if direction == "long" else 0.7
        elif self.market_regime == "extreme_greed":
            return 0.5 if direction == "long" else 1.2
        elif self.market_regime == "greed":
            return 0.7 if direction == "long" else 1.1
        return 1.0

    def _confidence_level(self, score: float) -> str:
        if score >= 0.85:
            return "very_high"
        elif score >= 0.75:
            return "high"
        elif score >= 0.60:
            return "medium"
        elif score >= 0.40:
            return "low"
        return "very_low"

    def _build_thinking(self) -> list[str]:
        steps = []
        steps.append(f"Piyasa rejimi: {self.market_regime}")
        steps.append(f"Toplam sinyal sayisi: {len(self.decisions)}")
        approved = [d for d in self.decisions if d["approved"]]
        steps.append(f"Onaylanan: {len(approved)}, Reddedilen: {len(self.decisions) - len(approved)}")
        if approved:
            best = approved[0]
            steps.append(f"En guclu sinyal: {best['symbol']} {best['direction'].upper()} - Composite: %{best['composite_score']*100:.0f}")
        return steps

    def _build_summary(self) -> str:
        approved = [d for d in self.decisions if d["approved"]]
        if not approved:
            return "Bu dongude onaylanan sinyal yok. Piyasa kosullari uygun degil."
        syms = ", ".join([f"{d['symbol']} {d['direction'].upper()}" for d in approved[:5]])
        return f"{len(approved)} sinyal onaylandi: {syms}"

    def log_decision(self, decision: dict):
        self.trade_log.append({
            **decision,
            "logged_at": time.time(),
        })
        if len(self.trade_log) > 500:
            self.trade_log = self.trade_log[-500:]
