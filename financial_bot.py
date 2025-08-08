"""
🧠 Bot d'Analyse Financière IA - Version Simplifiée Compatible
Version sans TA-Lib pour déploiement facile sur Render
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Configuration des alertes"""
    whatsapp_enabled: bool = True
    auto_frequency_minutes: int = 15
    min_breakout_strength: float = 0.6
    ai_confidence_threshold: float = 65.0
    volume_spike_threshold: float = 1.5

@dataclass
class MarketAlert:
    """Structure d'une alerte de marché"""
    asset: str
    alert_type: str
    message: str
    confidence: float
    timestamp: datetime
    timeframe: str
    price: float

class SimplifiedMarketAnalyzer:
    """Analyseur de marché simplifié sans TA-Lib"""
    
    ASSETS = {
        'NASDAQ': '^IXIC',
        'SP500': '^GSPC', 
        'GOLD': 'GC=F',
        'WTI_OIL': 'CL=F',
        'VIX': '^VIX',
        'DXY': 'DX-Y.NYB'
    }
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.market_data_cache = {}
        
    def calculate_simple_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les indicateurs techniques sans TA-Lib"""
        try:
            # RSI simplifié
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Moyennes mobiles
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD simplifié
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands simplifiées
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Support/Résistance
            df['resistance'] = df['High'].rolling(window=20).max()
            df['support'] = df['Low'].rolling(window=20).min()
            
            # Volume
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_spike'] = df['Volume'] / df['volume_sma']
            
            # Volatilité (ATR simplifié)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            return df
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs: {e}")
            return df
    
    async def get_market_data(self, symbol: str, period: str = '5d', interval: str = '15m') -> pd.DataFrame:
        """Récupère les données de marché avec cache"""
        cache_key = f"{symbol}_{period}_{interval}"
        
        if cache_key in self.market_data_cache:
            cached_data, timestamp = self.market_data_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=5):
                return cached_data
        
        try:
            data = yf.download(symbol, period=period, interval=interval)
            if not data.empty:
                data = self.calculate_simple_indicators(data)
                self.market_data_cache[cache_key] = (data, datetime.now())
            return data
        except Exception as e:
            logger.error(f"❌ Erreur récupération données {symbol}: {e}")
            return pd.DataFrame()
    
    def detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """Détecte la structure de marché"""
        if len(df) < 50:
            return {"trend": "insufficient_data", "structure": "unknown"}
        
        current_price = df['Close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df and not pd.isna(df['sma_20'].iloc[-1]) else current_price
        ema_12 = df['ema_12'].iloc[-1] if 'ema_12' in df and not pd.isna(df['ema_12'].iloc[-1]) else current_price
        
        # Détection de tendance
        short_trend = "bullish" if current_price > ema_12 else "bearish"
        medium_trend = "bullish" if current_price > sma_20 else "bearish"
        
        # Détection de range
        recent_highs = df['High'].tail(20).max()
        recent_lows = df['Low'].tail(20).min()
        range_size = (recent_highs - recent_lows) / current_price * 100
        
        is_ranging = range_size < 3.0
        
        # Force de la tendance
        price_momentum = (current_price / df['Close'].iloc[-20] - 1) * 100 if len(df) >= 20 else 0
        
        # Détection de breakout
        breakout_strength = 0
        if 'resistance' in df and 'support' in df and 'atr' in df:
            resistance = df['resistance'].iloc[-2] if not pd.isna(df['resistance'].iloc[-2]) else current_price
            support = df['support'].iloc[-2] if not pd.isna(df['support'].iloc[-2]) else current_price
            atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else 1.0
            
            if current_price > resistance and atr > 0:
                breakout_strength = min((current_price - resistance) / atr, 3.0)
            elif current_price < support and atr > 0:
                breakout_strength = -min((support - current_price) / atr, 3.0)
        
        return {
            "trend": short_trend,
            "trend_strength": abs(price_momentum),
            "is_ranging": is_ranging,
            "range_size": range_size,
            "breakout_strength": breakout_strength,
            "support": df['support'].iloc[-1] if 'support' in df else current_price * 0.95,
            "resistance": df['resistance'].iloc[-1] if 'resistance' in df else current_price * 1.05,
            "current_price": current_price
        }
    
    async def initialize_ai_models(self):
        """Initialise les modèles IA simplifiés"""
        logger.info("🤖 Initialisation des modèles IA...")
        
        for asset_name, symbol in self.ASSETS.items():
            try:
                data = yf.download(symbol, period='3mo', interval='1h')
                if len(data) > 100:
                    features, targets = self._prepare_simple_features(data)
                    if len(features) > 50:
                        model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
                        model.fit(features, targets)
                        self.ml_models[asset_name] = model
                        logger.info(f"✅ Modèle IA créé pour {asset_name}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur modèle IA {asset_name}: {e}")
    
    def _prepare_simple_features(self, data: pd.DataFrame):
        """Prépare les features simplifiées pour l'IA"""
        df = data.copy()
        df = self.calculate_simple_indicators(df)
        
        # Features simples
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['price_change'] = df['Close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # Target
        df['future_return'] = df['Close'].shift(-2) / df['Close'] - 1
        df['target'] = (df['future_return'] > 0.002).astype(int)
        
        df = df.dropna()
        
        feature_cols = ['rsi', 'macd', 'sma_20', 'ema_12', 'hour', 'day_of_week', 'volatility']
        available_cols = [col for col in feature_cols if col in df.columns and not df[col].isna().all()]
        
        if len(available_cols) < 3:
            return np.array([]), np.array([])
            
        return df[available_cols].values, df['target'].values
    
    def get_ai_prediction(self, asset_name: str, df: pd.DataFrame) -> Dict:
        """Génère une prédiction IA simplifiée"""
        if asset_name not in self.ml_models or len(df) < 20:
            return {"direction": "neutral", "confidence": 50.0}
        
        try:
            # Préparer les features actuelles
            current_features = []
            
            # RSI
            if 'rsi' in df and not pd.isna(df['rsi'].iloc[-1]):
                current_features.append(df['rsi'].iloc[-1])
            else:
                current_features.append(50.0)
            
            # MACD
            if 'macd' in df and not pd.isna(df['macd'].iloc[-1]):
                current_features.append(df['macd'].iloc[-1])
            else:
                current_features.append(0.0)
            
            # SMA 20
            if 'sma_20' in df and not pd.isna(df['sma_20'].iloc[-1]):
                current_features.append(df['sma_20'].iloc[-1])
            else:
                current_features.append(df['Close'].iloc[-1])
            
            # EMA 12
            if 'ema_12' in df and not pd.isna(df['ema_12'].iloc[-1]):
                current_features.append(df['ema_12'].iloc[-1])
            else:
                current_features.append(df['Close'].iloc[-1])
            
            # Heure et jour
            current_features.append(datetime.now().hour)
            current_features.append(datetime.now().weekday())
            
            # Volatilité
            volatility = df['Close'].pct_change().rolling(window=10).std().iloc[-1]
            current_features.append(volatility if not pd.isna(volatility) else 0.02)
            
            # Prédiction
            model = self.ml_models[asset_name]
            prediction_proba = model.predict_proba([current_features])[0]
            
            direction = "bullish" if prediction_proba[1] > 0.5 else "bearish"
            confidence = max(prediction_proba) * 100
            
            return {
                "direction": direction,
                "confidence": round(confidence, 1),
                "bullish_prob": round(prediction_proba[1] * 100, 1),
                "bearish_prob": round(prediction_proba[0] * 100, 1)
            }
        except Exception as e:
            logger.error(f"Erreur prédiction IA {asset_name}: {e}")
            return {"direction": "neutral", "confidence": 50.0}
    
    async def analyze_asset(self, asset_name: str, symbol: str) -> List[MarketAlert]:
        """Analyse complète d'un actif"""
        alerts = []
        
        try:
            data_15m = await self.get_market_data(symbol, period='5d', interval='15m')
            data_1h = await self.get_market_data(symbol, period='10d', interval='1h')
            
            if data_15m.empty or data_1h.empty:
                return alerts
            
            structure_15m = self.detect_market_structure(data_15m)
            structure_1h = self.detect_market_structure(data_1h)
            ai_prediction = self.get_ai_prediction(asset_name, data_1h)
            
            current_price = structure_15m['current_price']
            
            # ALERTE BREAKOUT
            if abs(structure_15m['breakout_strength']) > self.config.min_breakout_strength:
                direction = "📈 HAUSSIERE" if structure_15m['breakout_strength'] > 0 else "📉 BAISSIERE"
                alert = MarketAlert(
                    asset=asset_name,
                    alert_type="BREAKOUT",
                    message=f"💥 BREAKOUT {direction}\n"
                           f"Force: {abs(structure_15m['breakout_strength']):.1f}\n"
                           f"Prix: ${current_price:.2f}",
                    confidence=min(abs(structure_15m['breakout_strength']) * 30, 95),
                    timestamp=datetime.now(),
                    timeframe="15m",
                    price=current_price
                )
                alerts.append(alert)
            
            # ALERTE RANGE
            if structure_15m['is_ranging'] and structure_1h['is_ranging']:
                alert = MarketAlert(
                    asset=asset_name,
                    alert_type="RANGE",
                    message=f"🧱 RANGE DETECTE\n"
                           f"Support: ${structure_15m['support']:.2f}\n"
                           f"Résistance: ${structure_15m['resistance']:.2f}\n"
                           f"Taille: {structure_15m['range_size']:.1f}%",
                    confidence=70.0,
                    timestamp=datetime.now(),
                    timeframe="15m-1h",
                    price=current_price
                )
                alerts.append(alert)
            
            # ALERTE IA
            if ai_prediction['confidence'] > self.config.ai_confidence_threshold:
                emoji = "🔮📈" if ai_prediction['direction'] == "bullish" else "🔮📉"
                alert = MarketAlert(
                    asset=asset_name,
                    alert_type="AI_SIGNAL",
                    message=f"{emoji} SIGNAL IA\n"
                           f"Direction: {ai_prediction['direction'].upper()}\n"
                           f"Confiance: {ai_prediction['confidence']:.1f}%",
                    confidence=ai_prediction['confidence'],
                    timestamp=datetime.now(),
                    timeframe="1h",
                    price=current_price
                )
                alerts.append(alert)
            
            # ALERTE VOLUME
            if 'volume_spike' in data_15m:
                latest_volume_spike = data_15m['volume_spike'].iloc[-1]
                if not pd.isna(latest_volume_spike) and latest_volume_spike > self.config.volume_spike_threshold:
                    alert = MarketAlert(
                        asset=asset_name,
                        alert_type="VOLUME",
                        message=f"📊 SPIKE DE VOLUME\n"
                               f"Volume: {latest_volume_spike:.1f}x la moyenne\n"
                               f"Prix: ${current_price:.2f}",
                        confidence=min(latest_volume_spike * 25, 90),
                        timestamp=datetime.now(),
                        timeframe="15m",
                        price=current_price
                    )
                    alerts.append(alert)
                    
        except Exception as e:
            logger.error(f"Erreur analyse {asset_name}: {e}")
        
        return alerts
    
    async def generate_market_report(self) -> str:
        """Génère un rapport de marché complet"""
        logger.info("📊 Génération du rapport de marché...")
        
        report = f"🧠 RAPPORT MARCHE IA - {datetime.now().strftime('%H:%M')}\n"
        report += "=" * 40 + "\n\n"
        
        all_alerts = []
        
        for asset_name, symbol in self.ASSETS.items():
            try:
                alerts = await self.analyze_asset(asset_name, symbol)
                all_alerts.extend(alerts)
                
                data = await self.get_market_data(symbol, period='2d', interval='1h')
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    change_24h = (current_price / data['Close'].iloc[-24] - 1) * 100 if len(data) >= 24 else 0
                    
                    emoji = "📈" if change_24h > 0 else "📉"
                    report += f"{emoji} {asset_name}: ${current_price:.2f} ({change_24h:+.2f}%)\n"
                    
                    asset_alerts = [a for a in alerts if a.asset == asset_name]
                    for alert in asset_alerts[:2]:
                        report += f"   • {alert.message.replace(chr(10), ' | ')}\n"
                    
                    report += "\n"
                    
            except Exception as e:
                logger.error(f"Erreur rapport {asset_name}: {e}")
        
        high_confidence_alerts = [a for a in all_alerts if a.confidence > 75]
        if high_confidence_alerts:
            report += "\n🔥 ALERTES IMPORTANTES:\n"
            for alert in high_confidence_alerts[:3]:
                report += f"• {alert.asset} - {alert.alert_type} ({alert.confidence:.0f}%)\n"
        
        report += f"\n⏰ Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        report += f"\n🤖 Modèles IA actifs: {len(self.ml_models)}"
        
        return report

class NotificationManager:
    """Gestionnaire des notifications"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.whatsapp_api_key = os.getenv('CALLMEBOT_API_KEY')
        self.whatsapp_phone = os.getenv('WHATSAPP_PHONE')
    
    async def send_whatsapp(self, message: str) -> bool:
        """Envoie un message WhatsApp via CallMeBot"""
        if not self.config.whatsapp_enabled or not self.whatsapp_api_key:
            return False
            
        try:
            url = "https://api.callmebot.com/whatsapp.php"
            params = {
                'phone': self.whatsapp_phone,
                'text': message[:1000],
                'apikey': self.whatsapp_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            success = response.status_code == 200
            
            if success:
                logger.info("✅ Message WhatsApp envoyé")
            else:
                logger.error(f"❌ Erreur WhatsApp: {response.status_code}")
                
            return success
            
        except Exception as e:
            logger.error(f"Erreur WhatsApp: {e}")
            return False

class FinancialBot:
    """Bot principal d'analyse financière simplifié"""
    
    def __init__(self):
        self.config = AlertConfig()
        self.analyzer = SimplifiedMarketAnalyzer(self.config)
        self.notifier = NotificationManager(self.config)
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
    async def initialize(self):
        """Initialise le bot"""
        logger.info("🚀 Initialisation du Bot d'Analyse Financière IA")
        
        await self.analyzer.initialize_ai_models()
        
        self.scheduler.add_job(
            self.auto_analysis,
            'interval',
            minutes=self.config.auto_frequency_minutes,
            id='auto_analysis'
        )
        
        logger.info(f"⏰ Analyse automatique programmée toutes les {self.config.auto_frequency_minutes} minutes")
    
    async def auto_analysis(self):
        """Analyse automatique périodique"""
        try:
            logger.info("🔄 Début de l'analyse automatique...")
            
            report = await self.analyzer.generate_market_report()
            await self.notifier.send_whatsapp(report)
            
            logger.info("✅ Analyse automatique terminée")
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse automatique: {e}")
    
    async def manual_analysis(self, asset: str = None) -> str:
        """Analyse manuelle sur demande"""
        try:
            if asset and asset.upper() in self.analyzer.ASSETS:
                symbol = self.analyzer.ASSETS[asset.upper()]
                alerts = await self.analyzer.analyze_asset(asset.upper(), symbol)
                
                response = f"📊 ANALYSE {asset.upper()}\n" + "="*25 + "\n"
                
                if alerts:
                    for alert in alerts[:3]:
                        response += f"\n{alert.message}\n"
                else:
                    response += "\n✅ Aucune alerte particulière pour le moment."
                
                return response
            else:
                return await self.analyzer.generate_market_report()
                
        except Exception as e:
            logger.error(f"Erreur analyse manuelle: {e}")
            return f"❌ Erreur lors de l'analyse: {str(e)}"
    
    async def start(self):
        """Démarre le bot"""
        if self.is_running:
            return
            
        logger.info("🟢 Démarrage du bot...")
        
        await self.initialize()
        self.scheduler.start()
        self.is_running = True
        
        # Première analyse immédiate
        await self.auto_analysis()
        
        logger.info("✅ Bot démarré avec succès!")
    
    async def stop(self):
        """Arrête le bot"""
        if not self.is_running:
            return
            
        logger.info("🔴 Arrêt du bot...")
        self.scheduler.shutdown()
        self.is_running = False
        logger.info("✅ Bot arrêté")

# API Web simple
from aiohttp import web
import aiohttp_cors

class WebAPI:
    """API web pour contrôle du bot"""
    
    def __init__(self, bot: FinancialBot):
        self.bot = bot
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Configure les routes de l'API"""
        
        async def health_check(request):
            return web.json_response({
                "status": "running" if self.bot.is_running else "stopped",
                "models": len(self.bot.analyzer.ml_models),
                "timestamp": datetime.now().isoformat()
            })
        
        async def manual_analysis(request):
            asset = request.query.get('asset')
            result = await self.bot.manual_analysis(asset)
            return web.Response(text=result, content_type='text/plain; charset=utf-8')
        
        async def start_bot(request):
            await self.bot.start()
            return web.json_response({"message": "Bot démarré"})
        
        async def stop_bot(request):
            await self.bot.stop()
            return web.json_response({"message": "Bot arrêté"})
        
        self.app.router.add_get('/', health_check)
        self.app.router.add_get('/health', health_check)
        self.app.router.add_get('/analyze', manual_analysis)
        self.app.router.add_post('/start', start_bot)
        self.app.router.add_post('/stop', stop_bot)
        
        # CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)

async def main():
    """Fonction principale"""
    
    bot = FinancialBot()
    web_api = WebAPI(bot)
    
    try:
        await bot.start()
        
        # Démarrer le serveur web
        runner = web.AppRunner(web_api.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8000)))
        await site.start()
        
        logger.info(f"🌐 API web démarrée sur le port {os.getenv('PORT', 8000)}")
        logger.info("🤖 Bot d'analyse financière en cours d'exécution...")
        
        # Maintenir le bot en vie
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
    finally:
        await bot.stop()

if __name__ == "__main__":
    # Vérifier les variables d'environnement
    required_vars = ['CALLMEBOT_API_KEY', 'WHATSAPP_PHONE']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Variables d'environnement manquantes: {missing_vars}")
    else:
        asyncio.run(main())
