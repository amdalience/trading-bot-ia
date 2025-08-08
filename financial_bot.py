"""
🧠 Bot d'Analyse Financière IA - Version Optimisée
Système intelligent d'analyse et d'alertes pour les marchés financiers
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Configuration des alertes personnalisable"""
    whatsapp_enabled: bool = True
    telegram_enabled: bool = False
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

class EnhancedMarketAnalyzer:
    """Analyseur de marché avec IA améliorée"""
    
    ASSETS = {
        'NASDAQ': '^IXIC',
        'SP500': '^GSPC', 
        'GOLD': 'GC=F',
        'WTI_OIL': 'CL=F',
        'BRENT': 'BZ=F',
        'VIX': '^VIX',  # Ajout volatilité
        'DXY': 'DX-Y.NYB'  # Index dollar
    }
    
    TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.alerts_history = []
        self.market_data_cache = {}
        
    async def initialize_ai_models(self):
        """Initialise les modèles IA pour chaque actif"""
        logger.info("🤖 Initialisation des modèles IA...")
        
        for asset_name, symbol in self.ASSETS.items():
            try:
                # Récupération données historiques pour entraînement
                data = yf.download(symbol, period='6mo', interval='1h')
                if len(data) > 100:
                    features, targets = self._prepare_ml_features(data)
                    if len(features) > 50:
                        model = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=10,
                            random_state=42
                        )
                        model.fit(features, targets)
                        self.ml_models[asset_name] = model
                        logger.info(f"✅ Modèle IA créé pour {asset_name}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur modèle IA {asset_name}: {e}")
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les features pour l'IA"""
        df = data.copy()
        
        # Indicateurs techniques
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['bb_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['bb_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Features temporelles
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        
        # Volume analysis
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price action
        df['price_change'] = df['Close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # Target: mouvement futur (1 = hausse, 0 = baisse)
        df['future_return'] = df['Close'].shift(-2) / df['Close'] - 1
        df['target'] = (df['future_return'] > 0.002).astype(int)  # +0.2% threshold
        
        # Nettoyage
        df = df.dropna()
        
        feature_cols = ['rsi', 'macd', 'sma_20', 'ema_12', 'atr', 'hour', 
                       'day_of_week', 'volume_ratio', 'volatility']
        
        return df[feature_cols].values, df['target'].values
    
    async def get_market_data(self, symbol: str, period: str = '5d', interval: str = '15m') -> pd.DataFrame:
        """Récupère les données de marché avec cache"""
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Vérifier cache (valide 5 min)
        if cache_key in self.market_data_cache:
            cached_data, timestamp = self.market_data_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=5):
                return cached_data
        
        try:
            data = yf.download(symbol, period=period, interval=interval)
            if not data.empty:
                # Calcul des indicateurs
                data = self._add_technical_indicators(data)
                self.market_data_cache[cache_key] = (data, datetime.now())
            return data
        except Exception as e:
            logger.error(f"❌ Erreur récupération données {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute tous les indicateurs techniques"""
        try:
            # Moyennes mobiles
            df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
            
            # Oscillateurs
            df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # Support/Résistance dynamiques
            df['resistance'] = df['High'].rolling(window=20).max()
            df['support'] = df['Low'].rolling(window=20).min()
            
            # Volume
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_spike'] = df['Volume'] / df['volume_sma']
            
            # Volatilité
            df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
            return df
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs: {e}")
            return df
    
    def detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """Détecte la structure de marché avancée"""
        if len(df) < 50:
            return {"trend": "insufficient_data", "structure": "unknown"}
        
        current_price = df['Close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        ema_12 = df['ema_12'].iloc[-1]
        
        # Détection de tendance multi-timeframe
        short_trend = "bullish" if current_price > ema_12 else "bearish"
        medium_trend = "bullish" if current_price > sma_20 else "bearish"
        
        # Détection de range
        recent_highs = df['High'].tail(20).max()
        recent_lows = df['Low'].tail(20).min()
        range_size = (recent_highs - recent_lows) / current_price * 100
        
        is_ranging = range_size < 3.0  # Range si volatilité < 3%
        
        # Force de la tendance
        price_momentum = (current_price / df['Close'].iloc[-20] - 1) * 100
        
        # Détection de breakout
        breakout_strength = 0
        if current_price > df['resistance'].iloc[-2]:
            breakout_strength = min((current_price - df['resistance'].iloc[-2]) / df['atr'].iloc[-1], 3.0)
        elif current_price < df['support'].iloc[-2]:
            breakout_strength = -min((df['support'].iloc[-2] - current_price) / df['atr'].iloc[-1], 3.0)
        
        return {
            "trend": short_trend,
            "trend_strength": abs(price_momentum),
            "is_ranging": is_ranging,
            "range_size": range_size,
            "breakout_strength": breakout_strength,
            "support": df['support'].iloc[-1],
            "resistance": df['resistance'].iloc[-1],
            "current_price": current_price
        }
    
    def get_ai_prediction(self, asset_name: str, df: pd.DataFrame) -> Dict:
        """Génère une prédiction IA"""
        if asset_name not in self.ml_models or len(df) < 20:
            return {"direction": "neutral", "confidence": 0.0}
        
        try:
            # Préparer les features actuelles
            current_features = [
                df['rsi'].iloc[-1],
                df['macd'].iloc[-1],
                df['sma_20'].iloc[-1],
                df['ema_12'].iloc[-1],
                df['atr'].iloc[-1],
                datetime.now().hour,
                datetime.now().weekday(),
                df['volume_spike'].iloc[-1] if 'volume_spike' in df else 1.0,
                df['Close'].pct_change().rolling(window=20).std().iloc[-1]
            ]
            
            # Vérifier les NaN
            if any(pd.isna(current_features)):
                return {"direction": "neutral", "confidence": 0.0}
            
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
            return {"direction": "neutral", "confidence": 0.0}
    
    async def analyze_asset(self, asset_name: str, symbol: str) -> List[MarketAlert]:
        """Analyse complète d'un actif"""
        alerts = []
        
        try:
            # Récupérer données multi-timeframes
            data_15m = await self.get_market_data(symbol, period='5d', interval='15m')
            data_1h = await self.get_market_data(symbol, period='10d', interval='1h')
            
            if data_15m.empty or data_1h.empty:
                return alerts
            
            # Analyse structure de marché
            structure_15m = self.detect_market_structure(data_15m)
            structure_1h = self.detect_market_structure(data_1h)
            
            # Prédiction IA
            ai_prediction = self.get_ai_prediction(asset_name, data_1h)
            
            current_price = structure_15m['current_price']
            
            # 🔥 ALERTE BREAKOUT
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
            
            # 📊 ALERTE RANGE
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
            
            # 🤖 ALERTE IA
            if ai_prediction['confidence'] > self.config.ai_confidence_threshold:
                emoji = "🔮📈" if ai_prediction['direction'] == "bullish" else "🔮📉"
                alert = MarketAlert(
                    asset=asset_name,
                    alert_type="AI_SIGNAL",
                    message=f"{emoji} SIGNAL IA\n"
                           f"Direction: {ai_prediction['direction'].upper()}\n"
                           f"Confiance: {ai_prediction['confidence']:.1f}%\n"
                           f"Probabilités: 📈{ai_prediction['bullish_prob']:.0f}% | 📉{ai_prediction['bearish_prob']:.0f}%",
                    confidence=ai_prediction['confidence'],
                    timestamp=datetime.now(),
                    timeframe="1h",
                    price=current_price
                )
                alerts.append(alert)
            
            # 📈 ALERTE VOLUME
            latest_volume_spike = data_15m['volume_spike'].iloc[-1] if 'volume_spike' in data_15m else 1.0
            if latest_volume_spike > self.config.volume_spike_threshold:
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
        
        # Analyse de chaque actif
        for asset_name, symbol in self.ASSETS.items():
            try:
                alerts = await self.analyze_asset(asset_name, symbol)
                all_alerts.extend(alerts)
                
                # Statut rapide de l'actif
                data = await self.get_market_data(symbol, period='2d', interval='1h')
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    change_24h = (current_price / data['Close'].iloc[-24] - 1) * 100 if len(data) >= 24 else 0
                    
                    emoji = "📈" if change_24h > 0 else "📉"
                    report += f"{emoji} {asset_name}: ${current_price:.2f} ({change_24h:+.2f}%)\n"
                    
                    # Ajouter alertes pour cet actif
                    asset_alerts = [a for a in alerts if a.asset == asset_name]
                    for alert in asset_alerts[:2]:  # Max 2 alertes par actif
                        report += f"   • {alert.message.replace(chr(10), ' | ')}\n"
                    
                    report += "\n"
                    
            except Exception as e:
                logger.error(f"Erreur rapport {asset_name}: {e}")
        
        # Résumé des alertes importantes
        high_confidence_alerts = [a for a in all_alerts if a.confidence > 75]
        if high_confidence_alerts:
            report += "\n🔥 ALERTES IMPORTANTES:\n"
            for alert in high_confidence_alerts[:3]:
                report += f"• {alert.asset} - {alert.alert_type} ({alert.confidence:.0f}%)\n"
        
        # Footer avec timestamp
        report += f"\n⏰ Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        report += f"\n🤖 Modèles IA actifs: {len(self.ml_models)}"
        
        return report

class NotificationManager:
    """Gestionnaire des notifications multi-canaux"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.whatsapp_api_key = os.getenv('CALLMEBOT_API_KEY')
        self.whatsapp_phone = os.getenv('WHATSAPP_PHONE')
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    async def send_whatsapp(self, message: str) -> bool:
        """Envoie un message WhatsApp via CallMeBot"""
        if not self.config.whatsapp_enabled or not self.whatsapp_api_key:
            return False
            
        try:
            url = "https://api.callmebot.com/whatsapp.php"
            params = {
                'phone': self.whatsapp_phone,
                'text': message[:1000],  # Limite de caractères
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
    
    async def send_telegram(self, message: str) -> bool:
        """Envoie un message Telegram"""
        if not self.config.telegram_enabled or not self.telegram_bot_token:
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=data, timeout=10)
            success = response.status_code == 200
            
            if success:
                logger.info("✅ Message Telegram envoyé")
            else:
                logger.error(f"❌ Erreur Telegram: {response.status_code}")
                
            return success
            
        except Exception as e:
            logger.error(f"Erreur Telegram: {e}")
            return False
    
    async def broadcast_alert(self, message: str):
        """Diffuse une alerte sur tous les canaux activés"""
        tasks = []
        
        if self.config.whatsapp_enabled:
            tasks.append(self.send_whatsapp(message))
            
        if self.config.telegram_enabled:
            tasks.append(self.send_telegram(message))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.info(f"📱 Alertes envoyées: {success_count}/{len(tasks)}")

class FinancialBot:
    """Bot principal d'analyse financière"""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config = self._load_config(config_path)
        self.analyzer = EnhancedMarketAnalyzer(self.config)
        self.notifier = NotificationManager(self.config)
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
    def _load_config(self, config_path: str) -> AlertConfig:
        """Charge la configuration depuis un fichier JSON"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    return AlertConfig(**data)
        except Exception as e:
            logger.warning(f"Erreur chargement config: {e}")
        
        return AlertConfig()  # Configuration par défaut
    
    async def initialize(self):
        """Initialise le bot et ses composants"""
        logger.info("🚀 Initialisation du Bot d'Analyse Financière IA")
        
        # Initialiser les modèles IA
        await self.analyzer.initialize_ai_models()
        
        # Configurer le planificateur
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
            
            # Générer le rapport
            report = await self.analyzer.generate_market_report()
            
            # Envoyer les notifications
            await self.notifier.broadcast_alert(report)
            
            logger.info("✅ Analyse automatique terminée")
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse automatique: {e}")
    
    async def manual_analysis(self, asset: str = None) -> str:
        """Analyse manuelle sur demande"""
        try:
            if asset and asset.upper() in self.analyzer.ASSETS:
                # Analyse d'un actif spécifique
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
                # Rapport complet
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

# API Web simple pour contrôle manuel
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
            return web.Response(text=result, content_type='text/plain')
        
        async def start_bot(request):
            await self.bot.start()
            return web.json_response({"message": "Bot démarré"})
        
        async def stop_bot(request):
            await self.bot.stop()
            return web.json_response({"message": "Bot arrêté"})
        
        # Routes
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

# Point d'entrée principal
async def main():
    """Fonction principale"""
    
    # Configuration par défaut (à modifier selon vos besoins)
    config = AlertConfig(
        whatsapp_enabled=True,
        auto_frequency_minutes=15,
        ai_confidence_threshold=65.0,
        min_breakout_strength=0.6
    )
    
    # Créer et démarrer le bot
    bot = FinancialBot()
    bot.config = config
    
    # Démarrer l'API web (optionnel)
    web_api = WebAPI(bot)
    
    try:
        # Démarrer le bot
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
    # Variables d'environnement requises
    required_vars = [
        'CALLMEBOT_API_KEY',  # Clé API CallMeBot
        'WHATSAPP_PHONE'      # Numéro WhatsApp
    ]
    
    # Vérifier les variables d'environnement
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Variables d'environnement manquantes: {missing_vars}")
        logger.info("Créez un fichier .env avec ces variables ou définissez-les dans votre système")
    
    # Démarrer le bot
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"💥 Erreur de démarrage: {e}")
