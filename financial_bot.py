"""
🧠 Bot d'Analyse Financière - Version Minimale
Compatible Render - Sans opérations pandas complexes
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
import requests
import json
import yfinance as yf
from aiohttp import web
import aiohttp_cors

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalBot:
    """Bot minimal qui fonctionne sur Render"""
    
    ASSETS = {
        'NASDAQ': '^IXIC',
        'SP500': '^GSPC', 
        'GOLD': 'GC=F',
        'OIL': 'CL=F'
    }
    
    def __init__(self):
        self.api_key = os.getenv('CALLMEBOT_API_KEY', '6823355')
        self.phone = os.getenv('WHATSAPP_PHONE', '33782707825')
        self.is_running = False
        self.last_analysis = None
        
    def get_price_data(self, symbol: str) -> dict:
        """Récupère juste les prix sans calculs complexes"""
        try:
            # Récupération simple
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d', interval='1h')
            
            if hist.empty:
                return {"error": "Pas de données"}
            
            # Conversion simple en listes Python (pas de pandas operations)
            closes = hist['Close'].tolist()
            volumes = hist['Volume'].tolist()
            
            current_price = closes[-1]
            prev_price = closes[-24] if len(closes) >= 24 else closes[0]
            change_24h = ((current_price / prev_price) - 1) * 100
            
            # Moyennes simples avec Python pur
            recent_closes = closes[-10:] if len(closes) >= 10 else closes
            sma_10 = sum(recent_closes) / len(recent_closes)
            
            longer_closes = closes[-20:] if len(closes) >= 20 else closes
            sma_20 = sum(longer_closes) / len(longer_closes)
            
            # Volume simple
            recent_volumes = volumes[-20:] if len(volumes) >= 20 else volumes
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
            
            return {
                "price": round(current_price, 2),
                "change_24h": round(change_24h, 2),
                "sma_10": round(sma_10, 2),
                "sma_20": round(sma_20, 2),
                "volume_ratio": round(volume_ratio, 2)
            }
            
        except Exception as e:
            logger.error(f"Erreur prix {symbol}: {e}")
            return {"error": str(e)}
    
    def analyze_simple(self, data: dict) -> dict:
        """Analyse ultra-simple"""
        if "error" in data:
            return data
            
        price = data['price']
        sma_10 = data['sma_10']
        sma_20 = data['sma_20']
        change_24h = data['change_24h']
        volume_ratio = data['volume_ratio']
        
        # Tendance simple
        if price > sma_10 > sma_20:
            trend = "📈 HAUSSIERE"
            signal = "🟢 BULLISH"
        elif price < sma_10 < sma_20:
            trend = "📉 BAISSIERE"
            signal = "🔴 BEARISH"
        else:
            trend = "➡️ NEUTRE"
            signal = "🟡 NEUTRE"
        
        # Alertes simples
        alerts = []
        if abs(change_24h) > 2:
            alerts.append(f"💥 Mouvement fort: {change_24h:+.1f}%")
        
        if volume_ratio > 2:
            alerts.append(f"📊 Volume élevé: {volume_ratio:.1f}x")
        
        return {
            "trend": trend,
            "signal": signal,
            "alerts": alerts,
            "strength": abs(change_24h)
        }
    
    def send_whatsapp(self, message: str) -> bool:
        """Envoie message WhatsApp"""
        try:
            url = "https://api.callmebot.com/whatsapp.php"
            params = {
                'phone': self.phone,
                'text': message[:1000],
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            success = response.status_code == 200
            
            if success:
                logger.info("✅ Message envoyé")
            else:
                logger.error(f"❌ Erreur: {response.status_code}")
                
            return success
            
        except Exception as e:
            logger.error(f"Erreur WhatsApp: {e}")
            return False
    
    def generate_report(self) -> str:
        """Génère un rapport simple"""
        report = f"🧠 MARCHE ANALYSE - {datetime.now().strftime('%H:%M')}\n"
        report += "=" * 25 + "\n\n"
        
        important_alerts = []
        
        for asset_name, symbol in self.ASSETS.items():
            try:
                # Récupération et analyse
                price_data = self.get_price_data(symbol)
                
                if "error" not in price_data:
                    analysis = self.analyze_simple(price_data)
                    
                    # Affichage
                    report += f"{analysis['trend']} {asset_name}\n"
                    report += f"Prix: ${price_data['price']} ({price_data['change_24h']:+.1f}%)\n"
                    report += f"Signal: {analysis['signal']}\n"
                    
                    # Alertes
                    for alert in analysis['alerts']:
                        report += f"   {alert}\n"
                        if analysis['strength'] > 3:
                            important_alerts.append(f"{asset_name}: {alert}")
                    
                    report += "\n"
                else:
                    report += f"❌ {asset_name}: Erreur données\n\n"
                    
            except Exception as e:
                report += f"❌ {asset_name}: {str(e)[:50]}\n\n"
        
        # Résumé des alertes importantes
        if important_alerts:
            report += "🔥 ALERTES IMPORTANTES:\n"
            for alert in important_alerts[:3]:
                report += f"• {alert}\n"
            report += "\n"
        
        report += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        return report
    
    async def auto_analysis(self):
        """Analyse automatique"""
        try:
            logger.info("🔄 Analyse en cours...")
            
            report = self.generate_report()
            self.last_analysis = report
            
            success = self.send_whatsapp(report)
            
            if success:
                logger.info("✅ Rapport envoyé")
            else:
                logger.error("❌ Échec envoi")
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse: {e}")
    
    async def start(self):
        """Démarre le bot"""
        if self.is_running:
            return
            
        logger.info("🚀 Démarrage...")
        self.is_running = True
        
        # Message de démarrage
        start_msg = "🤖 Bot d'analyse financière démarré!\n\nVous recevrez des rapports automatiques."
        self.send_whatsapp(start_msg)
        
        # Première analyse
        await self.auto_analysis()
        
        logger.info("✅ Bot démarré")
    
    def stop(self):
        """Arrête le bot"""
        self.is_running = False
        logger.info("🔴 Bot arrêté")

# Instance globale
bot = MinimalBot()

# API Web minimal
async def health_check(request):
    return web.json_response({
        "status": "running" if bot.is_running else "stopped",
        "timestamp": datetime.now().isoformat()
    })

async def analyze_endpoint(request):
    asset = request.query.get('asset')
    
    if asset and asset.upper() in bot.ASSETS:
        symbol = bot.ASSETS[asset.upper()]
        price_data = bot.get_price_data(symbol)
        
        if "error" not in price_data:
            analysis = bot.analyze_simple(price_data)
            
            response = f"📊 {asset.upper()}\n"
            response += f"Prix: ${price_data['price']} ({price_data['change_24h']:+.1f}%)\n"
            response += f"Tendance: {analysis['trend']}\n"
            response += f"Signal: {analysis['signal']}"
            
            return web.Response(text=response, content_type='text/plain; charset=utf-8')
        else:
            return web.Response(text=f"❌ Erreur: {price_data['error']}", status=500)
    else:
        # Rapport complet
        if bot.last_analysis:
            return web.Response(text=bot.last_analysis, content_type='text/plain; charset=utf-8')
        else:
            report = bot.generate_report()
            return web.Response(text=report, content_type='text/plain; charset=utf-8')

async def start_endpoint(request):
    await bot.start()
    return web.json_response({"message": "Bot démarré"})

async def stop_endpoint(request):
    bot.stop()
    return web.json_response({"message": "Bot arrêté"})

async def manual_trigger(request):
    """Déclenche une analyse manuelle"""
    await bot.auto_analysis()
    return web.json_response({"message": "Analyse déclenchée"})

# Configuration de l'app web
def create_app():
    app = web.Application()
    
    # Routes
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    app.router.add_get('/analyze', analyze_endpoint)
    app.router.add_post('/start', start_endpoint)
    app.router.add_post('/stop', stop_endpoint)
    app.router.add_post('/trigger', manual_trigger)
    
    # CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

# Fonction principale
async def main():
    """Point d'entrée principal"""
    
    # Démarrer le bot
    await bot.start()
    
    # Créer l'app web
    app = create_app()
    
    # Démarrer le serveur
    runner = web.AppRunner(app)
    await runner.setup()
    
    port = int(os.getenv('PORT', 8000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"🌐 Serveur démarré sur le port {port}")
    
    # Planifier les analyses périodiques (toutes les 15 minutes)
    while True:
        try:
            await asyncio.sleep(900)  # 15 minutes
            if bot.is_running:
                await bot.auto_analysis()
        except Exception as e:
            logger.error(f"Erreur loop: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Arrêt demandé")
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
