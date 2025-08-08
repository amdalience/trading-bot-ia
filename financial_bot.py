"""
🧠 Bot d'Analyse Financière - Version Ultra-Simplifiée pour Replit
Fonctionne partout sans dépendances complexes
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
import requests
import json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import yfinance as yf
import pandas as pd
import numpy as np

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraSimpleBot:
    """Bot ultra-simplifié qui fonctionne partout"""
    
    ASSETS = {
        'NASDAQ': '^IXIC',
        'SP500': '^GSPC', 
        'GOLD': 'GC=F',
        'OIL': 'CL=F'
    }
    
    def __init__(self):
        # Vos credentials CallMeBot
        self.api_key = "6823355"
        self.phone = "33782707825"
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
    def get_simple_analysis(self, symbol: str) -> dict:
        """Analyse ultra-simple d'un actif"""
        try:
            # Récupérer les données
            data = yf.download(symbol, period='5d', interval='1h')
            
            if data.empty:
                return {"error": "Pas de données"}
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-24] if len(data) >= 24 else data['Close'].iloc[0]
            change_24h = (current_price / prev_price - 1) * 100
            
            # Moyennes simples
            sma_10 = data['Close'].tail(10).mean()
            sma_20 = data['Close'].tail(20).mean()
            
            # Tendance simple
            trend = "📈 HAUSSIERE" if current_price > sma_10 > sma_20 else "📉 BAISSIERE"
            
            # Volume
            avg_volume = data['Volume'].tail(20).mean()
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                "price": current_price,
                "change_24h": change_24h,
                "trend": trend,
                "sma_10": sma_10,
                "sma_20": sma_20,
                "volume_ratio": volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse {symbol}: {e}")
            return {"error": str(e)}
    
    def send_whatsapp(self, message: str) -> bool:
        """Envoie un message WhatsApp"""
        try:
            url = "https://api.callmebot.com/whatsapp.php"
            params = {
                'phone': self.phone,
                'text': message[:1000],
                'apikey': self.api_key
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
    
    def generate_report(self) -> str:
        """Génère un rapport simple"""
        report = f"🧠 RAPPORT MARCHE - {datetime.now().strftime('%H:%M')}\n"
        report += "=" * 30 + "\n\n"
        
        for asset_name, symbol in self.ASSETS.items():
            analysis = self.get_simple_analysis(symbol)
            
            if "error" not in analysis:
                report += f"{analysis['trend']} {asset_name}\n"
                report += f"Prix: ${analysis['price']:.2f} ({analysis['change_24h']:+.2f}%)\n"
                
                if analysis['volume_ratio'] > 1.5:
                    report += f"📊 Volume élevé: {analysis['volume_ratio']:.1f}x\n"
                
                # Signal simple
                if analysis['price'] > analysis['sma_10'] > analysis['sma_20']:
                    report += "🟢 Signal: BULLISH\n"
                elif analysis['price'] < analysis['sma_10'] < analysis['sma_20']:
                    report += "🔴 Signal: BEARISH\n"
                else:
                    report += "🟡 Signal: NEUTRE\n"
                
                report += "\n"
            else:
                report += f"❌ {asset_name}: {analysis['error']}\n\n"
        
        report += f"⏰ Mise à jour: {datetime.now().strftime('%H:%M:%S')}"
        return report
    
    async def auto_analysis(self):
        """Analyse automatique"""
        try:
            logger.info("🔄 Analyse automatique...")
            
            report = self.generate_report()
            self.send_whatsapp(report)
            
            logger.info("✅ Analyse terminée")
            
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
    
    async def start(self):
        """Démarre le bot"""
        if self.is_running:
            return
            
        logger.info("🚀 Démarrage du bot...")
        
        # Programmer les analyses toutes les 15 minutes
        self.scheduler.add_job(
            self.auto_analysis,
            'interval',
            minutes=15,
            id='auto_analysis'
        )
        
        self.scheduler.start()
        self.is_running = True
        
        # Première analyse immédiate
        await self.auto_analysis()
        
        logger.info("✅ Bot démarré!")
        
        # Test immédiat
        test_msg = "🤖 Bot d'analyse financière démarré!\nVous recevrez des rapports toutes les 15 minutes."
        self.send_whatsapp(test_msg)
    
    async def stop(self):
        """Arrête le bot"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("🔴 Bot arrêté")

# API Web simple pour Replit
from flask import Flask, jsonify, request

app = Flask(__name__)
bot = UltraSimpleBot()

@app.route('/')
def home():
    return jsonify({
        "status": "running" if bot.is_running else "stopped",
        "message": "Bot d'Analyse Financière IA",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "running" if bot.is_running else "stopped",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze')
def analyze():
    asset = request.args.get('asset')
    
    if asset and asset.upper() in bot.ASSETS:
        symbol = bot.ASSETS[asset.upper()]
        analysis = bot.get_simple_analysis(symbol)
        
        if "error" not in analysis:
            response = f"📊 ANALYSE {asset.upper()}\n"
            response += f"Prix: ${analysis['price']:.2f} ({analysis['change_24h']:+.2f}%)\n"
            response += f"Tendance: {analysis['trend']}\n"
            response += f"Volume: {analysis['volume_ratio']:.1f}x normal"
            return response
        else:
            return f"❌ Erreur: {analysis['error']}"
    else:
        return bot.generate_report()

@app.route('/start', methods=['POST'])
def start_bot():
    asyncio.create_task(bot.start())
    return jsonify({"message": "Bot démarré"})

@app.route('/stop', methods=['POST'])
def stop_bot():
    asyncio.create_task(bot.stop())
    return jsonify({"message": "Bot arrêté"})

# Fonction principale pour Replit
async def main():
    """Démarre automatiquement le bot"""
    await bot.start()
    
    # Maintenir le bot en vie
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        await bot.stop()

if __name__ == "__main__":
    # Pour Replit - démarre automatiquement
    import threading
    
    def run_bot():
        asyncio.run(main())
    
    # Démarrer le bot en arrière-plan
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Démarrer le serveur web
    app.run(host='0.0.0.0', port=8080, debug=False)
