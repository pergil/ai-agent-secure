# 🤖 AI Agent API

FastAPI aplikace pro AI chat, plánování úkolů a odesílání emailů pomocí DeepSeek AI.

## 🚀 Deployment na Render.com

1. **Forkni tento repozitář**
2. **Jdi na [Render.com](https://render.com)**
3. **Propoj s GitHub účtem**
4. **Vytvoř nový Web Service**
5. **Nastav Environment Variables v Render dashboardu**

## 🔒 Environment Variables

Nastav následující proměnné v Render dashboardu:

- `DEEPSEEK_API_KEY` - TVŮJ_API_KLÍČ
- `SMTP_PASSWORD` - HESLO_PRO_SMTP
- `API_MASTER_TOKEN` - tomas-hulman-ai-agents-790912

## 📧 SMTP Configuration

Email služba používá Hostinger SMTP:
- Server: smtp.hostinger.com
- Port: 465
- Username: info@byte-wings.com
- From: ai-agent@byte-wings.com

## 🛡️ Bezpečnost

**NIKDY neukládej API klíče do kódu!** Vždy používej environment variables.
