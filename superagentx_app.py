from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import os
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
import json
import re
from email_validator import validate_email, EmailNotValidError

app = FastAPI(title="SuperAgentX AI API", version="2.1.0")

# ==================== KONFIGURACE ====================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
API_MASTER_TOKEN = os.environ.get("API_MASTER_TOKEN", "7909127138")  # ZMĚNA: Použij tvůj token

# Kontrola API klíče
if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
    raise ValueError("❌ CRITICAL: Please set DEEPSEEK_API_KEY environment variable!")

# SMTP KONFIGURACE
SMTP_CONFIG = {
    "server": "smtp.hostinger.com",
    "port": 465,
    "username": "info@byte-wings.com",
    "password": SMTP_PASSWORD,
    "from_email": "ai-agent@byte-wings.com"
}

# Povolení CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DEEPSEEK CLIENT ====================
class DeepSeekClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
    
    async def execute(self, query_instruction: str, additional_params: dict = None) -> str:
        if additional_params is None:
            additional_params = {}
        
        max_tokens = additional_params.get('max_tokens', 800)
        temperature = additional_params.get('temperature', 0.7)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": query_instruction}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API Error {response.status}: {error_text}")

# Inicializace
deepseek_client = DeepSeekClient(DEEPSEEK_API_KEY)

# ==================== MODELY ====================
class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "fast"
    token: str

class TaskRequest(BaseModel):
    task: str
    schedule_time: str
    priority: Optional[str] = "medium"
    token: str

class EmailRequest(BaseModel):
    to: EmailStr
    subject: str
    body: str
    token: str

# ==================== MIDDLEWARE PRO AUTENTIZACI ====================
@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    # Seznam veřejných endpointů které NEPOTŘEBUJÍ token
    public_endpoints = [
        "/api/health",
        "/",
        "/docs", 
        "/redoc",
        "/openapi.json",
        "/favicon.ico"
    ]
    
    # Pokud je to veřejný endpoint, přeskoč autentizaci
    if request.url.path in public_endpoints:
        return await call_next(request)
    
    # OPTIONS requesty pro CORS také přeskoč
    if request.method == "OPTIONS":
        return await call_next(request)
    
    # Zkus získat token z různých zdrojů
    token = None
    
    # 1. Z Authorization headeru
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
    
    # 2. Z query parametrů (pro GET requesty)
    if not token and request.query_params.get("token"):
        token = request.query_params.get("token")
    
    # 3. Z JSON body (pro POST requesty)
    if not token and request.method == "POST":
        try:
            body = await request.body()
            if body:
                data = json.loads(body)
                token = data.get('token')
        except:
            pass
    
    # Ověř token
    if token and token == API_MASTER_TOKEN:
        return await call_next(request)
    
    # Pokud token chybí nebo je špatný
    return JSONResponse(
        status_code=401, 
        content={"error": "Unauthorized - invalid token"}
    )

# ==================== SMTP SLUŽBA ====================
class SMTPService:
    def __init__(self, config: dict):
        self.config = config
        self.enabled = bool(config.get("password"))
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        if not self.enabled:
            print("❌ Email service disabled - SMTP_PASSWORD not set")
            return False
        
        try:
            if not self.is_valid_email(to):
                print(f"❌ Neplatná emailová adresa: {to}")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.config['from_email']
            msg['To'] = to
            msg['Subject'] = subject
            
            html_body = f"""
            <!DOCTYPE html><html><head><meta charset="UTF-8"><style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 20px; border-radius: 0 0 10px 10px; }}
            .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}</style></head>
            <body><div class="container"><div class="header"><h2>🤖 AI Agent Notification</h2></div>
            <div class="content">{body.replace('\n', '<br>')}</div>
            <div class="footer">Tento email byl vygenerován automaticky AI agentem<br>{datetime.now().strftime('%d.%m.%Y %H:%M')}</div></div></body></html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP_SSL(self.config['server'], self.config['port']) as server:
                server.login(self.config['username'], self.config['password'])
                server.send_message(msg)
            
            print(f"✅ Email úspěšně odeslán na: {to}")
            return True
            
        except Exception as e:
            print(f"❌ SMTP Error: {e}")
            return False
    
    def is_valid_email(self, email: str) -> bool:
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False

smtp_service = SMTPService(SMTP_CONFIG)

# ==================== PLÁNOVAČ ÚKOLŮ ====================
class TaskScheduler:
    def __init__(self):
        self.scheduled_tasks = []
    
    async def schedule_task(self, task_description: str, schedule_time: str, priority: str = "medium"):
        task_id = f"task_{len(self.scheduled_tasks) + 1}_{datetime.now().strftime('%H%M%S')}"
        task = {
            'id': task_id, 'description': task_description, 'scheduled_time': schedule_time,
            'priority': priority, 'status': 'scheduled', 'created_at': datetime.now().isoformat(),
            'created_at_readable': datetime.now().strftime('%d.%m.%Y %H:%M')
        }
        self.scheduled_tasks.append(task)
        print(f"✅ Úkol naplánován: {task_description} na {schedule_time}")
        return task_id
    
    async def get_scheduled_tasks(self):
        now = datetime.now()
        self.scheduled_tasks = [task for task in self.scheduled_tasks if (now - datetime.fromisoformat(task['created_at'].replace('Z', '+00:00'))).days < 7]
        return self.scheduled_tasks
    
    async def complete_task(self, task_id: str):
        for task in self.scheduled_tasks:
            if task['id'] == task_id:
                task['status'] = 'completed'
                task['completed_at'] = datetime.now().isoformat()
                task['completed_at_readable'] = datetime.now().strftime('%d.%m.%Y %H:%M')
                print(f"✅ Úkol dokončen: {task['description']}")
                return True
        return False

task_scheduler = TaskScheduler()

# ==================== API ENDPOINTS ====================
@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        print(f"💬 Chat request: {request.message[:50]}... | Mode: {request.mode}")
        
        additional_params = {}
        if request.mode == "deep":
            additional_params = {"max_tokens": 2000, "temperature": 0.3}
            print("🧠 Deep Thinking mode activated")
        else:
            additional_params = {"max_tokens": 800, "temperature": 0.7}
            print("⚡ Fast mode activated")
        
        response = await deepseek_client.execute(
            query_instruction=request.message,
            additional_params=additional_params
        )
        
        processed_response = await process_special_commands(request.message, response)
        
        return {
            "success": True,
            "response": processed_response,
            "mode": request.mode,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chyba: {str(e)}")

@app.post("/api/schedule-task")
async def schedule_task(request: TaskRequest):
    try:
        print(f"📅 Scheduling task: {request.task} for {request.schedule_time}")
        
        optimized_task = await deepseek_client.execute(
            query_instruction=f"Optimalizuj a uprav tento úkol pro lepší provedení: {request.task}",
            additional_params={"max_tokens": 300}
        )
        
        task_id = await task_scheduler.schedule_task(optimized_task, request.schedule_time, request.priority)
        
        confirmation = await deepseek_client.execute(
            query_instruction=f"Vytvoř přátelské potvrzení naplánování úkolu: {optimized_task} na čas: {request.schedule_time}",
            additional_params={"max_tokens": 150}
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "optimized_task": optimized_task,
            "confirmation": confirmation,
            "scheduled_time": request.schedule_time
        }
        
    except Exception as e:
        print(f"❌ Schedule error: {e}")
        raise HTTPException(status_code=500, detail=f"Chyba plánování: {str(e)}")

@app.post("/api/send-email")
async def send_email_api(request: EmailRequest):
    try:
        print(f"📧 Sending email to: {request.to}")
        
        if not smtp_service.enabled:
            raise HTTPException(status_code=503, detail="Email service is not configured")
        
        enhanced_email = await deepseek_client.execute(
            query_instruction=f"Vylepši tento email (předmět: {request.subject}): {request.body}",
            additional_params={"max_tokens": 500}
        )
        
        success = smtp_service.send_email(request.to, request.subject, enhanced_email)
        
        if success:
            return {
                "success": True,
                "message": "Email úspěšně odeslán",
                "to": request.to,
                "subject": request.subject
            }
        else:
            raise HTTPException(status_code=500, detail="Nepodařilo se odeslat email")
            
    except Exception as e:
        print(f"❌ Email error: {e}")
        raise HTTPException(status_code=500, detail=f"Chyba odesílání emailu: {str(e)}")

@app.get("/api/tasks")
async def get_tasks(token: str):
    if token != API_MASTER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    tasks = await task_scheduler.get_scheduled_tasks()
    return {"success": True, "tasks": tasks, "count": len(tasks)}

@app.post("/api/complete-task")
async def complete_task(request: dict):
    try:
        if request.get('token') != API_MASTER_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        task_id = request.get('task_id')
        if not task_id:
            raise HTTPException(status_code=400, detail="Task ID is required")
        
        success = await task_scheduler.complete_task(task_id)
        
        if success:
            return {"success": True, "message": "Úkol dokončen"}
        else:
            raise HTTPException(status_code=404, detail="Úkol nenalezen")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SuperAgentX API",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0",
        "features": {
            "chat": True,
            "scheduling": True,
            "email": smtp_service.enabled,
            "authentication": True,
            "deepseek_connected": bool(DEEPSEEK_API_KEY)
        },
        "environment": {
            "python_version": os.environ.get("PYTHON_VERSION", "Unknown"),
            "render": True
        }
    }

@app.get("/")
async def root():
    return {
        "message": "🤖 SuperAgentX API is running!",
        "version": "2.1.0",
        "endpoints": {
            "chat": "POST /api/chat",
            "schedule": "POST /api/schedule-task",
            "email": "POST /api/send-email",
            "tasks": "GET /api/tasks?token=YOUR_TOKEN",
            "health": "GET /api/health"
        },
        "security": {
            "authentication_required": True,
            "environment_variables_configured": {
                "DEEPSEEK_API_KEY": bool(DEEPSEEK_API_KEY),
                "SMTP_PASSWORD": bool(SMTP_PASSWORD),
                "API_MASTER_TOKEN": bool(API_MASTER_TOKEN)
            }
        }
    }

# ==================== POMOCNÉ FUNKCE ====================
async def process_special_commands(user_message: str, ai_response: str) -> str:
    """Automatické zpracování speciálních příkazů v chatu"""
    
    # Detekce emailového příkazu
    email_patterns = [
        r"pošli email (na|do) (.+?) (s předmětem|o) (.+?) (s obsahem|s textem) (.+)",
        r"send email to (.+?) (with subject|about) (.+?) (with content|with text) (.+)",
        r"napiš mail (na|do) (.+?) (předmět) (.+?) (text) (.+)"
    ]
    
    for pattern in email_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            email = match.group(2).strip()
            subject = match.group(4).strip()
            content = match.group(6).strip()
            
            # Ověření emailové adresy
            if smtp_service.is_valid_email(email):
                if smtp_service.enabled:
                    # Odeslání emailu
                    success = smtp_service.send_email(email, subject, content)
                    if success:
                        return f"{ai_response}\n\n📧 Email byl úspěšně odeslán na adresu: {email}"
                    else:
                        return f"{ai_response}\n\n❌ Nepodařilo se odeslat email. Zkontroluj SMTP nastavení."
                else:
                    return f"{ai_response}\n\n❌ Email služba není nakonfigurována. SMTP_PASSWORD není nastaveno."
            else:
                return f"{ai_response}\n\n❌ Neplatná emailová adresa: {email}"
    
    # Detekce plánování úkolu
    schedule_patterns = [
        r"naplánuj úkol (.+?) (na|pro) (.+)",
        r"schedule task (.+?) (for|on) (.+)",
        r"připomeň mi (.+?) (v|ve|na) (.+)"
    ]
    
    for pattern in schedule_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            task = match.group(1).strip()
            schedule_time = match.group(3).strip()
            
            try:
                task_id = await task_scheduler.schedule_task(task, schedule_time)
                return f"{ai_response}\n\n✅ Úkol naplánován: '{task}' na {schedule_time}\nID: {task_id}"
            except Exception as e:
                return f"{ai_response}\n\n❌ Chyba při plánování úkolu: {str(e)}"
    
    return ai_response

# ==================== ERROR HANDLING ====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "success": False}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "success": False}
    )

# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting SuperAgentX API Server...")
    print(f"🔐 API Master Token: {'Set' if API_MASTER_TOKEN else 'Not set'}")
    print(f"🤖 DeepSeek API Key: {'Set' if DEEPSEEK_API_KEY else 'Not set'}")
    print(f"📧 SMTP Email: {'Enabled' if smtp_service.enabled else 'Disabled'}")
    print(f"🌐 Environment: {os.environ.get('RENDER', 'Development')}")
    print("📋 Available endpoints:")
    print("   POST /api/chat - Chat with AI")
    print("   POST /api/schedule-task - Schedule tasks")
    print("   POST /api/send-email - Send emails")
    print("   GET  /api/tasks - List scheduled tasks")
    print("   GET  /api/health - Health check")
    print("   GET  / - API information")
    print("   GET  /docs - API documentation")
    print("   GET  /redoc - Alternative documentation")

# ==================== PERFORMANCE OPTIMIZATION ====================
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    response.headers["X-Process-Time"] = f"{process_time:.3f}s"
    print(f"⏱️  Request {request.method} {request.url.path} took {process_time:.3f}s")
    return response

# ==================== RENDER.COM SPECIFIC OPTIMIZATION ====================
# Pro Render.com free tier - optimalizace pro cold starts
@app.get("/api/warmup")
async def warmup():
    """Endpoint pro zahřátí aplikace na Render.com free tier"""
    try:
        # Rychlý test DeepSeek připojení
        test_response = await deepseek_client.execute(
            "Hello", 
            additional_params={"max_tokens": 10, "temperature": 0.1}
        )
        return {
            "success": True,
            "message": "API warmed up successfully",
            "deepseek_connected": True,
            "response_time": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Warmup failed: {str(e)}",
            "deepseek_connected": False
        }

if __name__ == "__main__":
    import uvicorn
    print("🔧 Running in development mode...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
