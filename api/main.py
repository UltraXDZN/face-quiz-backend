import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Set or clear emulator environment variables BEFORE any Firebase imports
if os.getenv("ENVIRONMENT") == "production":
    if "FIRESTORE_EMULATOR_HOST" in os.environ:
        del os.environ["FIRESTORE_EMULATOR_HOST"]
    if "FIREBASE_AUTH_EMULATOR_HOST" in os.environ:
        del os.environ["FIREBASE_AUTH_EMULATOR_HOST"]
    print("🔥 Production mode: Cleared emulator environment variables")
else:
    if os.getenv("FIRESTORE_EMULATOR_HOST"):
        os.environ["FIRESTORE_EMULATOR_HOST"] = os.getenv("FIRESTORE_EMULATOR_HOST")
    if os.getenv("FIREBASE_AUTH_EMULATOR_HOST"):
        os.environ["FIREBASE_AUTH_EMULATOR_HOST"] = os.getenv("FIREBASE_AUTH_EMULATOR_HOST")
    print("🔥 Development mode: Using emulator")

import firebase_admin
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials
from api.middleware.request_counter import RequestCounterMiddleware
from api.users.routes import router as users_router
from api.solutions.routes import router as solutions_router
from api.exams.routes import router as exams_router
from api.leaderboard.routes import router as leaderboard_router
from api.auth.routes import router as auth_router
from api.live.session import LiveSessionRegistry, BroadcastHub, Reaper

# Optional routers — may exist on production but not in local dev checkout
try:
    from api.media.routes import router as media_router
except ModuleNotFoundError:
    media_router = None

try:
    from api.tags.routes import router as tags_router
except ModuleNotFoundError:
    tags_router = None

from api.proctoring.routes import router as proctoring_router


# Initialize Firebase
if os.getenv("ENVIRONMENT") == "production":
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"))
    options = {}
    project_id = os.getenv("FIREBASE_TESTING_PROJECT_ID")
    if project_id:
        options["projectId"] = project_id
    storage_bucket = os.getenv("FIREBASE_TESTING_STORAGE_BUCKET")
    if storage_bucket:
        options["storageBucket"] = storage_bucket
    messaging_id = os.getenv("FIREBASE_TESTING_MESSAGING_SENDER_ID")
    if messaging_id:
        options["messagingSenderId"] = messaging_id
    app_id = os.getenv("FIREBASE_TESTING_APP_ID")
    if app_id:
        options["appId"] = app_id

    api_key = os.getenv("FIREBASE_TESTING_API_KEY")
    if api_key:
        os.environ["FIREBASE_TESTING_API_KEY"] = api_key
    auth_domain = os.getenv("FIREBASE_TESTING_AUTH_DOMAIN")
    if auth_domain:
        os.environ["FIREBASE_TESTING_AUTH_DOMAIN"] = auth_domain

    if options:
        firebase_admin.initialize_app(cred, options=options)
        print(f"🚀 Using Production Firebase with options: {options}")
    else:
        firebase_admin.initialize_app(cred)
        print("🚀 Using Production Firebase")
else:
    os.environ["FIRESTORE_EMULATOR_HOST"] = os.getenv("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8080")
    firebase_admin.initialize_app(options={
        'projectId': os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
    })
    print(f"🔥 Using Firebase Emulator: {os.environ.get('FIRESTORE_EMULATOR_HOST')}")


BACKEND_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = BACKEND_ROOT / "workers" / "photo_analysis_worker.py"
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 30


@asynccontextmanager
async def lifespan(app: FastAPI):
    proc = subprocess.Popen(
        [sys.executable, str(WORKER_SCRIPT)],
        cwd=str(BACKEND_ROOT),
        env={**os.environ},
    )
    app.state.worker_proc = proc
    print(f"👁 Photo analysis worker started (pid={proc.pid})")

    # Live exam monitoring primitives. Routes added in BE-2/BE-3 reach
    # these via `request.app.state.live_*`.
    app.state.live_registry = LiveSessionRegistry()
    app.state.live_hub = BroadcastHub()
    app.state.live_reaper = Reaper(app.state.live_registry, app.state.live_hub)
    app.state.live_reaper.start()
    print("📡 Live session reaper started")

    try:
        yield
    finally:
        try:
            await app.state.live_reaper.stop()
            print("📡 Live session reaper stopped")
        except Exception as e:
            print(f"📡 Live reaper shutdown error: {e}")
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"👁 Worker did not exit in {WORKER_SHUTDOWN_TIMEOUT_SECONDS}s, killing")
            proc.kill()
            proc.wait()
        except Exception as e:
            print(f"👁 Worker shutdown error: {e}")
        print("👁 Photo analysis worker stopped")


app = FastAPI(title="Face Quiz Backend", version="1.0.0", root_path="/api", lifespan=lifespan)

# CORS
allowed_origins = [os.environ.get("FRONTEND_URL", "http://localhost:3000")]
print(f"Allowed CORS origins: {allowed_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestCounterMiddleware)

# Routers
app.include_router(users_router)
app.include_router(solutions_router)
app.include_router(exams_router)
app.include_router(leaderboard_router)
app.include_router(auth_router)
if media_router:
    app.include_router(media_router)
if tags_router:
    app.include_router(tags_router)
app.include_router(proctoring_router)


@app.get("/")
async def root():
    return {"message": "This is Face Quiz Backend!"}
