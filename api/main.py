import asyncio
import os
from contextlib import asynccontextmanager
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
from api.users.routes import router as users_router
from api.solutions.routes import router as solutions_router
from api.exams.routes import router as exams_router
from api.leaderboard.routes import router as leaderboard_router
from api.auth.routes import router as auth_router
from api.media.routes import router as media_router
from api.tags.routes import router as tags_router
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    from api.proctoring.worker import worker_loop
    task = loop.create_task(worker_loop())
    print("👁 Proctoring analysis worker started")
    yield
    from api.proctoring.worker import _running
    _running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    print("👁 Proctoring analysis worker stopped")


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

# Routers
app.include_router(users_router)
app.include_router(solutions_router)
app.include_router(exams_router)
app.include_router(leaderboard_router)
app.include_router(auth_router)
app.include_router(media_router)
app.include_router(tags_router)
app.include_router(proctoring_router)


@app.get("/")
async def root():
    return {"message": "This is Face Quiz Backend!"}
