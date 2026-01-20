import os
from dotenv import load_dotenv

load_dotenv()

# Set or clear emulator environment variables BEFORE any Firebase imports
if os.getenv("ENVIRONMENT") == "production":
    # Clear emulator variables in production mode
    if "FIRESTORE_EMULATOR_HOST" in os.environ:
        del os.environ["FIRESTORE_EMULATOR_HOST"]
    if "FIREBASE_AUTH_EMULATOR_HOST" in os.environ:
        del os.environ["FIREBASE_AUTH_EMULATOR_HOST"]
    print("🔥 Production mode: Cleared emulator environment variables")
else:
    # Set emulator variables for development
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

app = FastAPI(title="Face Quiz Backend", version="1.0.0")

# Initialize Firebase
if os.getenv("ENVIRONMENT") == "production":
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"))
    # Build options from production env vars when available
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

    # API key and auth domain are typically client-side config, but expose
    # them in the process env so other parts of the app can read them if needed.
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
    # Emulator mode - no credentials needed
    os.environ["FIRESTORE_EMULATOR_HOST"] = os.getenv("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8080")
    firebase_admin.initialize_app(options={
        'projectId': os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
    })
    print(f"🔥 Using Firebase Emulator: {os.environ.get('FIRESTORE_EMULATOR_HOST')}")

# Add CORS middleware to allow frontend access in development
allowed_origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(users_router)
app.include_router(solutions_router)
app.include_router(exams_router)
app.include_router(leaderboard_router)
app.include_router(auth_router)
app.include_router(leaderboard_router)

@app.get("/")
async def root():
    return {"message": "This is Face Quiz Backend!"}