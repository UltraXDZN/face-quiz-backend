import os
from dotenv import load_dotenv

load_dotenv()

# Set emulator environment variables BEFORE any Firebase imports
if os.getenv("ENVIRONMENT") != "production":
    if os.getenv("FIRESTORE_EMULATOR_HOST"):
        os.environ["FIRESTORE_EMULATOR_HOST"] = os.getenv("FIRESTORE_EMULATOR_HOST")
    if os.getenv("FIREBASE_AUTH_EMULATOR_HOST"):
        os.environ["FIREBASE_AUTH_EMULATOR_HOST"] = os.getenv("FIREBASE_AUTH_EMULATOR_HOST")

import firebase_admin
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials
from api.users.routes import router as users_router
from api.solutions.routes import router as solutions_router
from api.exams.routes import router as exams_router
from api.leaderboard.routes import router as leaderboard_router

app = FastAPI(title="Face Quiz Backend", version="1.0.0")

# Initialize Firebase
if os.getenv("ENVIRONMENT") == "production":
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"))
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

@app.get("/")
async def root():
    return {"message": "This is Face Quiz Backend!"}