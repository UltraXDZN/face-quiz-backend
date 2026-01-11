import os
import uvicorn
import firebase_admin
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from api.users.routes import router as users_router
from firebase_admin import credentials, firestore, auth
from google.auth.credentials import AnonymousCredentials

load_dotenv()

production = os.getenv("ENVIRONMENT") == "production"

app = FastAPI(title="Face Quiz Backend", version="1.0.0")

print(f"Environment: {'Production' if production else 'Development'}")

# Initialize Firebase
try:
    # Check if using emulator
    if not production and os.getenv("FIRESTORE_EMULATOR_HOST"):
        if not production and os.getenv("FIRESTORE_EMULATOR_HOST"):
            os.environ["FIRESTORE_EMULATOR_HOST"] = os.getenv("FIRESTORE_EMULATOR_HOST")
            
            if os.getenv("FIREBASE_AUTH_EMULATOR_HOST"):
                os.environ["FIREBASE_AUTH_EMULATOR_HOST"] = os.getenv("FIREBASE_AUTH_EMULATOR_HOST")
            print(f"🔧 Emulator hosts: Firestore={os.environ['FIRESTORE_EMULATOR_HOST']}, Auth={os.environ.get('FIREBASE_AUTH_EMULATOR_HOST')}")
        
        firebase_admin.initialize_app(
            options={
                "projectId": os.getenv("FIREBASE_TESTING_PROJECT_ID")
            },
            credential=AnonymousCredentials()
        )
        print("🔥 Using Firebase Emulator")
    else:
        # Use production credentials
        cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"))
        firebase_admin.initialize_app(cred)
        print("🚀 Using Production Firebase")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    import traceback
    traceback.print_exc()
    raise

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

@app.get("/")
async def root():
    return {"message": "This is Face Quiz Backend!"}