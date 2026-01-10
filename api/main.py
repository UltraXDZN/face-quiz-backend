import os
import uvicorn
from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials, firestore, auth
from dotenv import load_dotenv
from api.users.routes import router as users_router

load_dotenv()

app = FastAPI(title="Face Quiz Backend", version="1.0.0")

# Initialize Firebase
try:
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"))
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()

# Include routers
app.include_router(users_router)

@app.get("/")
async def root():
    return {"message": "This is Face Quiz Backend!"}