import os
import uvicorn
from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials, firestore, auth
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Initialize Firebase
try:
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"))
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()

@app.get("/")
async def root():
    return {"message": "This is Face Quiz Backend!"}

@app.get("/api/users/")
async def get_users():
    users_ref = db.collection("users").stream()
    users = [user.to_dict() for user in users_ref]
    return users