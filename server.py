from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import run_model
import os

app = FastAPI()

# allow React frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "backend running"}

@app.post("/run-model")
async def run_model_api(
    forecast: UploadFile = File(...),
    inventory: UploadFile = File(...),
    shelf: UploadFile = File(...),
    delivery: UploadFile = File(...)
):
    print(f"Starting model execution")
    print(50*"=")
    paths = []

    for file in [forecast, inventory, shelf, delivery]:
        path = f"temp_{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        paths.append(path)

    result = run_model(*paths)
    print(type(result))

    # cleanup
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

    return result
