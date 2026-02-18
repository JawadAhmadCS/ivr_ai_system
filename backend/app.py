
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine
import models
from routes import restaurant, call_logs, dashboard

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="IVR AI Admin System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (dev mode)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(restaurant.router)
app.include_router(call_logs.router)
app.include_router(dashboard.router)

@app.get("/")
def root():
    return {"status": "Server running"}
