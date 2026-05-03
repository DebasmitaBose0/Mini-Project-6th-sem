from fastapi import FastAPI
from app.routes import rewrite, analyze, history

app = FastAPI()

app.include_router(rewrite.router)
app.include_router(analyze.router)
app.include_router(history.router)