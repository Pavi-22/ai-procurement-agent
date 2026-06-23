from fastapi import FastAPI

app = FastAPI()

@app.get("/memory")
def memory():
    return {"status": "ok"}