from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/predict")
def predict():
    return {
        "prediction": 150,
        "status": "success"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
