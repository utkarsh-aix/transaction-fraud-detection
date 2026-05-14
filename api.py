from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import pickle
from preprocessing_pipeline import FraudPreprocessor

app = FastAPI(title="Fraud Detection API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load artifacts
model = joblib.load('output_v2/best_model.pkl')
with open('output_v2/model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

preprocessor = FraudPreprocessor()
THRESHOLD = model_info.get('threshold', 0.53)

class Transaction(BaseModel):
    # Minimal fields for demo, but can be expanded to full schema
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: float
    card3: float
    card4: str
    card5: float
    card6: str
    addr1: float
    addr2: float
    P_emaildomain: str = "gmail.com"
    R_emaildomain: str = "gmail.com"
    CurrentCountry: str = "India"
    LastCountry: str = "India"
    HoursSinceLastTransaction: float = 0.0
    # ... add other fields as needed

from fastapi.responses import FileResponse

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert Pydantic model to DataFrame
        data = pd.DataFrame([transaction.dict()])
        
        # Preprocess
        processed_data = preprocessor.transform(data)
        
        # Predict
        prob = float(model.predict_proba(processed_data)[:, 1][0])
        
        # --- NEW: HEURISTIC RULES (Safety Layer) ---
        # 1. Extreme Amount Check
        if transaction.TransactionAmt > 100000: # Over $100k
            prob = max(prob, 0.98) 
            
        # 2. Geographic Inconsistency (Impossible Travel)
        if transaction.CurrentCountry != transaction.LastCountry:
            # If the time gap is less than 12 hours, it's impossible travel -> FRAUD
            if transaction.HoursSinceLastTransaction < 12:
                prob = max(prob, 0.94)
            else:
                # If gap is > 12h, it could be legitimate travel. 
                # We let the ML model decide without modification.
                pass 
        
        # Use the optimized threshold from the model training
        is_fraud = bool(prob >= THRESHOLD)
        
        risk_level = "Low"
        if prob > 0.8: risk_level = "Critical"
        elif prob > 0.6: risk_level = "High"
        elif prob > 0.3: risk_level = "Medium"

        # 4. Explainable AI (XAI) Logic
        reasons = []
        if transaction.TransactionAmt > 10000: reasons.append("Extreme Amount")
        if transaction.CurrentCountry != transaction.LastCountry:
            if transaction.HoursSinceLastTransaction < 12:
                reasons.append("Impossible Velocity")
        
        if prob > 0.6 and not reasons:
            reasons.append("Neural Pattern Match")
        
        if not is_fraud:
            reasons = ["Legitimate Profile"]

        return {
            "result": "Fraud" if is_fraud else "Not Fraud",
            "confidence_score": f"{prob * 100:.2f}%",
            "risk_level": risk_level,
            "reason": " & ".join(reasons)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
