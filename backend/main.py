from fastapi import FastAPI
from services.kpi_service import *
from fastapi.middleware.cors import CORSMiddleware

app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "RetentionFlow AI Backend Running"}

@app.get("/analytics/kpis")
def get_kpis():
    df=load_data()
    df=validate_data(df)
    df=engagement(df)
    return calculate_kpis(df)

@app.get("/analytics/high_risk")
def high_risk():
    df=load_data()
    df=validate_data(df)
    return get_high_risk_users(df).to_dict(orient="records")

@app.get("/analytics/stickiness")
def get_stickiness():
    df=load_data()
    df=validate_data(df)
    df=customer_stickiness(df)

    return df[['CustomerId','Stickiness_score']].head(10).to_dict(orient="records")

@app.get("/analytics/rsi")
def get_rsi():
    df=load_data()
    df=validate_data(df)
    df=Rsi(df)

    return df[['CustomerId','RSI']].head(10).to_dict(orient="records")

@app.get("/analytics/segment")
def get_segment():
    df=load_data()
    df=validate_data(df)
    df=Rsi(df)
    df=segment(df)
    

    return df[['CustomerId','RSI','Segment']].head(10).to_dict(orient="records")

@app.get("/analytics/churn")
def get_churn_prediction():
    df=load_data()
    df=validate_data(df)
    model = train_churn(df)
    df=predict_churn(df,model)

    return df[['CustomerId','ChurnProbability','ChurnPrediction']].head(10).to_dict(orient="records")

