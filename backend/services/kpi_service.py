import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path

def load_data():
    # df = pd.read_csv(r"D:\unified\RetentionFlow AI\backend\data\European_Bank.csv")
    file_path = Path(__file__).resolve().parent.parent / "data" / "European_Bank.csv"
    df = pd.read_csv(file_path)
    return df
 
def validate_data(df):
    df = df.dropna()
    df['Exited'] = df['Exited'].astype(int)
    df['IsActiveMember'] = df['IsActiveMember'].astype(int)
    return df

def engagement(df):
    def classify(row):
        if row['IsActiveMember'] == 1 and row['NumOfProducts'] >1:
            return "Engaged"
        elif row['IsActiveMember']==0:
            return "Disengaged"
        else: 
            return "Low Engagement"
    df['EngagementLevel'] = df.apply(classify,axis =1)
    return df

def creditcard_stickiness(df):
    churn_without_card = df[df['HasCrCard']==0]['Exited'].mean()
    churn_with_card = df[df['HasCrCard']==1]['Exited'].mean()
    score = churn_without_card-churn_with_card
    return {
        "churn_without_card":round(churn_without_card,2),
        "churn_with_card":round(churn_with_card,2),
        "score" : round(score,2)
    }


def relationship_strength(df):
    df['RelationScore']=(df['IsActiveMember']*2+df['NumOfProducts']*1.5 +df['HasCrCard']*1)
    avg_score=df['RelationScore'].mean()
    high_strength = df[df['RelationScore']>avg_score].shape[0]

    return {
        "avg_score": round(avg_score,2),
        "high_strength": high_strength
    }
def customer_stickiness(df):
    df['Balance_normalize']=df['Balance']/df['Balance'].max() 
    df['Stickiness_score']=round((0.4*df['Tenure']+0.3*df['NumOfProducts']+0.3*df['Balance_normalize']),4)
    return df

def Rsi(df):
    df['Balance_normalize']=df['Balance']/df['Balance'].max()
    df['RSI']=(
        0.3 *df['Tenure']+0.2*df['NumOfProducts']+0.2*df['IsActiveMember']+0.3*df['Balance_normalize']
    )
    return df

def segment(df):
    def label(score):
        if score>3:
            return 'High Value'
        elif score>2:
            return 'Medium Value'
        else:
            return 'Low Value'
    df['Segment']=df['RSI'].apply(label)
    return df


def train_churn(df):
    x=df[['Tenure','Balance','NumOfProducts','IsActiveMember']]
    y=df['Exited']

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(x_train,y_train)

    return model

def predict_churn(df,model):
    x=df[['Tenure','Balance','NumOfProducts','IsActiveMember']]
    df['ChurnProbability']=model.predict_proba(x)[:,1]
    df['ChurnPrediction']=model.predict(x)
    return df

def engagement_vs_churn(df):
    result = df.groupby("IsActiveMember")["Exited"].mean().reset_index()
    result.columns = ["IsActiveMember", "ChurnRate"]
    return result.to_dict(orient="records")

def product_impact(df):
    result = df.groupby("NumOfProducts")["Exited"].mean().reset_index()
    result.columns = ["NumOfProducts", "ChurnRate"]
    return result.to_dict(orient="records")

def high_value_risk(df):
    return df[
        (df["Balance"] > df["Balance"].median()) &
        (df["IsActiveMember"] == 0) &
        (df["ChurnProbability"] > 0.6)
    ][["CustomerId", "Balance", "ChurnProbability"]].to_dict(orient="records")




def calculate_kpis(df):
    total_users=len(df)
    avgbalance=df['Balance'].mean()
    avgproducts= df['NumOfProducts'].mean()
    churn_rate=df['Exited'].mean()
    active_rate =df[df['IsActiveMember']==1]['Exited'].mean()
    inactive_rate =df[df['IsActiveMember']==0]['Exited'].mean()
    product_churn = df.groupby('NumOfProducts')['Exited'].mean().to_dict()
    high_balance_risk=df[(df['Balance']>100000)&(df['IsActiveMember']==0)].shape[0]
    card=creditcard_stickiness(df)
    relation=relationship_strength(df)

    return {
        "total_users": total_users,
        "avgbalance":round(avgbalance,2),
        "avgproducts":round(avgproducts,2),
        "churn_rate" : round(churn_rate,2),
        "active_rate": round(active_rate,2),
        "inactive_rate": round(inactive_rate,2),
        "product_churn": product_churn,
        "high_balance_risk" : high_balance_risk,
        "card":card,
        "relation":relation
    }

def get_high_risk_users(df):
    high_risk=df[(df['Balance']>100000)& (df['IsActiveMember']==0)]
    return high_risk[['CustomerId', 'Balance', 'NumOfProducts']]


