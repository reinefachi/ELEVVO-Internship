import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting",page_icon="📈",layout="wide")
st.title("📈 Walmart Sales Forecasting")
st.write("Train regression models on historical weekly sales and evaluate on a later time period.")
file=st.file_uploader("Upload Walmart train.csv",type="csv")
if file:
    df=pd.read_csv(file); st.subheader("Dataset preview"); st.dataframe(df.head(10),use_container_width=True)
    if not {'Date','Weekly_Sales'}.issubset(df.columns): st.error("CSV must contain Date and Weekly_Sales columns.")
    else:
        df['Date']=pd.to_datetime(df['Date'],errors='coerce'); df['Weekly_Sales']=pd.to_numeric(df['Weekly_Sales'],errors='coerce'); df=df.dropna(subset=['Date','Weekly_Sales'])
        if 'Store' in df and df['Store'].nunique()>1:
            stores=sorted(df.Store.dropna().unique()); store=st.sidebar.selectbox('Store',stores); df=df[df.Store==store]
        if 'Dept' in df and df.Dept.nunique()>1:
            depts=sorted(df.Dept.dropna().unique()); dept=st.sidebar.selectbox('Department',depts); df=df[df.Dept==dept]
        df=df.sort_values('Date').copy()
        if len(df)<15: st.error("At least 15 dated observations are recommended for a chronological holdout.")
        else:
            df['year']=df.Date.dt.year; df['month']=df.Date.dt.month; df['week']=df.Date.dt.isocalendar().week.astype(int); df['quarter']=df.Date.dt.quarter
            df['lag_1']=df.Weekly_Sales.shift(1); df['lag_2']=df.Weekly_Sales.shift(2); df['rolling_mean_4']=df.Weekly_Sales.shift(1).rolling(4).mean()
            if 'IsHoliday' in df: df['IsHoliday']=df['IsHoliday'].astype(str).map({'True':1,'False':0,'1':1,'0':0}).fillna(0)
            features=['year','month','week','quarter','lag_1','lag_2','rolling_mean_4']+(['IsHoliday'] if 'IsHoliday' in df else [])
            data=df.dropna(subset=features).copy()
            if len(data)<10: st.error("Not enough rows remain after creating lag features.")
            else:
                split=int(len(data)*0.8); train=data.iloc[:split]; test=data.iloc[split:]; Xtr=train[features]; Xte=test[features]; ytr=train.Weekly_Sales; yte=test.Weekly_Sales
                models={'Random Forest':RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42,n_jobs=-1),'HistGradientBoosting':HistGradientBoostingRegressor(random_state=42)}
                rows=[]; predictions={}
                for name,m in models.items():
                    m.fit(Xtr,ytr); p=m.predict(Xte); predictions[name]=p; rows.append({'Model':name,'MAE':mean_absolute_error(yte,p),'RMSE':np.sqrt(mean_squared_error(yte,p))})
                result=pd.DataFrame(rows).set_index('Model'); st.subheader("Forecast evaluation"); st.dataframe(result.style.format('{:,.2f}'),use_container_width=True)
                selected=st.selectbox('View predictions',list(models)); fig,ax=plt.subplots(figsize=(10,4)); ax.plot(test.Date,yte,label='Actual'); ax.plot(test.Date,predictions[selected],label='Predicted'); ax.set_title(selected+' — actual vs predicted'); ax.set_ylabel('Weekly sales'); ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate(); st.pyplot(fig)
                pred_df=pd.DataFrame({'Date':test.Date,'Actual':yte,'Predicted':predictions[selected]}); st.dataframe(pred_df,use_container_width=True)
                st.download_button('Download test predictions',pred_df.to_csv(index=False),'sales_predictions.csv','text/csv')
                st.info("This is a holdout evaluation, not a recursive forecast beyond the last date. Lagged features use prior observed sales.")
else: st.info("Upload the Walmart Recruiting Store Sales Forecasting train.csv file.")
