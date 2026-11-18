import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Loan Approval Prediction",page_icon="🏦",layout="wide")
st.title("🏦 Loan Approval Prediction")
st.warning("Educational demonstration only. Do not use this model to make real lending decisions.")
st.write("Compare Logistic Regression and Decision Tree classifiers using applicant attributes.")
file=st.file_uploader("Upload loan dataset CSV",type="csv")
if file:
    df=pd.read_csv(file); st.subheader("Dataset preview"); st.dataframe(df.head(10),use_container_width=True)
    target=st.selectbox("Select target column",df.columns,index=list(df.columns).index('Loan_Status') if 'Loan_Status' in df.columns else len(df.columns)-1)
    if df[target].nunique(dropna=True)!=2: st.error("This app currently supports a binary target (two classes).")
    else:
        work=df.dropna(subset=[target]).copy(); y=work[target].astype(str); X=work.drop(columns=[target])
        id_cols=[c for c in X.columns if c.lower() in ['loan_id','id','customer_id']]
        X=X.drop(columns=id_cols,errors='ignore')
        if X.empty: st.error("No predictor columns remain after excluding identifiers.")
        else:
            test_size=st.sidebar.slider("Test set proportion",0.1,0.4,0.2,0.05)
            try:
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)
                nums=Xtr.select_dtypes(include=np.number).columns.tolist(); cats=[c for c in Xtr.columns if c not in nums]
                num_pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
                cat_pipe=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
                prep=ColumnTransformer([('num',num_pipe,nums),('cat',cat_pipe,cats)])
                models={'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced'),'Decision Tree':DecisionTreeClassifier(random_state=42,class_weight='balanced',max_depth=5)}
                metrics=[]; fitted={}
                for name,clf in models.items():
                    pipe=Pipeline([('preprocessor',prep),('model',clf)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); fitted[name]=pipe
                    metrics.append({'Model':name,'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred,pos_label=sorted(y.unique())[-1],zero_division=0),'Recall':recall_score(yte,pred,pos_label=sorted(y.unique())[-1],zero_division=0),'F1-score':f1_score(yte,pred,pos_label=sorted(y.unique())[-1],zero_division=0)})
                met=pd.DataFrame(metrics).set_index('Model'); st.subheader("Model comparison"); st.dataframe(met.style.format('{:.3f}'),use_container_width=True)
                chosen=st.selectbox("Inspect model",list(fitted)); pred=fitted[chosen].predict(Xte)
                col1,col2=st.columns(2)
                with col1:
                    fig,ax=plt.subplots(); ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax); ax.set_title(chosen+" — confusion matrix"); st.pyplot(fig)
                with col2: st.text("Classification report"); st.code(classification_report(yte,pred,zero_division=0))
                st.download_button("Download model comparison",met.to_csv(),"loan_model_comparison.csv","text/csv")
                st.caption("Precision/recall are calculated for the lexicographically last target label. Confirm which label represents loan approval in your dataset.")
            except ValueError as e: st.error(f"Training could not proceed: {e}. Check class counts and missing/invalid data.")
else: st.info("Upload a loan approval CSV. Select the approval outcome column as the target.")
