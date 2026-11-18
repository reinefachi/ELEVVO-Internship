import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay,precision_recall_curve

st.set_page_config(page_title="Predictive Maintenance",page_icon="🏭",layout="wide")
st.title("🏭 Industrial Predictive Maintenance")
st.write("Predict machine failure from sensor readings. The app emphasises false-alarm control.")
st.warning("Educational prototype only. Validate with domain experts before operational use.")
file=st.file_uploader("Upload AI4I 2020 predictive maintenance CSV",type='csv')
if file:
    df=pd.read_csv(file); st.subheader("Dataset preview"); st.dataframe(df.head(10),use_container_width=True)
    target=st.selectbox('Target column',df.columns,index=list(df.columns).index('Machine failure') if 'Machine failure' in df.columns else len(df.columns)-1)
    y=pd.to_numeric(df[target],errors='coerce'); X=df.drop(columns=[target])
    # Exclude identifiers and failure-mode outcome labels to avoid leakage.
    excluded=[c for c in X.columns if c.lower() in ['udi','product id','machine failure','twf','hdf','pwf','osf','rnf']]
    X=X.drop(columns=excluded,errors='ignore'); X=X.select_dtypes(include=np.number)
    valid=y.notna(); X=X.loc[valid].replace([np.inf,-np.inf],np.nan); y=y.loc[valid]
    X=X.fillna(X.median(numeric_only=True))
    if y.nunique()!=2: st.error("Target must contain exactly two classes, such as 0 and 1.")
    elif len(X)<20: st.error("Dataset is too small for a meaningful train/validation/test split.")
    else:
        recall_goal=st.sidebar.slider('Minimum validation recall target',0.1,0.95,0.5,0.05)
        X_trainval,X_test,y_trainval,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        X_train,X_val,y_train,y_val=train_test_split(X_trainval,y_trainval,test_size=.25,random_state=42,stratify=y_trainval)
        model=RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=-1,min_samples_leaf=2)
        model.fit(X_train,y_train); val_prob=model.predict_proba(X_val)[:,1]
        thresholds=np.linspace(.05,.95,91); candidates=[]
        for t in thresholds:
            pred=(val_prob>=t).astype(int); precision=precision_score(y_val,pred,zero_division=0); recall=recall_score(y_val,pred,zero_division=0); fdr=1-precision if pred.sum() else 0
            candidates.append((t,fdr,recall,precision))
        eligible=[x for x in candidates if x[2]>=recall_goal]
        chosen=min(eligible,key=lambda x:x[1])[0] if eligible else max(candidates,key=lambda x:x[2])[0]
        final=RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=-1,min_samples_leaf=2)
        final.fit(X_trainval,y_trainval); prob=final.predict_proba(X_test)[:,1]; pred=(prob>=chosen).astype(int)
        precision=precision_score(y_test,pred,zero_division=0); recall=recall_score(y_test,pred,zero_division=0); fdr=1-precision if pred.sum() else 0
        a,b,c,d=st.columns(4); a.metric('Accuracy',f'{accuracy_score(y_test,pred):.3f}'); b.metric('Precision',f'{precision:.3f}'); c.metric('Recall',f'{recall:.3f}'); d.metric('False discovery rate',f'{fdr:.3f}')
        st.metric('Threshold selected on validation set',f'{chosen:.2f}')
        fig,ax=plt.subplots(); ConfusionMatrixDisplay.from_predictions(y_test,pred,ax=ax); ax.set_title('Test-set confusion matrix'); st.pyplot(fig)
        imp=pd.Series(final.feature_importances_,index=X.columns).sort_values(); fig,ax=plt.subplots(); imp.tail(15).plot(kind='barh',ax=ax); ax.set_title('Top sensor feature importances'); st.pyplot(fig)
        pred_out=pd.DataFrame({'Actual':y_test,'Failure_probability':prob,'Predicted_failure':pred},index=y_test.index); st.download_button('Download test predictions',pred_out.to_csv(index=False),'maintenance_predictions.csv','text/csv')
        st.caption("FDR = FP / (TP + FP) = 1 − precision. Failure-mode columns (TWF/HDF/PWF/OSF/RNF) are excluded from predictors when present. This app predicts overall failure only.")
else: st.info("Upload the AI4I 2020 dataset to train and evaluate the failure classifier.")
