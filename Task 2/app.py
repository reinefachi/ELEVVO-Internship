import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("🛍️ Customer Segmentation")
st.write("Explore customer groups using K-Means clustering on annual income and spending score.")
file=st.file_uploader("Upload Mall Customer CSV", type="csv")
if file:
    df=pd.read_csv(file)
    st.subheader("Dataset preview"); st.dataframe(df.head(10), use_container_width=True)
    required=["Annual Income (k$)","Spending Score (1-100)"]
    if not all(c in df.columns for c in required):
        st.error("CSV must contain: " + ", ".join(required))
    else:
        data=df[required].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data)<3: st.error("At least three complete customer records are required.")
        else:
            max_k=min(10,len(data)-1)
            k=st.sidebar.slider("Number of clusters",2,min(10,max_k),min(5,max_k))
            scaler=StandardScaler(); X=scaler.fit_transform(data)
            scores=[]; inertias=[]
            for n in range(2,max_k+1):
                m=KMeans(n_clusters=n,random_state=42,n_init=10).fit(X)
                inertias.append(m.inertia_); scores.append(silhouette_score(X,m.labels_))
            model=KMeans(n_clusters=k,random_state=42,n_init=10)
            labels=model.fit_predict(X)
            result=df.loc[data.index].copy(); result["Cluster"]=labels
            c1,c2,c3=st.columns(3); c1.metric("Customers analysed",f"{len(data):,}"); c2.metric("Clusters",k); c3.metric("Silhouette score",f"{silhouette_score(X,labels):.3f}")
            st.subheader("Customer segments")
            fig,ax=plt.subplots(); sc=ax.scatter(data[required[0]],data[required[1]],c=labels,cmap="tab10",alpha=.8); ax.set_xlabel(required[0]); ax.set_ylabel(required[1]); ax.set_title("K-Means customer clusters"); ax.grid(alpha=.2); st.pyplot(fig)
            st.subheader("Selecting the number of clusters")
            f,(a,b)=plt.subplots(1,2,figsize=(10,3)); ks=list(range(2,max_k+1)); a.plot(ks,inertias,marker='o'); a.set_title('Elbow method'); a.set_xlabel('k'); a.set_ylabel('Inertia'); b.plot(ks,scores,marker='o'); b.set_title('Silhouette score'); b.set_xlabel('k'); b.set_ylabel('Score'); st.pyplot(f)
            st.subheader("Cluster profile")
            profile=result.groupby("Cluster")[required].mean().round(2); st.dataframe(profile,use_container_width=True)
            st.download_button("Download customers with cluster labels",result.to_csv(index=False),"customer_segments.csv","text/csv")
            with st.expander("Optional DBSCAN comparison"):
                eps=st.slider("DBSCAN eps",0.1,2.0,0.5,0.1); min_samples=st.slider("Minimum samples",2,20,5)
                db=DBSCAN(eps=eps,min_samples=min_samples).fit_predict(X); st.write(pd.Series(db).value_counts().rename_axis('Label').to_frame('Customers'))
else: st.info("Upload the Mall Customer Segmentation CSV to begin.")
