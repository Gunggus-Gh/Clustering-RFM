import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="RFM Clustering App", layout="wide")

st.title("📊 Aplikasi Clustering Pelanggan (RFM + KMeans)")
st.write("Aplikasi ini melakukan clustering pelanggan menggunakan metode **RFM** dan **K-Means**.")

# =========================
# UPLOAD DATA
# =========================
st.sidebar.header("1️⃣ Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil dimuat")

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    # =========================
    # PREPROCESSING
    # =========================
    st.header("2️⃣ Preprocessing & RFM")

    df = df.rename(columns={
        "Purchase Amount (USD)": "Purchase_Amount",
        "Frequency of Purchases": "Frequency_of_Purchases",
        "Previous Purchases": "Previous_Purchases"
    })

    # --- R ---
    def recency_score(x):
        if x <= 10: return 1
        elif x <= 20: return 2
        elif x <= 30: return 3
        elif x <= 40: return 4
        else: return 5

    df["R"] = df["Previous_Purchases"].apply(recency_score)

    # --- F ---
    def frequency_score(x):
        if x == "Weekly": return 5
        elif x in ["Bi-Weekly", "Fortnightly"]: return 4
        elif x == "Monthly": return 3
        elif x in ["Quarterly", "Every 3 Months"]: return 2
        else: return 1

    df["F"] = df["Frequency_of_Purchases"].apply(frequency_score)

    # --- M ---
    def monetary_score(x):
        if x > 40: return 5
        elif x > 30: return 4
        elif x > 20: return 3
        elif x > 10: return 2
        else: return 1

    df["M"] = df["Purchase_Amount"].apply(monetary_score)

    st.write("Contoh hasil RFM:")
    st.dataframe(df[["Customer ID", "R", "F", "M"]].head())

    # =========================
    # AGREGASI RFM
    # =========================
    rfm = df.groupby("Customer ID").agg({
        "R": "max",
        "F": "max",
        "M": "max"
    }).reset_index()

    rfm["RFM_score"] = (rfm["R"] + rfm["F"] + rfm["M"]) / 3

    # Normalisasi
    scaler = MinMaxScaler()

    rfm_scaled = rfm.copy()
    rfm_scaled[["R", "F", "M"]] = scaler.fit_transform(
    rfm[["R", "F", "M"]]
    )
    X = rfm_scaled[["R", "F", "M"]]

    # =========================
    # PILIH JUMLAH CLUSTER
    # =========================
    st.header("3️⃣ Penentuan Jumlah Cluster")

    mode = st.radio(
        "Pilih metode jumlah cluster:",
        ["Otomatis (Silhouette)", "Manual"]
    )

    if mode == "Otomatis (Silhouette)":
        K_range = range(2, 10)
        sil_scores = []

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sil_scores.append(silhouette_score(X, labels))

        best_k = list(K_range)[np.argmax(sil_scores)]
        st.success(f"Jumlah cluster terbaik: **{best_k}**")

        fig, ax = plt.subplots()
        ax.plot(list(K_range), sil_scores, marker="o")
        ax.set_title("Silhouette Score")
        ax.set_xlabel("Jumlah Cluster")
        ax.set_ylabel("Score")
        st.pyplot(fig)

    else:
        best_k = st.slider("Pilih jumlah cluster", 2, 10, 3)

    # =========================
    # CLUSTERING
    # =========================
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm_scaled["Cluster"] = kmeans.fit_predict(X)

    st.header("4️⃣ Hasil Clustering")

    # =========================
    # PCA VISUALIZATION
    # =========================
    pca = PCA(n_components=2)
    pca_vals = pca.fit_transform(X)
    rfm_scaled["PCA1"] = pca_vals[:, 0]
    rfm_scaled["PCA2"] = pca_vals[:, 1]

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        rfm_scaled["PCA1"],
        rfm_scaled["PCA2"],
        c=rfm_scaled["Cluster"],
        cmap="tab10"
    )
    ax.set_title("Visualisasi PCA Cluster")
    st.pyplot(fig)

    # =========================
    # RINGKASAN CLUSTER
    # =========================
    st.subheader("Ringkasan Cluster")

    cluster_summary = rfm_scaled.groupby("Cluster").agg({
        "Customer ID": "count",
        "R": "mean",
        "F": "mean",
        "M": "mean",
        "RFM_score": "mean"
    }).rename(columns={"Customer ID": "Jumlah Customer"})

    st.dataframe(cluster_summary)

    # =========================
    # CEK CLUSTER CUSTOMER (INTERAKTIF)
    # =========================
    st.header("5️⃣ Cek Cluster Berdasarkan Input Anda")

    col1, col2, col3 = st.columns(3)
    with col1:
        R_input = st.slider("Recency (R)", 1, 5, 3)
    with col2:
        F_input = st.slider("Frequency (F)", 1, 5, 3)
    with col3:
        M_input = st.slider("Monetary (M)", 1, 5, 3)

    input_df = pd.DataFrame([[R_input, F_input, M_input]], columns=["R", "F", "M"])
    input_scaled = scaler.transform(input_df)

    cluster_pred = kmeans.predict(input_scaled)

    st.success(f"Customer ini masuk ke **Cluster {cluster_pred[0]}**")

    # =========================
    # VISUALISASI BAR: INPUT vs CLUSTER (RAPI)
    # =========================
    st.subheader("📊 Perbandingan Nilai RFM")

    cluster_id = cluster_pred[0]

    cluster_avg = rfm_scaled[
        rfm_scaled["Cluster"] == cluster_id
    ][["R", "F", "M"]].mean()

    plot_df = pd.DataFrame({
        "RFM": ["R", "F", "M"],
        "Input": [R_input, F_input, M_input],
        "Rata-rata Cluster": [
            cluster_avg["R"] * 5,
            cluster_avg["F"] * 5,
            cluster_avg["M"] * 5
        ]
    })

    # Ukuran lebih kecil & rapi
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    plot_df.set_index("RFM").plot(
        kind="bar",
        ax=ax,
        width=0.6
    )

    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Skor", fontsize=9)
    ax.set_title(
        f"Input vs Rata-rata Cluster {cluster_id}",
        fontsize=10
    )

    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.tick_params(axis='y', labelsize=8)

    # Grid halus
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    st.pyplot(fig, use_container_width=False)


else:
    st.info("Silakan upload dataset CSV terlebih dahulu.")
