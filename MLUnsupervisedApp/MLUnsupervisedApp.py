import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub
import os
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
breast_cancer_df = pd.DataFrame(data = breast_cancer.data, columns = breast_cancer.feature_names)
breast_cancer_df['target'] = breast_cancer.target

path = kagglehub.dataset_download("rohan0301/unsupervised-learning-on-country-data")
file_path = os.path.join(path, 'Country-data.csv')
print("Path to dataset files:", path)
countries_unsupervised_learning = pd.read_csv(file_path)

path2 = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
file_path2 = os.path.join(path2, "Mall_Customers.csv")
print("Path to dataset files:", path2)
Customer_Data = pd.read_csv(file_path2)

datasets = {
    "Breast Cancer": breast_cancer_df,
    "Country Dataset": countries_unsupervised_learning,
    "Customer Dataset": Customer_Data
}



# Title and PCA

# Create title and description
st.title("Unsupervised Machine Learning Streamlit App")
st.header("Dimensionality Reduction (Principal Component Analysis)")
st.info("This app allows the user to explore various datasets and study their properties through different supervised machine learning models. " \
            "It focuses on dimensionality reduction, specifically through Principal Component Analysis (PCA)." \
            "It then focuses on clustering techniques, KMeans and Hierarchical Clustering respectively." \
            "The Principal Component Analysis reduces the number of features in a dataset into a few informative dimensions." \
            "PCA reduces the dataset's dimensionality while retaining the majority of the variance in the data, making it easier to visualize and interpret." \
            "Clustering is a type of unsupervised learning that groups similar data points together. " 
            )

# Initialize a Data Frame and uploaded_file 
df = None
uploaded_file = None

## Sidebar
# Dataset selection and parameter tuning
st.sidebar.header("Step 1: Upload or Select a Dataset")
choice = st.sidebar.selectbox(
        "Select Dataset or Upload Your Own Below",
        ("Breast Cancer", "Country Dataset", "Customer Dataset", "Upload Your Own")
    )
st.sidebar.info("Upload your own dataset in CSV or Excel format. The dataset should contain numerical features for PCA.")

# Selecting dataset
if choice == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        # Determine file type
        file_type = "csv" if uploaded_file.type == "text/csv" else "excel"
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "excel":
            df = pd.read_excel(uploaded_file)
    else:
        st.warning("Upload a CSV file to proceed.")
        st.stop()
else:
    df = datasets[choice] 


        
## Preprocess the data and display uploaded dataset
# Remove missing values and categorical variables
df = df.dropna() 
numeric_columns = df.select_dtypes(include=[np.number]).columns
df = df[numeric_columns]
# Standardize the data
X = df[numeric_columns].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Display uploaded dataset
choice = uploaded_file.name if uploaded_file else choice
st.subheader(f"{choice} Dataset Preview")
st.write(df.head()) 


## Sidebar for PCA parameters
st.sidebar.header("Step 2: PCA Parameters")
n_components = st.sidebar.slider("Select number of components for PCA", min_value=1, max_value=min(X_scaled.shape)-1,)


## PCA Analysis
# PCA Transformation
pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(X_scaled)

# Plot explained variance ratio
explained_variance = pca.explained_variance_ratio_
st.subheader("PCA Explained Variance (Individual and Cumulative)")
st.write("The explained variance ratio indicates the proportion of the dataset's variance that is captured by each principal component.")
# Plot explained variance ratio
st.info("Explained Variance Ratio: This plot shows the proportion of variance explained by each principal component." \
"The user can select the number of components to visualize the explained variance ratio.")
st.bar_chart(explained_variance, width=0, height=0, use_container_width=True)


# Bar Plot: Variance Explained by Each Component
st.info("The bar plot on the left shows the variance explained by each principal component. The cumulative variance plot on the right shows the total variance explained as more components are added.")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
pca_full = PCA(n_components = min(X_scaled.shape)-1)
pca_full.fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
components = range(1, len(pca_full.explained_variance_ratio_) + 1)
plt.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='teal')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Variance Explained by Each Principal Component (Individual)')
plt.xticks(components)
plt.grid(True, axis='y')

# Scree Plot: Cumulative Explained Variance
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Variance Explained (Cumulative)')
plt.xticks(range(1, len(cumulative_variance)+1))
plt.grid(True)

plt.tight_layout()
st.pyplot(plt)
plt.clf()






   



## Clustering
# Sidebar for clustering parameters
st.sidebar.header("Step 3: Clustering Settings")
model_choice = st.sidebar.radio("Choose clustering model", ["KMeans", "Hierarchical"])
n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=3)

if model_choice == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif model_choice == "Hierarchical":
    model = AgglomerativeClustering(n_clusters=n_clusters)
labels = model.fit_predict(pca_data)


# Add cluster labels to the PCA DataFrame
pca_df = pd.DataFrame(data=pca_data, columns=[f"PC{i+1}" for i in range(n_components)])
pca_df["Cluster"] = labels

if model_choice == "KMeans":
    selected_model = KMeans(n_clusters=n_clusters, random_state=42)
    selected_model.fit(X_scaled)
    selected_labels = selected_model.labels_
    selected_silhouette = silhouette_score(X_scaled, selected_labels)
    selected_wcss = selected_model.inertia_
    
    st.sidebar.metric("Silhouette Score", round(selected_silhouette, 2))
    st.sidebar.metric("WCSS", round(selected_wcss, 2))

    st.sidebar.info("Silhouette Score: A measure of how similar an object is to its own cluster compared to other clusters. A higher score indicates better-defined clusters.")
    st.sidebar.info("WCSS: Within-Cluster Sum of Squares, a measure of the compactness of the clusters. Lower values indicate more compact clusters.")

# Elbow and Silhouette Plot over Range
ks = range(2, min(X_scaled.shape)-1)
wcss = []
silhouette_scores = []

if model_choice == "KMeans":
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        labels_k = km.labels_
        silhouette_scores.append(silhouette_score(X_scaled, labels_k))


# Plotting the results
st.subheader("Clustering Results")


st.subheader("PCA Cluster Scatterplot")
plt.figure(figsize=(8, 6))
if "PC2" in pca_df.columns:
    for label in pca_df["Cluster"].unique():
        cluster_data = pca_df[pca_df["Cluster"] == label]
        plt.scatter(
            cluster_data["PC1"],
            cluster_data["PC2"],
            label=f"Cluster {label}",
            s=60,
            alpha=0.7,
            edgecolors="k"
        )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Clustering on PCA Projection")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()
else:
    st.warning("PCA requires at least 2 components to plot a scatter plot.")


if model_choice == "KMeans":
    st.subheader("KMeans Clustering Results")

    # Plot the Elbow Method result
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ks, wcss, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)

    # Plot the Silhouette Score result
    plt.subplot(1, 2, 2)
    plt.plot(ks, silhouette_scores, marker='o', color='green')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

else:
    st.subheader("Dendrogram (Hierarchical Clustering)")
    linked = linkage(pca_data, method='ward')
    # from scipy.cluster.hierarchy import linkage, dendrogram
    plt.figure(figsize=(8, 4))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    st.pyplot(plt)
    plt.clf()

# Cluster Assignment Output
st.header("Cluster Assignments")
df_with_clusters = df.copy()
df_with_clusters["Cluster"] = labels
st.dataframe(df_with_clusters)

    