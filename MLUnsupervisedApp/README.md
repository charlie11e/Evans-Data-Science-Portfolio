# Machine Learning App in Streamlit

## In this project, I created an [Unsupervised Machine Learning Streamlit App](charlie11e-evans-data-mlunsupervisedappmlunsupervisedapp-696f3w.streamlit.app/) that utilizes unsupervised machine learning models to study the various properties of sample datasets as well as datasets that the user uploads.

In this app, users can upload their own datasets or use provided datasets. Users utilize unsupervised machine learning models, specifically Principal Component Analysis (PCA), KMeans clustering, and Hierarchical clustering to explore various datasets. The app preprocesses the data, performs dimensionality reduction via PCA, and visualizes the explained variance to help understand the underlying structure.

Once PCA is applied, users can experiment with clustering methods by adjusting parameters such as the number of components and clusters. The application provides cluster assignments, PCA scatterplots, and evaluation metrics like silhouette score and WCSS for KMeans. Hierarchical clustering is visualized using a dendrogram. 

## Images:

## PCA Scatterplot
![PCA Scatterplot](/MLUnsupervisedApp/pca-scatterplot.png)

## Dendrogram (Hierarchical Clustering)
![Dendrogram](/MLUnsupervisedApp/dendrogram.png)
