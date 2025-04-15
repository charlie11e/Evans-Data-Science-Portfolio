import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing() 
california_housing_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_housing_data['target'] = california_housing.target

import os
import kagglehub
path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
file_path = os.path.join(path, "Student_Performance.csv")
print("Path to dataset files:", path)
Student_Performance = pd.read_csv(file_path)
# Convert categorical variable to numerical
Student_Performance["Extracurricular Activities"] = Student_Performance["Extracurricular Activities"].map({"Yes": 1, "No": 0})

# Load datasets
datasets = {
    "California Housing": california_housing_data,
    "Student Performance": Student_Performance
}

tab1, tab2 = st.tabs(["Regression", "Classification"])

with tab1:
    # Create title and description
    st.title("Machine Learning Streamlit App")
    st.header("Linear Regression")
    st.info("This app allows the user to explore various datasets and study their properties through different machine learning models. " \
    "This tab focuses on linear regression models. These models are used to predict continuous numerical values (like prices or scores) " \
    "based on input features. Their purpose is to find patterns and relationships in data so that an outcome can be estimated, such as " \
    "predicting housing prices from size and location. The other tab focuses on classification models.")

    # Create a selectbox for dataset selection
    choice = st.selectbox(
        "Select Dataset or Upload Your Own Below",
        ("California Housing", "Student Performance", "Upload Your Own")
    )

    # Initialize an empty DataFrame
    df = None

    # Preloaded datasets
    if choice != "Upload Your Own":
        df = datasets[choice] 
        df = df.dropna() 
        st.subheader(f"{choice} Dataset Preview")
        st.write(df.head()) 

    # External file upload
    else:
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
        if uploaded_file is not None:
            # Determine file type
            file_type = "csv" if uploaded_file.type == "text/csv" else "excel"
            if file_type == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_type == "excel":
                df = pd.read_excel(uploaded_file)

            df = df.dropna() 
        
            # Remove categorical variables
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns]

            choice = uploaded_file.name

            # Display uploaded dataset
            st.subheader(f"{choice} Dataset Preview")
            st.write(df.head()) 


    # Linear Regression Supervised Learning Model
    st.subheader("Linear Regression Supervised Learning Model")
    st.write(f"The following model is trained on the {choice} dataset. This model predicts the target variable using liner regression. Please select the target variable and features for the model.")

    # Create a selectbox for target variable selection
    target_variable = st.selectbox("Select Target Variable", df.columns.tolist())

    # Create a multiselect for feature selection
    features = st.multiselect("Select Features",
        [col for col in df.columns.tolist() if col != target_variable],
    )


    # Training the model
    if features: 
        X = df[features]
        y = df[target_variable]
    
        # Scale the Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
        # Split the data into training and testing sets
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X, y, 
                                                                                        test_size=0.2, random_state=42
        )
    
        model_scaled = LinearRegression()
        # Train the model with training data
        model_scaled.fit(X_train_scaled, y_train_scaled)
        # Predict the labels for the test data
        y_pred_scaled = model_scaled.predict(X_test_scaled)
    
        # Plotting the results
        plt.figure(figsize = (10,6))
        plt.scatter(y_test_scaled, y_pred_scaled, color = "blue")
        plt.plot([y_test_scaled.min(), y_test_scaled.max()], [y_test_scaled.min(), y_test_scaled.max()], 'r--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Model Predictions vs Actual Values")
        st.pyplot(plt)

        # Regression Metrics (R^2 Score, MSE, RMSE)
        st.header("Regression Metrics")

        r2 = r2_score(y_test_scaled, y_pred_scaled)
        mse = mean_squared_error(y_test_scaled, y_pred_scaled)
        rmse = root_mean_squared_error(y_test_scaled, y_pred_scaled)

        # Create a DataFrame for metrics
        metrics_data = {
            "Metric": ["R² Score", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
            "Value": [f"{r2:.2f}", f"{mse:.2f}", f"{rmse:.2f}"]
        }
        metrics_df = pd.DataFrame(metrics_data)

        # Display the metrics as a table
        st.table(metrics_df)

        # Display model coefficients and intercept separately
        coefficients_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model_scaled.coef_
        })
        st.subheader("Coefficients")
        st.table(coefficients_df)

        st.subheader("Intercept")
        st.write(f"{model_scaled.intercept_:.2f}")


        # Residuals Plot
        st.header("Residuals Plot")
        st.write("Residuals are the differences between the actual and predicted values. " \
        "Ideally, they should be uniformly distributed around zero. This measure of " \
        "homeoscedasticity is another way to check the performance of the model.")
        residuals = y_test_scaled - y_pred_scaled
        plt.figure(figsize=(10,6))
        plt.scatter(y_pred_scaled, residuals, alpha=0.7)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs. Predicted Values")
        st.pyplot(plt)

    else:
        st.warning("Select at least one feature to train the model.")





# Tab 2 KNN Classification
with tab2:
    # Create title and description
    st.header("K-Nearest Neighbors Classification")
    st.info("This app allows the user to explore various datasets and study their properties " \
    "through K-Nearest Neighbors classifcation models, which predict categorical outcomes "
    "(like species) based on similar features in the data")

    titanic = sns.load_dataset("titanic")
    iris = sns.load_dataset("iris")
    penguins = sns.load_dataset("penguins")

    datasets2 = {
        "Titanic": titanic,
        "Iris": iris,
        "Penguins": penguins
    }

    # Create a selectbox for dataset selection
    choice2 = st.selectbox(
        "Select Dataset or Upload Your Own Below",
        ("Titanic", "Iris", "Penguins", "Upload Your Own")
    )

    # Initialize an empty DataFrame
    df2 = None

    # Preloaded datasets
    if choice2 != "Upload Your Own":
        df2 = datasets2[choice2] 
        df2 = df2.dropna() 
        st.subheader(f"{choice2} Dataset Preview")
        st.write(df2.head()) 

    # External file upload
    else:
        uploaded_file2 = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
        if uploaded_file2 is not None:
            # Determine file type
            file_type = "csv" if uploaded_file2.type == "text/csv" else "excel"
            if file_type == "csv":
                df2 = pd.read_csv(uploaded_file2)
            elif file_type == "excel":
                df2 = pd.read_excel(uploaded_file2)

            df2 = df2.dropna()

            choice2 = uploaded_file2.name

            # Display uploaded dataset
            st.subheader(f"{choice2} Dataset Preview")
            st.write(df2.head()) 

    # Linear Regression Supervised Learning Model
    st.subheader("K-Nearest Neighbor Classification Supervised Learning Model")
    st.write(f"The following model is trained on the {choice2} dataset. This model " \
             "predicts the target variable using liner regression. Please select the " \
             "target variable and features for the model.")

    # Create a selectbox for target variable selection
    target_variable2 = st.selectbox("Select Target Variable", 
                                    df2.columns.tolist())

    # Create a multiselect for feature selection with only numerical variables
    features2 = st.multiselect(
    "Select Features",
    df2.select_dtypes(include=[np.number]).columns.tolist(), 
    default=[col for col in df2.select_dtypes(include=[np.number]).columns.tolist() if col != target_variable2], 
    help="Select only numerical features for KNN classification. Categorical features will not work with the model."
)

    # Create a slider for the number of neighbors (k)
    st.write("Select the number of neighbors (k) for the KNN model. It should be an odd number to avoid ties.")
    k = st.slider("Select number of neighbors (k, odd values only)", min_value=1, max_value=21, step=2, value=5)


    # Training the model
    if features2:
        X2 = df2[features2]
        y2 = df2[target_variable2]
        # Scale the Data
        scaler2 = StandardScaler()
        X_scaled2 = scaler2.fit_transform(X2)
        X_scaled2 = pd.DataFrame(X_scaled2, columns = X2.columns)
        # Split the data into training and testing sets
        X_train_scaled2, X_test_scaled2, y_train_scaled2, y_test_scaled2 = train_test_split(X2, 
                                                                                    y2, 
                                                                                    test_size=0.2, 
                                                                                    random_state = 42)
        knn = KNeighborsClassifier(n_neighbors = k)
        # Train the model with training data
        knn.fit(X_train_scaled2, y_train_scaled2)
        # Predict the labels for the test data
        y_pred_scaled_knn = knn.predict(X_test_scaled2)

        # Get and display accuracy score
        accuracy = accuracy_score(y_test_scaled2, y_pred_scaled_knn)
        st.subheader(f"Accuracy Score: {accuracy:.2f}")

        # Display the confusion matrix and classification report side by side in two columns
        col1, col2 = st.columns(2)

        # Confusion Matrix
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test_scaled2, y_pred_scaled_knn)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('KNN Confusion Matrix (k = {k})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(plt)

        # Classification Report
        with col2:
            st.subheader("Classification Report")
            st.text(classification_report(y_test_scaled2, y_pred_scaled_knn))


        # Plot the accuracy at each k value
        st.subheader("Accuracy at Different k Values")
        # Define a range of k values to explore for all odd numbers
        k_values = range(1, 21, 2)
        accuracies_scaled = []

        # Loop through different values of k, train a KNN model on scaled data, and record the accuracy for each
        for i in k_values:
            knn = KNeighborsClassifier(n_neighbors = i)
            # Train the model with training data
            knn.fit(X_train_scaled2, y_train_scaled2)
            # Predict the labels for the test data
            y_pred_scaled_knn = knn.predict(X_test_scaled2)
            accuracies_scaled.append(accuracy_score(y_test_scaled2, y_pred_scaled_knn))

        # Plot the accuracy scores against the different k values
        plt.figure(figsize=(10,6))
        plt.plot(k_values, accuracies_scaled, marker = "o")
        plt.xlabel("Number of Neighbors (k)")
        plt.ylabel ("Accuracy Score")
        plt.xticks(k_values)
        st.pyplot(plt)

    else:
        st.warning("Select at least one feature to train the model.")









      

