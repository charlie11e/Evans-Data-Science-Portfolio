# Machine Learning App in Streamlit

<h2> In this project, I created an app in streamlit that utilizes supervised machine learning models to study the various properties of sample datasets as well as datasets that the user uploads. </h2>
<p></p>
In this app, users can upload their own datasets or use datasets that I have provided. The app allows users to use a regression or classification model. The first tab focuses on linear regression. The user is given a preview of the dataset and chooses the target variable and the features to run the regression model on. The app provides the graph of the regression, regression metrics (R² score, MSE, and RMSE), and the plot of the residuals with a small description of what to look for in the residuals plot. 
<p></p>
The second tab has a K-Nearest Neighbors Classification model. It predicts categorical outcomes, either binary or multi-class. The user can once again either use provided datasets or their own. The user is given a preview of the dataset and chooses the target variable and the features to run the classification model on. The user also chooses the number of neighbors, or k. The app then provides an accuracy score, confusion matrix, and classification report. It also provides a graph of the accuracy scores at different number of neighbors, showing which k is the most accurate. 
<p></p>
<h2> Images: </h2>

## Regression Model Output

![Regression Model Results](/MLStreamlitApp/MLAppLinearReg.png)

This plot shows the predicted vs. actual values and includes key regression metrics like R², MSE, and RMSE.

---

## K-Nearest Neighbors Classification Output

![KNN Classification Results](MLStreamlitApp/MLAppClassification.png)

This section includes an interactive K value selector, confusion matrix, accuracy score, and classification report.
