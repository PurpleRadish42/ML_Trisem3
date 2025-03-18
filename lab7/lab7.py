# %% [markdown]
# <center> <h3><b>Lab 7: LDA</b></h3></center>
# <center><p> R Abhijit Srivathsan <br> 2448044 </p></center>

# %% [markdown]
# ## Importing the libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# %%


# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Convert to a pandas DataFrame for easier exploration
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Print basic information
print("Data shape:", df.shape)
print("\nFeatures:\n", data.feature_names)
print("\nTarget names:", data.target_names)

# Inspect the first few rows
df.head()


# %%
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# ## EDA

# %%
df.describe()

# %%
df['target'].value_counts().plot(kind='bar')
plt.title("Distribution of Breast Cancer Classes")
plt.xlabel("Class (0 = Malignant, 1 = Benign)")
plt.ylabel("Count")
plt.show()


# %%
features_to_plot = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
df[features_to_plot].hist(figsize=(10, 6), bins=15)
plt.tight_layout()
plt.show()


# %%
df[features_to_plot].plot(kind='box', subplots=True, layout=(2,2), figsize=(10, 8), sharex=False, sharey=False)
plt.tight_layout()
plt.show()


# %%
corr_matrix = df.drop('target', axis=1).corr()
print("Correlation matrix shape:", corr_matrix.shape)

# Let's plot a correlation heatmap for the first 10 features to keep it more readable
import numpy as np

features_subset = df.columns[:10]  # pick the first 10 features for demonstration
subset_corr_matrix = df[features_subset].corr()

plt.figure(figsize=(8,6))
plt.imshow(subset_corr_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(features_subset)), features_subset, rotation=90)
plt.yticks(range(len(features_subset)), features_subset)
plt.title("Correlation Heatmap of First 10 Features")
plt.show()


# %%
target_corr = df.corr()['target'].drop('target').sort_values(ascending=False)
target_corr.head(10)


# %%
# For demonstration, pick 4 features that are somewhat correlated or interesting
subset_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'target']
sub_df = df[subset_features]

import seaborn as sns  # Typically used, though user might want standard matplotlib only
sns.pairplot(sub_df, hue='target', diag_kind='hist')
plt.show()


# %% [markdown]
# ## Key Observations from EDA
# 
# - **Class Distribution**: The dataset is not heavily imbalanced, but benign samples outnumber malignant.
# - **Feature Distributions**: Many features (like `mean area`, `mean radius`, etc.) appear skewed; scaling might help in modeling.
# - **Correlation**: Certain features are highly correlated (e.g., `mean radius` with `mean perimeter`, `mean area`). This could indicate redundancy.
# - **Class Separation**: Even simple scatter plots (e.g., `mean radius` vs. `mean texture`) can show partial separation between classes.
# 

# %%
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1),
    df['target'],
    test_size=0.2,
    random_state=42
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# %%
# Standardize features for best performance with LDA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
# Instantiate the LDA model
lda = LinearDiscriminantAnalysis()

# Fit the model on training data
lda.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = lda.predict(X_test_scaled)


# %%
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))


# %% [markdown]
# ## **Model Performance Summary**
# 
# ### **Classification Report Analysis**
# - **Malignant Class (0)**:  
#   - **Precision**: 0.97 – When the model predicts malignant, it is correct 97% of the time.  
#   - **Recall**: 0.91 – The model correctly identifies 91% of all actual malignant cases.  
#   - **F1-score**: 0.94 – A good balance between precision and recall.
# 
# - **Benign Class (1)**:  
#   - **Precision**: 0.95 – When the model predicts benign, it is correct 95% of the time.  
#   - **Recall**: 0.99 – The model correctly identifies 99% of all actual benign cases.  
#   - **F1-score**: 0.97 – High effectiveness in identifying benign cases.
# 
# ### **Overall Performance**
# - **Accuracy**: **96%** – The model correctly classifies 96% of the samples.  
# - **Macro Average F1-score**: **0.95** – A balanced measure of performance across both classes.  
# - **Weighted Average F1-score**: **0.96** – Accounts for class imbalance, showing an overall strong performance.
# 
# ### **Key Takeaways**
# - The model **performs exceptionally well** with high accuracy and recall.
# - It is **more confident in predicting benign cases (high recall: 99%)** but still does well in identifying malignant cases.
# - The **high F1-scores** indicate a balanced model with minimal trade-offs between precision and recall.
# 

# %%
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# %%
y_scores = lda.decision_function(X_test_scaled)

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for LDA")
plt.legend(loc="lower right")
plt.show()


# %% [markdown]
# ## **Inference from the ROC Curve for LDA**
# 
# 1. **High AUC Value (0.99)**:  
#    - The **Area Under the Curve (AUC) of 0.99** indicates that the **LDA model performs exceptionally well** in distinguishing between malignant and benign breast cancer cases.  
#    - AUC values close to **1.0** suggest a near-perfect classifier, meaning very few false positives or false negatives.
# 
# 2. **Steep Initial Rise in TPR**:  
#    - The ROC curve shows a **sharp increase** in True Positive Rate (TPR) at very low False Positive Rate (FPR), which suggests that the model **correctly identifies most malignant cases with minimal false alarms**.
# 
# 3. **Close to Ideal Performance**:  
#    - The ideal ROC curve would **hug the top-left corner**, indicating a perfect classifier.  
#    - Since the **LDA ROC curve is very close to this ideal shape**, the model has **excellent predictive power**.
# 

# %% [markdown]
# ## **Final Conclusion**
# 
# - The **Linear Discriminant Analysis (LDA) classifier** performed exceptionally well on the **Breast Cancer dataset**.
# - **Exploratory Data Analysis (EDA)** revealed that the dataset has **no missing values**, some highly **correlated features**, and a **slight class imbalance** favoring benign cases.
# - **Feature Scaling** improved model performance by ensuring all features contributed equally.
# - The **ROC Curve and AUC score (0.99)** indicate that the model has **high discriminatory power**, making it highly effective in differentiating between malignant and benign tumors.
# - The **classification report and confusion matrix** confirmed **high accuracy, precision, and recall**, showing minimal false negatives (which is crucial for medical diagnosis).
# - Given its strong performance, **LDA is a suitable model for this dataset**, though further refinements like **feature selection, hyperparameter tuning, or alternative models** could be explored for even better results.
# 


