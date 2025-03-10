# %% [markdown]
# <center><h3>Lab 4: Feature Selection - Wrapper and Embedded Approach</h3></center>
# <p style='text-align:center'>R Abhijit Srivathsan<br>
# 2448044</p>

# %% [markdown]
# # Importing the libraries and loading the dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"hotel_booking.csv")

# %% [markdown]
# # Checking the data

# %%
df.head()

# %%
df.info()

# %% [markdown]
# # Choosing the target variable

# %%
y = df["is_canceled"]

# %% [markdown]
# # Choosing the best features

# %%
import seaborn as sns

k = 17
cols = df.select_dtypes(include=['number']).corr().nlargest(k, 'is_canceled')['is_canceled'].index
cm = np.corrcoef(df[cols].values.T)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, cbar=True, annot=True, xticklabels=cols.values, yticklabels=cols.values)
plt.show()


# %%

X = df[[ 'lead_time', 'previous_cancellations', 'adults',
       'days_in_waiting_list', 'adr', 'stays_in_week_nights',
       'arrival_date_year', 'arrival_date_week_number', 'children',
       'stays_in_weekend_nights', 'arrival_date_day_of_month',
       'babies', 'previous_bookings_not_canceled',
       'is_repeated_guest', 'booking_changes', 'required_car_parking_spaces',
       'total_of_special_requests']]

# %%
X.info()

# %% [markdown]
# # Checking for null values

# %%
X.isnull().sum()

# %% [markdown]
# # Dealing with the null values

# %%
X['children'].fillna(1,inplace=True)


# %% [markdown]
# # Double Checking for null values

# %%
X.isnull().sum()

# %% [markdown]
# # Train Test Split , 20% of total data for test , 42 as the randomizer

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42,
                                                    stratify=y)

# %% [markdown]
# # Wrapper Method: Recursive Feature Elimination (RFE)
# ## We'll use a Logistic Regression as the base estimator for RFE

# %%


logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=5)  # Example: select top 10 features
rfe.fit(X_train, y_train)

# Get the boolean mask of selected features
selected_features_rfe = X_train.columns[rfe.support_]
print("Selected features (Wrapper - RFE):")
print(selected_features_rfe.tolist())

# Evaluate performance with these selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

logreg.fit(X_train_rfe, y_train)
y_pred_rfe = logreg.predict(X_test_rfe)
print("Accuracy with RFE-selected features:",
      accuracy_score(y_test, y_pred_rfe))


# %% [markdown]
# # **Observation**
# 
# - The Recursive Feature Elimination (RFE) process selected **five** features:
#   - previous_cancellations
#   - previous_bookings_not_canceled
#   - is_repeated_guest
#   - required_car_parking_spaces
#   - total_of_special_requests
# 
# - Using **only these five features** to predict cancellation status yields an **accuracy of approximately 0.678** on the test set.
# 
# - These features capture booking history (previous cancellations, previous non-cancellations, repeated guest status) and customer requests (parking spaces, special requests), suggesting that past behavior and engagement play a significant role in cancellation patterns.
# 

# %% [markdown]
# # **Inference**
# 
# - **Booking history** features (`previous_cancellations`, `previous_bookings_not_canceled`, `is_repeated_guest`) likely **reflect customer loyalty** and historical behavior—key indicators of whether a guest might cancel again.
# - **Engagement or commitment** factors (`required_car_parking_spaces`, `total_of_special_requests`) may correlate with a traveler’s seriousness about the trip; customers with more requests or reserved parking might be **less likely** to cancel.
# - Achieving an accuracy near **67.8%** with just five features suggests these are **highly predictive** relative to other features. However, one should compare this performance with the full feature set or other methods to confirm if **simplicity (fewer features)** outweighs any **marginal drop** in predictive performance.
# - In practice, **domain expertise** could further validate why these features stand out (e.g., special requests may signal stronger commitment), and whether **additional** or **alternative** features could improve the model’s accuracy or interpretability.
# 

# %% [markdown]
# # Embedded Method: Random Forest Feature Importance
# ## Train a Random Forest and retrieve feature importances

# %%

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Feature Importances (Random Forest):")
display(feature_importance_df)

# Let's select top N features based on importance
N = 5
top_features_embedded = feature_importance_df.head(N)['feature'].values
print(f"Top {N} features (Embedded - Random Forest):", top_features_embedded)

# Evaluate performance using only these top features
X_train_emb = X_train[top_features_embedded]
X_test_emb = X_test[top_features_embedded]

rf.fit(X_train_emb, y_train)
y_pred_emb = rf.predict(X_test_emb)
print("Accuracy with Embedded-selected features:",
      accuracy_score(y_test, y_pred_emb))


# %% [markdown]
# # **Observation**
# 
# - Random Forest classifier was trained on the training set, and its feature importances were computed.
# - The features were ranked by importance, and the top 5 features were selected based on this ranking.
# - The model was then re-trained using only these top 5 features, and the resulting accuracy on the test set was recorded.
# - The accuracy obtained using this reduced feature set demonstrates that these features capture a substantial portion of the predictive information.

# %% [markdown]
# # **Inference**
# 
# - The embedded feature selection approach using Random Forest shows that a small subset of features can effectively predict the target variable.
# - The top features identified by the model are considered the most influential in determining hotel booking cancellations.
# - Reducing the feature set can lead to simpler, more interpretable models and may help in reducing overfitting.
# - This technique not only improves computational efficiency but also provides valuable insights into the key factors affecting cancellation behavior.
# 

# %% [markdown]
# # Conclusion
# 
# Both the wrapper method (RFE) and the embedded method (Random Forest feature importance) successfully identified a concise set of features that are key to predicting hotel booking cancellations. 
# 
# - **Wrapper Approach (RFE):**  
#   - Selected features primarily reflecting booking history and customer engagement.  
#   - Achieved an accuracy of approximately 67.8%, indicating that these features are strong predictors.
# 
# - **Embedded Approach (Random Forest):**  
#   - Identified the top 5 most influential features based on the model’s internal importance metrics.  
#   - The reduced feature set maintained competitive predictive performance, demonstrating that much of the necessary information is captured by these features.
# 
# Overall, focusing on a reduced subset of features not only simplifies the model but also enhances interpretability and computational efficiency, making it a practical strategy for real-world applications. This approach underscores the importance of selecting quality features that directly contribute to model performance. 
# 


