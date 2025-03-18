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
# Embedded Methods<br>
# Embedded methods perform feature selection as part of the model training process. In this study, we use LassoCV, which applies L1 regularization to shrink coefficients to zero for less important features.

# %%
from sklearn.linear_model import LassoCV

# Use LassoCV to perform embedded feature selection with cross-validation
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)

# Create a series with feature coefficients
coef = pd.Series(lasso.coef_, index=X_train.columns)

# Select features with non-zero coefficients
selected_features_lasso = coef[coef != 0].index
print("Selected features (LassoCV):", list(selected_features_lasso))

# %% [markdown]
# Visualizing the Lasso coefficients can help interpret the relative importance of each feature.

# %%
import matplotlib.pyplot as plt

# Plot the coefficients from the Lasso model
plt.figure(figsize=(10, 5))
coef.plot(kind='bar')
plt.title("Lasso Coefficients")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.show()


# %% [markdown]
# Inferences<br>
# Key Consistent Features:<br>
# Across all methods, volatile acidity, chlorides, and sulphates are consistently selected. This reinforces their importance in determining wine quality.
# 
# Method-Specific Insights:<br>
# 
# Wrapper Methods: Provide a concise and highly interpretable feature set, which is advantageous when model simplicity is desired.<br>
# RFE: Its divergence suggests sensitivity to the method of eliminating features based on coefficient stability.<br>
# LassoCV: Offers a more nuanced view by retaining additional features with small but non-zero impacts, potentially leading to improved predictive performance when these subtle effects are valuable.<br>
# Practical Implication:<br>
# The choice of feature selection method should depend on your goals. If you aim for the most interpretable model with a minimal set of features, the consensus from the wrapper methods is compelling. If your priority is to capture all influential signals—even those with modest effects—an embedded method like LassoCV may be preferable.

# %% [markdown]
# # Conclusion
# 
# Both the wrapper method (RFE) and the embedded method successfully identified a concise set of features that are key to predicting hotel booking cancellations. 
# 
# - **Wrapper Approach (RFE):**  
#   - Selected features primarily reflecting booking history and customer engagement.  
#   - Achieved an accuracy of approximately 67.8%, indicating that these features are strong predictors.
# 
# 
# Overall, focusing on a reduced subset of features not only simplifies the model but also enhances interpretability and computational efficiency, making it a practical strategy for real-world applications. This approach underscores the importance of selecting quality features that directly contribute to model performance. 
# 


