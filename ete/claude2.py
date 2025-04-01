# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# For modeling and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# For warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style for visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# =============================================================================
# DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

# Fetch dataset from UCI ML Repository
real_estate_valuation = fetch_ucirepo(id=477) 

# Extract features and target
X = real_estate_valuation.data.features 
y = real_estate_valuation.data.targets 

# Display metadata and variables information
print("=== Dataset Metadata ===")
print(real_estate_valuation.metadata)
print("\n=== Variable Information ===")
print(real_estate_valuation.variables)

# Combine features and target for easier data exploration
df = pd.concat([X, y], axis=1)

# Display the first few rows of the dataset
print("\n=== First 5 rows of the dataset ===")
print(df.head())

# Display basic information about the dataset
print("\n=== Dataset Information ===")
print(df.info())

# Display summary statistics
print("\n=== Summary Statistics ===")
print(df.describe())

# Check for missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n=== EXPLORATORY DATA ANALYSIS ===")

# Understanding the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['Y house price of unit area'], kde=True)
plt.title('Distribution of House Price of Unit Area', fontsize=15)
plt.xlabel('House Price of Unit Area', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Check if the target variable is normally distributed using Q-Q plot
plt.figure(figsize=(10, 6))
sm.qqplot(df['Y house price of unit area'], line='45')
plt.title('Q-Q Plot for House Price of Unit Area', fontsize=15)
plt.show()

# Explore the distribution of each numerical feature
plt.figure(figsize=(18, 15))
for i, column in enumerate(X.columns):
    plt.subplot(3, 2, i+1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}', fontsize=12)
plt.tight_layout()
plt.show()

# Explore the relationship between features and target
plt.figure(figsize=(20, 16))
for i, column in enumerate(X.columns):
    plt.subplot(3, 2, i+1)
    sns.scatterplot(x=df[column], y=df['Y house price of unit area'])
    plt.title(f'House Price vs {column}', fontsize=12)
    plt.xlabel(column, fontsize=10)
    plt.ylabel('House Price of Unit Area', fontsize=10)
plt.tight_layout()
plt.show()

# Correlation analysis
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=15)
plt.show()

# Print the correlation with target
print("\n=== Correlation with Target Variable ===")
correlations = df.corr()['Y house price of unit area'].sort_values(ascending=False)
print(correlations)

# Pairplot for visualizing relationships between variables
plt.figure(figsize=(14, 12))
sns.pairplot(df, height=2.5)
plt.suptitle('Pairwise Relationships Between Variables', y=1.02, fontsize=15)
plt.show()

# Box plots to identify outliers
plt.figure(figsize=(18, 15))
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df[column])
    plt.title(f'Box Plot of {column}', fontsize=12)
plt.tight_layout()
plt.show()

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

print("\n=== DATA PREPROCESSING ===")

# Function to detect outliers using IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Check for outliers in each feature
print("Outliers in each column:")
for column in df.columns:
    outliers = detect_outliers(df, column)
    print(f"{column}: {len(outliers)} outliers")

# We'll handle outliers by capping them at the boundaries
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

print("\nOutliers have been handled by capping at IQR boundaries")

# Check for skewness in the data
print("\nSkewness in each column before transformation:")
print(df.skew())

# Apply log transformation to highly skewed columns (skewness > 0.5)
skewed_columns = df.skew()[df.skew() > 0.5].index.tolist()
print("\nApplying log transformation to highly skewed columns:", skewed_columns)

for column in skewed_columns:
    # Add a small constant to handle zero values
    if (df[column] <= 0).any():
        df[column] = np.log1p(df[column] - df[column].min() + 1)
    else:
        df[column] = np.log1p(df[column])

print("\nSkewness after transformation:")
print(df.skew())

# Split the data back into features and target
X_processed = df.drop(columns=['Y house price of unit area'])
y_processed = df['Y house price of unit area']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_processed.columns)

print("\nFeatures have been scaled. Sample of scaled features:")
print(X_scaled_df.head())

# =============================================================================
# FEATURE SELECTION / FEATURE EXTRACTION
# =============================================================================

print("\n=== FEATURE SELECTION / EXTRACTION ===")

# Method 1: Correlation-based feature selection
print("\n1. Correlation-based Feature Selection:")
correlation_target = abs(correlation_matrix['Y house price of unit area']).sort_values(ascending=False)
print(correlation_target)

# Method 2: Statistical feature selection using SelectKBest
print("\n2. Statistical Feature Selection (SelectKBest with f_regression):")
k_best_selector = SelectKBest(f_regression, k=3)
X_k_best = k_best_selector.fit_transform(X_scaled, y_processed)
selected_features_mask = k_best_selector.get_support()
selected_features = X_processed.columns[selected_features_mask]
print("Selected features by SelectKBest:", selected_features.tolist())
print("Feature scores:", k_best_selector.scores_)

# Method 3: Recursive Feature Elimination (RFE)
print("\n3. Recursive Feature Elimination (RFE):")
lr = LinearRegression()
rfe = RFE(estimator=lr, n_features_to_select=3)
X_rfe = rfe.fit_transform(X_scaled, y_processed)
selected_features_rfe = X_processed.columns[rfe.support_]
print("Selected features by RFE:", selected_features_rfe.tolist())
print("Feature ranking (lower is better):", rfe.ranking_)

# Method 4: Check for multicollinearity using VIF
print("\n4. Multicollinearity Analysis using VIF:")
X_const = sm.add_constant(X_processed)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print(vif_data)

# Based on the analyses, select the most important features
final_features = selected_features_rfe.tolist()
print("\nFinal selected features for modeling:", final_features)

# Feature importance visualization
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(k_best_selector.scores_, index=X_processed.columns)
feature_importance.sort_values(ascending=False).plot(kind='barh')
plt.title('Feature Importance Based on f_regression', fontsize=15)
plt.xlabel('F-Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.show()

# =============================================================================
# MODEL DEVELOPMENT
# =============================================================================

print("\n=== MODEL DEVELOPMENT ===")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_processed, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 1. Linear Regression (Baseline Model)
print("\n1. Baseline Model: Multiple Linear Regression")
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Evaluate the baseline model
y_train_pred = linear_reg.predict(X_train)
y_test_pred = linear_reg.predict(X_test)

print("Training R²:", r2_score(y_train, y_train_pred))
print("Testing R²:", r2_score(y_test, y_test_pred))
print("Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Testing MSE:", mean_squared_error(y_test, y_test_pred))
print("Training RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Testing RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Training MAE:", mean_absolute_error(y_train, y_train_pred))
print("Testing MAE:", mean_absolute_error(y_test, y_test_pred))

# Display the coefficients
coef_df = pd.DataFrame({
    'Feature': X_processed.columns,
    'Coefficient': linear_reg.coef_
})
print("\nModel Coefficients:")
print(coef_df.sort_values(by='Coefficient', ascending=False))

# 2. Ridge Regression (L2 Regularization)
print("\n2. Ridge Regression with Cross-Validation")
ridge_alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_cv = GridSearchCV(
    Ridge(), 
    param_grid={'alpha': ridge_alphas},
    cv=5, 
    scoring='neg_mean_squared_error'
)
ridge_cv.fit(X_train, y_train)

print("Best Ridge alpha:", ridge_cv.best_params_)
print("Best Ridge CV score:", -ridge_cv.best_score_)

# Evaluate Ridge regression with the best alpha
ridge_best = Ridge(alpha=ridge_cv.best_params_['alpha'])
ridge_best.fit(X_train, y_train)

y_train_pred_ridge = ridge_best.predict(X_train)
y_test_pred_ridge = ridge_best.predict(X_test)

print("Ridge Training R²:", r2_score(y_train, y_train_pred_ridge))
print("Ridge Testing R²:", r2_score(y_test, y_test_pred_ridge))
print("Ridge Training RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred_ridge)))
print("Ridge Testing RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred_ridge)))

# 3. Lasso Regression (L1 Regularization)
print("\n3. Lasso Regression with Cross-Validation")
lasso_alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
lasso_cv = GridSearchCV(
    Lasso(max_iter=10000), 
    param_grid={'alpha': lasso_alphas},
    cv=5, 
    scoring='neg_mean_squared_error'
)
lasso_cv.fit(X_train, y_train)

print("Best Lasso alpha:", lasso_cv.best_params_)
print("Best Lasso CV score:", -lasso_cv.best_score_)

# Evaluate Lasso regression with the best alpha
lasso_best = Lasso(alpha=lasso_cv.best_params_['alpha'], max_iter=10000)
lasso_best.fit(X_train, y_train)

y_train_pred_lasso = lasso_best.predict(X_train)
y_test_pred_lasso = lasso_best.predict(X_test)

print("Lasso Training R²:", r2_score(y_train, y_train_pred_lasso))
print("Lasso Testing R²:", r2_score(y_test, y_test_pred_lasso))
print("Lasso Training RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred_lasso)))
print("Lasso Testing RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred_lasso)))

# Check Lasso features (coefficients that are not zero)
lasso_coef_df = pd.DataFrame({
    'Feature': X_processed.columns,
    'Coefficient': lasso_best.coef_
})
print("\nLasso Coefficients (features with non-zero coefficients are selected):")
print(lasso_coef_df.sort_values(by='Coefficient', ascending=False))

# 4. Polynomial Regression
print("\n4. Polynomial Regression")
poly_degrees = [2, 3]
poly_pipelines = {}
poly_scores = {}

for degree in poly_degrees:
    # Create polynomial features
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Create pipeline
    pipeline = Pipeline([
        ('poly_features', polynomial_features),
        ('ridge', Ridge(alpha=ridge_cv.best_params_['alpha']))
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Store pipeline
    poly_pipelines[degree] = pipeline
    
    # Evaluate the model
    y_train_pred_poly = pipeline.predict(X_train)
    y_test_pred_poly = pipeline.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred_poly)
    test_r2 = r2_score(y_test, y_test_pred_poly)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_poly))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_poly))
    
    poly_scores[degree] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }
    
    print(f"\nPolynomial Regression (degree {degree}):")
    print(f"Training R²: {train_r2}")
    print(f"Testing R²: {test_r2}")
    print(f"Training RMSE: {train_rmse}")
    print(f"Testing RMSE: {test_rmse}")

# Find the best polynomial model
best_degree = max(poly_scores, key=lambda k: poly_scores[k]['test_r2'])
print(f"\nBest Polynomial Degree: {best_degree} with Test R²: {poly_scores[best_degree]['test_r2']}")

# Select the best model based on test R²
models = {
    'Multiple Linear Regression': (linear_reg, y_test_pred),
    'Ridge Regression': (ridge_best, y_test_pred_ridge),
    'Lasso Regression': (lasso_best, y_test_pred_lasso),
    f'Polynomial Regression (degree {best_degree})': (poly_pipelines[best_degree], poly_pipelines[best_degree].predict(X_test))
}

model_scores = {
    model_name: r2_score(y_test, pred) 
    for model_name, (_, pred) in models.items()
}

best_model_name = max(model_scores, key=model_scores.get)
best_model, best_pred = models[best_model_name]

print(f"\nBest Model: {best_model_name} with Test R²: {model_scores[best_model_name]}")

# =============================================================================
# MODEL EVALUATION
# =============================================================================

print("\n=== MODEL EVALUATION ===")

# Comprehensive evaluation of the best model
print(f"\nEvaluation of {best_model_name}:")

# Cross-validation for robustness check
cv_scores = cross_val_score(
    best_model, X_scaled_df, y_processed, 
    cv=5, scoring='r2'
)
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R²: {np.mean(cv_scores)}")
print(f"Standard Deviation of CV R²: {np.std(cv_scores)}")

# If the best model is a polynomial model, we need to use the pipeline
if 'Polynomial' in best_model_name:
    y_train_pred_best = best_model.predict(X_train)
    y_test_pred_best = best_model.predict(X_test)
else:
    y_train_pred_best = best_model.predict(X_train)
    y_test_pred_best = best_model.predict(X_test)

# Calculate evaluation metrics
train_r2 = r2_score(y_train, y_train_pred_best)
test_r2 = r2_score(y_test, y_test_pred_best)
train_mse = mean_squared_error(y_train, y_train_pred_best)
test_mse = mean_squared_error(y_test, y_test_pred_best)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred_best)
test_mae = mean_absolute_error(y_test, y_test_pred_best)

print(f"\nTraining R²: {train_r2}")
print(f"Testing R²: {test_r2}")
print(f"Training MSE: {train_mse}")
print(f"Testing MSE: {test_mse}")
print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {test_rmse}")
print(f"Training MAE: {train_mae}")
print(f"Testing MAE: {test_mae}")

# Visualize actual vs predicted values
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_test_pred_best, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual House Prices', fontsize=12)
plt.ylabel('Predicted House Prices', fontsize=12)
plt.title(f'Actual vs Predicted House Prices - {best_model_name}', fontsize=15)
plt.show()

# Visualize residuals
residuals = y_test - y_test_pred_best
plt.figure(figsize=(12, 8))
plt.scatter(y_test_pred_best, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted House Prices', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot - Checking for Homoscedasticity', fontsize=15)
plt.show()

# Residual histogram
plt.figure(figsize=(12, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Residuals', fontsize=15)
plt.show()

# QQ plot for residuals
plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals', fontsize=15)
plt.show()

# Feature importance of the best model (if applicable)
if hasattr(best_model, 'coef_'):
    # For linear models
    if 'Polynomial' in best_model_name:
        # For polynomial models, it's complicated to interpret coefficients
        print("\nFor polynomial models, coefficients are not directly interpretable.")
    else:
        coef_df = pd.DataFrame({
            'Feature': X_processed.columns,
            'Coefficient': best_model.coef_
        })
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df)
        plt.title(f'Feature Importance - {best_model_name}', fontsize=15)
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.show()
        
        print("\nFeature Coefficients (Importance):")
        print(coef_df[['Feature', 'Coefficient']])

# =============================================================================
# INSIGHTS AND CONCLUSIONS
# =============================================================================

print("\n=== INSIGHTS AND CONCLUSIONS ===")

print("\nKey Findings from EDA:")
print("1. The distribution of house prices shows some right skewness, indicating more properties at lower price points.")
print("2. Distance to the nearest MRT station and age of the house show strong correlations with house prices.")
print("3. The dataset has a good spread of properties across different areas, ages, and price points.")

print("\nInsights from Feature Selection:")
print(f"1. The most important features for predicting house prices are: {', '.join(final_features)}")
print("2. Distance to nearest MRT station has a negative correlation with house prices, indicating locations closer to public transport command higher prices.")
print("3. Number of convenience stores in the vicinity positively impacts house prices, reflecting the premium for convenient locations.")

print("\nModel Selection and Performance:")
print(f"1. The best performing model is {best_model_name} with a test R² of {model_scores[best_model_name]:.4f}")
print(f"2. The model can explain approximately {model_scores[best_model_name]*100:.2f}% of the variance in house prices.")
print(f"3. The average prediction error (RMSE) is {test_rmse:.4f}, which represents the average deviation of predictions from actual house prices.")

print("\nRecommendations and Applications:")
print("1. For real estate investors: Focus on properties closer to MRT stations and with more convenience stores nearby.")
print("2. For property developers: Consider the house age and location characteristics as key determinants of pricing strategy.")
print("3. For homebuyers: Use this model to evaluate if a property is reasonably priced based on its features.")

print("\nLimitations and Future Improvements:")
print("1. The model could be enhanced by incorporating more features such as property condition, school districts, crime rates, etc.")
print("2. Non-linear relationships might be better captured using more complex models like gradient boosting or neural networks.")
print("3. Temporal analysis could be valuable to understand how real estate prices evolve over time.")
