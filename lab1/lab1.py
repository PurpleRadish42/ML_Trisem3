# %% [markdown]
# <center><h3>Lab 1: Data Preprocessing</h3></center>
# <p style='text-align:center'>R Abhijit Srivathsan<br>
# 2448044</p>

# %% [markdown]
# ### Importing pandas

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# %% [markdown]
# ### Loading the datasets

# %%
data = pd.read_csv('Housing_Price.csv')
data.head()

# %% [markdown]
# ### Calculating the total *missing values*

# %%
missing_values = data.isnull().sum()
missing_percent = (missing_values / len(data)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_data = missing_data[missing_data['Missing Values'] > 0]
missing_data

# %% [markdown]
# <div class="alert alert-block alert-info"><b>Note: </b>Dropped columns with more than 80% missing values </div>

# %%
# Drop columns with more than 80% missing values
data.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'], inplace=True)

# %% [markdown]
# ### Filling in $median$ values for <code>LotFrontage</code> column, and filling 0's for the other numerical columns

# %%
# Impute numerical missing values
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

# %% [markdown]
# ### Filling <code>None</code> for categorical missing values

# %%
# Impute categorical missing values
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
data[garage_cols] = data[garage_cols].fillna('None')

bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
data[bsmt_cat_cols] = data[bsmt_cat_cols].fillna('None')

data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
data['MasVnrType'] = data['MasVnrType'].fillna('None')

# Impute mode for categorical variables
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

# %% [markdown]
# ### Computing *Total missing values*

# %%
# Verify missing values handled
print("Total missing values left:", data.isnull().sum().sum())

# %% [markdown]
# ### Plotting *SalesPrice* Distribution

# %%
# Plot SalePrice distribution
plt.figure(figsize=(8, 5))
sb.histplot(data['SalePrice'], kde=True, bins=30)
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()


# %%
# Convert categorical features to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

print("New dataset shape after encoding:", data.shape)

# %%
# Check correlation with SalePrice
corr_matrix = data.corr()
top_corr_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(10)

# Plot top correlated features
plt.figure(figsize=(10, 5))
sb.barplot(x=top_corr_features.index, y=top_corr_features.values)
plt.xticks(rotation=45)
plt.title("Top 10 Features Correlated with SalePrice")
plt.ylabel("Correlation")
plt.show()

# Display top correlated features
top_corr_features

# %% [markdown]
# ### Scatterplot: GrLivArea vs SalePrice

# %%
plt.figure(figsize=(8, 5))
sb.scatterplot(x='GrLivArea', y='SalePrice', data=data)
plt.title('GrLivArea vs SalePrice')
plt.xlabel('GrLivArea (Above ground living area)')
plt.ylabel('SalePrice')
plt.show()
# Scatterplot: SalePrice vs LotArea
plt.figure(figsize=(8, 5))
sb.scatterplot(x=data['LotArea'], y=data['SalePrice'], alpha=0.6, color='purple')
plt.title("SalePrice vs LotArea")
plt.xlabel("LotArea")
plt.ylabel("SalePrice")
plt.show()

# %% [markdown]
# ### Detecting Outliers

# %%
#detecting outliers
# Plot boxplots for highly correlated numerical features
top_numeric_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
plt.figure(figsize=(15, 8))

for i, col in enumerate(top_numeric_features):
    plt.subplot(2, 3, i+1)
    sb.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()



