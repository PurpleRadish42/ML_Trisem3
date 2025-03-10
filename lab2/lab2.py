# %% [markdown]
# <center><h3>Lab 2: Regression Analysis</h3></center>
# <p style='text-align:center'>R Abhijit Srivathsan<br>
# 2448044</p>

# %% [markdown]
# <center><i><h2>Simple Linear Regression</h2></i></center>

# %% [markdown]
# ### Importing Libraries

# %%
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error,mean_absolute_percentage_error,r2_score # type: ignore

# %% [markdown]
# ### Importing Data

# %%
df = pd.read_csv(r"Salary_Data.csv")

# %% [markdown]
# ### Exploratory Data Analysis

# %%
df.head()

# %%
df.describe()

# %%
df.isnull().sum()

# %% [markdown]
# ### Dropping `Na` Values

# %%
df = df.dropna()

# %% [markdown]
# ### Double Checking `Na` values

# %%
df.isnull().sum()

# %% [markdown]
# ### Scatter Plot

# %%
import matplotlib.pyplot as plt #type: ignore
# Scatter plot
plt.scatter(df['Years of Experience'], df['Salary'], alpha=0.8)

# Adding labels and title
plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True)

# Display the plot
plt.show()

# %% [markdown]
# ### Boxplot

# %%
plt.boxplot(df['Salary'], vert=False, patch_artist=True)
plt.title('Salary Distribution (Box Plot)')
plt.xlabel('Salary')
plt.show()

# %% [markdown]
# ### Removing Outliers

# %%
from scipy.stats import zscore #type: ignore

# Compute Z-scores for each column
z_scores = df.apply(zscore)

# Remove rows where Z-scores are above the threshold
threshold = 3
filtered_df = df[(z_scores['Years of Experience'].abs() < threshold) & 
                 (z_scores['Salary'].abs() < threshold)]

print("Filtered DataFrame using Z-score:")
filtered_df.head()
df = filtered_df
df.head()

# %% [markdown]
# ### Defining $Feature$ & $Target$ Variable

# %%
X = df[['Years of Experience']]  # Features (independent variable)
y = df['Salary']        # Target (dependent variable)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train.head()

# %% [markdown]
# ### Model Creation & Fitting

# %%
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# %% [markdown]
# ### Making Predictions

# %%
y_pred = model.predict(X_test)
y_pred

# %% [markdown]
# ### Model Evaluation

# %%
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean Absolute Error = ', mae)
print('Mean Absolute Percentage Error = ', mape)
print('Mean Squared Error = ', mse)
print('R2 Square = ', r_squared)

# %% [markdown]
# ### Plotting the Model

# %%
# Plot the data points
plt.scatter(X, y, color='blue', label='Data points')

# Plot the regression line
plt.plot(X, model.predict(X), color='red', label='Regression line')

plt.title('Experience vs Salary (Linear Regression)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# %% [markdown]
# ### Actual vs Predicted graph

# %%
c = [i for i in range(1,len(y_test)+1,1)]         
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Actual', fontsize=18)                              
plt.ylabel('Predicted', fontsize=16)    

# %% [markdown]
# ### Predicting our Data

# %%
model.predict([[10.0]])

# %% [markdown]
# <center><i><h2>Multiple Linear Regression</h2></i></center>

# %% [markdown]
# ### Importing Libraries

# %%
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns# type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from sklearn.metrics import mean_absolute_percentage_error # type: ignore  
from sklearn.metrics import mean_squared_error # type: ignore
from sklearn.metrics import r2_score # type: ignore

# %% [markdown]
# ### Reading data

# %%
df = pd.read_csv(r"auto-mpg.csv")

# %% [markdown]
# ### Exploratory Data Analysis

# %%
df.head()

# %%
df.describe()

# %% [markdown]
# ### Checking for `Na` values

# %%
df.isnull().sum()

# %% [markdown]
# ### Histogram

# %%
# Histogram
plt.hist(df['mpg'], bins=5, alpha=0.8, color='blue', edgecolor='black')
plt.title('Distribution of MPG')
plt.xlabel('Miles Per Gallon (MPG)')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ### Scatter Plot

# %%
plt.scatter(df['weight'], df['mpg'], color='green', alpha=0.7)
plt.title('Weight vs MPG')
plt.xlabel('Weight')
plt.ylabel('Miles Per Gallon (MPG)')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Boxplot

# %%
# Boxplot for 'mpg' across different 'cylinders'
plt.figure(figsize=(8, 6))
sns.boxplot(x='cylinders', y='mpg', data=df)
plt.title('MPG across different Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('MPG')
plt.show()

# %% [markdown]
# ### Checking unique values

# %%
df.horsepower.unique()

# %% [markdown]
# ### Handling the <code>?</code>

# %%
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())  # Filling with mean
df['horsepower'] = df['horsepower'].astype(int)

# %% [markdown]
# ### Defining *Features* & *Targets*

# %%
X = df[['horsepower', 'weight', 'cylinders']]  # Independent variables
y = df['mpg']  # Dependent variable

# %% [markdown]
# ### Train-Test split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### Initialize and fit the model

# %%
model = LinearRegression()
model.fit(X_train, y_train)

# Display coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# %% [markdown]
# ### Making predictions on the test set

# %%
y_pred = model.predict(X_test)
y_pred[:5]

# %% [markdown]
# ### Evaluating the model

# %%
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean Absolute Error = ', mae)
print('Mean Absolute Percentage Error = ', mape)
print('Mean Squared Error = ', mse)
print('R2 Square = ', r_squared)

# %% [markdown]
# ### Checking Regression Line

# %%
import matplotlib.pyplot as plt

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Perfect prediction line
plt.title('Actual vs Predicted MPG')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Plotting Actual vs Predicted

# %%
c = [i for i in range(1,len(y_test)+1,1)]         
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Actual', fontsize=18)                              
plt.ylabel('Predicted', fontsize=16)

# %% [markdown]
# ### Model Prediction

# %%
model.predict([[110,2720,3]])


