import kagglehub

# Download latest version
path = kagglehub.dataset_download("aadyasingh55/impact-of-social-media-on-suicide-rates")

print("Path to dataset files:", path)

import os

# Path to the dataset directory
dataset_path = r"C:\Users\User\.cache\kagglehub\datasets\aadyasingh55\impact-of-social-media-on-suicide-rates\versions\1"

# List all files in the dataset directory
dataset_files = os.listdir(dataset_path)
print(dataset_files)

import pandas as pd

# Example filename (change if necessary)
file_path = os.path.join(dataset_path, 'social-media-impact-on-suicide-rates.csv')

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())

# Get a summary of the data (data types, missing values, etc.)
print(df.info())

# Check for any missing values
print(df.isnull().sum())

# Display basic statistics for numerical columns
print(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Scatter plot for Suicide Rate % change vs Twitter User Count % change
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Twitter user count % change since 2010", y="Suicide Rate % change since 2010", hue="sex", palette="coolwarm")
plt.title("Suicide Rate % Change vs Twitter User Count % Change")
plt.xlabel("Twitter User Count % Change since 2010")
plt.ylabel("Suicide Rate % Change since 2010")
plt.show()

# Scatter plot for Suicide Rate % change vs Facebook User Count % change
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Facebook user count % change since 2010", y="Suicide Rate % change since 2010", hue="sex", palette="coolwarm")
plt.title("Suicide Rate % Change vs Facebook User Count % Change")
plt.xlabel("Facebook User Count % Change since 2010")
plt.ylabel("Suicide Rate % Change since 2010")
plt.show()

# Line plot for Suicide Rate % change over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="Suicide Rate % change since 2010", hue="sex", marker="o")
plt.title("Suicide Rate % Change Over Time")
plt.xlabel("Year")
plt.ylabel("Suicide Rate % Change since 2010")
plt.show()

# Line plot for Twitter User Count % change over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="Twitter user count % change since 2010", hue="sex", marker="o")
plt.title("Twitter User Count % Change Over Time")
plt.xlabel("Year")
plt.ylabel("Twitter User Count % Change since 2010")
plt.show()

# Line plot for Facebook User Count % change over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="Facebook user count % change since 2010", hue="sex", marker="o")
plt.title("Facebook User Count % Change Over Time")
plt.xlabel("Year")
plt.ylabel("Facebook User Count % Change since 2010")
plt.show()


# Line plot for Suicide Rate % change over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="Suicide Rate % change since 2010", hue="sex", marker="o")
plt.title("Suicide Rate % Change Over Time")
plt.xlabel("Year")
plt.ylabel("Suicide Rate % Change since 2010")
plt.show()

# Line plot for Twitter User Count % change over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="Twitter user count % change since 2010", hue="sex", marker="o")
plt.title("Twitter User Count % Change Over Time")
plt.xlabel("Year")
plt.ylabel("Twitter User Count % Change since 2010")
plt.show()


# Line plot for Facebook User Count % change over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="Facebook user count % change since 2010", hue="sex", marker="o")
plt.title("Facebook User Count % Change Over Time")
plt.xlabel("Year")
plt.ylabel("Facebook User Count % Change since 2010")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Compute the correlation matrix
corr = df[['Suicide Rate % change since 2010',
           'Twitter user count % change since 2010',
           'Facebook user count % change since 2010']].corr()

# Create the heatmap
plt.figure(figsize=(10, 8))  # Increase figure size for better label spacing
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1, cbar=True)

# Rotate x and y labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x-axis labels and adjust font size
plt.yticks(rotation=45, ha='right', fontsize=12)  # Rotate y-axis labels and adjust font size

# Set title with a larger font size
plt.title("Correlation Matrix", fontsize=16)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


from sklearn.linear_model import LinearRegression

# Prepare the data for regression
X = df[['Twitter user count % change since 2010']]  # Independent variable
y = df['Suicide Rate % change since 2010']  # Dependent variable

# Instantiate and fit the regression model
reg_model = LinearRegression()
reg_model.fit(X, y)

# Print out the regression coefficients
print(f"Intercept: {reg_model.intercept_}")
print(f"Coefficient (Twitter User Count): {reg_model.coef_[0]}")

# Predicting values
y_pred = reg_model.predict(X)

# Plotting the regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X['Twitter user count % change since 2010'], y=y, color="blue")
plt.plot(X['Twitter user count % change since 2010'], y_pred, color="red")
plt.title("Linear Regression: Suicide Rate vs Twitter User Count")
plt.xlabel("Twitter User Count % Change since 2010")
plt.ylabel("Suicide Rate % Change since 2010")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Filter the dataset for males
df_male = df[df['sex'] == 'Male']

# Create a figure with multiple subplots for better comparison
plt.figure(figsize=(14, 6))

# Plot 1: Suicide Rate % Change vs Twitter User Count % Change
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_male,
                x="Twitter user count % change since 2010",
                y="Suicide Rate % change since 2010",
                color="blue")
plt.title("Male Suicide Rate vs Twitter User Count % Change")
plt.xlabel("Twitter User Count % Change since 2010")
plt.ylabel("Suicide Rate % Change since 2010")
plt.grid(True)


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the column names are clean
df.columns = df.columns.str.strip()  # Strip any unwanted spaces

# Filter the dataset for MLE (Male)
df_male = df[df['sex'] == 'MLE']

# Check if the filtering works correctly
print(f"Rows for MLE (Male): {df_male.shape[0]}")
print(df_male.head())

# Create a figure with multiple subplots for better comparison
plt.figure(figsize=(14, 6))

# Plot 1: Suicide Rate % Change vs Twitter User Count % Change
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_male,
                x="Twitter user count % change since 2010",
                y="Suicide Rate % change since 2010",
                color="blue")
plt.title("Male Suicide Rate vs Twitter User Count % Change")
plt.xlabel("Twitter User Count % Change since 2010")
plt.ylabel("Suicide Rate % Change since 2010")
plt.grid(True)


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Filter the dataset for MLE (Male)
df_male = df[df['sex'] == 'MLE']

# Let's focus on data from 2010 to 2018
df_male = df_male[(df_male['year'] >= 2010) & (df_male['year'] <= 2018)]

# Create a line plot with three lines for each variable
plt.figure(figsize=(10, 6))

# Plot Suicide Rate % change since 2010
sns.lineplot(data=df_male, x='year', y='Suicide Rate % change since 2010', label='Suicide Rate % Change', color='blue', marker='o')

# Plot Twitter User Count % change since 2010
sns.lineplot(data=df_male, x='year', y='Twitter user count % change since 2010', label='Twitter User Count % Change', color='green', marker='o')

# Plot Facebook User Count % change since 2010
sns.lineplot(data=df_male, x='year', y='Facebook user count % change since 2010', label='Facebook User Count % Change', color='red', marker='o')

# Set plot labels and title
plt.title("Male Suicide Rate vs Social Media User Count % Change (2010-2018)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Percentage Change from 2010", fontsize=14)

# Show grid
plt.grid(True)

# Show the legend
plt.legend(title="Legend", loc="upper left")

# Show the plot
plt.tight_layout()
plt.show()

