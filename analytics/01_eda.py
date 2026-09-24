import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. LOAD DATA (Exactly once)
print("Loading Titanic dataset...")
df = sns.load_dataset('titanic')
df.to_csv('titanic.csv', index=False)
print("Saved to titanic.csv for offline fallback.")

# 2. MISSING VALUE HANDLING
# --- YOUR LOGIC HERE ---
# Calculate missing % for each column.
# If <5%: Drop. If 5-30%: Impute. If >30%: Drop or encode 'missing'.

# 3. UNIVARIATE ANALYSIS
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df['age'], kde=True, ax=axes[0, 0])
sns.boxplot(x=df['age'], ax=axes[0, 1])
sns.histplot(df['fare'], kde=True, ax=axes[1, 0])
sns.boxplot(x=df['fare'], ax=axes[1, 1])
plt.tight_layout()
plt.savefig('univariate.png')

# IQR Outliers
def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((series < lower) | (series > upper)).sum()

print(f"Age outliers: {count_outliers(df['age'].dropna())}")
print(f"Fare outliers: {count_outliers(df['fare'].dropna())}")

# 4. BIVARIATE ANALYSIS
print("\nSurvival by Sex:\n", df.groupby('sex')['survived'].mean())
print("\nSurvival by Pclass:\n", df.groupby('pclass')['survived'].mean())
print("\nSurvival by Sex & Pclass:\n", df.groupby(['sex', 'pclass'])['survived'].mean())

# Correlation Matrix (exactly 6 columns)
cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df[cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation.png')

# 5. MULTIVARIATE CHARTS
# --- YOUR 4 CHARTS WITH WRITTEN INTERPRETATION HERE ---
