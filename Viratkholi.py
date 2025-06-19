# Virat Kohli Performance Prediction Script (Enhanced with Improvement Analysis)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1: Load and Clean Data ---
kohli_stats_file = "Virat Kohli Dataset.xlsx"
sheet_name = "Data"
kohli_df = pd.read_excel(kohli_stats_file, sheet_name=sheet_name)
kohli_df.columns = kohli_df.columns.str.strip().str.replace(" ", "_")

kohli_df['Runs'] = kohli_df['Runs'].astype(str).str.replace("*", "", regex=False)
kohli_df['Runs'] = pd.to_numeric(kohli_df['Runs'], errors='coerce')

if 'Match_Date' in kohli_df.columns:
    kohli_df['Match_Date'] = pd.to_datetime(kohli_df['Match_Date'], errors='coerce')
    kohli_df['Match_Year'] = kohli_df['Match_Date'].dt.year
    kohli_df['Match_Month'] = kohli_df['Match_Date'].dt.month
else:
    kohli_df['Match_Year'] = np.nan
    kohli_df['Match_Month'] = np.nan

kohli_df.sort_values('Match_Date', inplace=True)
kohli_df['Game_Number'] = range(1, len(kohli_df) + 1)

kohli_df['Avg_Last_5'] = kohli_df['Runs'].rolling(5).mean().shift(1)
kohli_df['Low_Score_Count'] = (kohli_df['Runs'] < 20).rolling(5).sum().shift(1).fillna(0)
kohli_df['Improvement_Score'] = kohli_df['Game_Number'] / kohli_df['Game_Number'].max()
kohli_df['Recovery_Adjustment'] = kohli_df['Improvement_Score'] * kohli_df['Low_Score_Count']

kohli_df.rename(columns={'Batting_Position': 'Bat_Order'}, inplace=True)
kohli_df['Bat_Order'] = pd.to_numeric(kohli_df['Bat_Order'], errors='coerce')

kohli_df.dropna(subset=['Runs', 'Avg_Last_5', 'Match_Year', 'Match_Month', 'Bat_Order'], inplace=True)

# --- Step 2: Feature Encoding ---
categorical_cols = ['Opposition', 'Ground', 'Dismissal']
categorical_cols = [col for col in categorical_cols if col in kohli_df.columns]
kohli_encoded_data = pd.get_dummies(kohli_df, columns=categorical_cols, drop_first=True)

core_features = [
    'Innings', 'Avg_Last_5', 'Match_Year', 'Match_Month', 'Bat_Order',
    'Game_Number', 'Low_Score_Count', 'Recovery_Adjustment']

encoded_category_columns = []
for col in kohli_encoded_data.columns:
    if isinstance(col, str) and any(key in col for key in ['Opposition_', 'Ground_', 'Dismissal_']):
        encoded_category_columns.append(col)

final_feature_set = [col for col in core_features if col in kohli_encoded_data.columns] + encoded_category_columns

X_features = kohli_encoded_data[final_feature_set]
y_target = kohli_encoded_data['Runs']

# --- Step 3: Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Random Forest Model
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
forest_rmse = np.sqrt(mean_squared_error(y_test, forest_predictions))
forest_r2 = r2_score(y_test, forest_predictions)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

print("\n--- Model Performance Comparison ---")
print("Random Forest RMSE:", round(forest_rmse, 2), "| R²:", round(forest_r2, 2))
print("Linear Regression RMSE:", round(linear_rmse, 2), "| R²:", round(linear_r2, 2))

# --- Step 4: Future Match Prediction ---
next_match_data = pd.DataFrame([np.zeros(len(X_features.columns))], columns=X_features.columns)

if 'Innings' in X_features.columns: next_match_data.at[0, 'Innings'] = 1
if 'Avg_Last_5' in X_features.columns: next_match_data.at[0, 'Avg_Last_5'] = 50.0
if 'Match_Year' in X_features.columns: next_match_data.at[0, 'Match_Year'] = 2025
if 'Match_Month' in X_features.columns: next_match_data.at[0, 'Match_Month'] = 5
if 'Bat_Order' in X_features.columns: next_match_data.at[0, 'Bat_Order'] = 2
if 'Game_Number' in X_features.columns: next_match_data.at[0, 'Game_Number'] = len(kohli_df) + 1
if 'Low_Score_Count' in X_features.columns: next_match_data.at[0, 'Low_Score_Count'] = 5
if 'Recovery_Adjustment' in X_features.columns: next_match_data.at[0, 'Recovery_Adjustment'] = 0.6

for col in next_match_data.columns:
    if isinstance(col, str):
        if 'Opposition_Australia' in col:
            next_match_data.at[0, col] = 1
        if 'Ground_Mumbai' in col:
            next_match_data.at[0, col] = 1
        if 'Dismissal_bowled' in col:
            next_match_data.at[0, col] = 1

next_match_data = next_match_data[X_features.columns]
predicted_forest = forest_model.predict(next_match_data)[0]
predicted_linear = linear_model.predict(next_match_data)[0]

print("\n--- Future Match Prediction ---")
print(f"Random Forest Predicted Runs: {round(predicted_forest)}")
print(f"Linear Regression Predicted Runs: {round(predicted_linear)}")

# --- Step 5: Graphs and Career Trends ---
plt.figure(figsize=(10, 5))
plt.plot(kohli_df['Game_Number'], kohli_df['Runs'], label='Actual Runs', color='blue', alpha=0.6)
plt.title("Career Runs Trend")
plt.xlabel("Career Match Number")
plt.ylabel("Runs Scored")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Comparison of Model RMSEs
plt.figure(figsize=(6, 4))
plt.bar(['Random Forest', 'Linear Regression'], [forest_rmse, linear_rmse], color=['green', 'orange'])
plt.title("Model RMSE Comparison")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()

# Histogram of Predicted Runs from Random Forest Model
plt.figure(figsize=(7, 4))
plt.hist(forest_predictions, bins=20, color='teal', edgecolor='black')
plt.title("Distribution of Predicted Runs (Random Forest)")
plt.xlabel("Predicted Runs")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(['Random Forest', 'Linear Regression'], [forest_r2, linear_r2], color=['green', 'orange'])
plt.title("Model R² Score Comparison")
plt.ylabel("R² Score")
plt.tight_layout()
plt.show()


importances = forest_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.title("Top Feature Importances (Random Forest)")
plt.bar([X_features.columns[i] for i in indices[:10]], importances[indices[:10]], color='skyblue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.axvline(predicted_forest, color='red', linestyle='--', label='Future Match Prediction')
plt.legend()

# --- Average Runs by Batting Position ---
if 'Bat_Order' in kohli_df.columns:
    plt.figure(figsize=(8, 5))
    kohli_df.groupby('Bat_Order')['Runs'].mean().plot(kind='bar', color='teal')
    plt.title('Average Runs by Batting Position')
    plt.xlabel('Batting Position')
    plt.ylabel('Average Runs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Average Runs by Dismissal Type ---
if 'Dismissal' in kohli_df.columns:
    plt.figure(figsize=(8, 5))
    kohli_df.groupby('Dismissal')['Runs'].mean().sort_values().plot(kind='bar', color='coral')
    plt.title('Average Runs by Dismissal Type')
    plt.ylabel('Average Runs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Runs Over Time ---
if 'Match_Date' in kohli_df.columns:
    plt.figure(figsize=(10, 4))
    kohli_df.sort_values('Match_Date').set_index('Match_Date')['Runs'].plot(color='purple')
    plt.title('Runs Over Time')
    plt.xlabel('Date')
    plt.ylabel('Runs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Ground-wise Performance ---
if 'Ground' in kohli_df.columns:
    top_grounds = kohli_df.groupby('Ground')['Runs'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    top_grounds.plot(kind='barh', color='forestgreen')
    plt.title('Top 10 Grounds with Highest Average Runs')
    plt.xlabel('Average Runs')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# --- Monthly Performance Trend ---
if 'Match_Month' in kohli_df.columns:
    plt.figure(figsize=(8, 4))
    kohli_df.groupby('Match_Month')['Runs'].mean().plot(kind='line', marker='o', color='chocolate')
    plt.title('Monthly Average Performance')
    plt.xlabel('Month')
    plt.ylabel('Average Runs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

top_grounds.to_csv("Top_Grounds.csv")
future_details = pd.DataFrame({
    'Model': ['Random Forest'],
    'Opposition': ['Australia'],
    'Ground': ['Mumbai'],
    'Month': [11],
    'Year': [2025],
    'Batting Position': [3],
    'Recent Form Average': [75.0],
    'Predicted Runs': [round(predicted_forest, 2)]
})

future_details.to_csv("Future_Prediction_Detailed.csv", index=False)

