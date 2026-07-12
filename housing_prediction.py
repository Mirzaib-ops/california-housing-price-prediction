"""
Task 3: California Housing Price Prediction
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
print("Shape:", df.shape)
print(df.head())

# Step 2: Check Missing Values
print("\nMissing values:\n", df.isnull().sum())
df.dropna(inplace=True)

# Step 3: Feature Engineering
df["RoomsPerOccup"] = df["AveRooms"] / df["AveOccup"].clip(lower=1)
df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]
df["PopPerOccup"] = df["Population"] / df["AveOccup"].clip(lower=1)

# Step 4: Train/Test Split
features = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
    "RoomsPerOccup", "BedroomsPerRoom", "PopPerOccup"
]
X = df[features]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nFeature matrix: {X.shape} | Target range: ${y.min()*100000:,.0f} - ${y.max()*100000:,.0f}")

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Step 6: Train & Evaluate Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
print("\n--- Model Training & Evaluation ---")
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}
    print(f"\n{name}")
    print(f"  MAE  : {mae:.4f}  (~${mae*100000:,.0f} avg error)")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

best_name = max(results, key=lambda k: results[k]["r2"])
print(f"\nBest Model: {best_name} (R² = {results[best_name]['r2']:.4f})")

# Step 7: Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].hist(df["MedHouseVal"], bins=50, color="steelblue")
axes[0, 0].axvline(df["MedHouseVal"].mean(), color="red", linestyle="--", label="Mean")
axes[0, 0].set_title("House Price Distribution")
axes[0, 0].legend()

corr = df[features + ["MedHouseVal"]].corr()["MedHouseVal"].sort_values(ascending=False)
sns.heatmap(corr.to_frame(), annot=True, cmap="RdYlGn", ax=axes[0, 1], cbar=True)
axes[0, 1].set_title("Feature Correlation with Price")

names = list(results.keys())
r2_scores = [results[n]["r2"] for n in names]
axes[0, 2].bar(names, r2_scores, color=["red", "orange", "green", "purple"])
axes[0, 2].set_title("Model R² Score Comparison")
axes[0, 2].tick_params(axis="x", rotation=20)

best_pred = results[best_name]["y_pred"]
axes[1, 0].scatter(y_test, best_pred, alpha=0.3)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect prediction")
axes[1, 0].set_title(f"Actual vs Predicted ({best_name})")
axes[1, 0].set_xlabel("Actual Price ($100K)")
axes[1, 0].set_ylabel("Predicted Price ($100K)")
axes[1, 0].legend()

residuals = y_test - best_pred
axes[1, 1].scatter(best_pred, residuals, alpha=0.3, color="orange")
axes[1, 1].axhline(0, color="red", linestyle="--")
axes[1, 1].set_title("Residual Plot")
axes[1, 1].set_xlabel("Predicted Price")
axes[1, 1].set_ylabel("Residuals")

gb_model = results["Gradient Boosting"]["model"]
importances = gb_model.feature_importances_
idx_sort = np.argsort(importances)
axes[1, 2].barh(np.array(features)[idx_sort], importances[idx_sort], color="teal")
axes[1, 2].set_title("Feature Importance (Gradient Boosting)")

plt.suptitle("Task 3: California Housing Price Prediction")
plt.tight_layout()
plt.savefig("task3_housing_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nKey Findings:")
print(f"- {best_name} was the best model (R²={results[best_name]['r2']:.4f})")
print("- Median Income (MedInc) is typically the strongest predictor of housing price")
print("- Ensemble methods generally outperform linear models on this dataset")
