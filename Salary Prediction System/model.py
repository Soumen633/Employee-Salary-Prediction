# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Data Scientist Job Data Analysis with Random Forest Regressor ===\n")

# Load the dataset
print("1. Loading Dataset...")
df = pd.read_csv('data/eda_data.csv')
print(f"Dataset loaded successfully! Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Basic information about the dataset
print("\n2. Dataset Information:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Missing values per column:")
print(df.isnull().sum().head(10))

# Exploratory Data Analysis
print("\n3. Exploratory Data Analysis...")

# Check data types
print("\nData types:")
print(df.dtypes.head(10))

# Statistical summary for numerical columns
print("\nStatistical Summary for numerical columns:")
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(df[numerical_cols].describe())

# Data Cleaning and Preprocessing
print("\n4. Data Cleaning and Preprocessing...")

# Create a copy for processing
df_processed = df.copy()

# Handle missing values in target variable (avg_salary)
print(f"Missing values in avg_salary before cleaning: {df_processed['avg_salary'].isnull().sum()}")

# Remove rows where target variable is missing
df_processed = df_processed.dropna(subset=['avg_salary'])
print(f"Rows after removing missing target values: {df_processed.shape[0]}")

# Select relevant features for prediction
# Based on the data, we'll use numerical and some categorical features
features_to_use = [
    'min_salary', 'max_salary', 'Rating', 'hourly', 'employer_provided',
    'python_yn', 'R_yn', 'spark', 'aws', 'excel', 'same_state', 'age',
    'desc_len', 'num_comp'
]

# Check which features exist in the dataset
available_features = [col for col in features_to_use if col in df_processed.columns]
print(f"\nAvailable features for modeling: {available_features}")

# Create feature matrix and target vector
X = df_processed[available_features].copy()
y = df_processed['avg_salary'].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Handle missing values in features
print("\n5. Handling Missing Values...")
print("Missing values in features before imputation:")
print(X.isnull().sum())

# Impute missing values
# For numerical features, use median
# For binary features, use mode (most frequent)
imputer_num = SimpleImputer(strategy='median')
imputer_mode = SimpleImputer(strategy='most_frequent')

# Separate numerical and binary columns
binary_cols = ['hourly', 'employer_provided', 'python_yn', 'R_yn', 'spark', 'aws', 'excel', 'same_state']
numerical_cols = [col for col in available_features if col not in binary_cols]

# Apply imputation
if numerical_cols:
    X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

if binary_cols:
    # Only impute binary columns that exist in our dataset
    existing_binary_cols = [col for col in binary_cols if col in X.columns]
    if existing_binary_cols:
        X[existing_binary_cols] = imputer_mode.fit_transform(X[existing_binary_cols])

print("Missing values after imputation:")
print(X.isnull().sum())

# Feature Engineering
print("\n6. Feature Engineering...")

# Create salary range feature
if 'min_salary' in X.columns and 'max_salary' in X.columns:
    X['salary_range'] = X['max_salary'] - X['min_salary']
    print("Created salary_range feature")

# Create experience level indicator
if 'age' in X.columns:
    X['experience_level'] = pd.cut(X['age'], bins=[0, 5, 15, 30, 100], 
                                  labels=['New', 'Moderate', 'Established', 'Legacy'])
    # Convert to numerical
    le = LabelEncoder()
    X['experience_level_encoded'] = le.fit_transform(X['experience_level'].astype(str))
    X.drop('experience_level', axis=1, inplace=True)
    print("Created experience_level feature")

# Create tech skills count
tech_skills = ['python_yn', 'R_yn', 'spark', 'aws', 'excel']
existing_tech_skills = [col for col in tech_skills if col in X.columns]
if existing_tech_skills:
    X['tech_skills_count'] = X[existing_tech_skills].sum(axis=1)
    print("Created tech_skills_count feature")

print(f"Final feature matrix shape: {X.shape}")

# Split the data
print("\n7. Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Feature Scaling (optional for Random Forest, but good practice)
print("\n8. Feature Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Model Training
print("\n9. Training Random Forest Regressor...")

# Initialize Random Forest with basic parameters
rf_basic = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_basic.fit(X_train, y_train)
print("Basic Random Forest model trained successfully!")

# Hyperparameter Tuning
print("\n10. Hyperparameter Tuning...")

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform grid search
rf_grid = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf_grid,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Use the best model
rf_best = grid_search.best_estimator_

# Model Evaluation
print("\n11. Model Evaluation...")

# Make predictions
y_pred_basic = rf_basic.predict(X_test)
y_pred_best = rf_best.predict(X_test)

# Calculate metrics for basic model
mse_basic = mean_squared_error(y_test, y_pred_basic)
mae_basic = mean_absolute_error(y_test, y_pred_basic)
r2_basic = r2_score(y_test, y_pred_basic)

# Calculate metrics for best model
mse_best = mean_squared_error(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("\nBasic Random Forest Performance:")
print(f"Mean Squared Error: {mse_basic:.2f}")
print(f"Mean Absolute Error: {mae_basic:.2f}")
print(f"R² Score: {r2_basic:.4f}")
print(f"RMSE: {np.sqrt(mse_basic):.2f}")

print("\nOptimized Random Forest Performance:")
print(f"Mean Squared Error: {mse_best:.2f}")
print(f"Mean Absolute Error: {mae_best:.2f}")
print(f"R² Score: {r2_best:.4f}")
print(f"RMSE: {np.sqrt(mse_best):.2f}")

# Cross-validation
print("\n12. Cross-Validation...")
cv_scores = cross_val_score(rf_best, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature Importance Analysis
print("\n13. Feature Importance Analysis...")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

# Visualization
print("\n14. Creating Visualizations...")

# Create a comprehensive visualization dashboard
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Random Forest Regressor Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Actual vs Predicted scatter plot
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.6, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Salary')
axes[0, 0].set_ylabel('Predicted Salary')
axes[0, 0].set_title(f'Actual vs Predicted Salaries\nR² = {r2_best:.4f}')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals plot
residuals = y_test - y_pred_best
axes[0, 1].scatter(y_pred_best, residuals, alpha=0.6, color='green')
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted Salary')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals Plot')
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature importance plot
top_features = feature_importance.head(10)
axes[0, 2].barh(top_features['feature'], top_features['importance'])
axes[0, 2].set_xlabel('Importance')
axes[0, 2].set_title('Top 10 Feature Importances')
axes[0, 2].grid(True, alpha=0.3)

# 4. Distribution of actual vs predicted
axes[1, 0].hist(y_test, bins=30, alpha=0.7, label='Actual', color='blue')
axes[1, 0].hist(y_pred_best, bins=30, alpha=0.7, label='Predicted', color='red')
axes[1, 0].set_xlabel('Salary')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution: Actual vs Predicted')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Model comparison
models = ['Basic RF', 'Optimized RF']
r2_scores = [r2_basic, r2_best]
rmse_scores = [np.sqrt(mse_basic), np.sqrt(mse_best)]

x_pos = np.arange(len(models))
width = 0.35

axes[1, 1].bar(x_pos - width/2, r2_scores, width, label='R² Score', alpha=0.8)
axes[1, 1].set_xlabel('Models')
axes[1, 1].set_ylabel('R² Score')
axes[1, 1].set_title('Model Performance Comparison')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(models)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Learning curve (simplified)
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
val_scores = []

for size in train_sizes:
    # Sample training data
    sample_size = int(size * len(X_train))
    X_sample = X_train.iloc[:sample_size]
    y_sample = y_train.iloc[:sample_size]
    
    # Train model
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_temp.fit(X_sample, y_sample)
    
    # Calculate scores
    train_pred = rf_temp.predict(X_sample)
    val_pred = rf_temp.predict(X_test)
    
    train_scores.append(r2_score(y_sample, train_pred))
    val_scores.append(r2_score(y_test, val_pred))

axes[1, 2].plot(train_sizes, train_scores, 'o-', label='Training Score', color='blue')
axes[1, 2].plot(train_sizes, val_scores, 'o-', label='Validation Score', color='red')
axes[1, 2].set_xlabel('Training Set Size')
axes[1, 2].set_ylabel('R² Score')
axes[1, 2].set_title('Learning Curve')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional Insights
print("\n15. Additional Insights...")

# Prediction intervals
print(f"Prediction range: ${y_pred_best.min():.2f} - ${y_pred_best.max():.2f}")
print(f"Actual range: ${y_test.min():.2f} - ${y_test.max():.2f}")

# Calculate prediction accuracy within different ranges
within_10k = np.abs(y_test - y_pred_best) <= 10000
within_20k = np.abs(y_test - y_pred_best) <= 20000

print(f"Predictions within $10k: {within_10k.mean():.2%}")
print(f"Predictions within $20k: {within_20k.mean():.2%}")

# Summary Report
print("\n" + "="*60)
print("RANDOM FOREST REGRESSOR - FINAL REPORT")
print("="*60)
print(f"Dataset: {df.shape[0]} job postings")
print(f"Features used: {len(X.columns)}")
print(f"Model: Random Forest Regressor")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Final R² Score: {r2_best:.4f}")
print(f"Final RMSE: ${np.sqrt(mse_best):.2f}")
print(f"Mean Absolute Error: ${mae_best:.2f}")
print("="*60)

# Save the model (optional)
import joblib
joblib.dump(rf_best, 'random_forest_salary_predictor.pkl')
joblib.dump(scaler, 'salary_scaler.pkl')
print("\nModel and scaler saved successfully!")

print("\n✅ Analysis completed successfully!")
