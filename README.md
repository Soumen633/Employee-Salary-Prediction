🎯 Random Forest Regressor for Data Scientist Salary Prediction
This project implements a Random Forest Regressor to predict data scientist salaries using a dataset of job postings. It includes data cleaning, feature engineering, hyperparameter tuning, model evaluation, and rich visualizations.

📁 Files
eda_data.csv: Dataset of data science job postings and features

rf_salary_predictor.py: Main script for preprocessing, training, and visualizations

random_forest_salary_predictor.pkl: Trained Random Forest model

salary_scaler.pkl: Feature scaler for new predictions

🛠️ Requirements
Python 3.x

pandas

numpy

scikit-learn

seaborn

matplotlib

joblib

Install dependencies:

bash
Copy
Edit
pip install pandas numpy scikit-learn seaborn matplotlib joblib
🚀 Usage
1. Prepare the Data
Place eda_data.csv in your working directory.

2. Run the Script
bash
Copy
Edit
python rf_salary_predictor.py
This will:

Load and preprocess the data

Handle missing values

Engineer new features (e.g., salary range, skill count)

Train and tune a Random Forest Regressor

Evaluate the model (MAE, RMSE, R², etc.)

Visualize results

Save the trained model and scaler

📈 Outputs
Console summary of steps, metrics, and key findings

Visualizations:

Actual vs Predicted Salary (scatter plot)

Residual Error Plot

Feature Importance Plot

Salary Distribution (Actual vs Predicted)

Model Comparison and Learning Curve

Saved Files:

random_forest_salary_predictor.pkl

salary_scaler.pkl

🧠 What the Model Does
Predicts average salary for data science roles based on job attributes

Identifies top salary influencers via feature importance

Evaluates performance with cross-validation and regression metrics

Provides insights into how skills like Python, R, Spark, AWS impact salary

📝 Notes
Code is fully commented and modular for easy editing

Easily adaptable to other job roles or datasets

Built with reproducibility and deployment in mind

💡 Example Results
Best Parameters: Found via GridSearchCV

R² Score: Shows model fit

Top Features: Shown in feature importance plot

Salary Predictions: Visualized against actual values

📦 Model Serving (Optional)
Use the exported .pkl model and scaler in deployment scripts or web apps to predict salaries on new job listings.

🤝 Contributions
Feel free to fork, open issues, or submit PRs. Contributions are welcome!
