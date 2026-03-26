import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,accuracy_score,precision_score
from data_preprocessing import load_and_preprocess

# Set MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Churn Prediction")

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess("data/churn.csv")

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# MLflow logging
with mlflow.start_run():

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)

    mlflow.sklearn.log_model(
        best_model,
        name="churn_model",
        registered_model_name="ChurnModel"
    )

    print("Best Params:", grid.best_params_)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)
    print("Precision:", precision)