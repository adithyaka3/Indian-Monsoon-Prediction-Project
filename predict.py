# import torch
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
# from xgboost import XGBRegressor  # ← Import XGBoost

# def prediction(train_rain_file_tensor, test_rain_file_tensor, train_feature_data_path, test_feature_data_path):
#     # Load and convert training data
#     X_train = torch.load(train_feature_data_path).numpy()
#     y_train = torch.load(train_rain_file_tensor).squeeze().numpy()

#     # Load and convert test data
#     X_test = torch.load(test_feature_data_path).numpy()
#     y_test = torch.load(test_rain_file_tensor).squeeze().numpy()


#     # Handle potential NaNs in first row
#     if torch.isnan(torch.tensor(X_train[0])).any():
#         X_train = X_train[1:, :]
#         y_train = y_train[1:]

#     # XGBoost model
#     model = XGBRegressor(
#     n_estimators=2000,
#     learning_rate=0.05,
#     max_depth=5,
#     subsample=0.8,
#     colsample_bytree=0.9,
# )
#     model.fit(X_train, y_train)

#     # Predict and evaluate
#     y_pred = model.predict(X_test)
#     lpa = 730.5
#     # y_pred = (y_pred - lpa) / lpa * 100  
#     # y_test = (y_test - lpa) / lpa * 100  
#     #two decimal places
#     y_pred = y_pred.round(2)
#     y_test = y_test.round(2)

#     mape = mean_absolute_percentage_error(y_test, y_pred) * 100
#     mse = mean_squared_error(y_test, y_pred)

#     return mse, mape, y_pred, y_test

import torch
# import optuna <-- REMOVED
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
# cross_val_score is now handled internally by GridSearchCV
from sklearn.model_selection import GridSearchCV 
from xgboost import XGBRegressor

# ------------------------------------
# Function to tune hyperparameters using GridSearchCV
# ------------------------------------
def tune_xgb_grid_search(X_train, y_train):
    """
    Performs hyperparameter tuning for XGBoost using GridSearchCV.
    """
    # Define the grid of hyperparameters to search.
    # WARNING: The number of combinations can grow very quickly!
    # Start with a smaller grid to test.
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
    }
    
    # Calculate and print the number of fits
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"GridSearchCV will perform {n_combinations} * 5 (cv) = {n_combinations * 5} fits.")

    # Instantiate the XGBoost regressor
    model = XGBRegressor()

    # Set up GridSearchCV
    # - n_jobs=-1 uses all available CPU cores to speed up the search.
    # - verbose=2 provides progress updates during the search.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error', # Maximizing this is the same as minimizing MSE
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    # Run the grid search
    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)

    print(f"Best cross-validation score (negative MSE): {grid_search.best_score_}")
    print(f"Best parameters found: {grid_search.best_params_}")
    
    return grid_search.best_params_

# ------------------------------------
# Main Prediction Function
# ------------------------------------
def prediction(train_rain_file_tensor, test_rain_file_tensor, train_feature_data_path, test_feature_data_path, use_tuning=False):
    # Load and convert training data
    X_train = torch.load(train_feature_data_path).numpy()
    y_train = torch.load(train_rain_file_tensor).squeeze().numpy()

    # Load and convert test data
    X_test = torch.load(test_feature_data_path).numpy()
    y_test = torch.load(test_rain_file_tensor).squeeze().numpy()

    # Handle potential NaNs in first row
    if torch.isnan(torch.tensor(X_train[0])).any():
        X_train = X_train[1:, :]
        y_train = y_train[1:]

    # Tune hyperparameters if needed
    if use_tuning:
        # Call the new GridSearchCV function
        best_params = tune_xgb_grid_search(X_train, y_train)
    else:
        # Default parameters remain the same
        best_params = {
            'n_estimators': 3000,
            'learning_rate': 0.01,
            'max_depth': 8,
            'subsample': 1,
            'colsample_bytree': 0.9
        }

    # Train XGBoost model
    print("\nTraining final model with best parameters...")
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Optional: Convert to percentage deviation from LPA (commented out)
    # lpa = 730.5
    # y_pred = (y_pred - lpa) / lpa * 100
    # y_test = (y_test - lpa) / lpa * 100

    # Round predictions and true values
    y_pred = y_pred.round(2)
    y_test = y_test.round(2)

    # Metrics
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Final Metrics -> MSE: {mse:.4f}, MAPE: {mape:.4f}%")

    return mse, mape, y_pred, y_test
