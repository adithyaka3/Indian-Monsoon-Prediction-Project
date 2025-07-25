import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def prediction(rain_file_tensor, feature_data_path):

    X = torch.load(feature_data_path)  # Load the feature data tensor
    X = X.numpy()  # Convert to NumPy array
    y = torch.load(rain_file_tensor)  # Load the rainfall data tensor
    y = y.squeeze().numpy()  # Convert to NumPy array


    # Split data for validation (optional)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and fit

    model = RandomForestRegressor(
        n_estimators=100,   # number of trees in the forest
        max_depth=10,       # depth of each tree
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred) * 100  # Convert to percentage
    mse = mean_squared_error(y_val, y_pred)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Mean Squared Error (MSE): {mse:.2f}")


