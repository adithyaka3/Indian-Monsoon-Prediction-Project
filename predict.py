import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def prediction(train_rain_file_tensor, test_rain_file_tensor, train_feature_data_path, test_feature_data_path):

    X_train = torch.load(train_feature_data_path)  # Load the training feature data tensor
    X_train = X_train.numpy()  # Convert to NumPy array
    y_train = torch.load(train_rain_file_tensor)  # Load the rainfall data tensor
    y_train = y_train.squeeze().numpy()  # Convert to NumPy array

    X_test = torch.load(test_feature_data_path)  # Load the test feature data tensor
    X_test = X_test.numpy()  # Convert to NumPy array
    y_test = torch.load(test_rain_file_tensor)  # Load the rainfall data tensor for
    y_test = y_test.squeeze().numpy()  # Convert to NumPy array

    print("Training data shape:", X_train.shape, y_train.shape)
    print("Test data shape:", X_test.shape, y_test.shape)
    for i in range(X_train.shape[1]):
        if X_train[0][i] == float('nan'):
            X_train = X_train[1:, :]
            y_train = y_train[1:]
            break

    model = RandomForestRegressor(
        n_estimators=50,   # number of trees in the forest
        max_depth=50,       # depth of each tree
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(" First 10 predictions:", y_pred[:10])
    print(" First 10 actual values:", y_test[:10])
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to percentage
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Mean Squared Error (MSE): {mse:.2f}")


