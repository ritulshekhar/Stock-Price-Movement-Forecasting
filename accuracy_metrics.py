from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def accuracy_predictions(y_true, y_pred):
    """
    Evaluate regression model predictions with common metrics.
    
    Parameters:
    - y_true: true values (numpy array or list)
    - y_pred: predicted values (numpy array or list)
    
    Prints RMSE, MAE, and R^2 score.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("Evaluation Metrics:")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - MAE:  {mae:.4f}")
    print(f" - R^2:  {r2:.4f}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2}
