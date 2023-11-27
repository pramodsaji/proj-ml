from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 
def encodeLabels(data, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col]) + 1
    return data

# Function to evaluate the model
def evaluate(model, predictions, X_test, y_test, isSequential = False):
    # Evaluate the model
    if(isSequential):
        score = model.evaluate(X_test, y_test)
    else:
        score = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model Performance")
    print("Score:", score)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2):", r2)

# Function to plot the predictions
def plot_predictions(model, data, selected_date, selected_menu_item, poly = None, isXReshaped = False):
    delta_data = data[(data['DateTime'].dt.date == pd.to_datetime(selected_date).date()) & (data['MenuItem'] == selected_menu_item)][3:]
    
    X_d = delta_data[['NumericDate', 'TimeOfDay', 'MenuItem']]
    y_d = delta_data['Quantity']
    if(poly != None):
        X_d = poly.transform(X_d)

    if(isXReshaped):
        X_d = np.reshape(X_d, (X_d.shape[0], 1, X_d.shape[1]))
    # Predicting for the specific menu item using the already trained model
    predictions_d = model.predict(X_d)
    
    plt.figure(figsize=(10, 6))
    plt.plot(delta_data['DateTime'].dt.strftime('%H:%M:%S'), y_d, marker='o', linestyle='-', label=selected_menu_item)
    plt.plot(delta_data['DateTime'].dt.strftime('%H:%M:%S'), predictions_d, marker='o', linestyle='-', label=selected_menu_item)
    plt.xlabel('Date and Time')
    plt.ylabel('MenuItem')
    plt.title(f'MenuItem "{selected_menu_item}" over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()