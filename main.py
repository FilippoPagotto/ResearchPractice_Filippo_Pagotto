import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Load the dataset
csv_path = r'E:\PyCharm\pythonProject4\dft_traffic_counts_raw_counts3.csv'
data = pd.read_csv(csv_path)

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Label Encoding for categorical columns
label_encoder = LabelEncoder()
for column in ['Direction_of_travel', 'Region_id', 'Local_authority_id', 'Road_category', 'Road_type',
               'Road_name', 'Start_junction_road_name', 'End_junction_road_name']:
    data[column] = label_encoder.fit_transform(data[column])

# Extract features and target variable
X = data.drop(['All_motor_vehicles'], axis=1)  # Features (excluding the target variable)
y = data['All_motor_vehicles']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using correlation matrix
print("Performing feature selection using correlation matrix...")
selected_features_corr = SelectKBest(f_regression, k=8).fit(X_train, y_train).get_support()
selected_columns_corr = X_train.columns[selected_features_corr]
X_train_selected_corr = X_train[selected_columns_corr]
X_test_selected_corr = X_test[selected_columns_corr]

# Feature selection using ANOVA
print("Performing feature selection using ANOVA...")
selected_features_anova = SelectKBest(f_classif, k=8).fit(X_train, y_train).get_support()
selected_columns_anova = X_train.columns[selected_features_anova]
X_train_selected_anova = X_train[selected_columns_anova]
X_test_selected_anova = X_test[selected_columns_anova]

# Create and train the Neural Network model using correlation-based features
print("Training Neural Network model with correlation-based features...")
model_corr = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_selected_corr.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_corr.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model_corr.fit(X_train_selected_corr, y_train, validation_data=(X_test_selected_corr, y_test),
               epochs=50, batch_size=32, callbacks=[early_stopping])

# Create and train the Neural Network model using ANOVA-based features
print("Training Neural Network model with ANOVA-based features...")
model_anova = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_selected_anova.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_anova.compile(optimizer='adam', loss='mean_squared_error')

model_anova.fit(X_train_selected_anova, y_train, validation_data=(X_test_selected_anova, y_test),
                epochs=50, batch_size=32, callbacks=[early_stopping])

# Make predictions on the test set for correlation-based features
print("Making predictions on the test set with correlation-based features...")
y_pred_corr = model_corr.predict(X_test_selected_corr).flatten()

# Make predictions on the test set for ANOVA-based features
print("Making predictions on the test set with ANOVA-based features...")
y_pred_anova = model_anova.predict(X_test_selected_anova).flatten()

# Calculate RMSE and R2 for correlation-based features
rmse_corr = np.sqrt(mean_squared_error(y_test, y_pred_corr))
r2_corr = r2_score(y_test, y_pred_corr)

# Calculate RMSE and R2 for ANOVA-based features
rmse_anova = np.sqrt(mean_squared_error(y_test, y_pred_anova))
r2_anova = r2_score(y_test, y_pred_anova)

# Display results for correlation-based features
print("\nResults for Correlation-based Feature Selection with Neural Network:")
print(f"Selected Features: {selected_columns_corr}")
print(f"Root Mean Squared Error (RMSE): {rmse_corr}")
print(f"R-squared (R2): {r2_corr}")

# Display results for ANOVA-based features
print("\nResults for ANOVA-based Feature Selection with Neural Network:")
print(f"Selected Features: {selected_columns_anova}")
print(f"Root Mean Squared Error (RMSE): {rmse_anova}")
print(f"R-squared (R2): {r2_anova}")

# Scatter plot comparing predicted vs actual values for correlation-based features
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred_corr, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter plot for Actual vs Predicted Values - Correlation-based")
plt.show()

# Scatter plot comparing predicted vs actual values for ANOVA-based features
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred_anova, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter plot for Actual vs Predicted Values - ANOVA-based")
plt.show()
