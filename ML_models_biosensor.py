import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor 
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.linear_model import LinearRegression



# Load data from the Excel file
file_path = 'file_path'
data = pd.read_excel(file_path)

# Data Filtering: Limiting Vg range
#data = data[(data['Vg'] >= -1) & (data['Vg'] <= -0.5)]
data = data.reset_index(drop=True)
print(data)


unique_concentrations = data['ro'].unique()
for concentration in unique_concentrations:
    concentration_data = data[data['ro'] == concentration]
    plt.plot(concentration_data['Vg'], concentration_data['Isd'], label=f'Concentration {concentration} pg/ml', marker='o')

plt.rcParams.update({'font.size': 14}) 
plt.xlabel('Vg (V)')
plt.ylabel('Isd (A)')
plt.title('Vg vs Isd Curves for Different Hormone Concentrations')
plt.yscale('log')  
plt.legend()
plt.grid(True)
plt.show()


#Feature Engineering
# Polynomial and interaction terms
poly_features = pd.DataFrame()
poly_features['Vg'] = data['Vg']
poly_features['Isd'] = data['Isd']
poly_features['Vg^2'] = data['Vg']**2
poly_features['Vg Isd'] = data['Vg'] * data['Isd']
poly_features['Isd^2'] = data['Isd']**2

# First and second-order derivatives
poly_features['dIsd/dVg'] = np.gradient(data['Isd'], data['Vg'])
poly_features['d^2Isd/dVg^2'] = np.gradient(poly_features['dIsd/dVg'], data['Vg'])

#  Area Under the Curve (AUC) using the trapezoidal rule for each concentration
unique_concentrations = data['ro'].unique()
auc_values = []

for concentration in unique_concentrations:
    concentration_data = data[data['ro'] == concentration]
    auc = np.trapz(concentration_data['Isd'], concentration_data['Vg'])
    print(f'Area under the curve for {concentration}: {auc}')
    auc_values.extend([auc] * len(concentration_data))

poly_features['AUC'] = auc_values


# Combination of all data
processed_data = pd.concat([data[['Vg', 'ro', 'Isd']], poly_features], axis=1)
processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]

print(processed_data)


# Voltage shifts calculations

processed_data_sorted = processed_data.sort_values(by=['Isd', 'Vg'])
baseline_concentration = 0
concentrations_to_compare = [30, 100, 300]


def interpolate_vg_at_isd(concentration_data, target_isd):
    return np.interp(target_isd, concentration_data['Isd'], concentration_data['Vg'])


shifts_list = []

#The maximum shift definition 
max_total_shift = 0
best_isd = None
best_shift_info = {}


for isd_value in processed_data_sorted['Isd'].unique():
    shift_info = {'Isd': isd_value}
    
    # Vg interpolation for the baseline concentration (0 pg/ml) at the current Isd
    baseline_data = processed_data_sorted[processed_data_sorted['ro'] == baseline_concentration]
    baseline_vg = interpolate_vg_at_isd(baseline_data, isd_value)
    shift_info['Baseline Vg'] = baseline_vg
    
    #Vg shift for each concentration relative to the baseline
    total_shift = 0 
    for concentration in concentrations_to_compare:
        concentration_data = processed_data_sorted[processed_data_sorted['ro'] == concentration]
        interpolated_vg = interpolate_vg_at_isd(concentration_data, isd_value)
        vg_shift = interpolated_vg - baseline_vg  # Calculate the shift
        shift_info[f'Vg shift for {concentration} pg/ml'] = vg_shift
        total_shift += abs(vg_shift) 
    
    # Check if this Isd has the largest total shift so far
    if total_shift > max_total_shift:
        max_total_shift = total_shift
        best_isd = isd_value
        best_shift_info = shift_info
    
    shifts_list.append(shift_info)

shift_results_df = pd.DataFrame(shifts_list)


merged_data = pd.merge(processed_data, shift_results_df, on='Isd', how='left')

shift_results_df = shift_results_df.drop(columns=['Baseline Vg'])

merged_data = pd.merge(processed_data, shift_results_df, on='Isd', how='left')
print(merged_data)


concentrations = concentrations_to_compare
voltage_shifts = [best_shift_info[f'Vg shift for {conc} pg/ml'] for conc in concentrations]

plt.plot(concentrations, voltage_shifts, marker='o', linestyle='-')
plt.xlabel('Concentration (pg/ml)')
plt.ylabel('Voltage Shift (V)')
plt.title(f'Voltage Shift vs. Concentration for Isd = {best_isd}')
plt.grid(True)
plt.show()

print(f'The Isd value with the largest voltage shift is: {best_isd}')
print(f'Voltage shifts for different concentrations at Isd = {best_isd}: {voltage_shifts}')


# ML
# Training and Testing Sets
X = processed_data.drop(columns=['ro'])  # Features
y = processed_data['ro']  # Target (concentration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'Linear Regression': LinearRegression(),
    'XGBoost Regressor': XGBRegressor(n_estimators=100, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(kernel='rbf'),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    'CatBoost Regressor': CatBoostRegressor(iterations=100, learning_rate=0.1, verbose=0, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Model Training and Cross-Validation
results = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'Mean Squared Error': mse,
        'R^2 Score': r2,
        'Cross-Validation R^2': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    }

    dump(model, f'{name.lower().replace(" ", "_")}_model.joblib')

results_df = pd.DataFrame(results).T
print(results_df)


plt.rcParams.update({'font.size': 14}) 

# Plotting Actual vs. Predicted concentrations for each model

for name, model in models.items():
    y_pred = model.predict(X_test)
    plt.errorbar(y_test, y_pred,  fmt='o', label=f'{name} Predictions', alpha=0.7)


min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Concentration (pg/ml)')
plt.ylabel('Predicted Concentration (pg/ml)')
plt.title('Actual vs. Predicted Concentration')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Top 4 models based on the R² score from the results DataFrame
top_models = results_df.sort_values(by='R^2 Score', ascending=False).head(4).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, name in enumerate(top_models):
    model = models[name]
    y_pred = model.predict(X_test)

    axes[i].scatter(y_test, y_pred, label=f'{name} Predictions', alpha=0.7)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
    
    axes[i].set_title(f'{name} Model')
    axes[i].set_xlabel('Actual Concentration (pg/ml)')
    axes[i].set_ylabel('Predicted Concentration (pg/ml)')
    axes[i].legend(loc='best')
    axes[i].grid(True)

plt.tight_layout()
plt.suptitle('Actual vs. Predicted Concentration for Top Models', fontsize=16, y=1.02)
plt.show()


# Feature Importance

models_with_importance = [(name, model) for name, model in models.items() if hasattr(model, 'feature_importances_')]
n_models = len(models_with_importance)

n_cols = 2 
n_rows = (n_models + n_cols - 1) // n_cols 

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
axes = axes.flatten()

for i, (name, model) in enumerate(models_with_importance):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance) 

    axes[i].barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    axes[i].set_yticks(range(len(sorted_idx)))
    axes[i].set_yticklabels(np.array(X.columns)[sorted_idx])
    axes[i].set_xlabel('Importance Score')
    axes[i].set_title(f'Feature Importance for {name}')
    axes[i].grid(True)


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Feature Importance for Different Models', fontsize=16, y=1.02)
plt.show()

#Predictions comparation
predictions_dict = {}

for name, model in models.items():
    y_pred = model.predict(X_test)

    predictions_dict[name] = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

# Display the predictions for each model
for name, df in predictions_dict.items():
    print(f"\n{name} Model Predictions:\n")
    print(df.head()) 


# The model with the best R^2 score
best_model_name = results_df['R^2 Score'].idxmax()
print(f"The best performing model is: {best_model_name}")

best_model = models[best_model_name]

# Dictionary to store the relevant results for regression metrics
performance_metrics = {
    'Model': [],
    'MSE': [],
    'RMSE': [],  
    'R² Score': []
}

# Regression metrics
for name, model in models.items():
    # Predictions on the test set
    y_pred = model.predict(X_test)
 
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  
    r2 = r2_score(y_test, y_pred)
    
    performance_metrics['Model'].append(name)
    performance_metrics['MSE'].append(mse)
    performance_metrics['RMSE'].append(rmse)  
    performance_metrics['R² Score'].append(r2)


performance_df = pd.DataFrame(performance_metrics)
print(performance_df)

acceptable_models_df = performance_df[performance_df['RMSE'] <= 10]

# Plot RMSE and R² Score for models with RMSE <= 10
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(acceptable_models_df['Model'], acceptable_models_df['RMSE'], color='skyblue', label='RMSE')
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE (pg/ml)')
ax1.set_title('RMSE Comparison for Models with RMSE ≤ 10 pg/ml')

ax1.grid(True)
plt.tight_layout()
plt.show()

# Print acceptable models
print("Models with RMSE ≤ 10 pg/ml:")
print(acceptable_models_df[['Model', 'RMSE', 'R² Score']])