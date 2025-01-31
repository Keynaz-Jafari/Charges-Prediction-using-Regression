import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # visualizing data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

#phase 1 ==================================================================================================

data_file = './DataSet.csv'
df = pd.read_csv(data_file)

print("=== Dataset Overview ===\n")
df.info()

# Check for Missing Data
missing_counts = df.isnull().sum()
print(f"\nMissing Values per Column:\n{missing_counts}")

#Box Plots
plt.figure(figsize=(10, 6))
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for idx, feature in enumerate(numeric_cols, start=1):
    plt.subplot(2, 2, idx)
    sns.boxplot(y=df[feature])
    plt.title(f"Boxplot: {feature}")
plt.tight_layout()
plt.show()

#  Histograms
df.select_dtypes(include=['float64', 'int64']).hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Distributions of Numeric Features", fontsize=14)
plt.show()

all_stats = {}
for col in numeric_cols:
    all_stats[col] = {
        "Mean": df[col].mean(),
        "Median": df[col].median(),
        "Variance": df[col].var(),
        "StdDev": df[col].std()
    }

print("\n=== Descriptive Statistics for Numerical Columns ===")
for col_name, col_stats in all_stats.items():
    print(f"\nColumn: {col_name}")
    for stat_label, stat_val in col_stats.items():
        print(f"  {stat_label}: {stat_val}")


sns.pairplot(df, vars=["age", "bmi", "children", "charges"], hue="smoker", palette="coolwarm")
plt.suptitle("Pairwise Plots with 'smoker' Highlighted", y=1.02, fontsize=15)
plt.show()


corr_mat = df.select_dtypes(include=['float64', 'int64']).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap for Numerical Features", fontsize=15)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['bmi'], bins=30, kde=True, color='purple')
plt.title("BMI Distribution", fontsize=14)
plt.show()

#phase 2 ==================================================================================================

#  Handle Missing Rows ( tu in data set ndrim missing data)
df_clean = df.dropna()


df_encoded = pd.get_dummies(df_clean, columns=["sex", "smoker", "region"], drop_first=True).astype(float)


# D. Remove Outliers via IQR
#numeric_in_encoded = df_encoded.select_dtypes(include=['float64', 'int64']).columns
#Q1 = df_encoded[numeric_in_encoded].quantile(0.20)
#Q3 = df_encoded[numeric_in_encoded].quantile(0.80)
#IQR = Q3 - Q1

#outlier_mask = (
 #       (df_encoded[numeric_in_encoded] < (Q1 - 1.5 * IQR)) |
    #    (df_encoded[numeric_in_encoded] > (Q3 + 1.5 * IQR))
#)
#df_encoded = df_encoded[~outlier_mask.any(axis=1)]

# E. Feature Scaling (MinMax)
scaler_main = MinMaxScaler()
num_features_all = df_encoded.select_dtypes(include=['float64', 'int64']).columns
num_features_ex_target = num_features_all.drop('charges', errors='ignore')  # Keep the target separate

df_encoded[num_features_ex_target] = scaler_main.fit_transform(df_encoded[num_features_ex_target])

# Scale the target (charges) separately
scaler_target = MinMaxScaler()
df_encoded['charges'] = scaler_target.fit_transform(df_encoded[['charges']])

# F. Train-Test Split
X = df_encoded.drop(["charges", "sex_male"], axis=1)  # Example: dropping 'sex_male' to avoid collinearity
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n=== Sample X_train Head ===")
print(X_train.head())


#phase 3 ==================================================================================================

def cost_function(X, y, w):

    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64).reshape(-1, 1)
    w_arr = np.array(w, dtype=np.float64)

    m = len(y_arr)
    pred = X_arr.dot(w_arr)
    residuals = pred - y_arr
    return (1 / (2 * m)) * np.sum(residuals ** 2)


def gradient_descent_optim(X, y, w_init, lr, num_iters): # teta bedast miare be soorate iterative

    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64).reshape(-1, 1)

    m = len(y_arr)
    theta = w_init.copy()
    cost_list = []
    param_path = []

    for i in range(num_iters):
        preds = X_arr.dot(theta)
        errs = preds - y_arr
        grad = (1 / m) * (X_arr.T.dot(errs))

        theta = theta - lr * grad

        current_cost = (1 / (2 * m)) * np.sum(errs ** 2)
        cost_list.append(current_cost)
        param_path.append(theta.copy())

        if np.isnan(current_cost):
            print(f"Stopping early at iteration {i} due to NaN cost.")
            break

    return theta, cost_list, param_path


def newton_1st_order(X, y, w_init, lr, num_iters):

    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64).reshape(-1, 1)

    m = len(y_arr)
    theta = w_init.copy()
    cost_list = []
    param_path = []

    for i in range(num_iters):
        preds = X_arr.dot(theta)
        diff = preds - y_arr
        grad = (1 / m) * X_arr.T.dot(diff)

        theta -= lr * grad

        current_cost = (1 / (2 * m)) * np.sum(diff ** 2)
        cost_list.append(current_cost)
        param_path.append(theta.copy())

        if np.isnan(current_cost):
            print(f"NaN encountered at iteration {i}, halting.")
            break

    return theta, cost_list, param_path


def newton_2nd_order(X, y, w_init, num_iters):

    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64).reshape(-1, 1)

    m = len(y_arr)
    theta = w_init.copy()
    cost_list = []
    param_path = []

    for i in range(num_iters):
        preds = X_arr.dot(theta)
        diff = preds - y_arr
        grad = (1 / m) * X_arr.T.dot(diff)
        hessian = (1 / m) * X_arr.T.dot(X_arr)

        # Attempt standard Newton update (may be singular if hessian is not invertible)
        theta = theta - np.linalg.inv(hessian).dot(grad)

        current_cost = (1 / (2 * m)) * np.sum(diff ** 2)
        cost_list.append(current_cost)
        param_path.append(theta.copy())

        if np.isnan(current_cost):
            print(f"NaN cost at iteration {i}, stopping.")
            break

    return theta, cost_list, param_path


# Helper function for contour plotting (for the first two features)
def contour_plot_2D(X, y, param_history, title_str):

    # Restrict to first two features for visualization (plus intercept)
    X_2d = X[:, :2]
    w_2d = np.array(param_history)[:, :2]

    w0_vals = np.linspace(-2, 2, 100)
    w1_vals = np.linspace(-2, 2, 100)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    Z = np.zeros_like(W0)

    # Evaluate cost at each point in the grid
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            test_w = np.array([[W0[i, j]], [W1[i, j]]])
            Z[i, j] = cost_function(X_2d, y, test_w)

    plt.figure(figsize=(9, 6))
    cont = plt.contourf(W0, W1, Z, levels=50, cmap='viridis')
    plt.colorbar(cont).set_label('Cost Value')

    # Plot the trajectory of the parameter updates
    plt.plot(w_2d[:, 0], w_2d[:, 1], 'r-o', label='Update Trajectory')
    plt.title(title_str)
    plt.xlabel('Weight 0')
    plt.ylabel('Weight 1')
    plt.legend()
    plt.show()


# Prepare design matrices with bias
X_train_aug = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_aug = np.c_[np.ones((X_test.shape[0], 1)), X_test]

y_train_arr = y_train.values.reshape(-1, 1)
y_test_arr = y_test.values.reshape(-1, 1)

# Initialize
np.random.seed(42)
init_weights = np.zeros((X_train_aug.shape[1], 1))

learning_rate = 0.015
num_iters = 1000

# ========================
# Training the Models
# ========================
print("\n=== Training Models ===")

# 1. Gradient Descent
theta_gd, costs_gd, path_gd = gradient_descent_optim(X_train_aug, y_train_arr, init_weights.copy(), learning_rate,
                                                     num_iters)

# 2. Newton’s 1st Order
theta_nf, costs_nf, path_nf = newton_1st_order(X_train_aug, y_train_arr, init_weights.copy(), learning_rate, num_iters)

# 3. Newton’s 2nd Order
theta_ns, costs_ns, path_ns = newton_2nd_order(X_train_aug, y_train_arr, init_weights.copy(), num_iters)

# Convert path lists to arrays
path_gd = np.array(path_gd)
path_nf = np.array(path_nf)
path_ns = np.array(path_ns)

# Plot the weight updates across epochs (3 subplots)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
# Plot GD
for i in range(path_gd.shape[1]):
    ax[0].plot(path_gd[:, i], label=f'Param {i + 1}')
ax[0].set_title("Weight Evolution: Gradient Descent")
ax[0].legend()
ax[0].grid(True)

# Plot Newton 1st
for i in range(path_nf.shape[1]):
    ax[1].plot(path_nf[:, i], label=f'Param {i + 1}')
ax[1].set_title("Weight Evolution: Newton 1st Order")
ax[1].legend()
ax[1].grid(True)

# Plot Newton 2nd
for i in range(path_ns.shape[1]):
    ax[2].plot(path_ns[:, i], label=f'Param {i + 1}')
ax[2].set_title("Weight Evolution: Newton 2nd Order")
ax[2].legend()
ax[2].grid(True)

plt.tight_layout()
plt.show()

# Contour plots for each method
contour_plot_2D(X_train_aug, y_train_arr, path_gd, "Contour Plot: Gradient Descent")
contour_plot_2D(X_train_aug, y_train_arr, path_nf, "Contour Plot: Newton 1st Order")
contour_plot_2D(X_train_aug, y_train_arr, path_ns, "Contour Plot: Newton 2nd Order")


#phase 4  ==================================================================================================
def assess_performance(y_true, y_hat):

    mse_val = mean_squared_error(y_true, y_hat)
    r2_val = r2_score(y_true, y_hat)
    return mse_val, r2_val


# Predictions for each approach
pred_gd = X_test_aug.dot(theta_gd)
pred_nf = X_test_aug.dot(theta_nf)
pred_ns = X_test_aug.dot(theta_ns)

# Baseline (predict average y)
baseline_pred = np.full_like(y_test_arr, np.mean(y_test_arr))
baseline_mse = mean_squared_error(y_test_arr, baseline_pred)
print(f"\n=== Baseline MSE ===\n{baseline_mse}")

# Evaluate MSE, R2
mse_gd, r2_gd = assess_performance(y_test_arr, pred_gd)
mse_nf, r2_nf = assess_performance(y_test_arr, pred_nf)
mse_ns, r2_ns = assess_performance(y_test_arr, pred_ns)

print(f"\n=== Model Performance Metrics ===")
print(f"Gradient Descent   -> MSE: {mse_gd}, R²: {r2_gd}")
print(f"Newton First Order -> MSE: {mse_nf}, R²: {r2_nf}")
print(f"Newton Second Order-> MSE: {mse_ns}, R²: {r2_ns}")

# Visualize cost histories
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(range(len(costs_gd)), costs_gd, color='blue')
plt.title("GD Cost vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(len(costs_nf)), costs_nf, color='red')
plt.title("Newton 1st Order Cost vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(len(costs_ns)), costs_ns, color='green')
plt.title("Newton 2nd Order Cost vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)

plt.suptitle("Cost Convergence for All Methods", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
