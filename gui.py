import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# Set Streamlit page config
# -----------------------
st.set_page_config(page_title="Insurance Charges Prediction", layout="wide")

# ---------------------------------------------------------
# Phase 3 functions: cost function, GD, Newton methods, etc.
# ---------------------------------------------------------
def cost_function(X, y, w):
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64).reshape(-1, 1)
    w_arr = np.array(w, dtype=np.float64)

    m = len(y_arr)
    pred = X_arr.dot(w_arr)
    residuals = pred - y_arr
    return (1 / (2 * m)) * np.sum(residuals ** 2)

def gradient_descent_optim(X, y, w_init, lr, num_iters):
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

        # Newton update
        # May fail if hessian is singular
        try:
            theta = theta - np.linalg.inv(hessian).dot(grad)
        except np.linalg.LinAlgError:
            st.warning("Hessian is singular; cannot invert. Stopping early.")
            break

        current_cost = (1 / (2 * m)) * np.sum(diff ** 2)
        cost_list.append(current_cost)
        param_path.append(theta.copy())

        if np.isnan(current_cost):
            break

    return theta, cost_list, param_path

def contour_plot_2D(X, y, param_history, title_str):
    """
    Creates contour lines of the cost function in the plane of the first two weights.
    Plots the path from the parameter history.
    """
    fig = plt.figure(figsize=(6, 4))

    # Restrict to first two features for visualization (plus intercept).
    # X[:, :2] => intercept column + first real feature
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

    cont = plt.contourf(W0, W1, Z, levels=50, cmap='viridis')
    plt.colorbar(cont).set_label('Cost Value')
    plt.plot(w_2d[:, 0], w_2d[:, 1], 'r-o', label='Update Trajectory')
    plt.title(title_str)
    plt.xlabel('Weight 0')
    plt.ylabel('Weight 1')
    plt.legend()

    st.pyplot(fig)


def assess_performance(y_true, y_hat):
    mse_val = mean_squared_error(y_true, y_hat)
    r2_val = r2_score(y_true, y_hat)
    return mse_val, r2_val

# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.title("Charges Prediction Analysis")

    st.markdown(
        """
        This app performs an exploratory data analysis on the dataset
        and trains a **Linear Regression** model using **Gradient Descent** and **Newton’s Method** variants.
        """
    )

    # ---------------------------------------------
    # Phase 1: File Upload and Data Exploration
    # ---------------------------------------------
    st.header("1. Upload and Explore Data")

    uploaded_file = st.file_uploader("Upload the CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Overview")
        st.dataframe(df)    # This creates a scrollable table in the Streamlit app


        with st.expander("Data Info"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()  # get the text
            st.text(info_str)
        missing_counts = df.isnull().sum()
        st.write("**Missing Values per Column:**")
        st.write(missing_counts)

        # Check numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # Box plots
        if st.checkbox("Show Box Plots"):
            fig_box, axes_box = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 4))
            if len(numeric_cols) == 1:
                axes_box = [axes_box]  # make it iterable
            for ax, feature in zip(axes_box, numeric_cols):
                sns.boxplot(y=df[feature], ax=ax)
                ax.set_title(f"Boxplot: {feature}")
            st.pyplot(fig_box)

        if st.checkbox("Show Histograms"):
            numeric_data = df.select_dtypes(include=['float64', 'int64'])
            fig_hist, ax_hist = plt.subplots(figsize=(12, 8))
            numeric_data.hist(ax=ax_hist, bins=20, edgecolor='black')
            st.pyplot(fig_hist)

        # Descriptive stats
        if st.checkbox("Show Descriptive Statistics"):
            all_stats = {}
            for col in numeric_cols:
                all_stats[col] = {
                    "Mean": df[col].mean(),
                    "Median": df[col].median(),
                    "Variance": df[col].var(),
                    "StdDev": df[col].std()
                }
            st.write(pd.DataFrame(all_stats).T)

        # Pairplot
        if st.checkbox("Show Pairplot (age, bmi, children, charges) by smoker"):
            st.info("This might take a moment to render.")
            fig_pair = sns.pairplot(df, vars=["age", "bmi", "children", "charges"], hue="smoker", palette="coolwarm")
            st.pyplot(fig_pair)

        # Correlation heatmap
        if st.checkbox("Show Correlation Heatmap"):
            corr_mat = df.select_dtypes(include=['float64', 'int64']).corr()
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            ax_corr.set_title("Correlation Heatmap for Numerical Features", fontsize=15)
            st.pyplot(fig_corr)

        # Additional BMI distribution
        if "bmi" in df.columns and st.checkbox("Show BMI Distribution"):
            fig_bmi = plt.figure(figsize=(8, 5))
            sns.histplot(df['bmi'], bins=30, kde=True, color='purple')
            plt.title("BMI Distribution", fontsize=14)
            st.pyplot(fig_bmi)

        # ---------------------------------------------
        # Phase 2: Data Cleaning / Encoding / Splitting
        # ---------------------------------------------
        st.header("2. Data Preprocessing")
        st.markdown("We drop rows with missing data and perform one-hot encoding for categorical columns.")

        df_clean = df.dropna()
        df_encoded = pd.get_dummies(df_clean, columns=["sex", "smoker", "region"], drop_first=True).astype(float)

        # Additional Features
        if "age" in df_encoded.columns and "bmi" in df_encoded.columns:
            df_encoded["age_bmi_interact"] = df_encoded["age"] * df_encoded["bmi"]
        if "smoker_yes" in df_encoded.columns and "bmi" in df_encoded.columns:
            df_encoded["smoker_bmi_interact"] = df_encoded.get("smoker_yes", 0) * df_encoded["bmi"]
        if "children" in df_encoded.columns and "age" in df_encoded.columns:
            df_encoded["child_age_interact"] = df_encoded["children"] * df_encoded["age"]

        # Feature Scaling
        numeric_features = df_encoded.select_dtypes(include=['float64', 'int64']).columns
        num_features_ex_target = numeric_features.drop('charges', errors='ignore')

        scaler_main = MinMaxScaler()
        df_encoded[num_features_ex_target] = scaler_main.fit_transform(df_encoded[num_features_ex_target])

        scaler_target = MinMaxScaler()
        if 'charges' in df_encoded.columns:
            df_encoded['charges'] = scaler_target.fit_transform(df_encoded[['charges']])

        # Train-Test Split
        if 'charges' in df_encoded.columns:
            # Drop 'sex_male' as in your original example if it exists
            features_to_drop = ["charges", "sex_male"] if "sex_male" in df_encoded.columns else ["charges"]
            X = df_encoded.drop(features_to_drop, axis=1, errors='ignore')
            y = df_encoded["charges"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            st.write("**X_train shape**:", X_train.shape)
            st.write("**y_train shape**:", y_train.shape)

            # Show sample
            st.markdown("**Sample X_train Head**")
            st.write(X_train.head())

            # Allow user to train models
            st.header("3. Model Training and Evaluation")

            if st.button("Train Models (GD, Newton 1st, Newton 2nd)"):
                # Prepare design matrices
                X_train_aug = np.c_[np.ones((X_train.shape[0], 1)), X_train]
                X_test_aug = np.c_[np.ones((X_test.shape[0], 1)), X_test]

                y_train_arr = y_train.values.reshape(-1, 1)
                y_test_arr = y_test.values.reshape(-1, 1)

                init_weights = np.zeros((X_train_aug.shape[1], 1))
                learning_rate = 0.015
                num_iters = 1000

                # Train each model
                theta_gd, costs_gd, path_gd = gradient_descent_optim(
                    X_train_aug, y_train_arr, init_weights.copy(), learning_rate, num_iters
                )
                theta_nf, costs_nf, path_nf = newton_1st_order(
                    X_train_aug, y_train_arr, init_weights.copy(), learning_rate, num_iters
                )
                theta_ns, costs_ns, path_ns = newton_2nd_order(
                    X_train_aug, y_train_arr, init_weights.copy(), num_iters
                )

                # Convert paths
                path_gd = np.array(path_gd)
                path_nf = np.array(path_nf)
                path_ns = np.array(path_ns)

                # Plot param evolution
                st.subheader("Parameter Evolution")
                fig, ax = plt.subplots(1, 3, figsize=(18, 4))

                # 1) GD
                for i in range(path_gd.shape[1]):
                    ax[0].plot(path_gd[:, i], label=f'Param {i}')
                ax[0].set_title("Gradient Descent")
                ax[0].legend()
                ax[0].grid(True)

                # 2) Newton 1st
                for i in range(path_nf.shape[1]):
                    ax[1].plot(path_nf[:, i], label=f'Param {i}')
                ax[1].set_title("Newton 1st Order")
                ax[1].legend()
                ax[1].grid(True)

                # 3) Newton 2nd
                for i in range(path_ns.shape[1]):
                    ax[2].plot(path_ns[:, i], label=f'Param {i}')
                ax[2].set_title("Newton 2nd Order")
                ax[2].legend()
                ax[2].grid(True)

                st.pyplot(fig)

                # Contour Plots (2D) - uses only first two weights
                st.subheader("2D Contour Plots")
                st.info("Only plotting w0 and w1 (intercept and first feature).")
                contour_plot_2D(X_train_aug, y_train_arr, path_gd, "GD Contour")
                contour_plot_2D(X_train_aug, y_train_arr, path_nf, "Newton 1st Order Contour")
                contour_plot_2D(X_train_aug, y_train_arr, path_ns, "Newton 2nd Order Contour")

                # -----------------------
                # Phase 4: Performance
                # -----------------------
                st.subheader("Model Performance")

                pred_gd = X_test_aug.dot(theta_gd)
                pred_nf = X_test_aug.dot(theta_nf)
                pred_ns = X_test_aug.dot(theta_ns)

                # Baseline
                baseline_pred = np.full_like(y_test_arr, np.mean(y_test_arr))
                baseline_mse = mean_squared_error(y_test_arr, baseline_pred)

                st.write(f"**Baseline MSE (predict mean)**: {baseline_mse:.4f}")

                # Evaluate
                mse_gd, r2_gd = assess_performance(y_test_arr, pred_gd)
                mse_nf, r2_nf = assess_performance(y_test_arr, pred_nf)
                mse_ns, r2_ns = assess_performance(y_test_arr, pred_ns)

                st.write(f"**Gradient Descent** -> MSE: {mse_gd:.4f}, R2: {r2_gd:.4f}")
                st.write(f"**Newton 1st Order** -> MSE: {mse_nf:.4f}, R2: {r2_nf:.4f}")
                st.write(f"**Newton 2nd Order** -> MSE: {mse_ns:.4f}, R2: {r2_ns:.4f}")

                # Cost Convergence
                st.subheader("Cost Convergence Plots")
                fig_cost, ax_cost = plt.subplots(1, 3, figsize=(18, 5))

                # GD
                ax_cost[0].plot(range(len(costs_gd)), costs_gd, color='blue')
                ax_cost[0].set_title("GD Cost vs. Iterations")
                ax_cost[0].set_xlabel("Iterations")
                ax_cost[0].set_ylabel("Cost")
                ax_cost[0].grid(True)

                # Newton 1st
                ax_cost[1].plot(range(len(costs_nf)), costs_nf, color='red')
                ax_cost[1].set_title("Newton 1st Order Cost vs. Iterations")
                ax_cost[1].set_xlabel("Iterations")
                ax_cost[1].set_ylabel("Cost")
                ax_cost[1].grid(True)

                # Newton 2nd
                ax_cost[2].plot(range(len(costs_ns)), costs_ns, color='green')
                ax_cost[2].set_title("Newton 2nd Order Cost vs. Iterations")
                ax_cost[2].set_xlabel("Iterations")
                ax_cost[2].set_ylabel("Cost")
                ax_cost[2].grid(True)

                fig_cost.suptitle("Cost Convergence for All Methods", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                st.pyplot(fig_cost)

        else:
            st.error("No 'charges' column found in the dataset. Cannot proceed with training.")

    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
