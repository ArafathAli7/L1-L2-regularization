import gradio as gr
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io

def fig_to_numpy(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = plt.imread(buf, format="png")
    plt.close(fig)
    return img

def generate_data(n_samples, n_features, noise, effective_rank, random_state):
    X, y, coef = make_regression(n_samples=n_samples,
                                 n_features=n_features,
                                 noise=noise,
                                 effective_rank=effective_rank,
                                 coef=True,
                                 random_state=random_state)
    return X, y, coef

def compute_paths(X, y, alphas):
    lasso_coefs = []
    ridge_coefs = []
    for a in alphas:
        l = Lasso(alpha=a, max_iter=10000)
        r = Ridge(alpha=a, max_iter=10000)
        l.fit(X, y)
        r.fit(X, y)
        lasso_coefs.append(l.coef_.copy())
        ridge_coefs.append(r.coef_.copy())
    return np.array(lasso_coefs), np.array(ridge_coefs)

def plot_coefficient_paths(alphas, coefs, title):
    fig, ax = plt.subplots(figsize=(8,4))
    for j in range(coefs.shape[1]):
        ax.plot(alphas, coefs[:, j])
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('alpha')
    ax.set_ylabel('coefficient')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    return fig

def plot_bar_coeffs(names, coefs, title):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(coefs)), coefs)
    ax.set_xticks(range(len(coefs)))
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    ax.set_title(title)
    ax.axhline(0, color='k', linewidth=0.6)
    plt.tight_layout()
    return fig

def run_demo(n_samples=200, n_features=10, noise=10.0, effective_rank=10, standardize=True,
             alpha=1.0, n_alphas=30, random_state=42):
    X, y, true_coef = generate_data(n_samples, n_features, noise, effective_rank, random_state)
    feature_names = [f'feat_{i}' for i in range(n_features)]
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    lr = LinearRegression()
    ridge = Ridge(alpha=alpha, max_iter=10000)
    lasso = Lasso(alpha=alpha, max_iter=10000)

    lr.fit(X, y)
    ridge.fit(X, y)
    lasso.fit(X, y)

    lr_coef = lr.coef_
    ridge_coef = ridge.coef_
    lasso_coef = lasso.coef_

    lr_mse = mean_squared_error(y, lr.predict(X))
    ridge_mse = mean_squared_error(y, ridge.predict(X))
    lasso_mse = mean_squared_error(y, lasso.predict(X))

    alphas = np.logspace(-4, 2, n_alphas)
    lasso_paths, ridge_paths = compute_paths(X, y, alphas)

    img1 = fig_to_numpy(plot_coefficient_paths(alphas, lasso_paths, "Lasso Paths (L1)"))
    img2 = fig_to_numpy(plot_coefficient_paths(alphas, ridge_paths, "Ridge Paths (L2)"))
    img3 = fig_to_numpy(plot_bar_coeffs(feature_names, lr_coef, "OLS Coefficients"))
    img4 = fig_to_numpy(plot_bar_coeffs(feature_names, ridge_coef, f"Ridge Coefficients α={alpha}"))
    img5 = fig_to_numpy(plot_bar_coeffs(feature_names, lasso_coef, f"Lasso Coefficients α={alpha}"))

    summary = pd.DataFrame({
        "Method": ["OLS", "Ridge", "Lasso"],
        "Alpha": [0.0, alpha, alpha],
        "MSE": [lr_mse, ridge_mse, lasso_mse]
    })

    coef_table = pd.DataFrame({
        "Feature": feature_names,
        "TrueCoef": true_coef,
        "OLS": lr_coef,
        "Ridge": ridge_coef,
        "Lasso": lasso_coef
    })

    return summary, coef_table, img1, img2, img3, img4, img5

with gr.Blocks() as demo:
    gr.Markdown("# L1 vs L2 Regularization Demo")

    with gr.Row():
        with gr.Column(scale=1):
            n_samples = gr.Slider(50, 2000, value=200, step=50, label="Samples")
            n_features = gr.Slider(2, 40, value=10, step=1, label="Features")
            noise = gr.Slider(0.0, 50.0, value=10.0, step=0.5, label="Noise")
            effective_rank = gr.Slider(1, 40, value=10, step=1, label="Effective Rank")
            standardize = gr.Checkbox(value=True, label="Standardize")
            alpha = gr.Slider(0.0, 100.0, value=1.0, step=0.1, label="Alpha")
            n_alphas = gr.Slider(10, 80, value=30, step=1, label="Alphas for Path Plots")
            random_state = gr.Number(value=42, label="Random State")
            run_button = gr.Button("Run Demo")

        with gr.Column(scale=2):
            out_summary = gr.Dataframe(interactive=False)
            out_table = gr.Dataframe(interactive=False)
            out_img1 = gr.Image(type="numpy")
            out_img2 = gr.Image(type="numpy")
            out_img3 = gr.Image(type="numpy")
            out_img4 = gr.Image(type="numpy")
            out_img5 = gr.Image(type="numpy")

    run_button.click(
        fn=run_demo,
        inputs=[n_samples, n_features, noise, effective_rank, standardize, alpha, n_alphas, random_state],
        outputs=[out_summary, out_table, out_img1, out_img2, out_img3, out_img4, out_img5]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
