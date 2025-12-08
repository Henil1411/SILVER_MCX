# app.py - Streamlit app for streaming-style trend + magnitude predictions
# Usage:
#   pip install streamlit pandas numpy scikit-learn matplotlib joblib xgboost
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import os, re, joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

DEFAULT_PATHS = [
    "/mnt/data/streaming_model_results/streaming_sim_results.csv",
    "/mnt/data/MCX_SILVER.csv",
    "/content/MCX_SILVER.csv"
]

st.set_page_config(layout="wide", page_title="Streaming Trend Predictor")

st.title("Streaming Trend Predictor — Direction & Magnitude")

# ---------------------------
# Utilities
# ---------------------------
def safe_to_numeric(s):
    if isinstance(s, str):
        s2 = re.sub(r"[^\d\.\-eE]", "", s)
        try:
            return float(s2)
        except:
            return np.nan
    return s

def kalman_smooth(signal, q=1e-5, r=1e-2):
    n = len(signal)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = signal[0]
    P[0] = 1.0
    for k in range(1, n):
        xhat_minus = xhat[k-1]
        P_minus = P[k-1] + q
        K = P_minus / (P_minus + r + 1e-12)
        xhat[k] = xhat_minus + K * (signal[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return xhat

def default_load_csv():
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, p
            except:
                pass
    return None, None

def prepare_features(df):
    # Normalize column names
    df = df.copy()
    df.columns = [re.sub(r"[^0-9A-Za-z]+", "_", str(c)).lower() for c in df.columns]

    date_col = next((c for c in df.columns if "date" in c), df.columns[0])
    close_col = next((c for c in df.columns if "close" in c or "ltp" in c or "settle" in c), None)
    if close_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        close_col = numeric_cols[0] if numeric_cols else df.columns[1]

    df = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Close"] = df["Close"].apply(safe_to_numeric)
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    # smoothing + features
    df["close_kf"] = kalman_smooth(df["Close"].values, q=1e-5, r=1e-2)
    df["ret_1"] = df["close_kf"].pct_change()
    df["logret"] = np.log(df["close_kf"]).diff()

    for w in (5, 10, 20, 50):
        df[f"sma_{w}"] = df["close_kf"].rolling(w).mean()
        df[f"ema_{w}"] = df["close_kf"].ewm(span=w, adjust=False).mean()

    df["roc_5"] = df["close_kf"].pct_change(5)

    for w in (5, 20, 60):
        df[f"vol_{w}"] = df["ret_1"].rolling(w).std()

    # RSI-ish
    delta = df["close_kf"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(com=13).mean()
    roll_down = down.ewm(com=13).mean()
    df["rsi_14"] = 100 - 100 / (1 + roll_up / (roll_down + 1e-12))

    for lag in (1, 2, 3, 5, 10):
        df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)
        df[f"close_lag_{lag}"] = df["close_kf"].shift(lag)

    df = df.dropna().reset_index(drop=True)

    # targets
    df["target_ret_1"] = df["close_kf"].shift(-1) / df["close_kf"] - 1.0
    df["target_dir_1"] = (df["target_ret_1"] > 0).astype(int)
    df = df.dropna(subset=["target_ret_1", "target_dir_1"]).reset_index(drop=True)
    return df

# ---------------------------
# Sidebar - file + options
# ---------------------------
with st.sidebar:
    st.header("Input & Options")
    uploaded = st.file_uploader("Upload MCX CSV (optional)", type=["csv"])
    use_default = st.checkbox("Use default path if available (/mnt/data/...)", value=True)
    seq_len = st.slider("Sequence length for LSTM (if used)", min_value=10, max_value=120, value=30)
    train_frac = st.slider("Training fraction (initial batch)", min_value=0.5, max_value=0.9, value=0.7)
    use_lstm = st.checkbox("Attempt LSTM baseline (requires tensorflow)", value=False)
    run_button = st.button("Run streaming sim")

# ---------------------------
# Load data
# ---------------------------
df_raw = None
source = None

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        source = "uploaded file"
    except Exception as e:
        st.error("Failed to read uploaded file: " + str(e))
elif use_default:
    df_raw, source = default_load_csv()
    if df_raw is None:
        st.info("No default CSV found; please upload a CSV.")
else:
    st.info("Upload a CSV or enable default path.")

if df_raw is not None:
    st.success(f"Loaded data from: {source}")
    st.write("Preview:")
    st.dataframe(df_raw.head(6))

# ---------------------------
# Main pipeline + streaming simulation
# ---------------------------
if df_raw is not None and run_button:
    with st.spinner("Preparing features and running streaming simulation..."):
        df = prepare_features(df_raw)
        st.write(
            f"Prepared features. Date range: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}"
        )
        st.write("Sample after feature prep:")
        st.dataframe(df.head(5))

        # feature matrix
        exclude = ["Date", "Close", "close_kf", "target_ret_1", "target_dir_1"]
        feature_cols = [
            c
            for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[feature_cols].values
        y_reg = df["target_ret_1"].values
        y_clf = df["target_dir_1"].values
        dates = df["Date"].values

        n = len(X)
        train_n = int(n * train_frac)
        X_train, X_test = X[:train_n], X[train_n:]
        y_reg_train, y_reg_test = y_reg[:train_n], y_reg[train_n:]
        y_clf_train, y_clf_test = y_clf[:train_n], y_clf[train_n:]
        dates_train, dates_test = dates[:train_n], dates[train_n:]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Attempt to load pre-saved models (optional)
        saved_dir_candidates = [
            "/mnt/data/streaming_model_results",
            "/content/streaming_model_results",
        ]
        saved_loaded = False
        model_info = {"online_reg": None, "online_clf": None, "scaler": None, "lstm": None}
        for d in saved_dir_candidates:
            try:
                if os.path.exists(os.path.join(d, "scaler.joblib")):
                    model_info["scaler"] = joblib.load(os.path.join(d, "scaler.joblib"))
                if os.path.exists(os.path.join(d, "lstm_return_model.keras")) and use_lstm:
                    import tensorflow as tf
                    model_info["lstm"] = tf.keras.models.load_model(
                        os.path.join(d, "lstm_return_model.keras")
                    )
                saved_loaded = True
            except Exception:
                pass

        # Build online models and initialize
        online_reg = SGDRegressor(max_iter=1000, tol=1e-3)
        online_clf = SGDClassifier(max_iter=1000, tol=1e-3)
        init_n = min(50, len(X_train_s))
        if init_n <= 2:
            st.error(
                "Not enough training rows to initialize online model. Increase data or reduce train fraction."
            )
        online_reg.partial_fit(X_train_s[:init_n], y_reg_train[:init_n])
        online_clf.partial_fit(
            X_train_s[:init_n], y_clf_train[:init_n], classes=np.array([0, 1])
        )

        # If user requests LSTM baseline, try to train (may take time)
        lstm_model = None
        if use_lstm:
            try:
                import tensorflow as tf
                from tensorflow.keras import layers, models, callbacks

                tf.random.set_seed(42)

                def make_sequences(Xa, seq_len):
                    Xs = []
                    for i in range(seq_len, len(Xa)):
                        Xs.append(Xa[i - seq_len : i])
                    return np.array(Xs)

                scaled_all = scaler.transform(X)
                Xs_all = make_sequences(scaled_all, seq_len)
                yseq_all = y_reg[seq_len:]
                seq_train_n = int(len(Xs_all) * 0.7)
                Xs_train = Xs_all[:seq_train_n]
                yseq_train = yseq_all[:seq_train_n]

                inp = layers.Input(shape=(seq_len, Xs_train.shape[2]))
                x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(inp)
                x = layers.Dropout(0.2)(x)
                x = layers.Dense(16, activation="relu")(x)
                out = layers.Dense(1, activation="linear")(x)
                lstm_model = models.Model(inp, out)
                lstm_model.compile(optimizer="adam", loss="mse")
                es = callbacks.EarlyStopping(
                    monitor="val_loss", patience=4, restore_best_weights=True
                )
                lstm_model.fit(
                    Xs_train,
                    yseq_train,
                    validation_split=0.1,
                    epochs=20,
                    batch_size=32,
                    callbacks=[es],
                    verbose=0,
                )
                st.success("LSTM baseline trained.")
            except Exception as e:
                st.warning(
                    "LSTM training failed or tensorflow not available: " + str(e)
                )
                lstm_model = None

        # Streaming simulation over the test set
        results = []
        preds_online_reg = []
        preds_online_clf = []
        preds_lstm_reg = []

        for i in range(len(X_test_s)):
            x_t = X_test_s[i].reshape(1, -1)

            pred_mag_online = online_reg.predict(x_t)[0]
            pred_dir_online = online_clf.predict(x_t)[0]

            preds_online_reg.append(pred_mag_online)
            preds_online_clf.append(int(pred_dir_online))

            # LSTM pred if available (sequence ending at absolute index)
            if lstm_model is not None:
                abs_idx = train_n + i
                if abs_idx - seq_len + 1 >= 0:
                    seq = scaler.transform(X)[abs_idx - seq_len + 1 : abs_idx + 1]
                    if seq.shape[0] == seq_len:
                        seq = seq.reshape(1, seq_len, seq.shape[1])
                        try:
                            pred_l = float(lstm_model.predict(seq, verbose=0)[0, 0])
                        except Exception:
                            pred_l = np.nan
                    else:
                        pred_l = np.nan
                else:
                    pred_l = np.nan
                preds_lstm_reg.append(pred_l)

            true_mag = y_reg_test[i]
            true_dir = int(y_clf_test[i])

            results.append(
                {
                    "date": dates_test[i],
                    "true_mag": true_mag,
                    "true_dir": true_dir,
                    "pred_online_mag": pred_mag_online,
                    "pred_online_dir": int(pred_dir_online),
                    "pred_lstm_mag": preds_lstm_reg[-1] if lstm_model is not None else np.nan,
                }
            )

            # update online models with the true label (simulate learning)
            online_reg.partial_fit(x_t, np.array([true_mag]))
            online_clf.partial_fit(x_t, np.array([true_dir]))

        res_df = pd.DataFrame(results)

        # metrics
        mse_online = mean_squared_error(res_df["true_mag"], res_df["pred_online_mag"])
        mae_online = mean_absolute_error(res_df["true_mag"], res_df["pred_online_mag"])
        acc_online = accuracy_score(res_df["true_dir"], res_df["pred_online_dir"])
        f1_online = f1_score(res_df["true_dir"], res_df["pred_online_dir"])

        if lstm_model is not None and res_df["pred_lstm_mag"].notna().sum() > 0:
            valid_mask = ~np.isnan(res_df["pred_lstm_mag"])
            mse_lstm = mean_squared_error(
                res_df["true_mag"][valid_mask], res_df["pred_lstm_mag"][valid_mask]
            )
            mae_lstm = mean_absolute_error(
                res_df["true_mag"][valid_mask], res_df["pred_lstm_mag"][valid_mask]
            )
        else:
            mse_lstm = np.nan
            mae_lstm = np.nan

        # ---------------------------
        # Performance Summary tables
        # ---------------------------
        st.subheader("Performance Summary (Regression, stream-test)")
        summary_reg = pd.DataFrame([
            {"Model": "Online (SGDReg)", "MSE": mse_online, "MAE": mae_online},
            {"Model": "LSTM (batch)",    "MSE": mse_lstm,  "MAE": mae_lstm},
        ])
        st.dataframe(summary_reg.style.format({"MSE": "{:.3e}", "MAE": "{:.6f}"}))

        st.subheader("Performance Summary (Classification, stream-test)")
        summary_cls = pd.DataFrame([
            {"Model": "Online (SGDClf)", "Accuracy": acc_online, "F1": f1_online}
        ])
        st.dataframe(summary_cls.style.format({"Accuracy": "{:.4f}", "F1": "{:.4f}"}))

        # Latest prediction box
        st.subheader("Latest Prediction")
        latest = res_df.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Date", str(pd.to_datetime(latest["date"]).date()))
        col2.metric(
            "Predicted Direction",
            "UP" if latest["pred_online_dir"] == 1 else "DOWN",
        )
        col3.metric(
            "Predicted Next-Day Return",
            f"{latest['pred_online_mag'] * 100:.3f}%",
        )

        # Plots: pred vs true scatter, time series, rolling accuracy, cumulative returns
        st.subheader("Diagnostics & Plots")

        # 1) scatter
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(res_df["true_mag"], res_df["pred_online_mag"], alpha=0.4, s=8)
        ax.set_xlabel("True next-day return")
        ax.set_ylabel("Predicted (online)")
        ax.set_title("Predicted vs True (online reg)")
        st.pyplot(fig)

        # 2) timeseries (last 400 points)
        maxpts = min(400, len(res_df))
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(
            pd.to_datetime(res_df["date"][-maxpts:]),
            res_df["true_mag"][-maxpts:],
            label="true",
            linewidth=1,
        )
        ax2.plot(
            pd.to_datetime(res_df["date"][-maxpts:]),
            res_df["pred_online_mag"][-maxpts:],
            label="online_pred",
            linewidth=1,
        )
        if lstm_model is not None:
            ax2.plot(
                pd.to_datetime(res_df["date"][-maxpts:]),
                res_df["pred_lstm_mag"][-maxpts:],
                label="lstm_pred",
                linewidth=1,
            )
        ax2.legend()
        ax2.set_title("True vs Predictions (recent)")
        st.pyplot(fig2)

        # 3) rolling accuracy for direction
        res_df["correct_online_dir"] = (
            res_df["true_dir"] == res_df["pred_online_dir"]
        ).astype(int)
        res_df["rolling_acc_online"] = res_df["correct_online_dir"].rolling(
            50, min_periods=1
        ).mean()
        fig3, ax3 = plt.subplots(figsize=(10, 2))
        ax3.plot(
            pd.to_datetime(res_df["date"]),
            res_df["rolling_acc_online"],
            label="rolling_acc_online",
        )
        ax3.axhline(0.5, linestyle="--", color="k", alpha=0.6)
        ax3.set_ylim(0, 1)
        ax3.set_title("Rolling accuracy (window=50)")
        st.pyplot(fig3)

        # 4) cumulative returns: strategy vs buy & hold
        strat_online = (
            res_df["pred_online_dir"].astype(int) * res_df["true_mag"]
        ).fillna(0)
        cum_strat_online = np.cumprod(1 + strat_online) - 1
        cum_hold = np.cumprod(1 + res_df["true_mag"]) - 1
        fig4, ax4 = plt.subplots(figsize=(10, 3))
        ax4.plot(
            pd.to_datetime(res_df["date"]),
            cum_strat_online,
            label="online strategy",
        )
        ax4.plot(
            pd.to_datetime(res_df["date"]),
            cum_hold,
            label="buy & hold",
        )
        ax4.legend()
        ax4.set_title("Cumulative returns")
        st.pyplot(fig4)

        # export results
        out_dir = os.path.join(".", "streamlit_results")
        os.makedirs(out_dir, exist_ok=True)
        res_df.to_csv(os.path.join(out_dir, "stream_sim_results.csv"), index=False)
        joblib.dump(
            {
                "summary": {
                    "mse_online": mse_online,
                    "mae_online": mae_online,
                    "acc_online": acc_online,
                    "f1_online": f1_online,
                }
            },
            os.path.join(out_dir, "summary.joblib"),
        )
        st.success(
            f"Streaming sim finished. Results saved to {out_dir}/stream_sim_results.csv"
        )
        st.info(
            "Tip: Use transaction-cost thresholding before trading; only trade when predicted magnitude > threshold."
        )
