"""
MCX Silver Complete ML Pipeline with Streamlit Interface
Includes tomorrow's prediction capability
Data format: Date, Price, Open, High, Low, Vol., Change %

Run with: streamlit run app.py
"""

import os, re, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import SGDRegressor, SGDClassifier
import joblib
import streamlit as st

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
OUT_DIR = "mcx_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def kalman_smooth(signal, q=1e-5, r=1e-2):
    """Kalman filter for noise reduction"""
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

def safe_numeric(s):
    """Convert string to numeric, handling special characters"""
    if isinstance(s, str):
        s = re.sub(r"[^0-9.\-eE]", "", s)
        try:
            return float(s)
        except:
            return np.nan
    return s

def engineer_features(df):
    """Apply feature engineering to dataframe"""
    df = df.copy()
    
    # Apply Kalman smoothing
    df["close_kf"] = kalman_smooth(df["Close"].values, q=1e-5, r=1e-2)
    
    # Returns
    df["ret_1"] = df["close_kf"].pct_change()
    df["logret"] = np.log(df["close_kf"]).diff()
    df["ret_5"] = df["close_kf"].pct_change(5)
    df["ret_10"] = df["close_kf"].pct_change(10)
    
    # Moving averages
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["close_kf"].rolling(w).mean()
        df[f"ema_{w}"] = df["close_kf"].ewm(span=w, adjust=False).mean()
        df[f"price_to_sma_{w}"] = df["close_kf"] / df[f"sma_{w}"]
    
    # Volatility
    for w in [5, 20, 60]:
        df[f"vol_{w}"] = df["ret_1"].rolling(w).std()
    
    # RSI
    delta = df["close_kf"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + roll_up / (roll_down + 1e-12))
    
    # OHLC features (if available)
    if "High" in df.columns and "Low" in df.columns:
        df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["hl_avg"] = (df["High"] + df["Low"]) / 2
        
    if "Open" in df.columns:
        df["open_close_diff"] = (df["Close"] - df["Open"]) / df["Open"]
    
    if "Volume" in df.columns:
        df["vol_ma_5"] = df["Volume"].rolling(5).mean()
        df["vol_ratio"] = df["Volume"] / (df["vol_ma_5"] + 1)
    
    # Lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)
        df[f"close_lag_{lag}"] = df["close_kf"].shift(lag)
    
    # Momentum
    df["momentum_5"] = df["close_kf"] - df["close_kf"].shift(5)
    df["momentum_10"] = df["close_kf"] - df["close_kf"].shift(10)
    
    return df

def load_and_prepare_data(uploaded_file=None, file_path=None):
    """Load and prepare data from file"""
    # Load data
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    elif file_path is not None:
        df_raw = pd.read_csv(file_path)
    else:
        return None
    
    # Normalize column names
    df_raw.columns = [re.sub(r"[^0-9A-Za-z]+", "_", str(c)).strip("_").lower() for c in df_raw.columns]
    
    # Identify columns
    date_col = next((c for c in df_raw.columns if "date" in c), df_raw.columns[0])
    price_col = next((c for c in df_raw.columns if "price" in c or "close" in c), None)
    open_col = next((c for c in df_raw.columns if "open" in c), None)
    high_col = next((c for c in df_raw.columns if "high" in c), None)
    low_col = next((c for c in df_raw.columns if "low" in c), None)
    vol_col = next((c for c in df_raw.columns if "vol" in c), None)
    
    # Create clean dataframe
    cols_to_keep = [date_col]
    rename_map = {date_col: "Date"}
    
    if price_col:
        cols_to_keep.append(price_col)
        rename_map[price_col] = "Close"
    if open_col:
        cols_to_keep.append(open_col)
        rename_map[open_col] = "Open"
    if high_col:
        cols_to_keep.append(high_col)
        rename_map[high_col] = "High"
    if low_col:
        cols_to_keep.append(low_col)
        rename_map[low_col] = "Low"
    if vol_col:
        cols_to_keep.append(vol_col)
        rename_map[vol_col] = "Volume"
    
    df = df_raw[cols_to_keep].rename(columns=rename_map)
    
    # Parse and clean data
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    for col in df.columns:
        if col != "Date":
            df[col] = df[col].apply(safe_numeric)
    
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    
    return df

def train_models(df, train_frac=0.7, use_xgboost=True):
    """Train all models and return results"""
    
    # Engineer features
    df = engineer_features(df)
    df = df.dropna().reset_index(drop=True)
    
    # Create targets
    df["target_ret"] = df["close_kf"].shift(-1) / df["close_kf"] - 1.0
    df["target_dir"] = (df["target_ret"] > 0).astype(int)
    df = df.dropna(subset=["target_ret", "target_dir"]).reset_index(drop=True)
    
    # Feature matrix
    exclude = ["Date", "Close", "close_kf", "target_ret", "target_dir", "Open", "High", "Low", "Volume"]
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].values
    y_reg = df["target_ret"].values
    y_dir = df["target_dir"].values
    dates = df["Date"].values
    
    # Split data
    train_n = int(len(X) * train_frac)
    X_train = X[:train_n]
    X_test = X[train_n:]
    y_reg_train = y_reg[:train_n]
    y_reg_test = y_reg[train_n:]
    y_dir_train = y_dir[:train_n]
    y_dir_test = y_dir[train_n:]
    dates_test = dates[train_n:]
    
    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Save scaler and feature columns
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(OUT_DIR, "feature_cols.joblib"))
    
    # Initialize online models (loss='log_loss' enables predict_proba)
    online_reg = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    online_clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    
    init_n = min(50, len(X_train_s))
    online_reg.partial_fit(X_train_s[:init_n], y_reg_train[:init_n])
    online_clf.partial_fit(X_train_s[:init_n], y_dir_train[:init_n], classes=np.array([0, 1]))
    
    # Train on full training data
    for i in range(init_n, len(X_train_s), 100):
        batch_end = min(i + 100, len(X_train_s))
        online_reg.partial_fit(X_train_s[i:batch_end], y_reg_train[i:batch_end])
        online_clf.partial_fit(X_train_s[i:batch_end], y_dir_train[i:batch_end])
    
    # Streaming simulation
    online_results = []
    for i in range(len(X_test_s)):
        x_t = X_test_s[i].reshape(1, -1)
        
        pred_mag = online_reg.predict(x_t)[0]
        pred_dir = online_clf.predict(x_t)[0]
        
        true_mag = y_reg_test[i]
        true_dir = y_dir_test[i]
        
        online_results.append({
            "date": dates_test[i],
            "true_mag": true_mag,
            "true_dir": int(true_dir),
            "pred_mag": pred_mag,
            "pred_dir": int(pred_dir)
        })
        
        online_reg.partial_fit(x_t, np.array([true_mag]))
        online_clf.partial_fit(x_t, np.array([true_dir]))
    
    # Evaluate
    online_df = pd.DataFrame(online_results)
    mse = mean_squared_error(online_df["true_mag"], online_df["pred_mag"])
    mae = mean_absolute_error(online_df["true_mag"], online_df["pred_mag"])
    acc = accuracy_score(online_df["true_dir"], online_df["pred_dir"])
    f1 = f1_score(online_df["true_dir"], online_df["pred_dir"])
    
    # Save final models
    joblib.dump(online_reg, os.path.join(OUT_DIR, "online_reg_model.joblib"))
    joblib.dump(online_clf, os.path.join(OUT_DIR, "online_clf_model.joblib"))
    
    # XGBoost (optional)
    xgb_metrics = None
    if use_xgboost:
        try:
            import xgboost as xgb
            dtrain = xgb.DMatrix(X_train_s, label=y_dir_train)
            dtest = xgb.DMatrix(X_test_s, label=y_dir_test)
            
            params = {
                "objective": "binary:logistic",
                "eta": 0.05,
                "max_depth": 4,
                "eval_metric": "auc",
                "seed": 42
            }
            
            bst = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
            y_prob = bst.predict(dtest)
            y_pred = (y_prob >= 0.5).astype(int)
            
            xgb_acc = accuracy_score(y_dir_test, y_pred)
            xgb_f1 = f1_score(y_dir_test, y_pred)
            xgb_auc = roc_auc_score(y_dir_test, y_prob)
            
            xgb_metrics = {"accuracy": xgb_acc, "f1": xgb_f1, "auc": xgb_auc}
            bst.save_model(os.path.join(OUT_DIR, "xgb_model.json"))
        except:
            pass
    
    return {
        "online_df": online_df,
        "metrics": {"mse": mse, "mae": mae, "accuracy": acc, "f1": f1},
        "xgb_metrics": xgb_metrics,
        "feature_cols": feature_cols,
        "df_full": df
    }

def predict_tomorrow(df):
    """Predict tomorrow's price direction and magnitude (FIXED VERSION)"""

    # Load models
    try:
        scaler = joblib.load(os.path.join(OUT_DIR, "scaler.joblib"))
        online_reg = joblib.load(os.path.join(OUT_DIR, "online_reg_model.joblib"))
        online_clf = joblib.load(os.path.join(OUT_DIR, "online_clf_model.joblib"))
        feature_cols = joblib.load(os.path.join(OUT_DIR, "feature_cols.joblib"))
    except:
        return None

    # Engineer features
    df_feat = engineer_features(df).dropna()
    if len(df_feat) == 0:
        return None

    latest = df_feat.iloc[-1]

    # ✅ LAST TRADING DAY
    latest_date = latest["Date"]
    tomorrow_date = latest_date + timedelta(days=1)

    # ✅ USE RAW EXCEL CLOSE (NOT Kalman)
    current_price = latest["Close"]

    # Extract features
    X_latest = np.array([latest[feature_cols].values])
    X_latest_s = scaler.transform(X_latest)

    # Predict return
    pred_ret = online_reg.predict(X_latest_s)[0]

    # Predict direction
    pred_dir = online_clf.predict(X_latest_s)[0]

    # ✅ CONSISTENCY FIX (VERY IMPORTANT)
    if pred_ret < 0:
        pred_dir = 0
    else:
        pred_dir = 1

    # Direction probability
    try:
        pred_dir_proba = online_clf.predict_proba(X_latest_s)[0]
    except:
        pred_dir_proba = np.array([0.5, 0.5])

    # ✅ PRICE PREDICTION BASED ON RAW CLOSE
    pred_price = current_price * (1 + pred_ret)

    # XGBoost (unchanged)
    xgb_pred = None
    try:
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(os.path.join(OUT_DIR, "xgb_model.json"))
        dpredict = xgb.DMatrix(X_latest_s)
        xgb_prob = float(bst.predict(dpredict)[0])
        xgb_dir = 1 if xgb_prob >= 0.5 else 0
        xgb_pred = {"direction": xgb_dir, "probability": xgb_prob}
    except:
        pass

    return {
        "current_date": latest_date,
        "current_price": current_price,
        "predicted_return": pred_ret,
        "predicted_direction": int(pred_dir),
        "direction_probability": pred_dir_proba,
        "predicted_price": pred_price,
        "xgb_prediction": xgb_pred,
        "tomorrow_date": tomorrow_date
    }

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="MCX Silver Predictor", layout="wide", page_icon="📈")

st.title("📈 MCX Silver Price Prediction System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("Upload MCX Silver CSV", type=["csv"])
    
    st.markdown("### Training Parameters")
    train_frac = st.slider("Training Data Fraction", 0.5, 0.9, 0.7, 0.05)
    use_xgboost = st.checkbox("Use XGBoost (slower)", value=True)
    
    train_button = st.button("🚀 Train Models", type="primary", use_container_width=True)
    predict_button = st.button("🔮 Predict Tomorrow", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This system uses online learning and XGBoost to predict MCX Silver prices.")

# Main content
if uploaded_file is not None:
    
    # Load data
    df = load_and_prepare_data(uploaded_file=uploaded_file)
    
    if df is not None:
        st.success(f"✅ Data loaded: {len(df)} rows")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        col3.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
        col4.metric("Last Change", f"{((df['Close'].iloc[-1]/df['Close'].iloc[-2]-1)*100):.2f}%")
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.tail(10), use_container_width=True)
        
        # Train models
        if train_button:
            with st.spinner("Training models... This may take a minute."):
                results = train_models(df, train_frac=train_frac, use_xgboost=use_xgboost)
                
                st.session_state['results'] = results
                st.session_state['df'] = df
                
                st.success("✅ Models trained successfully!")
        
        # Display results if available
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            st.markdown("---")
            st.header("📊 Model Performance")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = results['metrics']
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("F1 Score", f"{metrics['f1']:.4f}")
            col3.metric("MAE", f"{metrics['mae']:.6f}")
            col4.metric("MSE", f"{metrics['mse']:.2e}")
            
            if results['xgb_metrics']:
                st.markdown("#### XGBoost Performance")
                col1, col2, col3 = st.columns(3)
                col1.metric("XGB Accuracy", f"{results['xgb_metrics']['accuracy']:.2%}")
                col2.metric("XGB F1", f"{results['xgb_metrics']['f1']:.4f}")
                col3.metric("XGB AUC", f"{results['xgb_metrics']['auc']:.4f}")
            
            # Visualizations
            st.markdown("---")
            st.header("📈 Performance Visualizations")
            
            online_df = results['online_df']
            
            # Time series
            tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Scatter Plot", "Rolling Accuracy", "Strategy Returns"])
            
            with tab1:
                window = min(300, len(online_df))
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(pd.to_datetime(online_df["date"][-window:]), 
                        online_df["true_mag"][-window:], 
                        label="True", linewidth=1.5, alpha=0.8)
                ax.plot(pd.to_datetime(online_df["date"][-window:]), 
                        online_df["pred_mag"][-window:], 
                        label="Predicted", linewidth=1.5, alpha=0.8)
                ax.legend(fontsize=12)
                ax.set_title(f"True vs Predicted Returns (Last {window} Days)", fontsize=14)
                ax.set_ylabel("Return", fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(online_df["true_mag"], online_df["pred_mag"], alpha=0.4, s=20)
                ax.plot([-0.05, 0.05], [-0.05, 0.05], 'r--', alpha=0.5, linewidth=2)
                ax.set_xlabel("True Return", fontsize=12)
                ax.set_ylabel("Predicted Return", fontsize=12)
                ax.set_title("Predicted vs True Returns", fontsize=14)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab3:
                online_df["correct"] = (online_df["true_dir"] == online_df["pred_dir"]).astype(int)
                online_df["rolling_acc"] = online_df["correct"].rolling(50, min_periods=1).mean()
                
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(pd.to_datetime(online_df["date"]), online_df["rolling_acc"], linewidth=2)
                ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, linewidth=2)
                ax.fill_between(pd.to_datetime(online_df["date"]), 0.5, online_df["rolling_acc"], 
                               where=(online_df["rolling_acc"] > 0.5), alpha=0.3, color='green')
                ax.set_ylim(0, 1)
                ax.set_ylabel("Accuracy", fontsize=12)
                ax.set_title("Rolling Direction Prediction Accuracy (Window=50)", fontsize=14)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab4:
                online_df["strategy_ret"] = online_df["pred_dir"] * online_df["true_mag"]
                online_df["cum_strategy"] = (1 + online_df["strategy_ret"]).cumprod() - 1
                online_df["cum_hold"] = (1 + online_df["true_mag"]).cumprod() - 1
                
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(pd.to_datetime(online_df["date"]), 
                        online_df["cum_strategy"] * 100, 
                        label="Strategy", linewidth=2.5)
                ax.plot(pd.to_datetime(online_df["date"]), 
                        online_df["cum_hold"] * 100, 
                        label="Buy & Hold", linewidth=2.5, alpha=0.7)
                ax.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
                ax.fill_between(pd.to_datetime(online_df["date"]), 0, online_df["cum_strategy"] * 100,
                               alpha=0.3)
                ax.set_ylabel("Cumulative Return (%)", fontsize=12)
                ax.set_title("Strategy Performance: Model vs Buy & Hold", fontsize=14)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Strategy stats
                final_strategy = online_df["cum_strategy"].iloc[-1] * 100
                final_hold = online_df["cum_hold"].iloc[-1] * 100
                outperformance = final_strategy - final_hold
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Strategy Return", f"{final_strategy:.2f}%")
                col2.metric("Buy & Hold Return", f"{final_hold:.2f}%")
                col3.metric("Outperformance", f"{outperformance:.2f}%", 
                           delta=f"{outperformance:.2f}%")
        
        # Tomorrow's prediction
        if predict_button:
            if 'results' not in st.session_state:
                st.error("⚠️ Please train the models first!")
            else:
                with st.spinner("Predicting tomorrow's price..."):
                    prediction = predict_tomorrow(df)
                    
                    if prediction:
                        st.markdown("---")
                        
                        # ONE-LINE PREDICTION BANNER
                        direction_emoji = "📈" if prediction['predicted_direction'] == 1 else "📉"
                        direction_text = "UP" if prediction['predicted_direction'] == 1 else "DOWN"
                        price_change_pct = prediction['predicted_return'] * 100
                        confidence = prediction['direction_probability'][prediction['predicted_direction']] * 100
                        
                        # Color based on direction
                        if prediction['predicted_direction'] == 1:
                            banner_color = "#00ff00"  # Green for UP
                            text_color = "#006400"
                        else:
                            banner_color = "#ff4444"  # Red for DOWN
                            text_color = "#8B0000"
                        
                        st.markdown(f"""
                        <div style="background-color: {banner_color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                            <h1 style="color: {text_color}; margin: 0; font-size: 2.5em;">
                                {direction_emoji} Tomorrow's Market Trend: <b>{direction_text}</b> by <b>{abs(price_change_pct):.2f}%</b> {direction_emoji}
                            </h1>
                            <p style="color: {text_color}; margin: 10px 0 0 0; font-size: 1.2em;">
                                Confidence: {confidence:.1f}% | Predicted Price: ₹{prediction['predicted_price']:.2f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.header("🔮 Detailed Prediction")
                        
                        # Prediction display
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📅 Current Status")
                            st.metric("Date", prediction['current_date'].strftime('%Y-%m-%d'))
                            st.metric("Current Price", f"₹{prediction['current_price']:.2f}")
                        
                        with col2:
                            st.markdown("### 🎯 Tomorrow's Forecast")
                            st.metric("Predicted Date", prediction['tomorrow_date'].strftime('%Y-%m-%d'))
                            
                            direction_text = "📈 UP" if prediction['predicted_direction'] == 1 else "📉 DOWN"
                            confidence = prediction['direction_probability'][prediction['predicted_direction']] * 100
                            
                            st.metric("Direction", direction_text)
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        # Price prediction
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        
                        price_change = prediction['predicted_price'] - prediction['current_price']
                        price_change_pct = (price_change / prediction['current_price']) * 100
                        
                        col1.metric("Predicted Price", f"₹{prediction['predicted_price']:.2f}")
                        col2.metric("Expected Change", f"₹{price_change:.2f}", 
                                   delta=f"{price_change_pct:.2f}%")
                        col3.metric("Expected Return", f"{prediction['predicted_return']*100:.2f}%")
                        
                        # XGBoost prediction if available
                        if prediction['xgb_prediction']:
                            st.markdown("---")
                            st.markdown("### 🎯 XGBoost Prediction")
                            xgb_dir = "📈 UP" if prediction['xgb_prediction']['direction'] == 1 else "📉 DOWN"
                            xgb_conf = prediction['xgb_prediction']['probability'] * 100
                            
                            col1, col2 = st.columns(2)
                            col1.metric("XGB Direction", xgb_dir)
                            col2.metric("XGB Confidence", f"{xgb_conf:.1f}%")
                        
                        # Recommendation
                        st.markdown("---")
                        st.markdown("### 💡 Trading Recommendation")
                        
                        if confidence > 65:
                            if prediction['predicted_direction'] == 1:
                                st.success(f"🟢 **STRONG BUY** signal with {confidence:.1f}% confidence. Expected gain: {price_change_pct:.2f}%")
                            else:
                                st.error(f"🔴 **STRONG SELL** signal with {confidence:.1f}% confidence. Expected loss: {price_change_pct:.2f}%")
                        elif confidence > 55:
                            if prediction['predicted_direction'] == 1:
                                st.info(f"🟡 **BUY** signal with moderate confidence ({confidence:.1f}%). Expected gain: {price_change_pct:.2f}%")
                            else:
                                st.warning(f"🟡 **SELL** signal with moderate confidence ({confidence:.1f}%). Expected loss: {price_change_pct:.2f}%")
                        else:
                            st.warning(f"⚪ **NEUTRAL** - Low confidence ({confidence:.1f}%). Consider waiting for clearer signals.")
                        
                        st.caption("⚠️ Happy Every Time !! ")
                    else:
                        st.error("Failed to generate prediction. Please ensure models are trained.")

else:
    st.info("👈 Please upload a CSV file to get started")
    
    st.markdown("### 📋 Required CSV Format")
    st.code("""
Date,Price,Open,High,Low,Vol.,Change %
2024-01-01,75000,74800,75200,74500,1000,0.5%
2024-01-02,75500,75000,76000,74900,1200,0.67%
...
    """)
    
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **Real-time prediction** for tomorrow's price
    - **Dual model approach**: Online Learning + XGBoost
    - **Comprehensive metrics**: Accuracy, F1, AUC, MAE, MSE
    - **Visual analytics**: Time series, scatter plots, rolling accuracy
    - **Strategy comparison**: Model predictions vs Buy & Hold
    - **Confidence scores** for each prediction
    """)

# End of SILVER_MCX_Stremlit.py