
# SILVER MCX Price Prediction – Streamlit App  
A complete Machine Learning + Streamlit project that predicts **Silver MCX price trend and magnitude** using historical data.  
This project includes:

- Data preprocessing  
- Feature engineering  
- Trend & magnitude prediction  
- Model training (XGBoost / Linear / other ML models)  
- Streaming/online evaluation  
- Interactive Streamlit web UI  

---

## 📌 Project Overview
This project loads historical Silver MCX data, generates technical indicators, trains ML models, and provides **real-time predictions** through a Streamlit dashboard.

It supports:

- Trend Prediction (Up/Down)
- Magnitude Prediction (Next price % movement)
- Online streaming: real-time prediction as new data arrives
- Model saving & loading using `.pkl`

This repository is based on the notebook: `SILVER_MCX_Stremlit.ipynb`, which you can later export into `app.py` for deployment.

---

## 📂 Suggested Folder Structure

You can organize your GitHub project like this:

```
├── data/
│   └── silver_data.csv        # Input dataset (your historical MCX data)
├── models/
│   ├── trend_model.pkl        # Saved trend classifier
│   ├── magnitude_model.pkl    # Saved regression model
├── SILVER_MCX_Stremlit.ipynb  # Original Jupyter notebook
├── app.py                     # (Optional) Streamlit UI script
├── train_model.py             # (Optional) Training pipeline script
├── utils.py                   # (Optional) Helper functions
├── requirements.txt
└── README.md
```

If you only have the notebook right now, you can still upload:

- `SILVER_MCX_Stremlit.ipynb`  
- `requirements.txt`  
- `README.md`

and later add `app.py` and `train_model.py`.

---

## 🚀 Features

### ✔ **1. Streamlit Web App** (when converted to app.py)
- Upload CSV file  
- View trends  
- Predict next candle  
- Plot historical series  
- Show momentum + features  

### ✔ **2. ML Models**
- XGBoost / RandomForest / Linear Regression  
- Trend classification  
- Magnitude regression  

### ✔ **3. Feature Engineering**
Typical indicators (as used in the notebook) such as:

- Simple & Exponential Moving Averages (SMA, EMA)  
- RSI  
- MACD  
- Bollinger Bands  
- Lag features  
- Rolling window statistics  

### ✔ **4. Streaming Predictions**
The notebook / app can be extended to continuously update predictions for new incoming rows.

---

## 🔧 Installation

### **1. Clone the repo**
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### **2. Create Environment & Install Dependencies**
Create a virtual environment (optional but recommended) and install required libraries:

```bash
pip install -r requirements.txt
```

### A minimal `requirements.txt` can be:

```txt
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
```

Add/remove packages depending on what you import in `SILVER_MCX_Stremlit.ipynb`.

---

## ▶️ How to Run the Notebook

1. Open a terminal / Anaconda Prompt in the project directory  
2. Launch Jupyter:

```bash
jupyter notebook
```

3. Open `SILVER_MCX_Stremlit.ipynb`  
4. Run all cells step by step  

---

## ▶️ How to Run as a Streamlit App (Optional)

If you convert your notebook logic into `app.py`, you can run:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal, for example:

```
http://localhost:8501


## 📊 Training the Models (Optional Script)

If you later move training code into `train_model.py`, you can run:

```bash
python train_model.py
```

The script can be designed to:

1. Load CSV  
2. Create features  
3. Split training vs streaming data  
4. Train ML models  
5. Save `.pkl` files automatically under `models/`  

Example console output:

```
Loaded rows: 2511
Rows after features: 2451
Train samples: 1715
Stream samples: 735
Model trained and saved.
Online MSE: 5.68
```

---

## 🧠 Real-Time Prediction Idea

You can extend this project to:

1. Read latest Silver MCX prices from an API or CSV append  
2. Generate the same features as training  
3. Feed the latest row into the saved model  
4. Display:
   - Trend → UP / DOWN  
   - Magnitude → % move prediction  

This can be done fully inside **Streamlit** for a smooth UI.

---

### ❌ Dataset errors?
Make sure your CSV contains at least:

```text
Date, Open, High, Low, Close
```

and that date is parseable.

---

## 🌟 Future Improvements

- Add LSTM or other deep learning models  
- Add live data API (e.g., broker or MCX feed)  
- Add backtesting module  
- Add auto retraining  
- Deploy app on cloud (Streamlit Cloud / Render / AWS / etc.)  

---

## 📄 License

This project is open-source under the **MIT License** .

---

## 👤 Author

**HENIL Patel**  

- GitHub: https://github.com/Henil1411/<Henil1411>  
- Email: *henilajpatel@gmail.com*  

Feel free to fork, open issues, or submit pull requests.

---

## ⭐ Support

If you find this project useful:

- Give the repo a ⭐ on GitHub  
- Share it with others  
- Contribute with ideas or code  