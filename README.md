# Firewall Frontend – README

## 📌 Overview
This project contains a **Machine Learning–powered Network Intrusion Detection System (ML-NIDS)** with a frontend dashboard, model training pipeline, ETL scripts, documentation, and sample datasets.  
It supports **supervised** and **unsupervised** anomaly detection models and includes full workflows for **data processing**, **training**, **evaluation**, and **real-time detection**.

---

## 📂 Project Structure

```
Firewall_Frontend-main/
│
├── ML-NIDS/
│   ├── dashboard/              # Frontend dashboard (Flask/Streamlit)
│   ├── data/                   # Raw & processed datasets
│   ├── etl/                    # Data ingestion, cleaning, visualization
│   ├── models/                 # Trained ML models (.pkl)
│   ├── models_comparison/      # Performance comparison charts
│   ├── results/                # Detection results & metrics
│   ├── scripts/                # Automation helpers
│   ├── src/                    # Core ML source code
│   ├── requirements.txt        # Python dependencies
│   └── README.md               # Module documentation
│
├── docs/                       # Architecture & algorithm documentation
├── LICENSE                     # License file
├── README.md                   # Replace with this file
├── USAGE_GUIDE.md              # Runtime instructions
└── testdata.ipynb              # Notebook for testing models
```

---

## 🚀 Features
- ✔️ Supervised Intrusion Detection (Random Forest, XGBoost, etc.)  
- ✔️ Unsupervised Anomaly Detection (Isolation Forest, One-Class SVM)  
- ✔️ Real-time traffic capture and classification  
- ✔️ Interactive monitoring dashboard  
- ✔️ PCAP processing & feature extraction  
- ✔️ Model comparison charts  
- ✔️ Pre-trained models included  

---

## 🧰 Installation

### 1. Extract or clone the project
```bash
git clone <repo-url>
cd ML-NIDS
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Start the Dashboard
```bash
cd ML-NIDS/dashboard
python app.py
```

### Quick Detection Test
```bash
python ML-NIDS/scripts/quick_start.py
```

### Train All Models
```bash
python ML-NIDS/scripts/train_models.py
```

### Verify Setup
```bash
python ML-NIDS/scripts/verify_setup.py
```

---

## 📊 Datasets
Included in the `data/` folder:
- `dataset_cleaned.csv`  
- `dataset_selection.csv`  
- `Network_logs.csv`  
- `Time-Series_Network_logs.csv`  

Process your own PCAP:
```bash
python ML-NIDS/etl/process_pcap.py <path_to_pcap>
```

---

## 🧠 Models
Stored in:

```
ML-NIDS/models/
```

Includes:
- `supervised_random_forest.pkl`  
- `supervised_xgboost.pkl`  
- `unsupervised_isolation_forest.pkl`  
- `unsupervised_one_class_svm.pkl`  

---

## 📘 Documentation
See the `docs/` folder for:
- System architecture  
- Feature extraction details  
- Fusion algorithm design  
- Model performance metrics  

---

## 🛠 Technologies Used
- Python 3.8+  
- Scikit-learn, XGBoost  
- Pandas, NumPy  
- Flask / Streamlit  
- Matplotlib  
- PyShark, Scapy  

---

## 📄 License
Refer to the `LICENSE` file.

---

## 🤝 Contributing
Pull requests are welcome — open an issue first to discuss major changes.

---

## 📬 Contact
For issues or support, open an issue or contact the project maintainer.
