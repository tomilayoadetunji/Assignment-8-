# Breast Cancer Classification Using Supervised Machine Learning

##  Project Overview
This project focuses on building a supervised machine learning classification model to predict whether a breast tumor is **benign** or **malignant**, using the **Breast Cancer Wisconsin (Diagnostic)** dataset. The project involves dataset exploration, preprocessing, model training, performance evaluation, interpretation, and outlining a deployment + monitoring strategy.

The dataset is publicly available via the **UCI Machine Learning Repository** and can be loaded easily using `sklearn.datasets.load_breast_cancer()` — no external downloads required.

---

##  Dataset Information
| Property | Value |
|---------|-------|
| Samples | 569 |
| Features | 30 numeric (tumor cell measurements) |
| Target | 0 = Benign, 1 = Malignant |
| Missing Values | None |

### Example Features
- **mean radius**
- **mean texture**
- **mean perimeter**
- **mean smoothness**

These features describe properties of cell nuclei found in digitized breast tissue images.

---

##  Exploratory Data Analysis (EDA)
EDA included:
- Class distribution visualization
- Feature correlation heatmap
- Pair plots for separating malignant vs benign groups
- Identification of key predictive features (e.g., mean radius, mean perimeter)

**Key Insight:** Tumor size-related measurements strongly correlate with malignancy.

---

##  Machine Learning Models Implemented
The following models were trained and compared:

| Model | Description |
|------|-------------|
| Logistic Regression | Baseline linear classifier |
| Support Vector Machine (RBF) | Nonlinear margin-based classifier |
| Decision Tree | Simple interpretable tree model |
| Random Forest ✅ | **Best performing ensemble classifier** |
| k-Nearest Neighbors | Instance-based learner |

###  Best Model: **Random Forest Classifier**
- **High Accuracy**
- **High Recall** (important for detecting malignant tumors)
- **Highest ROC-AUC score** among compared models

---

##  Evaluation Metrics Used
- Accuracy
- Precision
- Recall (critical for medical diagnosis)
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

---

##  Interpretation
- The Random Forest model performed best due to its robustness, ability to capture nonlinear relationships, and reduced overfitting.
- Feature importance analysis confirms medical relevance (e.g., tumors with larger radius/perimeter more likely malignant).

---

##  Deployment Strategy (Proposed)
1. Save trained model using **Joblib**
2. Serve it via a **FastAPI** or Flask REST endpoint
3. Accept patient / tumor measurement input in JSON format
4. Return predicted class + malignancy probability

### Monitoring Strategy
- Track input and output drift
- Periodically re-evaluate model on new clinical data
- Set automated alerts for performance degradation
- Retrain at scheduled intervals

---

## How to Set Up and Run the Notebook

### **Option 1: Run in Google Colab (Recommended ✅)**

1. Open the repository in your browser.
2. Click on the notebook file: `notebook.ipynb`
3. Click **"Open in Colab"** (or upload manually at https://colab.research.google.com/)
4. Run all cells top-to-bottom (no setup required — dataset loads automatically).

### **Option 2: Run Locally**

#### **Requirements**
```bash
python 3.8+
pip install numpy pandas matplotlib seaborn scikit-learn
