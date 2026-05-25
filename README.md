# 🏥 Smart Health Insurance Plan Recommendation System
### Data Analysis Project 2026 | Batch F7

---

## 👥 Team Members
| Roll No | Name |
|---------|------|
| 992401030333 | Pulkeshin Goyal |
| 992401030340 | Chetanya Mangla |
| 992401030343 | Smarth Khurana |
| 992401030344 | Aakash Jat |
| 992401030345 | Kunal Sharma |

---

## 📌 Problem Statement
Many individuals select health insurance plans randomly or based on incomplete information, leading to overpayment, insufficient coverage, or claims rejection. This project builds a **data-driven recommendation system** that analyzes user demographics, health indicators, and lifestyle to suggest the most suitable insurance plan.

**Project Balance:** 70–80% EDA & Data Analysis | 20–30% Simple ML

---

## 📁 Project Structure
```
health_insurance_project/
│
├── health_insurance_recommendation.py   ← Main project script
├── README.md                            ← This file
│
└── outputs/
    ├── health_insurance_dataset.csv     ← Generated dataset (1000 rows)
    ├── fig1_univariate.png              ← Univariate distributions
    ├── fig2_bivariate.png               ← Bivariate analysis
    ├── fig3_heatmap.png                 ← Correlation heatmap
    ├── fig4_deep_eda.png                ← Deep EDA cross-feature analysis
    ├── fig5_ml_results.png              ← ML confusion matrices + feature importance
    ├── fig6_decision_tree.png           ← Decision tree visualization
    └── fig7_dashboard.png               ← Final summary dashboard
```

---

## 🔧 Installation & Setup

### Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run the Project
```bash
python health_insurance_recommendation.py
```

---

## 📊 Dataset
- **Source:** Synthetic dataset (inspired by Kaggle Medical Insurance & UCI datasets)
- **Size:** 1,000 records
- **Features:**

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | 18–70 years |
| gender | Categorical | Male / Female |
| income | Numeric | Annual income (₹) |
| region | Categorical | North / South / East / West |
| bmi | Numeric | Body Mass Index |
| smoking | Categorical | Yes / No |
| exercise | Categorical | None / Light / Moderate / Heavy |
| medical_history | Categorical | None / Mild / Chronic |
| dependents | Numeric | Number of dependents |
| premium | Numeric | Calculated insurance premium (₹) |
| plan | Target | Basic / Standard / Premium |

---

## 🔍 Project Sections

### 1. Data Preprocessing
- Missing value imputation (median/mode)
- Outlier capping (1st–99th percentile)
- Label encoding of categorical features
- Standard scaling of numerical features

### 2. Exploratory Data Analysis (MAIN FOCUS — 70%)
- **Univariate:** Age, BMI, premium, income, gender, smoking, plan distributions
- **Bivariate:** Age vs premium, smoking vs premium, income vs plan, medical history vs premium
- **Correlation Heatmap:** Identifies strongest predictors of premium and plan
- **Deep EDA:** BMI by plan, age groups vs plan, regional analysis, income quintiles

### 3. Key Insights
- Smokers pay **~58% higher premiums** than non-smokers
- Chronic conditions raise average premium by **~₹4,800** vs no history
- Seniors (60+) pay **21% more** than young adults (18–30)
- Higher exercise frequency correlates with lower premiums

### 4. Minimal ML (Supporting Role — 30%)
- **Decision Tree** (Accuracy: ~97%) — primary model, interpretable
- **Logistic Regression** (Accuracy: ~84%) — baseline comparison
- Feature importance analysis confirms: smoking, medical history, age are top drivers

### 5. Recommendation Engine
Rule-based hybrid system combining EDA insights:

| Condition | Recommendation |
|-----------|---------------|
| Young (< 30), healthy, non-smoker, active | **Basic Plan** |
| Middle-aged, mild history OR smoker | **Standard Plan** |
| Senior (60+), chronic conditions, high BMI | **Premium Plan** |

---

## 📈 Outputs Generated
1. `fig1_univariate.png` — 9-panel distribution plots
2. `fig2_bivariate.png` — Key relationships (scatter, box, violin, bar)
3. `fig3_heatmap.png` — Correlation matrix
4. `fig4_deep_eda.png` — Cross-feature analysis (6 plots)
5. `fig5_ml_results.png` — Confusion matrices + feature importance
6. `fig6_decision_tree.png` — Decision tree visualization
7. `fig7_dashboard.png` — Executive summary dashboard

---

## ⚠️ Limitations & Future Work
- Dataset is synthetic; real-world validation required
- Rule-based engine could be enhanced with clustering (K-Means)
- Regional pricing variations not fully modeled
- Future: Add family floater plan type, chronic disease sub-categories

---

## 📚 References
- Kaggle: Health Insurance Cross-Sell Prediction Dataset
- UCI ML Repository: Health Insurance Dataset
- ResearchGate: Predictive Modelling of Healthcare Insurance Costs using ML
- WJARR: Predicting Health Insurance Premiums — Neural Network Model
