"""
=============================================================
  SMART HEALTH INSURANCE PLAN RECOMMENDATION SYSTEM
  Batch F7 | Data Analysis Project 2026
=============================================================
  Team Members:
  - Pulkeshin Goyal   (992401030333)
  - Chetanya Mangla   (992401030340)
  - Smarth Khurana    (992401030343)
  - Aakash Jat        (992401030344)
  - Kunal Sharma      (992401030345)

  Dataset: Kaggle Medical Insurance Dataset (insurance.csv)
  Source : https://www.kaggle.com/datasets/mirichoi0218/insurance
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import webbrowser
import shutil

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# Output directory for all figures
os.makedirs("outputs", exist_ok=True)

PALETTE = {
    "primary":   "#1A6B8A",
    "secondary": "#2ECC71",
    "accent":    "#E74C3C",
    "warn":      "#F39C12",
    "light":     "#ECF0F1",
    "dark":      "#2C3E50",
}
PLAN_COLORS = {
    "Basic":    "#3498DB",
    "Standard": "#2ECC71",
    "Premium":  "#E74C3C",
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":      130,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 1. LOAD & ENGINEER REAL DATASET
def load_dataset(filepath: str = "insurance.csv") -> pd.DataFrame:

    print(f"\n[STEP 1] Loading real dataset from '{filepath}'...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: '{filepath}'\n"
            "Place insurance.csv in the same folder as this script."
        )

    df = pd.read_csv(filepath)
    print(f"  Raw shape  : {df.shape}")
    print(f"  Columns    : {list(df.columns)}")

    # Rename columns
    df.rename(columns={
        "sex":      "gender",
        "children": "dependents",
        "smoker":   "smoking",
        "charges":  "premium",
    }, inplace=True)

    # Standardise categorical value
    df["gender"]  = df["gender"].str.capitalize()        
    df["smoking"] = df["smoking"].str.capitalize()        

    # Capitalise region labels for consistent display
    region_map = {
        "southwest": "Southwest", "southeast": "Southeast",
        "northwest": "Northwest", "northeast": "Northeast",
    }
    df["region"] = df["region"].map(region_map)

    # Engineer medical_history from actuarial risk factors
    def derive_medical_history(row):
        is_smoker  = row["smoking"] == "Yes"
        is_obese   = row["bmi"] >= 30
        is_overweight = row["bmi"] >= 27.5
        is_senior  = row["age"] >= 50
        is_middle  = row["age"] >= 40

        if is_smoker and (is_obese or is_senior):
            return "Chronic"
        if (is_overweight or is_middle) and not (is_smoker and is_obese):
            return "Mild"
        return "None"

    df["medical_history"] = df.apply(derive_medical_history, axis=1)

    p40 = df["premium"].quantile(0.40)
    p75 = df["premium"].quantile(0.75)

    def assign_plan(row):
        if row["medical_history"] == "Chronic" or row["premium"] > p75:
            return "Premium"
        if row["smoking"] == "Yes" or row["premium"] > p40 or row["bmi"] > 30:
            return "Standard"
        return "Basic"

    df["plan"] = df.apply(assign_plan, axis=1)

    print(f"  Final shape: {df.shape}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"\n  Plan distribution:")
    for plan, cnt in df["plan"].value_counts().items():
        print(f"    {plan:10s}: {cnt:4d}  ({cnt/len(df)*100:.1f}%)")

    print(f"\n  Medical history distribution:")
    for mh, cnt in df["medical_history"].value_counts().items():
        print(f"    {mh:10s}: {cnt:4d}  ({cnt/len(df)*100:.1f}%)")

    return df

# 2. DATA PREPROCESSING
def preprocess(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  SECTION 2: DATA PREPROCESSING")
    print("="*60)

    print(f"\n[INFO] Raw shape: {df.shape}")
    missing = df.isnull().sum()
    if missing.any():
        print(f"[INFO] Missing values:\n{missing[missing > 0]}")
    else:
        print("[INFO] No missing values found in the real dataset.")

    #Handle Missing Values
    if df["bmi"].isnull().any():
        df["bmi"].fillna(df["bmi"].median(), inplace=True)
    if df["premium"].isnull().any():
        df["premium"].fillna(df["premium"].median(), inplace=True)

    #Remove Duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Duplicates removed: {before - len(df)}")

    #Outlier Capping
    for col in ["premium", "bmi"]:
        q1, q3 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q1, q3)
    print("[INFO] Outliers capped at 1st–99th percentile for premium, bmi")

    # Encode Categoricals
    le = LabelEncoder()
    cat_cols = ["gender", "region", "smoking", "medical_history", "plan"]
    df_encoded = df.copy()
    encoders = {}
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    print(f"[INFO] Encoded columns: {cat_cols}")
    print(f"\n[INFO] Clean shape: {df.shape}")
    print(f"[INFO] Plan distribution:\n{df['plan'].value_counts()}")

    # Scale Numerical Features
    scaler = StandardScaler()
    scale_cols = ["age", "bmi", "premium", "dependents"]
    df_scaled = df_encoded.copy()
    df_scaled[scale_cols] = scaler.fit_transform(df_encoded[scale_cols])

    return df, df_encoded, df_scaled, encoders

# 3. EXPLORATORY DATA ANALYSIS
def eda(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  SECTION 3: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    #Overview
    print("\n[STATS] Descriptive Statistics:")
    print(df.describe().round(2).to_string())

    #  FIGURE 1 — Univariate Distributions
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Univariate Analysis — Feature Distributions\n(Kaggle Insurance Dataset, n=1,338)",
                 fontsize=18, fontweight="bold", y=1.01, color=PALETTE["dark"])

    # Age
    ax = axes[0, 0]
    ax.hist(df["age"], bins=25, color=PALETTE["primary"], edgecolor="white", alpha=0.9)
    ax.set_title("Age Distribution"); ax.set_xlabel("Age"); ax.set_ylabel("Count")
    ax.axvline(df["age"].mean(), color=PALETTE["accent"], linestyle="--",
               label=f'Mean: {df["age"].mean():.1f}')
    ax.legend()

    # BMI
    ax = axes[0, 1]
    ax.hist(df["bmi"], bins=25, color=PALETTE["secondary"], edgecolor="white", alpha=0.9)
    ax.set_title("BMI Distribution"); ax.set_xlabel("BMI"); ax.set_ylabel("Count")
    ax.axvline(25, color=PALETTE["accent"], linestyle="--", label="Normal BMI: 25")
    ax.axvline(30, color=PALETTE["warn"],   linestyle="--", label="Obese BMI: 30")
    ax.legend(fontsize=9)

    # Premium (charges)
    ax = axes[0, 2]
    ax.hist(df["premium"], bins=30, color=PALETTE["warn"], edgecolor="white", alpha=0.9)
    ax.set_title("Premium Distribution"); ax.set_xlabel("Premium ($)"); ax.set_ylabel("Count")
    ax.axvline(df["premium"].mean(), color=PALETTE["accent"], linestyle="--",
               label=f'Mean: ${df["premium"].mean():,.0f}')
    ax.legend()

    # Dependents (children)
    ax = axes[1, 0]
    counts = df["dependents"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values,
           color=PALETTE["primary"], edgecolor="white", alpha=0.9)
    ax.set_title("Dependents (Children)"); ax.set_xlabel("Number of Children"); ax.set_ylabel("Count")

    # Gender
    ax = axes[1, 1]
    counts = df["gender"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE["primary"], PALETTE["secondary"]], edgecolor="white")
    ax.set_title("Gender Distribution"); ax.set_ylabel("Count")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{v}', ha='center', fontweight='bold')

    # Smoking
    ax = axes[1, 2]
    counts = df["smoking"].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=[PALETTE["secondary"], PALETTE["accent"]],
           startangle=140, wedgeprops=dict(width=0.6))
    ax.set_title("Smoking Status")

    # Medical History
    ax = axes[2, 0]
    order_mh = ["None", "Mild", "Chronic"]
    counts = df["medical_history"].value_counts().reindex(order_mh, fill_value=0)
    colors_mh = [PALETTE["secondary"], PALETTE["warn"], PALETTE["accent"]]
    bars = ax.bar(counts.index, counts.values, color=colors_mh, edgecolor="white")
    ax.set_title("Medical History (Derived)"); ax.set_ylabel("Count")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{v}', ha='center', fontweight='bold')

    # Plan Distribution
    ax = axes[2, 1]
    counts = df["plan"].value_counts()
    colors_plan = [PLAN_COLORS.get(p, "#aaa") for p in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors_plan, edgecolor="white")
    ax.set_title("Insurance Plan Distribution"); ax.set_ylabel("Count")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9, fontweight='bold')

    # Region
    ax = axes[2, 2]
    counts = df["region"].value_counts()
    ax.bar(counts.index, counts.values, color=PALETTE["primary"], edgecolor="white", alpha=0.85)
    ax.set_title("Region Distribution"); ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig("outputs/fig1_univariate.png", bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/fig1_univariate.png")

    #  FIGURE 2 — Bivariate: Age vs Premium & Smoking Impact
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Bivariate Analysis — Key Relationships",
                 fontsize=18, fontweight="bold", color=PALETTE["dark"])

    # Age vs Premium (coloured by plan)
    ax = axes[0, 0]
    for plan, color in PLAN_COLORS.items():
        subset = df[df["plan"] == plan]
        ax.scatter(subset["age"], subset["premium"], c=color, label=plan,
                   alpha=0.55, s=30, edgecolors="none")
    ax.set_title("Age vs. Premium by Plan", fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Premium ($)")
    ax.legend(title="Plan")

    # Smoking vs Premium (box plot)
    ax = axes[0, 1]
    smoking_order = ["No", "Yes"]
    data_plot = [df[df["smoking"] == s]["premium"].values for s in smoking_order]
    bp = ax.boxplot(data_plot, labels=smoking_order, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5))
    bp["boxes"][0].set_facecolor(PALETTE["secondary"])
    bp["boxes"][1].set_facecolor(PALETTE["accent"])
    ax.set_title("Smoking Status vs. Premium", fontweight="bold")
    ax.set_xlabel("Smoker"); ax.set_ylabel("Premium ($)")
    for i, d in enumerate(data_plot):
        ax.text(i + 1, np.percentile(d, 75) + 200,
                f'Median:\n${np.median(d):,.0f}', ha='center', fontsize=9)

    # Dependents vs Plan (violin)
    ax = axes[1, 0]
    plan_order = ["Basic", "Standard", "Premium"]
    for i, plan in enumerate(plan_order):
        data = df[df["plan"] == plan]["dependents"].values.astype(float)
        parts = ax.violinplot(data, positions=[i], showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(list(PLAN_COLORS.values())[i])
            pc.set_alpha(0.75)
    ax.set_xticks(range(3)); ax.set_xticklabels(plan_order)
    ax.set_title("Dependents Distribution by Plan", fontweight="bold")
    ax.set_xlabel("Plan"); ax.set_ylabel("Number of Dependents")

    # Medical History vs Premium
    ax = axes[1, 1]
    med_order = ["None", "Mild", "Chronic"]
    means = [df[df["medical_history"] == m]["premium"].mean() for m in med_order]
    bars = ax.bar(med_order, means,
                  color=[PALETTE["secondary"], PALETTE["warn"], PALETTE["accent"]],
                  edgecolor="white")
    ax.set_title("Avg Premium by Medical History", fontweight="bold")
    ax.set_xlabel("Medical History"); ax.set_ylabel("Avg Premium ($)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'${val:,.0f}', ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig("outputs/fig2_bivariate.png", bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/fig2_bivariate.png")

    #  FIGURE 3 — Correlation Heatmap
    num_df = df[["age", "bmi", "premium", "dependents"]].copy()
    num_df["smoker_bin"]  = (df["smoking"] == "Yes").astype(int)
    num_df["chronic_bin"] = (df["medical_history"] == "Chronic").astype(int)
    num_df["mild_bin"]    = (df["medical_history"] == "Mild").astype(int)
    num_df["male_bin"]    = (df["gender"] == "Male").astype(int)
    plan_map = {"Basic": 0, "Standard": 1, "Premium": 2}
    num_df["plan_num"]    = df["plan"].map(plan_map)

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 10, "weight": "bold"},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap — Numerical & Binary Features",
                 fontsize=15, fontweight="bold", pad=15, color=PALETTE["dark"])
    plt.tight_layout()
    plt.savefig("outputs/fig3_heatmap.png", bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/fig3_heatmap.png")

    #  FIGURE 4 — Deep EDA: Plans, BMI & Regional Analysis
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Deep EDA — Cross-Feature Analysis",
                 fontsize=18, fontweight="bold", color=PALETTE["dark"])

    # BMI by plan
    ax = axes[0, 0]
    order = ["Basic", "Standard", "Premium"]
    for plan, color in PLAN_COLORS.items():
        subset = df[df["plan"] == plan]
        bp = ax.boxplot(subset["bmi"].values, positions=[order.index(plan)],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2))
        bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(order)
    ax.set_title("BMI Distribution by Plan", fontweight="bold")
    ax.set_xlabel("Plan"); ax.set_ylabel("BMI")

    # Smokers vs Non-smokers by plan
    ax = axes[0, 1]
    cross = pd.crosstab(df["plan"], df["smoking"], normalize="index") * 100
    cross = cross.reindex(["Basic", "Standard", "Premium"])
    # Ensure both Yes/No columns exist
    for col in ["No", "Yes"]:
        if col not in cross.columns:
            cross[col] = 0.0
    cross = cross[["No", "Yes"]]
    cross.plot(kind="bar", ax=ax, color=[PALETTE["secondary"], PALETTE["accent"]],
               edgecolor="white", width=0.65)
    ax.set_title("Smoking % Within Each Plan", fontweight="bold")
    ax.set_xlabel("Plan"); ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(["Basic", "Standard", "Premium"], rotation=0)
    ax.legend(title="Smoker")

    # Age brackets vs plan
    ax = axes[0, 2]
    df["age_group"] = pd.cut(df["age"], bins=[17, 30, 45, 60, 70],
                             labels=["18-30", "31-45", "46-60", "61-70"])
    cross2 = pd.crosstab(df["age_group"], df["plan"])
    for col in ["Basic", "Standard", "Premium"]:
        if col not in cross2.columns:
            cross2[col] = 0
    cross2 = cross2[["Basic", "Standard", "Premium"]]
    cross2.plot(kind="bar", ax=ax,
                color=[PLAN_COLORS["Basic"], PLAN_COLORS["Standard"], PLAN_COLORS["Premium"]],
                edgecolor="white", width=0.7)
    ax.set_title("Plan Selection by Age Group", fontweight="bold")
    ax.set_xlabel("Age Group"); ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Plan")

    # Regional premium heatmap
    ax = axes[1, 0]
    region_plan = df.groupby(["region", "plan"])["premium"].mean().unstack()
    for col in ["Basic", "Standard", "Premium"]:
        if col not in region_plan.columns:
            region_plan[col] = np.nan
    region_plan = region_plan[["Basic", "Standard", "Premium"]]
    sns.heatmap(region_plan, annot=True, fmt=".0f", cmap="Blues",
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Avg Premium ($) by Region & Plan", fontweight="bold")

    # BMI vs Premium scatter (coloured by smoking)
    ax = axes[1, 1]
    for smoke_val, color, label in [("No", PALETTE["secondary"], "Non-smoker"),
                                     ("Yes", PALETTE["accent"], "Smoker")]:
        sub = df[df["smoking"] == smoke_val]
        ax.scatter(sub["bmi"], sub["premium"], c=color, label=label,
                   alpha=0.45, s=20, edgecolors="none")
    ax.set_title("BMI vs Premium (by Smoking Status)", fontweight="bold")
    ax.set_xlabel("BMI"); ax.set_ylabel("Premium ($)")
    ax.legend()

    # Dependents vs plan choice
    ax = axes[1, 2]
    df["dep_group"] = pd.cut(df["dependents"], bins=[-1, 0, 1, 2, 5],
                              labels=["0", "1", "2", "3+"])
    cross3 = pd.crosstab(df["dep_group"], df["plan"], normalize="index") * 100
    for col in ["Basic", "Standard", "Premium"]:
        if col not in cross3.columns:
            cross3[col] = 0.0
    cross3 = cross3[["Basic", "Standard", "Premium"]]
    cross3.plot(kind="bar", stacked=True, ax=ax,
                color=[PLAN_COLORS["Basic"], PLAN_COLORS["Standard"], PLAN_COLORS["Premium"]],
                edgecolor="white", width=0.7)
    ax.set_title("Plan Choice by Number of Dependents (%)", fontweight="bold")
    ax.set_xlabel("Dependents"); ax.set_ylabel("Percentage (%)")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Plan", loc="upper left")

    plt.tight_layout()
    plt.savefig("outputs/fig4_deep_eda.png", bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/fig4_deep_eda.png")

    return df


# 4. INSIGHT GENERATION
def print_insights(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  SECTION 4: KEY INSIGHTS FROM EDA")
    print("="*60)

    smoker_avg    = df[df["smoking"] == "Yes"]["premium"].mean()
    nonsmoker_avg = df[df["smoking"] == "No"]["premium"].mean()
    chronic_avg   = df[df["medical_history"] == "Chronic"]["premium"].mean()
    mild_avg      = df[df["medical_history"] == "Mild"]["premium"].mean()
    none_avg      = df[df["medical_history"] == "None"]["premium"].mean()
    obese_avg     = df[df["bmi"] > 30]["premium"].mean()
    normal_avg    = df[df["bmi"] <= 25]["premium"].mean()
    young_avg     = df[df["age"] <= 30]["premium"].mean()
    senior_avg    = df[df["age"] >= 55]["premium"].mean()

    insights = {
        "Age → Premium":
            f"Avg premium for 18-30: ${young_avg:,.0f} "
            f"vs 55+: ${senior_avg:,.0f} "
            f"(+{(senior_avg/young_avg - 1)*100:.0f}%)",

        "Smoking Impact":
            f"Smokers pay ${smoker_avg:,.0f} avg "
            f"vs ${nonsmoker_avg:,.0f} for non-smokers "
            f"(+{(smoker_avg/nonsmoker_avg - 1)*100:.0f}% higher)",

        "Medical History":
            f"Chronic: ${chronic_avg:,.0f} | "
            f"Mild: ${mild_avg:,.0f} | "
            f"None: ${none_avg:,.0f}",

        "BMI Effect":
            f"Obese (BMI>30): ${obese_avg:,.0f} "
            f"vs Normal (BMI≤25): ${normal_avg:,.0f}",

        "Gender & Premium":
            f"Male avg: ${df[df['gender']=='Male']['premium'].mean():,.0f} "
            f"vs Female avg: ${df[df['gender']=='Female']['premium'].mean():,.0f}",

        "Plan Distribution":
            df["plan"].value_counts().to_dict().__str__(),

        "Dataset":
            f"Real Kaggle dataset — {len(df)} records, 0 missing values",
    }
    for k, v in insights.items():
        print(f"\n  ► {k}:\n    {v}")


# 5. ML — DECISION TREE + LOGISTIC REGRESSION
def run_ml(df: pd.DataFrame, df_encoded: pd.DataFrame):
    print("\n" + "="*60)
    print("  SECTION 5: MINIMAL ML (Supportive Role)")
    print("="*60)

    # Features available in the real dataset (income/exercise removed)
    features = ["age", "bmi", "smoking", "medical_history",
                "dependents", "gender", "region"]
    target   = "plan"

    X = df_encoded[features].copy()
    X = X.fillna(X.median())
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    results = {}

    # ── Decision Tree ──
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc  = accuracy_score(y_test, dt_pred)
    results["Decision Tree"] = {"model": dt, "pred": dt_pred, "acc": dt_acc}
    print(f"\n[DT] Decision Tree Accuracy: {dt_acc*100:.2f}%")
    print(classification_report(y_test, dt_pred, target_names=["Basic", "Premium", "Standard"]))

    # ── Logistic Regression ──
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc  = accuracy_score(y_test, lr_pred)
    results["Logistic Regression"] = {"model": lr, "pred": lr_pred, "acc": lr_acc}
    print(f"\n[LR] Logistic Regression Accuracy: {lr_acc*100:.2f}%")
    print(classification_report(y_test, lr_pred, target_names=["Basic", "Premium", "Standard"]))

    #  FIGURE 5 — ML Results
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("ML Model Results — Decision Tree & Logistic Regression",
                 fontsize=16, fontweight="bold", color=PALETTE["dark"])

    # Confusion Matrix — DT
    cm_dt = confusion_matrix(y_test, dt_pred)
    ConfusionMatrixDisplay(cm_dt, display_labels=["Basic", "Premium", "Standard"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Decision Tree\nAccuracy: {dt_acc*100:.1f}%", fontweight="bold")

    # Confusion Matrix — LR
    cm_lr = confusion_matrix(y_test, lr_pred)
    ConfusionMatrixDisplay(cm_lr, display_labels=["Basic", "Premium", "Standard"]).plot(
        ax=axes[1], colorbar=False, cmap="Greens")
    axes[1].set_title(f"Logistic Regression\nAccuracy: {lr_acc*100:.1f}%", fontweight="bold")

    # Feature Importance — Decision Tree
    importances = dt.feature_importances_
    feat_series = pd.Series(importances, index=features).sort_values(ascending=True)
    feat_series.plot(kind="barh", ax=axes[2],
                     color=[PALETTE["primary"] if v > 0.1 else PALETTE["secondary"]
                            for v in feat_series.values])
    axes[2].set_title("Feature Importance (Decision Tree)", fontweight="bold")
    axes[2].set_xlabel("Importance Score")
    for i, (val, name) in enumerate(zip(feat_series.values, feat_series.index)):
        axes[2].text(val + 0.002, i, f"{val:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/fig5_ml_results.png", bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/fig5_ml_results.png")

    #  FIGURE 6 — Decision Tree Visualization
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(dt, feature_names=features, class_names=["Basic", "Premium", "Standard"],
              filled=True, rounded=True, max_depth=3, ax=ax,
              fontsize=9, impurity=False, proportion=True)
    ax.set_title("Decision Tree Visualization (max_depth=3 shown)",
                 fontsize=15, fontweight="bold", color=PALETTE["dark"])
    plt.savefig("outputs/fig6_decision_tree.png", bbox_inches="tight", dpi=120)
    plt.close()
    print("[SAVED] outputs/fig6_decision_tree.png")

    return results, features, X_test, y_test

# 6. RULE-BASED RECOMMENDATION ENGINE
def recommend_plan(user: dict) -> dict:

    score   = 0
    reasons = []

    # Age rules
    if user["age"] >= 60:
        score += 3; reasons.append("Senior age (60+) → higher actuarial risk")
    elif user["age"] >= 45:
        score += 2; reasons.append("Middle-aged (45-59) → moderate risk")
    elif user["age"] >= 30:
        score += 1; reasons.append("Adult (30-44) → baseline risk")

    # Medical history
    if user["medical_history"] == "Chronic":
        score += 4; reasons.append("Chronic conditions → high coverage essential")
    elif user["medical_history"] == "Mild":
        score += 2; reasons.append("Mild medical history → moderate coverage")

    # Smoking (largest single driver in real data — ~3× premium increase)
    if user["smoking"] == "Yes":
        score += 3; reasons.append("Smoker → significantly elevated premium (~3× non-smoker)")

    # BMI
    if user["bmi"] >= 35:
        score += 2; reasons.append("Severely obese BMI (≥35) → elevated health risk")
    elif user["bmi"] >= 30:
        score += 1; reasons.append("Obese BMI (30-34) → moderate risk")

    # Dependents
    if user["dependents"] >= 3:
        score += 1; reasons.append("Multiple dependents → broader family coverage needed")

    # Map score → plan
    if score >= 7:
        plan, confidence = "Premium", "High"
        premium_range    = "$18,000 – $64,000"
        description      = "Comprehensive coverage with lowest out-of-pocket costs"
    elif score >= 4:
        plan, confidence = "Standard", "Medium"
        premium_range    = "$8,000 – $18,000"
        description      = "Balanced coverage suitable for moderate risk profiles"
    else:
        plan, confidence = "Basic", "High"
        premium_range    = "$1,100 – $8,000"
        description      = "Cost-effective plan for low-risk, young & healthy individuals"

    return {
        "recommended_plan":  plan,
        "confidence":        confidence,
        "risk_score":        score,
        "estimated_premium": premium_range,
        "plan_description":  description,
        "reasons":           reasons,
    }


def demo_recommendations():
    print("\n" + "="*60)
    print("  SECTION 6: RECOMMENDATION SYSTEM — DEMO PROFILES")
    print("="*60)

    profiles = [
        {"name": "riya",
         "age": 24, "bmi": 22.1, "smoking": "No",
         "medical_history": "None", "dependents": 0, "gender": "Female"},

        {"name": "Amit Verma",
         "age": 47, "bmi": 29.5, "smoking": "Yes",
         "medical_history": "Mild", "dependents": 2, "gender": "Male"},

        {"name": "Sunita Patel",
         "age": 63, "bmi": 33.2, "smoking": "No",
         "medical_history": "Chronic", "dependents": 1, "gender": "Female"},

        {"name": "Rahul Gupta",
         "age": 35, "bmi": 25.0, "smoking": "No",
         "medical_history": "None", "dependents": 1, "gender": "Male"},
    ]

    for p in profiles:
        rec = recommend_plan(p)
        print(f"\n  ┌─ {p['name']} (Age: {p['age']}, BMI: {p['bmi']}) ────────")
        print(f"  │  Recommended Plan : {rec['recommended_plan']} ({rec['confidence']} confidence)")
        print(f"  │  Risk Score       : {rec['risk_score']}/10")
        print(f"  │  Est. Premium     : {rec['estimated_premium']}")
        print(f"  │  Description      : {rec['plan_description']}")
        print(f"  │  Reasons:")
        for r in rec["reasons"]:
            print(f"  │    • {r}")
        print(f"  └{'─'*55}")

# 7. SUMMARY DASHBOARD
def plot_dashboard(df: pd.DataFrame):
    print("\n[INFO] Generating final dashboard...")

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

    fig.suptitle("Health Insurance Plan Recommendation System — Dashboard\n"
                 "(Kaggle Real Dataset · n=1,338)",
                 fontsize=20, fontweight="bold", color=PALETTE["dark"], y=0.98)

    # KPI row
    smoker_pct  = (df["smoking"] == "Yes").mean() * 100
    chronic_pct = (df["medical_history"] == "Chronic").mean() * 100
    kpis = [
        ("Total Records",  f"{len(df):,}",                     PALETTE["primary"]),
        ("Avg Premium",    f"${df['premium'].mean():,.0f}",     PALETTE["warn"]),
        ("Smoker Rate",    f"{smoker_pct:.1f}%",                PALETTE["accent"]),
        ("Chronic Cases",  f"{chronic_pct:.1f}%",               "#9B59B6"),
    ]
    for i, (label, val, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color)
        ax.text(0.5, 0.55, val, ha='center', va='center', fontsize=26,
                fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.18, label, ha='center', va='center', fontsize=11,
                color='white', transform=ax.transAxes, alpha=0.9)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])

    # Plan donut
    ax1 = fig.add_subplot(gs[1, 0])
    counts = df["plan"].value_counts()
    colors = [PLAN_COLORS.get(p, "#aaa") for p in counts.index]
    wedges, _, autotexts = ax1.pie(
        counts.values, labels=counts.index, autopct="%1.0f%%",
        colors=colors, startangle=140,
        wedgeprops=dict(width=0.55, edgecolor="white"),
        textprops=dict(fontsize=10))
    for at in autotexts: at.set_fontweight("bold")
    ax1.set_title("Plan Distribution", fontweight="bold", fontsize=12)

    # Age vs premium
    ax2 = fig.add_subplot(gs[1, 1])
    for plan, color in PLAN_COLORS.items():
        s = df[df["plan"] == plan]
        ax2.scatter(s["age"], s["premium"], c=color, label=plan,
                    alpha=0.45, s=18, edgecolors="none")
    ax2.set_title("Age vs Premium", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Age"); ax2.set_ylabel("Premium ($)")
    ax2.legend(title="Plan", fontsize=8, title_fontsize=9)

    # Smoking bar
    ax3 = fig.add_subplot(gs[1, 2])
    sm_data = df.groupby(["plan", "smoking"])["premium"].mean().unstack()
    sm_data = sm_data.reindex(["Basic", "Standard", "Premium"])
    for col in ["No", "Yes"]:
        if col not in sm_data.columns:
            sm_data[col] = np.nan
    sm_data = sm_data[["No", "Yes"]]
    sm_data.plot(kind="bar", ax=ax3,
                 color=[PALETTE["secondary"], PALETTE["accent"]],
                 edgecolor="white", width=0.65)
    ax3.set_title("Avg Premium: Smoker vs Non", fontweight="bold", fontsize=12)
    ax3.set_xlabel("Plan"); ax3.set_ylabel("Avg Premium ($)")
    ax3.tick_params(axis='x', rotation=0); ax3.legend(title="Smoker", fontsize=8)

    # Medical history stacked
    ax4 = fig.add_subplot(gs[1, 3])
    med_plan = pd.crosstab(df["medical_history"], df["plan"], normalize="index") * 100
    for col in ["Basic", "Standard", "Premium"]:
        if col not in med_plan.columns:
            med_plan[col] = 0.0
    med_plan = med_plan[["Basic", "Standard", "Premium"]]
    med_plan = med_plan.reindex(["None", "Mild", "Chronic"])
    med_plan.plot(kind="bar", stacked=True, ax=ax4,
                  color=[PLAN_COLORS["Basic"], PLAN_COLORS["Standard"], PLAN_COLORS["Premium"]],
                  edgecolor="white")
    ax4.set_title("Medical History → Plan (%)", fontweight="bold", fontsize=12)
    ax4.set_xlabel("Medical History"); ax4.set_ylabel("%")
    ax4.tick_params(axis='x', rotation=0); ax4.legend(title="Plan", fontsize=8)

    # BMI vs premium scatter (smoker colour)
    ax5 = fig.add_subplot(gs[2, 0:2])
    for smoke_val, color, label in [("No", PALETTE["secondary"], "Non-smoker"),
                                     ("Yes", PALETTE["accent"], "Smoker")]:
        sub = df[df["smoking"] == smoke_val]
        ax5.scatter(sub["bmi"], sub["premium"], c=color, label=label,
                    alpha=0.40, s=15, edgecolors="none")
    ax5.set_title("BMI vs Premium (Real Data — coloured by Smoking)", fontweight="bold", fontsize=12)
    ax5.set_xlabel("BMI"); ax5.set_ylabel("Premium ($)")
    ax5.legend(fontsize=9)

    # Recommendation profiles
    ax6 = fig.add_subplot(gs[2, 2:4])
    profiles_demo = [
        {"name": "Riya (24, Non-smoker)",  "score": 2,  "plan": "Basic"},
        {"name": "Rahul (35, Active)",      "score": 3,  "plan": "Basic"},
        {"name": "Amit (47, Smoker)",       "score": 8,  "plan": "Standard"},
        {"name": "Sunita (63, Chronic)",    "score": 10, "plan": "Premium"},
    ]
    names      = [p["name"]  for p in profiles_demo]
    scores     = [p["score"] for p in profiles_demo]
    plans      = [p["plan"]  for p in profiles_demo]
    bar_colors = [PLAN_COLORS[p] for p in plans]
    bars = ax6.barh(names, scores, color=bar_colors, edgecolor="white", height=0.5)
    ax6.set_title("Sample User Risk Scores → Plan Recommendation",
                  fontweight="bold", fontsize=12)
    ax6.set_xlabel("Risk Score")
    ax6.axvline(4, color=PALETTE["warn"],   linestyle="--", alpha=0.7, label="Standard threshold")
    ax6.axvline(7, color=PALETTE["accent"], linestyle="--", alpha=0.7, label="Premium threshold")
    for bar, plan, score in zip(bars, plans, scores):
        ax6.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                 f"→ {plan}", va='center', fontweight='bold', fontsize=10,
                 color=PLAN_COLORS[plan])
    ax6.legend(fontsize=9); ax6.set_xlim(0, 13)

    plt.savefig("outputs/fig7_dashboard.png", bbox_inches="tight", dpi=130)
    plt.close()
    print("[SAVED] outputs/fig7_dashboard.png")


# 8. EXPORT DASHBOARD JSON
def export_dashboard_json(df: pd.DataFrame, ml_results: dict):
    """
    Export all data needed by the HTML dashboard as a single JSON file.
    The dashboard reads outputs/dashboard_data.json at runtime.
    """
    import json as _json

    print("\n[INFO] Exporting dashboard JSON...")

    # ── KPIs ──
    smoker_pct  = round((df["smoking"] == "Yes").mean() * 100, 1)
    chronic_pct = round((df["medical_history"] == "Chronic").mean() * 100, 1)
    plan_vc     = df["plan"].value_counts()
    total       = len(df)

    # ── Plan distribution ──
    plan_dist = {p: int(plan_vc.get(p, 0)) for p in ["Basic", "Standard", "Premium"]}
    plan_pct  = {p: round(plan_dist[p] / total * 100, 1) for p in plan_dist}

    # ── Avg premium by medical history ──
    med_avgs = {m: round(float(df[df["medical_history"] == m]["premium"].mean()), 0)
                for m in ["None", "Mild", "Chronic"]}

    # ── Smoking stats ──
    smoker_avg    = round(float(df[df["smoking"] == "Yes"]["premium"].mean()), 0)
    nonsmoker_avg = round(float(df[df["smoking"] == "No"]["premium"].mean()), 0)
    smoker_uplift = round((smoker_avg / nonsmoker_avg - 1) * 100, 0)

    # ── Gender split ──
    gender_vc = df["gender"].value_counts()
    gender_dist = {g: int(gender_vc.get(g, 0)) for g in ["Male", "Female"]}

    # ── Age group distribution ──
    df["age_group"] = pd.cut(df["age"], bins=[17, 30, 45, 60, 64],
                             labels=["18-30", "31-45", "46-60", "61-64"])
    age_group_avgs = {
        ag: round(float(df[df["age_group"] == ag]["premium"].mean()), 0)
        for ag in ["18-30", "31-45", "46-60", "61-64"]
    }
    age_group_plan = {}
    for ag in ["18-30", "31-45", "46-60", "61-64"]:
        sub = df[df["age_group"] == ag]
        age_group_plan[ag] = {p: int((sub["plan"] == p).sum())
                               for p in ["Basic", "Standard", "Premium"]}

    # ── Correlation matrix (real) ──
    num_df = df[["age", "bmi", "premium", "dependents"]].copy()
    num_df["smoker_bin"]  = (df["smoking"] == "Yes").astype(int)
    num_df["chronic_bin"] = (df["medical_history"] == "Chronic").astype(int)
    num_df["mild_bin"]    = (df["medical_history"] == "Mild").astype(int)
    corr = num_df.corr().round(3)
    corr_features = list(corr.columns)
    corr_matrix   = corr.values.round(3).tolist()

    # ── Feature importances (from DT) ──
    dt_model     = ml_results["Decision Tree"]["model"]
    features_ml  = ["age", "bmi", "smoking", "medical_history",
                     "dependents", "gender", "region"]
    feat_imp     = [{"name": f, "value": round(float(v), 3)}
                    for f, v in zip(features_ml, dt_model.feature_importances_)]
    feat_imp.sort(key=lambda x: x["value"], reverse=True)

    # ── ML accuracies ──
    dt_acc = round(ml_results["Decision Tree"]["acc"] * 100, 2)
    lr_acc = round(ml_results["Logistic Regression"]["acc"] * 100, 2)

    # ── Age distribution histogram ──
    age_bins  = list(range(18, 65))
    age_freqs = [int((df["age"] == a).sum()) for a in age_bins]

    # ── Premium histogram ──
    bucket_size = 4000
    prem_buckets = list(range(0, 68000, bucket_size))
    prem_labels  = [f"${int(b/1000)}k" for b in prem_buckets]
    prem_freqs   = [int(((df["premium"] >= b) & (df["premium"] < b + bucket_size)).sum())
                    for b in prem_buckets]

    # ── Scatter (age vs premium by plan, 200 pts each) ──
    scatter = {}
    for p in ["Basic", "Standard", "Premium"]:
        sub = df[df["plan"] == p][["age", "premium"]].head(200)
        scatter[p] = [{"x": int(r.age), "y": round(float(r.premium), 0)}
                      for _, r in sub.iterrows()]

    # ── BMI vs premium scatter (by smoking, 200 pts each) ──
    bmi_scatter = {}
    for s in ["Yes", "No"]:
        sub = df[df["smoking"] == s][["bmi", "premium"]].head(200)
        bmi_scatter[s] = [{"x": round(float(r.bmi), 1), "y": round(float(r.premium), 0)}
                           for _, r in sub.iterrows()]

    # ── Smoker vs non-smoker avg premium by plan ──
    smoker_by_plan = {}
    for p in ["Basic", "Standard", "Premium"]:
        sub = df[df["plan"] == p]
        smoker_by_plan[p] = {
            "No":  round(float(sub[sub["smoking"] == "No"]["premium"].mean()), 0)
                   if len(sub[sub["smoking"] == "No"]) > 0 else 0,
            "Yes": round(float(sub[sub["smoking"] == "Yes"]["premium"].mean()), 0)
                   if len(sub[sub["smoking"] == "Yes"]) > 0 else 0,
        }

    # ── Regional avg premium ──
    region_avg = {}
    for reg in df["region"].dropna().unique():
        region_avg[reg] = round(float(df[df["region"] == reg]["premium"].mean()), 0)

    payload = {
        "meta": {
            "total_records": total,
            "dataset": "Kaggle Medical Insurance Dataset (insurance.csv)",
            "smoker_pct": smoker_pct,
            "chronic_pct": chronic_pct,
            "avg_premium": round(float(df["premium"].mean()), 0),
        },
        "plan_dist":       plan_dist,
        "plan_pct":        plan_pct,
        "med_avgs":        med_avgs,
        "smoker_avg":      smoker_avg,
        "nonsmoker_avg":   nonsmoker_avg,
        "smoker_uplift":   smoker_uplift,
        "gender_dist":     gender_dist,
        "age_group_avgs":  age_group_avgs,
        "age_group_plan":  age_group_plan,
        "corr_features":   corr_features,
        "corr_matrix":     corr_matrix,
        "feat_importances": feat_imp,
        "dt_accuracy":     dt_acc,
        "lr_accuracy":     lr_acc,
        "age_bins":        age_bins,
        "age_freqs":       age_freqs,
        "prem_labels":     prem_labels,
        "prem_freqs":      prem_freqs,
        "scatter":         scatter,
        "bmi_scatter":     bmi_scatter,
        "smoker_by_plan":  smoker_by_plan,
        "region_avg":      region_avg,
    }

    out_path = "outputs/dashboard_data.json"
    with open(out_path, "w") as fh:
        _json.dump(payload, fh, indent=2)
    print(f"[SAVED] {out_path}  ({os.path.getsize(out_path):,} bytes)")


# MAIN EXECUTION
def main():
    print("\n" + "█"*62)
    print("  SMART HEALTH INSURANCE PLAN RECOMMENDATION SYSTEM")
    print("  Batch F7 | Data Analysis Project 2026")
    print("  Dataset: Kaggle Medical Insurance Dataset (Real Data)")
    print("█"*62)

    # 1. Load Real Dataset
    df_raw = load_dataset("insurance.csv")

    # 2. Preprocess
    df, df_encoded, df_scaled, encoders = preprocess(df_raw.copy())

    # 3. EDA
    df = eda(df)

    # 4. Insights
    print_insights(df)

    # 5. ML
    ml_results, features, X_test, y_test = run_ml(df, df_encoded)

    # 6. Recommendation Demo
    demo_recommendations()

    # 7. Dashboard
    plot_dashboard(df)

    # 8. Save processed dataset
    df.to_csv("outputs/health_insurance_processed.csv", index=False)
    print("\n[SAVED] outputs/health_insurance_processed.csv")

    # 9. Export dashboard JSON (powers the HTML dashboard)
    export_dashboard_json(df, ml_results)

    print("\n" + "="*62)
    print("  PROJECT COMPLETE!")
    print("  All outputs saved to: ./outputs/")
    print("  Files:")
    for f in sorted(os.listdir("outputs")):
        print(f"    • {f}")
    print("="*62)

    # 10. Launch HTML Dashboard
    _launch_dashboard()


def _launch_dashboard():
    """
    Locate health_insurance_full_dashboard_v3.html relative to this script
    and open it in the system's default web browser automatically.
    """
    import pathlib

    script_dir         = pathlib.Path(__file__).resolve().parent
    dashboard_filename = "health_insurance_dashboard_final.html"

    search_paths = [
        script_dir / dashboard_filename,
        script_dir / "outputs" / dashboard_filename,
        pathlib.Path.cwd() / dashboard_filename,
        pathlib.Path.cwd() / "outputs" / dashboard_filename,
    ]

    dashboard_path = None
    for p in search_paths:
        if p.exists():
            dashboard_path = p
            break

    if dashboard_path is None:
        print(f"\n[WARNING] Dashboard HTML file not found.")
        print(f"  Please place '{dashboard_filename}' in the same folder as this script.")
        return

    dest = script_dir / "outputs" / dashboard_filename
    if dashboard_path != dest:
        shutil.copy2(dashboard_path, dest)
        print(f"\n[INFO] Dashboard copied to: outputs/{dashboard_filename}")

    url = dest.as_uri()
    print(f"\n{'='*62}")
    print(f"  OPENING DASHBOARD IN BROWSER...")
    print(f"  {url}")
    print(f"{'='*62}\n")
    webbrowser.open(url)


if __name__ == "__main__":
    main()
