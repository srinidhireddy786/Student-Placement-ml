import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, GradientBoostingRegressor
import joblib

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("data/processed.csv")

# ---------------- CLASSIFICATION ---------------- #
y_class = df['placement_status']
X = df.drop(['placement_status', 'salary'], axis=1)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_c, y_train_c)

# ---------------- REGRESSION (ONLY PLACED) ---------------- #
placed_df = df[df['placement_status'] == 1]

usd_to_inr = 83

salary_lpa = (placed_df['salary'] * usd_to_inr) / 100000

# Boost logic (important for realistic output)
salary_lpa += placed_df.get('college_tier_Tier 1', 0) * 2.5
salary_lpa += placed_df.get('dsa_score', 0) * 1.0
salary_lpa += placed_df.get('dsa_score', 0) * 0.02

# Optional: internships impact
salary_lpa += placed_df.get('internships', 0) * 0.3

# -------- FINAL NORMALIZATION -------- #

# Clip to realistic range
salary_lpa = salary_lpa.clip(lower=2, upper=25)

y_reg = salary_lpa

X_reg = placed_df.drop(['placement_status', 'salary'], axis=1)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
reg.fit(X_train_r, y_train_r)
# ---------------- SAVE ---------------- #
joblib.dump(clf, "models/placement_model.pkl")
joblib.dump(reg, "models/salary_model.pkl")
joblib.dump(X.columns, "models/features.pkl")

print("✅ Models trained successfully!")