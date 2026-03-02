import pandas as pd

# Load dataset
df = pd.read_csv("data/placement.csv")

# Drop unnecessary columns if present
df = df.dropna()

# Convert placement to numeric (if needed)
if 'placement_status' in df.columns:
    df['placement_status'] = df['placement_status'].map({'Placed': 1, 'Not Placed': 0})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Save processed data
df.to_csv("data/processed.csv", index=False)

print("✅ Preprocessing done!")