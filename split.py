import pandas as pd
from sklearn.model_selection import train_test_split

# ✅ Load the Excel file
df = pd.read_csv(r"C:\Users\cn230\OneDrive\Desktop\Final Project\Python files\adverse_drug_events_trained_datasets.csv")  # Update path if needed

# ✅ Split into train (80%) and test (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ✅ Save to CSV
train_df.to_csv(r"C:\Users\cn230\OneDrive\Desktop\Final Project\train.csv", index=False)
test_df.to_csv(r"C:\Users\cn230\OneDrive\Desktop\Final Project\test.csv", index=False)

print("✅ train.csv and test.csv saved successfully!")
