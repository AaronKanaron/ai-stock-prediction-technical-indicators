import pandas as pd
import numpy as np

# Ladda data
df = pd.read_csv('data/OMXS30.csv')

# 1. Kontrollera missing values
print("Missing values:")
print(df.isnull().sum())

# 2. Kontrollera Target-fördelning
print(f"\nTarget distribution:")
print(df['Target'].value_counts())
print(f"Target balance: {df['Target'].mean():.2%}")
