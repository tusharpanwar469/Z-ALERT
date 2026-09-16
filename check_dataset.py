import pandas as pd


# Load dataset
file_path = "data/survival_data.csv"
df = pd.read_csv(file_path)


print("=" * 60)
print("Z-ALERT DATASET VALIDATION")
print("=" * 60)


# 1. Dataset size
print("\n1. DATASET SIZE")
print("-" * 30)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# 2. Column names
print("\n2. COLUMNS")
print("-" * 30)

for column in df.columns:
    print("-", column)


# 3. Missing values
print("\n3. MISSING VALUES")
print("-" * 30)

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("STATUS: No missing values")


# 4. Duplicate rows
print("\n4. DUPLICATE ROWS")
print("-" * 30)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)


# 5. Survival distribution
print("\n5. SURVIVAL DISTRIBUTION")
print("-" * 30)

print(df["survived"].value_counts())

print("\nPercentage:")
print(df["survived"].value_counts(normalize=True) * 100)


# 6. Basic statistics
print("\n6. BASIC STATISTICS")
print("-" * 30)

print(df.describe())


# 7. Target validation
print("\n7. TARGET VALIDATION")
print("-" * 30)

print("Target column: survived")
print("Unique values:", sorted(df["survived"].unique()))

if set(df["survived"].unique()).issubset({0, 1}):
    print("STATUS: Target is valid")


# Final status
print("\n" + "=" * 60)
print("DATASET VALIDATION COMPLETE")
print("=" * 60)