import os
import numpy as np
import pandas as pd


# Make results reproducible
np.random.seed(42)

# Number of fictional survivors
N = 5000


# -----------------------------
# Generate fictional features
# -----------------------------

age = np.random.randint(16, 81, N)

health_score = np.random.randint(20, 101, N)

injury_severity = np.random.randint(0, 11, N)

infection_exposure = np.random.randint(0, 101, N)

food_days = np.random.randint(0, 31, N)

water_days = np.random.randint(0, 31, N)

shelter_quality = np.random.randint(0, 101, N)

medical_supplies = np.random.randint(0, 101, N)

weapon_availability = np.random.randint(0, 101, N)

group_size = np.random.randint(1, 11, N)

distance_to_safe_zone = np.random.randint(1, 101, N)

mobility_score = np.random.randint(20, 101, N)

days_since_outbreak = np.random.randint(1, 61, N)


# -----------------------------
# Calculate fictional survival
# score
# -----------------------------

score = (
    0.030 * health_score
    + 0.025 * food_days
    + 0.025 * water_days
    + 0.020 * shelter_quality
    + 0.018 * medical_supplies
    + 0.012 * weapon_availability
    + 0.015 * mobility_score
    + 0.008 * group_size
    - 0.020 * injury_severity
    - 0.015 * infection_exposure
    - 0.012 * distance_to_safe_zone
    - 0.008 * days_since_outbreak
    - 0.004 * age
)


# Convert score into probability
probability = 1 / (1 + np.exp(-score + 4.5))


# Add randomness so the model does not learn a perfect rule
survived = np.random.binomial(1, probability)


# -----------------------------
# Create DataFrame
# -----------------------------

data = pd.DataFrame({
    "age": age,
    "health_score": health_score,
    "injury_severity": injury_severity,
    "infection_exposure": infection_exposure,
    "food_days": food_days,
    "water_days": water_days,
    "shelter_quality": shelter_quality,
    "medical_supplies": medical_supplies,
    "weapon_availability": weapon_availability,
    "group_size": group_size,
    "distance_to_safe_zone": distance_to_safe_zone,
    "mobility_score": mobility_score,
    "days_since_outbreak": days_since_outbreak,
    "survived": survived
})


# -----------------------------
# Save dataset
# -----------------------------

os.makedirs("data", exist_ok=True)

file_path = "data/survival_data.csv"

data.to_csv(file_path, index=False)


# -----------------------------
# Display information
# -----------------------------

print("=" * 50)
print("Z-ALERT SURVIVAL DATASET CREATED")
print("=" * 50)

print(f"Records created : {len(data)}")
print(f"Columns         : {len(data.columns)}")
print(f"Saved to        : {file_path}")

print("\nSurvival distribution:")
print(data["survived"].value_counts())

print("\nFirst 5 records:")
print(data.head())

print("\nDataset information:")
print(data.info())