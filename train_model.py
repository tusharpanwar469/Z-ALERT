import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "data/survival_data.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("Z-ALERT SURVIVAL MODEL TRAINING")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

features = [
    "age",
    "health_score",
    "injury_severity",
    "infection_exposure",
    "food_days",
    "water_days",
    "shelter_quality",
    "medical_supplies",
    "weapon_availability",
    "group_size",
    "distance_to_safe_zone",
    "mobility_score",
    "days_since_outbreak"
]

target = "survived"


X = df[features]
y = df[target]


print(f"\nFeatures: {len(features)}")
print(f"Target: {target}")


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAIN / TEST SPLIT")
print("-" * 30)
print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training complete!")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. MODEL ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 30)
print(f"Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT")
print("-" * 30)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Did Not Survive",
            "Survived"
        ]
    )
)


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

print("\nCONFUSION MATRIX")
print("-" * 30)

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 30)

print(importance.to_string(index=False))


# ==========================================
# 11. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model_path = "models/survival_model.pkl"

joblib.dump(model, model_path)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(f"Saved model: {model_path}")