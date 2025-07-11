import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
np.random.seed(42)
n = 614
data = {
    "SleepHours": np.random.normal(6.5, 1.5, n).clip(0, 12),
    "ScreenTime": np.random.normal(5, 2, n).clip(0, 12),
    "StressLevel": np.random.randint(1, 6, n),
    "SocialActivity": np.random.randint(0, 8, n),
    "Grades": np.random.choice(["A", "B", "C", "D", "F"], size=n, p=[0.25, 0.3, 0.25, 0.15, 0.05])
}
df = pd.DataFrame(data)
df["MentalHealthRisk"] = ((df["SleepHours"] < 5) & (df["StressLevel"] > 3) & (df["ScreenTime"] > 6)).astype(int)
df["GradesEncoded"] = df["Grades"].map({"A": 4, "B": 3, "C": 2, "D": 1, "F": 0})

# Split and train
X = df[["SleepHours", "ScreenTime", "StressLevel", "SocialActivity", "GradesEncoded"]]
y = df["MentalHealthRisk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open("mental_health_model.pkl", "wb"))
print("✅ Model saved as mental_health_model.pkl")
