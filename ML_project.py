import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Adjust the path to wherever you placed the CSV file
file_path = "DBtrainrides.csv"  # or e.g., "data/DBtrainrides.csv"

# Try reading with UTF-8 first; if it fails, use latin1
try:
    df = pd.read_csv(file_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="latin1")

# Optional: parse datetime columns
df["departure_plan"] = pd.to_datetime(df["departure_plan"], errors="coerce")
df["arrival_plan"] = pd.to_datetime(df["arrival_plan"], errors="coerce")
df["departure_change"] = pd.to_datetime(df["departure_change"], errors="coerce")
df["arrival_change"] = pd.to_datetime(df["arrival_change"], errors="coerce")

# Display the first few rows of the dataset
print(df.head())
# Display the shape of the dataset
print(f"Dataset shape: {df.shape}")
# Display the columns of the dataset
print(f"Dataset columns: {df.columns.tolist()}")
# Display the data types of the columns
print(f"Dataset dtypes:\n{df.dtypes}")
# Display basic statistics of the dataset
print(f"Dataset statistics:\n{df.describe(include='all')}")
# Display the number of missing values in each column
print(f"Missing values:\n{df.isnull().sum()}")


# Preprocessing before splitting the dataset: removing duplicates, dropping columns.
# Making a "work" df variable in case the original dataset was needed
df_work = df.copy()

# Define the columns to keep
keep_cols = [
    "zip",
    "category",
    "arrival_plan",
    "departure_plan",
    "arrival_change",
    "departure_change",
    "arrival_delay_m"
]

# Subset to only those columns
df_work = df_work[keep_cols]

# Quick sanity check
print("Kept columns:", df_work.columns.tolist())
print("Shape before dropping dupes:", df_work.shape)

# Drop duplicate rows
df_work = df_work.drop_duplicates()
print("Shape after dropping dupes:", df_work.shape)


# Splitting the data into train, test, validation.
X = df_work.drop(columns="arrival_delay_m")
y = df_work["arrival_delay_m"]

# split into train / tmp 80%, then tmp into val 20% & train 80% (→ 64/16/20) —
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.20, random_state=42
)

# “Safe” feature engineering on each split
def add_time_and_delta(df):
    out = df.copy()
    out["arr_hour"]    = out["arrival_plan"].dt.hour
    out["arr_weekday"] = out["arrival_plan"].dt.weekday
    out["dep_hour"]    = out["departure_plan"].dt.hour
    out["dep_weekday"] = out["departure_plan"].dt.weekday
    # Replaces the arrival planned and change time to delta which is an integer that represents how much the delay was. It also replaces the NAN from "change" columns to zero.
    out["arr_change_delta"] = (
        (out["arrival_change"] - out["arrival_plan"])
        .dt.total_seconds() / 60
    ).fillna(0)
    out["dep_change_delta"] = (
        (out["departure_change"] - out["departure_plan"])
        .dt.total_seconds() / 60
    ).fillna(0)
    return out

X_train = add_time_and_delta(X_train)
X_val   = add_time_and_delta(X_val)
X_test  = add_time_and_delta(X_test)

# Step to drop the original columns which are currently not needed after applying the function.
to_drop = [
    "arrival_plan", "departure_plan",
    "arrival_change", "departure_change"
]

X_train = X_train.drop(columns=to_drop)
X_val   = X_val.drop(columns=to_drop)
X_test  = X_test.drop(columns=to_drop)

print("Training features now:", X_train.columns.tolist())

# Normalizing and encoding the features
NUM_COLS = ["arr_hour", "arr_weekday", "dep_hour", "dep_weekday",
            "arr_change_delta", "dep_change_delta"]
CAT_COLS = ["zip", "category"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
])

# TBD: Comparing models and chosing the best, feature engineering.

'''
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

models = {
    "SGD":      SGDRegressor(random_state=42),
    "Linear":   LinearRegression(),
    "KNN":      KNeighborsRegressor(),
    "RF":       RandomForestRegressor(random_state=42),
}

from sklearn.model_selection import cross_val_score
import numpy as np

results = {}
for name, mdl in models.items():
    pipe = Pipeline([("pre", preprocessor), ("reg", mdl)])
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=5, scoring="neg_mean_squared_error")
    results[name] = np.mean(-scores)

#Pick the model with the lowest training MSE
print("CV MSE:", results)

# Feature selector
from sklearn.feature_selection import SequentialFeatureSelector

pipe = Pipeline([("pre", preprocessor), ("reg", best_model)])
selector = SequentialFeatureSelector(pipe, 
             n_features_to_select="auto", direction="forward",
             scoring="neg_mean_squared_error", cv=5)
selector.fit(X_train, y_train)

# Retrain the chosen pipeline
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
final_pipe = Pipeline([
    ("pre", preprocessor),
    ("feat_sel", selector),      # if used
    ("reg", best_model)
])
final_pipe.fit(X_tv, y_tv)

# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
y_pred = final_pipe.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R²:",  r2_score(y_test, y_pred))

'''