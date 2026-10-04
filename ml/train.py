import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["order_count_30d", "avg_order_value_30d", "days_since_last_order", "view_count_7d"]

data = pd.read_csv("data/training_set.csv", parse_dates=["event_timestamp"])

# Split by time: learn from February to June, check on July and August.
# The test months come after the training months, as they would in real use.
train = data[data["event_timestamp"] < "2026-07-01"]
test = data[data["event_timestamp"] >= "2026-07-01"]
print(f"train rows: {len(train)}, test rows: {len(test)}")

# X = the inputs (features), y = the true answers (1 = bought within 30 days)
X_train, y_train = train[FEATURES], train["label"]
X_test, y_test = test[FEATURES], test["label"]


def report(name, predictions, scores=None):
    """Print how well a set of predictions matches the true answers."""
    print(f"\n{name}")
    # Of all predictions, how many were right
    print(f"  accuracy : {accuracy_score(y_test, predictions):.3f}")
    # Of the users predicted to buy, how many did (0 if nobody was predicted to buy)
    print(f"  precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    # Of the users who did buy, how many were predicted to
    print(f"  recall   : {recall_score(y_test, predictions):.3f}")
    if scores is not None:
        # How well the scores rank buyers above non-buyers: 0.5 = random, 1.0 = perfect
        print(f"  ROC AUC  : {roc_auc_score(y_test, scores):.3f}")


# Baseline 1: always predict "won't buy"
always_no = [0] * len(test)
report("Baseline 1: always no", always_no)

# Baseline 2: predict "will buy" if the user ordered in the last 30 days.
# The comparison gives True/False per row; astype(int) turns that into 1/0.
# As a score for ranking users, the order count itself is used.
ordered_recently = (test["order_count_30d"] > 0).astype(int)
report("Baseline 2: ordered in the last 30 days", ordered_recently, test["order_count_30d"])

# Model 1: logistic regression. A pipeline runs its steps in order, both when
# learning and when predicting, so raw feature values can go straight in.
model = make_pipeline(
    # Fill missing values with the column's median from the training data, and
    # add a 0/1 column per feature saying "this value was missing".
    SimpleImputer(strategy="median", add_indicator=True),
    # Put every column on the same scale (mean 0, spread 1), so a feature
    # measured in days does not outweigh one measured in counts.
    StandardScaler(),
    # Learn one weight per column; the weighted sum becomes a probability.
    LogisticRegression(),
)
model.fit(X_train, y_train)

# predict_proba gives [P(no), P(yes)] per row; column 1 is the probability of buying.
# predict turns that into 1 when the probability is above 0.5, otherwise 0.
probabilities = model.predict_proba(X_test)[:, 1]
report("Model 1: logistic regression", model.predict(X_test), probabilities)

# What the model learned: one weight per column. Positive pushes towards
# "will buy", negative towards "won't buy"; larger means stronger.
print("\nWeights")
columns = model[:-1].get_feature_names_out()
weights = model[-1].coef_[0]
for column, weight in sorted(zip(columns, weights, strict=True), key=lambda pair: -abs(pair[1])):
    print(f"  {column:<45} {weight:+.2f}")
