import numpy as np
from sklearn.model_selection import train_test_split
from fuzzy_knn import FuzzyKNN
from sklearn.metrics import accuracy_score
import xgboost as xgb

X = np.random.rand(100, 128)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = FuzzyKNN(k=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("FuzzyKNN Accuracy:", accuracy_score(y_test, preds))

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
