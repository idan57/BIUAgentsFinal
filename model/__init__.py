import os
from collections import Counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

COLS_TO_ENCODE = [
    "Attrition",
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "Over18",
    "OverTime"
]


class HREmployeeAttritionModel(object):
    def __init__(self):
        self._number_of_employees = 0
        self._hr_retention_data = self._get_hr_retention_data()

        self.classifiers = {
            "RandomForestClassifier_Gini": RandomForestClassifier(n_estimators=500, criterion="gini"),
            "RandomForestClassifier_Entropy": RandomForestClassifier(n_estimators=500, criterion="entropy"),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, loss='ls'),
            "BaggingClassifier": BaggingClassifier(n_estimators=1000),
            "XGBClassifier": XGBClassifier(n_estimators=1000, max_depth=3)
        }

        fitted_data = self._get_fitted_data()
        x_cols = [col for col in self._hr_retention_data.columns if col != "Attrition"]
        x = fitted_data[x_cols]
        y = fitted_data["Attrition"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=999)

    def _get_hr_retention_data(self):
        path_to_data = os.path.join(os.path.dirname(__file__), "data", "HR-Employee-Attrition.csv")
        data = pd.read_csv(path_to_data).fillna(0)
        self._number_of_employees = len(data)
        return data

    def _get_fitted_data(self):
        encoder = LabelEncoder()
        data = self._hr_retention_data.copy(deep=True)

        for col in COLS_TO_ENCODE:
            data[col] = encoder.fit_transform(data[col].astype(str))

        return data

    def train(self):
        for model_name, model in self.classifiers.items():
            print(f"Training model: {model_name}")
            model.fit(self.x_train, self.y_train)
            print(f"Training is done")
            self.log_train_res(model, model_name)

    def predict(self, data):
        predictions = []
        for model_name, model in self.classifiers.items():
            print(f"Predicting with model: {model_name}")
            prediction = model.predict(data)
            predictions.append(prediction)
            print(f"Predicted: {prediction}")

        counted_predictions = Counter(predictions)
        top_prediction = max(counted_predictions, key=counted_predictions.get)
        return top_prediction

    def log_train_res(self, model, model_name):
        print(f"Train accuracy for {model_name}: "
              f"{accuracy_score(self.y_train, model.predict(self.x_train))}")

        print(f"Test accuracy for {model_name}: "
              f"{accuracy_score(self.y_test, model.predict(self.x_test))}")
