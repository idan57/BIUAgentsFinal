import os
from collections import Counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

COLS_TO_ENCODE = {
    "Attrition": {},
    "BusinessTravel": {},
    "Department": {},
    "EducationField": {},
    "Gender": {},
    "JobRole": {},
    "MaritalStatus": {},
    "Over18": {},
    "OverTime": {}
}


class HREmployeeAttritionModel(object):
    """
    Class that will train over the HR data
    """
    COLS_INDEXES = {}

    def __init__(self):
        self._number_of_employees = 0
        self.hr_retention_data = self._get_hr_retention_data()
        self.x_cols = []
        self.reversed_mappings = {}

        self.classifiers = {
            "RandomForestClassifier_Gini": RandomForestClassifier(n_estimators=500, criterion="gini"),
            "RandomForestClassifier_Entropy": RandomForestClassifier(n_estimators=500, criterion="entropy"),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "BaggingClassifier": BaggingClassifier(n_estimators=1000),
            "XGBClassifier": XGBClassifier(n_estimators=1000, max_depth=3)
        }
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def _get_hr_retention_data(self):
        path_to_data = os.path.join(os.path.dirname(__file__), "data", "HR-Employee-Attrition.csv")
        data = pd.read_csv(path_to_data).fillna(0)
        self._number_of_employees = len(data)
        return data

    def _get_fitted_data(self):
        data = self.hr_retention_data.copy(deep=True)

        for col in self.hr_retention_data.columns:
            col_data = data[col].astype(str)
            col_data = set(col_data.tolist())
            reverse_mapping = {i: val for val, i in zip(col_data, range(len(col_data)))}
            self.reversed_mappings[col] = reverse_mapping

        for col in COLS_TO_ENCODE:
            if col in data.columns:
                col_data = data[col].astype(str)
                col_data = set(col_data.tolist())
                print(f"{col}: {col_data}")
                mapping = {val: i for val, i in zip(col_data, range(len(col_data)))}
                COLS_TO_ENCODE[col] = mapping
                data[col] = data[col].map(mapping)

        return data

    def train(self):
        """
        Train the model
        """
        HREmployeeAttritionModel.COLS_INDEXES = {col: 0
                                                 for col in self.hr_retention_data.columns
                                                 if col != "Attrition"}
        found_attrition = False
        for col in self.hr_retention_data.columns:
            if col == "Attrition":
                found_attrition = True
                continue

            if col in HREmployeeAttritionModel.COLS_INDEXES and not found_attrition:
                HREmployeeAttritionModel.COLS_INDEXES[col] = list(self.hr_retention_data.columns).index(col)
            elif col in HREmployeeAttritionModel.COLS_INDEXES and found_attrition:
                HREmployeeAttritionModel.COLS_INDEXES[col] = list(self.hr_retention_data.columns).index(col) - 1

        self.x_cols = [col for col in self.hr_retention_data.columns if col != "Attrition"]
        fitted_data = self._get_fitted_data()
        x = fitted_data[self.x_cols]
        y = fitted_data["Attrition"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=999)
        for model_name, model in self.classifiers.items():
            print(f"Training model: {model_name}")
            model.fit(self.x_train, self.y_train)
            print(f"Training is done")
            self.log_train_res(model, model_name)

    def predict(self, data):
        """
        Predict from given data
        """
        predictions = []
        end_predictions = []
        for model_name, model in self.classifiers.items():
            print(f"Predicting with model: {model_name}")
            prediction = model.predict(data)
            predictions.append(prediction)
            print(f"Predicted: {prediction}")

        num_of_classifiers = len(self.classifiers)

        for i in range(len(predictions[0])):
            preds = [predictions[j][i] for j in range(num_of_classifiers)]
            counted_predictions = Counter(preds)
            top_prediction = max(counted_predictions, key=counted_predictions.get)
            end_predictions.append(top_prediction)
        return end_predictions

    def log_train_res(self, model, model_name):
        """
        Log training data
        """
        print(f"Train accuracy for {model_name}: "
              f"{accuracy_score(self.y_train, model.predict(self.x_train))}")

        print(f"Test accuracy for {model_name}: "
              f"{accuracy_score(self.y_test, model.predict(self.x_test))}")

    @staticmethod
    def fit_data(vals: dict):
        """
        Fit given data
        """
        result = []
        for key, val in vals.items():
            if key in COLS_TO_ENCODE:
                result.append(COLS_TO_ENCODE[key][val])
            else:
                result.append(float(val))

        return result
