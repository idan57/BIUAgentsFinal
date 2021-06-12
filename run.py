import os
from copy import deepcopy

from model import HREmployeeAttritionModel
import matplotlib.pyplot as plt

plt.ioff()


def get_age_index_to_inc(age):
    if age <= 30:
        return 0
    elif 30 < age < 40:
        return 1
    else:
        return 2


def get_monthly_index_to_inc(monthly_income):
    if 1000 <= monthly_income < 2500:
        return 0
    if 2500 <= monthly_income < 5000:
        return 1
    elif 5000 <= monthly_income < 7500:
        return 2
    elif 7500 <= monthly_income < 10000:
        return 3
    elif 10000 <= monthly_income < 12500:
        return 4
    elif 12500 <= monthly_income < 15000:
        return 5
    elif 15000 <= monthly_income < 17500:
        return 6
    else:
        return 7


def get_hourly_rate_index_to_inc(hourly_rate):
    if 30 <= hourly_rate < 40:
        return 0
    if 40 <= hourly_rate < 50:
        return 1
    if 50 <= hourly_rate < 60:
        return 2
    if 60 <= hourly_rate < 70:
        return 3
    if 70 <= hourly_rate < 80:
        return 4
    if 80 <= hourly_rate < 90:
        return 5
    else:
        return 6


def save_figures(cols_for_plotting, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

    for key, val in cols_for_plotting.items():
        print(f"Saving figure: {key}")
        fig = plt.figure(figsize=(20, 10))

        # creating the bar plot
        plt.bar(val["langs"], val["vals"], color='maroon',
                width=0.4)

        plt.ylabel(f"{key} Values")
        plt.title(key)

        fig.savefig(os.path.join(directory, f"{key}.png"), dpi=fig.dpi)
        plt.close()


def visualize_classifier(model, directory):
    predictions = model.predict(model.x_test)
    mistakes = []
    cols_for_plotting = {
        "Age": {
            "langs": ["Age <= 30", "30 < Age < 40", "Age >= 40"],
            "vals": [0] * 3
        },
        "HourlyRate": {
            "langs": [f"{30 + i * 10} <= Hourly Rate < {30 + (i + 1) * 10}" for i in range(6)] +
                     ["90 <= Hourly Rate <= 100"],
            "vals": [0] * 7
        },
        "MonthlyIncome": {
            "langs": ["1000 - 2500", "2500 - 5000", "5000 - 7500", "7500 - 10000", "10000 - 12500", "12500 - 15000",
                      "15000 - 17500", "17500 - 20000"],
            "vals": [0] * 8
        },
    }
    not_needed_to_add = ["EmployeeNumber", "Age", "DailyRate", "EmployeeCount", "MonthlyRate", "Over18",
                         "MonthlyIncome"]

    for col in model.hr_retention_data.columns:
        if col not in not_needed_to_add:
            langs = list(set(model.hr_retention_data[col].tolist()))
            langs.sort()
            langs = [str(val) for val in langs]

            cols_for_plotting[col] = {
                "langs": langs,
                "vals": [0] * len(langs)
            }

    already_added = ["Age", "MonthlyIncome", "HourlyRate", "Attrition"]

    for attrition, x, y in zip(predictions, model.x_test.values, model.y_test.values):
        if attrition != y:
            mistakes.append(x)
        if attrition:
            monthly_income_index = model.COLS_INDEXES["MonthlyIncome"]
            hourly_rate_index = model.COLS_INDEXES["HourlyRate"]
            age = x[0]
            hourly_rate = x[hourly_rate_index]
            monthly_income = x[monthly_income_index]

            age_index_to_inc = get_age_index_to_inc(age)
            cols_for_plotting["Age"]["vals"][age_index_to_inc] += 1

            monthly_index_to_inc = get_monthly_index_to_inc(monthly_income)
            cols_for_plotting["MonthlyIncome"]["vals"][monthly_index_to_inc] += 1

            hourly_rate_index_to_inc = get_hourly_rate_index_to_inc(hourly_rate)
            cols_for_plotting["HourlyRate"]["vals"][hourly_rate_index_to_inc] += 1

            for col, index in model.COLS_INDEXES.items():
                if col in already_added or col not in cols_for_plotting:
                    continue

                val = cols_for_plotting[col]

                index -= 1
                option = x[index]
                if option in model.reversed_mappings[col]:
                    orig_op = model.reversed_mappings[col][option]
                else:
                    orig_op = str(option)
                index_to_inc = val["langs"].index(orig_op)
                val["vals"][index_to_inc] += 1

        val = cols_for_plotting["Attrition"]
        val["vals"][attrition] += 1

    save_figures(cols_for_plotting, directory)
    return predictions


if __name__ == '__main__':
    model_with_monthly_income = HREmployeeAttritionModel()
    model_without_monthly_income = deepcopy(model_with_monthly_income)
    del model_without_monthly_income.hr_retention_data["MonthlyIncome"]

    # Train model
    print("Training model with monthly income...")
    model_with_monthly_income.train()
    print("Done!")

    print("Training model without monthly income...")
    model_without_monthly_income.train()
    print("Done!")

    # Predict On Test
    results = []
    with_income_directory = os.path.join("Plots", "With Monthly Income")
    without_income_directory = os.path.join("Plots", "Without Monthly Income")

    # Visualize and save figures
    print("Visualizing model with monthly income...")
    predictions_with_monthly_income = visualize_classifier(model_with_monthly_income, with_income_directory)
    print("Done!")

    print("Visualizing model without monthly income...")
    predictions_without_monthly_income = visualize_classifier(model_without_monthly_income, without_income_directory)
    print("Done!")

    # Intersection
    employee_number_index = model_with_monthly_income.COLS_INDEXES["EmployeeNumber"]
    predictions_with_monthly_income = {str(emp[employee_number_index])
                                       for emp, attrition in zip(model_with_monthly_income.x_test.values,
                                                                 predictions_with_monthly_income)}
    predictions_without_monthly_income = {str(emp[employee_number_index])
                                          for emp, attrition in zip(model_with_monthly_income.x_test.values,
                                                                    predictions_without_monthly_income)}

    with_income_intersection = predictions_with_monthly_income.intersection(predictions_without_monthly_income)
    without_income_intersection = predictions_without_monthly_income.intersection(predictions_with_monthly_income)

    with open("with_income_intersection.txt", "w") as with_income_file:
        with_income_file.write("\n".join(with_income_intersection))

    with open("without_income_intersection.txt", "w") as without_income_file:
        without_income_file.write("\n".join(without_income_intersection))
