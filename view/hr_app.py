import logging
from datetime import datetime
from typing import List

import PySimpleGUI as sg

import pyautogui

import os
import numpy as np

from model import HREmployeeAttritionModel
import matplotlib.pyplot as plt

from view.checkers import MinMaxChecker, OptionsChecker, Checker
from view.timer import open_timer

plt.ioff()
sg.theme('DarkGrey6')

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


def get_age_index_to_inc(age):
    """
    Get index for an age
    """
    if age <= 30:
        return 0
    elif 30 < age < 40:
        return 1
    else:
        return 2


def get_monthly_index_to_inc(monthly_income):
    """
    Get index for a monthly income
    """
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
    """
    Get index for hourly rate
    """
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
    """
    Save cols values in files as graphs
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)

    for key, val in cols_for_plotting.items():
        logging.info(f"Saving figure: {key}")
        fig = plt.figure(figsize=(20, 10))

        # creating the bar plot
        plt.bar(val["langs"], val["vals"], color='maroon',
                width=0.4)

        plt.ylabel(f"{key} Values")
        plt.title(key)

        fig.savefig(os.path.join(directory, f"{key}.png"), dpi=fig.dpi)
        plt.close()


def visualize_classifier(model, directory):
    """
    Visualize a model
    """
    predictions = model.predict(model.x_test)
    mistakes = []
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

            if "Age" in model.COLS_INDEXES:
                age = x[0]
                age_index_to_inc = get_age_index_to_inc(age)
                cols_for_plotting["Age"]["vals"][age_index_to_inc] += 1

            if "MonthlyIncome" in model.COLS_INDEXES:
                monthly_income_index = model.COLS_INDEXES["MonthlyIncome"]
                monthly_income = x[monthly_income_index]
                monthly_index_to_inc = get_monthly_index_to_inc(monthly_income)
                cols_for_plotting["MonthlyIncome"]["vals"][monthly_index_to_inc] += 1

            if "HourlyRate" in model.COLS_INDEXES:
                hourly_rate_index = model.COLS_INDEXES["HourlyRate"]
                hourly_rate = x[hourly_rate_index]
                hourly_rate_index_to_inc = get_hourly_rate_index_to_inc(hourly_rate)
                cols_for_plotting["HourlyRate"]["vals"][hourly_rate_index_to_inc] += 1

            for col, index in model.COLS_INDEXES.items():
                if col in already_added or col not in cols_for_plotting:
                    continue

                val = cols_for_plotting[col]

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


def values_are_valid(values, checkers: List[Checker]):
    """
    Check if the given values are valid by their checkers
    """
    not_valid_vals = []
    for val, checker in zip(values, checkers):
        if val:
            logging.info(f"Checking '{checker.col}' value: {val}")
            if not checker.check(val):
                logging.warning(f"'{checker.col}' value '{val}' is not valid")
                not_valid_vals.append([checker.col, val])

    if not_valid_vals:
        msg = "The following values are invalid please try again (the left is the name of the field you provided):\n"
        lines = []
        for col, val in not_valid_vals:
            lines.append(f"{col} -> {val}")
        msg += "\n".join(lines)
        sg.popup(msg, title="Invalid Input!")
        return False
    return True


def open_gui():
    """
    Opens the GUI for the app
    """
    vals = {}
    model_with_monthly_income = HREmployeeAttritionModel()
    del model_with_monthly_income.hr_retention_data["EmployeeNumber"]
    del model_with_monthly_income.hr_retention_data["EmployeeCount"]
    del model_with_monthly_income.hr_retention_data["Over18"]
    features = [col for col in model_with_monthly_income.hr_retention_data.columns if col != "Attrition"]
    width, height = pyautogui.size()

    cols = []
    is_bool_dict = {}
    checkers = []
    for feature, i in zip(features, range(len(features))):
        if feature == "Attrition":
            continue

        langs = list(set(model_with_monthly_income.hr_retention_data[feature].tolist()))
        langs.sort()
        is_bool = False

        if type(langs[0]) == int or type(langs[0]) == float:
            min_val, max_val = min(langs), max(langs)
            checkers.append(MinMaxChecker(min_val, max_val, feature))
            options = f"{min_val} - {max_val}"
        elif type(langs[0]) == str:
            options = ", ".join(langs)
            checkers.append(OptionsChecker(langs, feature))
        else:
            is_bool = True
            options = "false / true"

        is_bool_dict[feature] = is_bool
        cols += [f"{feature} ({options})"]

    layout = [[sg.Text("Please enter your data below to get a recommendation if you need to quit your job",
                       font=('Helvetica', 24))]]
    layout += [[sg.Text("What's your name?", font=('Helvetica', 14))], [sg.Input()]]
    layout1 = []
    input_index = {"name": 0}
    for label, i in zip(cols, range(len(cols))):
        layout1 += [[sg.Text(f"What's your {label}?", font=('Helvetica', 12))], [sg.Input()]]
        input_index[label] = i

    layout += [[sg.Column(layout1, scrollable=True, key="Column", size=(width * 0.9, height * 0.7))]]
    layout += [[sg.Button('Submit', font=('Helvetica', 12))]]

    # Create the window
    window = sg.Window("HR App", layout, finalize=True)
    window.Maximize()

    while True:
        try:
            # Display and interact with the Window
            event, values = window.read()

            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            # Do something with the information gathered
            if values:
                if not values_are_valid(list(values.values())[1:], checkers):
                    continue

                timer_proc = open_timer()
                closed_timer = False
                try:
                    for i, feature in zip(range(len(values)), features):
                        val = values[i + 1]
                        if val:
                            if is_bool_dict[feature]:
                                val = bool(val)
                                if val:
                                    val = 1
                                else:
                                    val = 0
                            vals[feature] = val
                        else:
                            del model_with_monthly_income.hr_retention_data[feature]

                    # Train model
                    logging.info("Training model with monthly income...")
                    model_with_monthly_income.train()
                    logging.info("Done!")

                    # Predict On Test
                    now = datetime.now().strftime("%H_%M_%S_%f")
                    result_dir = os.path.join("Plots", now)

                    # Visualize and save figures
                    logging.info("Visualizing model with monthly income...")
                    predictions_with_monthly_income = visualize_classifier(model_with_monthly_income, result_dir)
                    logging.info("Done!")

                    logging.info("Predicting your values!!!")
                    vals = model_with_monthly_income.fit_data(vals)
                    vals = np.asarray([vals])
                    result = model_with_monthly_income.predict(vals)[0]
                    if result:
                        import psutil
                        p = psutil.Process(timer_proc.pid)
                        p.terminate()
                        sg.popup(f"{values[0]} it is best that you quit your job...", title="HR Result")
                    else:
                        import psutil
                        p = psutil.Process(timer_proc.pid)
                        p.terminate()
                        sg.popup(f"{values[0]} you shouldn't quit your job!", title="HR Result")
                    closed_timer = True
                    model_with_monthly_income = HREmployeeAttritionModel()
                finally:
                    if not closed_timer:
                        import psutil
                        p = psutil.Process(timer_proc.pid)
                        p.terminate()
        except Exception as e:
            sg.popup(f"There was a failure: {e}", title="Failure!")

    # Finish up by removing from the screen
    window.close()
