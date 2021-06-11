from model import HREmployeeAttritionModel, COLS_TO_ENCODE, COLS_INDEXES
import matplotlib.pyplot as plt


def visualize_classifier():
    predictions = model.predict(model.x_test)
    mistakes = []
    cols_for_plotting = {key: {"langs": list(val.keys()), "vals": [0 for _ in range(len(val))]}
                         for key, val in COLS_TO_ENCODE.items()}

    for attrition, x, y in zip(predictions, model.x_test.values, model.y_test.values):
        if attrition != y:
            mistakes.append(x)

        for col, index in COLS_INDEXES.items():
            val = cols_for_plotting[col]
            if col != "Attrition":
                index -= 1
                option = x[index]
                val["vals"][option] += 1
            else:
                val["vals"][attrition] += 1

    for key, val in cols_for_plotting.items():
        fig = plt.figure(figsize=(20, 10))

        # creating the bar plot
        plt.bar(val["langs"], val["vals"], color='maroon',
                width=0.4)

        plt.ylabel(f"{key} Values")
        plt.title(key)

        fig.savefig(f"{key}.png", dpi=fig.dpi)


if __name__ == '__main__':
    model = HREmployeeAttritionModel()

    # Train model
    model.train()

    # Predict On Test
    results = []
    visualize_classifier()
