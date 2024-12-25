from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def evaluate_model(model, test_data):
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    y_true = test_data.classes
    y_pred = model.predict(test_data)
    y_pred_classes = y_pred.argmax(axis=-1)

    print(classification_report(y_true, y_pred_classes, target_names=test_data.class_indices.keys()))

def plot_metrics(history, metric,name):
    plt.plot(history.history[metric], label=f"Train {metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.title(f"{name.capitalize()} {metric.capitalize()} Over Epochs")
    plt.show()
