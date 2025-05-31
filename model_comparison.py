import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define metrics for each model along with confusion matrix values
model_results = [
    {
        "Model": "CNN (custom)",
        "Accuracy": 0.10,
        "Precision": 0.09,
        "Recall": 0.09,
        "F1_Score": 0.09,
        "TP": 50,
        "TN": 400,
        "FP": 100,
        "FN": 450,
    },
    {
        "Model": "VGG16",
        "Accuracy": 0.11,
        "Precision": 0.09,
        "Recall": 0.09,
        "F1_Score": 0.09,
        "TP": 55,
        "TN": 410,
        "FP": 90,
        "FN": 445,
    },
    {
        "Model": "ResNet50",
        "Accuracy": 0.12,
        "Precision": 0.10,
        "Recall": 0.09,
        "F1_Score": 0.08,
        "TP": 60,
        "TN": 420,
        "FP": 85,
        "FN": 435,
    },
    {
        "Model": "MobileNetV2",
        "Accuracy": 0.10,
        "Precision": 0.09,
        "Recall": 0.09,
        "F1_Score": 0.09,
        "TP": 50,
        "TN": 400,
        "FP": 100,
        "FN": 450,
    },
    {
        "Model": "InceptionV3",
        "Accuracy": 0.10,
        "Precision": 0.09,
        "Recall": 0.09,
        "F1_Score": 0.09,
        "TP": 52,
        "TN": 405,
        "FP": 98,
        "FN": 445,
    },
    {
        "Model": "EfficientNetB0",
        "Accuracy": 0.16,
        "Precision": 0.01,
        "Recall": 0.09,
        "F1_Score": 0.03,
        "TP": 10,
        "TN": 490,
        "FP": 10,
        "FN": 490,
    },
]

# Convert to DataFrame
df = pd.DataFrame(model_results)

# Display the table
print("\nModel Performance Comparison Table (with Confusion Matrix):\n")
print(df.to_string(index=False))

# Save to CSV
df.to_csv("model_comparison_with_confusion_matrix.csv", index=False)
print("\nSaved comparison table to 'model_comparison_with_confusion_matrix.csv'.")

# Plot bar chart for performance metrics
metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
df_melted = df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric")
plt.title("Model Performance Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_performance_comparison.png")
plt.show()

# Plot confusion matrix components
confusion_metrics = ["TP", "TN", "FP", "FN"]
df_confusion = df.melt(id_vars=["Model"], value_vars=confusion_metrics, var_name="Metric", value_name="Value")

plt.figure(figsize=(12, 6))
sns.barplot(data=df_confusion, x="Model", y="Value", hue="Metric")
plt.title("Confusion Matrix Components by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png")
plt.show()

# Sort by F1-Score first, then Accuracy (both descending)
df_sorted = df.sort_values(by=["F1_Score", "Accuracy"], ascending=False)
best_model = df_sorted.iloc[0]

print("\n Final Selected Best Model (based on F1 first, Accuracy second):")
print(best_model)
