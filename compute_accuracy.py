import numpy as np
import json

with open("generated_sql_results.json", "r") as f:
    data = json.load(f)

data = np.array(data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def print_accuracy_reports(predictions, labels):
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    # Print metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


# Define a function to extract the "Response" section
def extract_response(text):
    return text.split("### Response:\n")[-1].strip()

vectorized_extract_response = np.vectorize(extract_response)

y_predict = vectorized_extract_response(data[:, 0])
y_test = data[:, 1]
print(y_predict[0])
print(y_test[0])
print_accuracy_reports(y_predict, y_test)
