import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler


# Load the optical recognition of handwritten digits dataset
digits = load_digits()
data = digits.data
labels = digits.target

# store value for auc and pauc for average
auc_values = []
pauc_values = []
# Iterate over all digits from 0 to 9
for normal_digit in range(10):
    print("\nNormal digit: ", normal_digit)
    # Create labels for anomaly detection
    # Normal class is labeled as 0, anomaly classes are labeled as 1
    labels_anomaly = np.where(labels == normal_digit, 0, 1)

    # Separate the data into normal and anomaly sets
    normal_data = data[labels_anomaly == 0]
    anomaly_data = data[labels_anomaly == 1]

    # Combine the normal and anomaly data
    combined_data = np.concatenate((normal_data, anomaly_data))
    combined_labels = np.concatenate(
        (np.zeros(len(normal_data)), np.ones(len(anomaly_data))))

    # Shuffle the combined dataset
    indices = np.arange(len(combined_data))
    np.random.shuffle(indices)
    combined_data = combined_data[indices]
    combined_labels = combined_labels[indices]

    # Normalize the data
    # Since the pixel values are in the range [0, 16]
    combined_data = combined_data / 16.0

    # normalize data
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    # combined_data = scaler.fit_transform(combined_data)

    # Split into training and test sets (e.g., 80-20 split)
    train_data, test_data, train_labels, test_labels = train_test_split(
        combined_data, combined_labels, test_size=0.2, random_state=42)

    # Initialize and fit the IsolationForest
    # Adjust contamination to reflect the proportion of anomalies
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(train_data)

    # Predict anomalies on the test set
    predictions = isolation_forest.predict(test_data)
    # Convert predictions to 0 for normal and 1 for anomaly
    predictions = np.where(predictions == 1, 0, 1)

    # Evaluate the results
    print(classification_report(test_labels, predictions,
          target_names=['Normal', 'Anomaly']))

    # calculate  auc and pauc on test data
    # Calculate AUC
    auc_test = roc_auc_score(test_labels, predictions)
    print("AUC on test data: ", auc_test)

    # Calculate pAUC
    pauc_test = roc_auc_score(test_labels, predictions, max_fpr=0.1)
    print("pAUC on test data (FPR range [0, 0.1]): ", pauc_test)

    auc_values.append(auc_test)
    pauc_values.append(pauc_test)

# report average auc and pauc
print("\nAverage AUC: ", np.mean(auc_values))
print("Average pAUC: ", np.mean(pauc_values))
