from knn_classifier import knn_classifier
import json
import logging

# Set up logging
logging.basicConfig(
    filename="knn.log",
    level=logging.INFO, 
    format="%(asctime)s - %(message)s"
)

TRAIN_DATA_PATH = 'artifacts/data/play_tennis.json'
TARGET_ATTRIBUTE_INDEX = -1
POSTIVE_CLASS_LABEL = 'Yes'
NEGATIVE_CLASS_LABEL = 'No'

# Load dataset from json file 
with open(TRAIN_DATA_PATH, 'r') as f:
    data = json.load(f)

# Drop day column from data
for d in data:
    d.pop('Day')
    
# Pretty print first 5 rows of the data
for d in data[:5]:
    print(d)
    
# Input K from the user
k = int(input('Enter the value of k: '))

# Make sure k is an odd number by exiting otherwise
# This is because we need to break ties when k is even
if k % 2 == 0:
    print("Please enter an odd number for k")
    exit()    

# Input distance metric from the user
distance_metric = input('Enter the distance metric (0: manhattan/1: euclidean): ')

# Get the target attribute
target_attribute_name = list(data[0].keys())[TARGET_ATTRIBUTE_INDEX]

# Init lists to hold the results
true_positives = []
true_negatives = []
false_positives = []
false_negatives = []

# Leave one out cross validation
for i in range(len(data)):
    # Split the data into train and test
    train_data = data[:i] + data[i+1:]
    test_data = data[i].copy() # Create a copy of the test data since we will be modifying it
    
    # Remove the target attribute from the test data
    actual_label = test_data.pop(target_attribute_name)
    
    # Create an instance of knn_classifier
    classifier = knn_classifier(train_data=train_data, k=k, distance_metric=distance_metric, target_attribute=target_attribute_name)
    
    # Predict the class of the test
    prediction = classifier.predict(test_data)
    print(f"Prediction: {prediction}, Actual: {actual_label}")
    # log the prediction
    logging.info(f"Prediction: {prediction}, Actual: {actual_label}")
    if prediction == actual_label:
        if prediction == POSTIVE_CLASS_LABEL:
            true_positives.append(test_data)
        else:
            true_negatives.append(test_data)
    else:
        if prediction == POSTIVE_CLASS_LABEL:
            false_positives.append(test_data)
        else:
            false_negatives.append(test_data)

# Print the confusion matrix
print("Confusion Matrix:")
print(f"True Positives (TP): {len(true_positives)}")
print(f"True Negatives (TN): {len(true_negatives)}")
print(f"False Positives (FP): {len(false_positives)}")
print(f"False Negatives (FN): {len(false_negatives)}")
# log confusion matrix as well
logging.info("Confusion Matrix:")
logging.info(f"True Positives (TP): {len(true_positives)}")
logging.info(f"True Negatives (TN): {len(true_negatives)}")
logging.info(f"False Positives (FP): {len(false_positives)}")
logging.info(f"False Negatives (FN): {len(false_negatives)}")


# Print accuracy
accuracy = (len(true_positives) + len(true_negatives)) / len(data)
print(f"Accuracy: {round(accuracy,2)}") # Round to 2 decimal places

