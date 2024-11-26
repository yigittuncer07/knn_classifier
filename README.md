# **k-Nearest Neighbors (k-NN) Classifier**

## **Overview**
This project implements a k-Nearest Neighbors (k-NN) classifier from scratch in Python to classify data from the `PlayTennis` dataset. The classifier supports:
- One-hot encoding for categorical features.
- Multiple distance metrics (Manhattan and Euclidean).
- Leave-one-out cross-validation for evaluation.
- Logging to track predictions, nearest neighbors, and performance metrics.

---

## **Features**
1. **Preprocessing**: Converts categorical features into one-hot encoded vectors for compatibility with distance-based computations.
2. **Distance Metrics**:
   - **Manhattan Distance**: Sum of absolute differences between feature values.
   - **Euclidean Distance**: Square root of the sum of squared differences.
3. **Cross-Validation**:
   - Implements leave-one-out cross-validation to evaluate the classifier on small datasets.
4. **Logging**:
   - Logs train and test data, nearest neighbors, and predictions to `knn.log`.

---

## **Setup and Execution**

### **Requirements**
- Python 3.7 or higher
- Required libraries: 
  - `pandas`
  - `json`
  - `logging`
  - `collections`

Install dependencies using:
```bash
pip install pandas
```

### **Dataset**
The dataset is stored as a JSON file at `artifacts/data/play_tennis.json`. Each record contains:
- **Features**: `Outlook`, `Temperature`, `Humidity`, `Wind`
- **Target**: `PlayTennis` (`Yes` or `No`)

### **Usage**
1. **Run the Program**:
   ```bash
   python main.py
   ```

2. **Input Parameters**:
   - Enter the value of `k` (must be an odd number).
   - Choose the distance metric:
     - `0` for Manhattan
     - `1` for Euclidean

3. **Output**:
   - Confusion Matrix:
     - True Positives (TP)
     - True Negatives (TN)
     - False Positives (FP)
     - False Negatives (FN)
   - Overall accuracy of the classifier.

---

## **Code Structure**

### **`knn_classifier` Class**
- **Initialization**: Takes training data, `k`, distance metric, and target attribute.
- **Methods**:
  - `_preprocess_data`: Converts categorical features into one-hot vectors.
  - `_preprocess_test_data`: Ensures test data matches the training data's structure.
  - `predict`: Predicts the class of a given test data instance.
  - `manhattan_distance` and `euclidean_distance`: Compute distances between feature vectors.
  - `store_model`: Saves preprocessed training data to a JSON file.

### **Main Script**
- **Steps**:
  1. Loads and preprocesses the dataset.
  2. Performs leave-one-out cross-validation.
  3. Initializes the `knn_classifier` for each test instance.
  4. Outputs predictions, confusion matrix, and accuracy.

---

## **Example Output**
```plaintext
Enter the value of k: 3
Enter the distance metric (0: manhattan/1: euclidean): 0
Prediction: No, Actual: No
Prediction: No, Actual: No
Prediction: Yes, Actual: Yes
...
Confusion Matrix:
True Positives (TP): 4
True Negatives (TN): 3
False Positives (FP): 2
False Negatives (FN): 5
Accuracy: 0.5
```

---

## **Logging**
Logs are saved in `knn.log` and include:
- Train and test data after preprocessing.
- Nearest neighbors and their distances for each test instance.
- Predictions and actual labels for each test instance.

---

## **Notes**
- Ensure the `artifacts/data/play_tennis.json` file exists and follows the correct structure.
- The program enforces `k` as an odd number to avoid ties during classification.
- Accuracy may vary based on the choice of `k` and distance metric.

---

## **Improvements**
- Implement weighted voting for neighbors to improve classification accuracy.
- Optimize distance calculations for large datasets.
- Experiment with other encoding methods for categorical data. 

---

## **License**
This project is licensed under the MIT License. Feel free to modify and distribute it as needed.