from typing import Sequence
import json
import pandas as pd
from collections import Counter
import logging

class knn_classifier:
    def __init__(self, train_data, k, distance_metric, target_attribute, log_file="knn.log") -> None:
        self.target_attribute = target_attribute
        self.train_data = self._preprocess_data(train_data) 
        self.store_model(f'artifacts/model/{target_attribute}.json')
        self.k = k
        self.distance_metric = distance_metric
        
        # Set up logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO, 
            format="%(asctime)s - %(message)s"
        )
        logging.info("kNN Classifier Initialized")

    # Turn nominal attributes into one hot vectors    
    def _preprocess_data(self, data) -> list:
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        
        # Extract the target attribute and keep it as-is
        target_values = df[self.target_attribute]
        
        # Identify nominal (categorical) columns excluding the target
        nominal_columns = df.select_dtypes(include=['object']).columns.drop(self.target_attribute)

        # Apply one-hot encoding to other categorical columns
        df_encoded = pd.get_dummies(df, columns=nominal_columns, drop_first=False)
        
        # Reattach the target attribute
        df_encoded[self.target_attribute] = target_values
        
        # Convert the DataFrame back to a list of dictionaries
        return df_encoded.to_dict(orient='records')

    # Turn nominal attributes into one hot vectors    
    def _preprocess_test_data(self, test_data: dict) -> dict:
        # Convert the test data to a DataFrame
        test_df = pd.DataFrame([test_data])  # Ensure test_data is wrapped in a list to create a DataFrame

        # Identify nominal columns in the test data
        nominal_columns = test_df.select_dtypes(include=['object']).columns

        # Apply one-hot encoding
        test_encoded = pd.get_dummies(test_df, columns=nominal_columns, drop_first=False)

        # Ensure the test encoding matches the training encoding structure
        train_columns = pd.DataFrame(self.train_data).columns.tolist()
        
        # Remove the target attribute from the list of columns
        train_columns.remove(self.target_attribute)
        
        for column in train_columns:
            if (column not in test_encoded.columns) and (column != self.target_attribute):
                test_encoded[column] = False

        # Ensure column order matches the training data
        test_encoded = test_encoded[train_columns]

        # Convert the DataFrame back to a dictionary
        return test_encoded.to_dict(orient='records')[0]

    
    # Predict the class of the test data
    def predict(self, test_data) -> str:
        # Preprocess the test data (convert nominal attributes into the same one-hot encoding)
        test_data_processed = self._preprocess_test_data(test_data)
        
        # Calculate distances between test_data and each training instance
        distances = []
        for train_instance in self.train_data:
            train_features = {
                key: value for key, value in train_instance.items() 
                if key != self.target_attribute
            }
            
            if self.distance_metric == '0':  # Manhattan Distance
                distance = self.manhattan_distance(
                    list(test_data_processed.values()), list(train_features.values())
                )
            elif self.distance_metric == '1':  # Euclidean Distance
                distance = self.euclidean_distance(
                    list(test_data_processed.values()), list(train_features.values())
                )
            
            # Store distance along with the class label of the training instance
            distances.append((distance, train_instance[self.target_attribute]))
        
        # Sort distances and get the k nearest neighbors
        distances.sort(key=lambda x: x[0])  # Sort by distance
        k_nearest_neighbors = distances[:self.k]
        
        # Get the most common class label among the k nearest neighbors
        class_labels = [label for _, label in k_nearest_neighbors]
        predicted_label = Counter(class_labels).most_common(1)[0][0]

        logging.info(f"Train Data: \n{pd.DataFrame(self.train_data)}")
        logging.info(f"Test Data: \n{test_data_processed}")
        logging.info(f"Nearest neighbors: {distances}")
        logging.info(f"------------------------------")
        print(f"Train Data: \n{pd.DataFrame(self.train_data)}")
        print(f"Test Data: \n{test_data_processed}")
        print(f"Nearest neighbors: {distances}")
        print(f"------------------------------")
        
        return predicted_label
                    
    # Set the train data              
    def set_train_data(self, train_data) -> None:
        self.train_data = train_data
        
    # Stores the objects train data into a json file
    def store_model(self, filename) -> None:
        with open(filename, 'w') as f:
            json.dump(self.train_data, f)
            
    # Get manhattan distance
    def manhattan_distance(self, x: Sequence[float], y: Sequence[float]) -> float:
        return sum(abs(a - b) for a, b in zip(x, y))

    # Get euclidean distance
    def euclidean_distance(self, x: Sequence[float], y: Sequence[float]) -> float:
        return sum((a - b) ** 2 for a, b in zip(x, y)) ** 0.5