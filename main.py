import os
import networkx as nx
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Function to load graphs from GraphML files in a directory
def load_graphs(graph_dir):
    graphs = []
    for file in os.listdir(graph_dir):
        if file.endswith(".graphml"):
            graph_path = os.path.join(graph_dir, file)
            graph = nx.read_graphml(graph_path)
            graphs.append(graph)
    return graphs

# Function to extract features from graphs for KNN classification
def extract_features_knn(graphs):
    features = []
    for graph in graphs:
        num_nodes = graph.number_of_nodes()
        features.append(num_nodes)
    return np.array(features).reshape(-1, 1)

# Function to extract features from graphs for vector-based classification
def extract_features_vector(graphs):
    features = []
    for graph in graphs:
        features.append(len(graph.nodes))
    return np.array(features).reshape(-1, 1)

# Function to perform KNN classification
def knn_classification(X_train, X_test, y_train, y_test, k=5):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return y_pred, report, confusion

# Function to perform vector-based classification
def vector_classification(X_train, X_test, y_train, y_test, kernel='linear'):
    svm_classifier = SVC(kernel=kernel)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return y_pred, report, confusion

def main():
    # Directory containing GraphML files
    graph_dir = "Graphs"

    # Load graphs from GraphML files
    graphs = load_graphs(graph_dir)

    # Generate labels for demonstration (replace this with your actual labels)
    y = np.random.randint(0, 2, size=len(graphs))

    # Extract features using KNN method
    X_knn = extract_features_knn(graphs)

    # Extract features using vector-based classification method
    X_vector = extract_features_vector(graphs)

    # Split data into training and testing sets for KNN
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X_knn, y, test_size=0.2, random_state=42)

    # Split data into training and testing sets for vector-based classification
    X_train_vector, X_test_vector, y_train_vector, y_test_vector = train_test_split(
        X_vector, y, test_size=0.2, random_state=42)

    # Perform KNN classification
    y_pred_knn, report_knn, confusion_knn = knn_classification(
        X_train_knn, X_test_knn, y_train_knn, y_test_knn)

    # Perform vector-based classification
    y_pred_vector, report_vector, confusion_vector = vector_classification(
        X_train_vector, X_test_vector, y_train_vector, y_test_vector)

    # Print results
    print("KNN Classification Results:")
    print("Classification Report:")
    print(report_knn)
    print("Confusion Matrix:")
    print(confusion_knn)

    print("\nVector-based Classification Results:")
    print("Classification Report:")
    print(report_vector)
    print("Confusion Matrix:")
    print(confusion_vector)

if __name__ == "__main__":
    main()
