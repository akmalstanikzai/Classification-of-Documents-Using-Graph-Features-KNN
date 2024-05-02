import os
import networkx as nx
from sklearn.neighbors import KNeighborsClassifier
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

# Function to extract features from graphs
def extract_features(graphs):
    features = []
    for graph in graphs:
        num_nodes = graph.number_of_nodes()
        features.append(num_nodes)
    return np.array(features).reshape(-1, 1)

# Function to perform KNN classification
def knn_classification(X_train, X_test, y_train, y_test, k=5):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return y_pred, report, confusion

def main():
    # Directory containing GraphML files
    graph_dir = "Graphs"
    
    # Load GraphML files from the directory
    graphs = load_graphs(graph_dir)
    print("Number of loaded graphs:", len(graphs))

    # Extract features from graphs
    X = extract_features(graphs)
    
    # Generate labels for demonstration (replace this with your actual labels)
    y = np.random.randint(0, 2, size=len(graphs))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform KNN classification
    y_pred, report, confusion = knn_classification(X_train, X_test, y_train, y_test)

    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion)

if __name__ == "__main__":
    main()
