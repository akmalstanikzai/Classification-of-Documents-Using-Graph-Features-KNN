import os
import pymongo
import networkx as nx

# Function to connect to MongoDB and retrieve preprocessed text data
def retrieve_preprocessed_data(uri, db_name, collection_name):
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    preprocessed_data = collection.find()
    return preprocessed_data

# Function to preprocess the text
def preprocess_text(text):
    return text.lower().split()

# Function to construct a directed graph from preprocessed text
def construct_graph(preprocessed_text):
    G = nx.DiGraph()
    # Add nodes for each unique word
    for word in preprocessed_text:
        G.add_node(word)
    # Add edges between consecutive words in the preprocessed text
    for i in range(len(preprocessed_text) - 1):
        current_word = preprocessed_text[i]
        next_word = preprocessed_text[i + 1]
        if G.has_edge(current_word, next_word):
            G[current_word][next_word]["weight"] += 1
        else:
            G.add_edge(current_word, next_word, weight=1)
    return G

# Directory to save GraphML files
output_dir = "./Graphs"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# MongoDB URI
mongo_uri = "mongodb://localhost:27017"
# MongoDB database name
db_name = "preprocessed_data"
# MongoDB collection name
collection_name = "documents"

# Retrieve preprocessed text data from MongoDB
preprocessed_data = retrieve_preprocessed_data(mongo_uri, db_name, collection_name)

# Counter for naming the GraphML files
file_counter = 1

# Iterate over each document
for doc in preprocessed_data:
    preprocessed_text = doc['preprocessed_text']
    graph = construct_graph(preprocessed_text)
    # Save the graph to a GraphML file
    output_file = os.path.join(output_dir, f"{file_counter}.graphml")
    nx.write_graphml(graph, output_file)
    file_counter += 1
