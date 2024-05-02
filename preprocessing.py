import os
import docx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient

# Initialize NLTK's WordNetLemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess the text
def preprocess_document(doc_text):
    # Tokenize the text
    tokens = word_tokenize(doc_text)
    # Convert tokens to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Function to save preprocessed data to MongoDB
def save_to_mongodb(directory_path, mongodb_uri, db_name, collection_name):
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    collection = db[collection_name]
    # Iterate over each topic directory
    for topic in os.listdir(directory_path):
        topic_dir = os.path.join(directory_path, topic)
        if os.path.isdir(topic_dir):
            print(f"Processing topic: {topic}")
            # Iterate over each document in the topic directory
            for filename in os.listdir(topic_dir):
                if filename.endswith('.docx'):
                    file_path = os.path.join(topic_dir, filename)
                    # Extract text from the .docx file
                    doc_text = extract_text_from_docx(file_path)
                    # Preprocess the text
                    preprocessed_text = preprocess_document(doc_text)
                    # Insert preprocessed data into MongoDB
                    document = {
                        "topic": topic,
                        "filename": filename,
                        "preprocessed_text": preprocessed_text
                    }
                    collection.insert_one(document)
    # Close the MongoDB connection
    client.close()

# Function to extract text from a .docx file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    all_text = []
    for paragraph in doc.paragraphs:
        all_text.append(paragraph.text)
    return ' '.join(all_text)

# Directory path containing the data
data_directory = "./Data"
# MongoDB connection URI
mongodb_uri = "mongodb://localhost:27017/"
# MongoDB database name
db_name = "preprocessed_data"
# MongoDB collection name
collection_name = "documents"

# Save preprocessed data to MongoDB
save_to_mongodb(data_directory, mongodb_uri, db_name, collection_name)
