import os
import docx

def extract_words_from_docx(file_path):
    doc = docx.Document(file_path)
    all_words = []
    for paragraph in doc.paragraphs:
        words = paragraph.text.split()
        all_words.extend(words)
    return all_words

def extract_words_from_folders(root_folder):
    all_words = []
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.docx'):
                    file_path = os.path.join(folder_path, filename)
                    words = extract_words_from_docx(file_path)
                    all_words.extend(words)
    return all_words

# Root folder containing subfolders with .docx files
root_folder = './Data'

# Extract words from all .docx files in all folders
all_words = extract_words_from_folders(root_folder)

print("Total words extracted:", len(all_words))

