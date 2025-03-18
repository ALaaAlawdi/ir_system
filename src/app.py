import os
import re
import string
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd

class IRSystem:
    def __init__(self, corpus_folder):
        """
        Initialize the IR System with the path to the corpus folder
        """
        self.corpus_folder = corpus_folder
        self.documents = {}
        self.normalized_docs = {}
        self.no_stopwords_docs = {}
        self.stemmed_docs = {}
        self.term_doc_matrix = None
        self.vocabulary = set()
        
        # Initialize Porter Stemmer and stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_documents(self):
        """
        Load all .txt documents from the corpus folder
        """
        for filename in os.listdir(self.corpus_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.corpus_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.documents[filename] = content
        
        return self.documents
    
    def normalize_documents(self):
        """
        Normalize documents by converting to lowercase and removing punctuation
        """
        for doc_name, content in self.documents.items():
            # Convert to lowercase
            content = content.lower()
            
            # Remove punctuation
            content = re.sub(f'[{string.punctuation}]', ' ', content)
            
            # Remove extra whitespace
            content = ' '.join(content.split())
            
            self.normalized_docs[doc_name] = content
        
        return self.normalized_docs
    
    def remove_stopwords(self):
        """
        Remove stopwords from the normalized documents
        """
        for doc_name, content in self.normalized_docs.items():
            words = content.split()
            filtered_words = [word for word in words if word not in self.stop_words]
            self.no_stopwords_docs[doc_name] = ' '.join(filtered_words)
        
        return self.no_stopwords_docs
    
    def stem_documents(self):
        """
        Apply Porter stemming to documents with stopwords removed
        """
        for doc_name, content in self.no_stopwords_docs.items():
            words = content.split()
            stemmed_words = [self.stemmer.stem(word) for word in words]
            self.stemmed_docs[doc_name] = ' '.join(stemmed_words)
            
            # Add stemmed words to vocabulary
            self.vocabulary.update(stemmed_words)
        
        return self.stemmed_docs
    
    def build_term_doc_matrix(self):
        """
        Build the term-document matrix
        """
        # Sort vocabulary for consistent ordering
        sorted_vocab = sorted(list(self.vocabulary))
        doc_names = sorted(list(self.stemmed_docs.keys()))
        
        # Create empty matrix
        matrix = np.zeros((len(sorted_vocab), len(doc_names)))
        
        # Fill matrix with term frequencies
        for j, doc_name in enumerate(doc_names):
            doc_words = self.stemmed_docs[doc_name].split()
            for i, term in enumerate(sorted_vocab):
                matrix[i, j] = doc_words.count(term)
        
        # Create DataFrame for better visualization
        self.term_doc_matrix = pd.DataFrame(matrix, index=sorted_vocab, columns=doc_names)
        
        return self.term_doc_matrix
    
    def process_all(self):
        """
        Process all steps in sequence
        """
        print("Loading documents...")
        self.load_documents()
        
        print("\nNormalizing documents...")
        self.normalize_documents()
        
        print("\nRemoving stopwords...")
        self.remove_stopwords()
        
        print("\nStemming documents...")
        self.stem_documents()
        
        print("\nBuilding term-document matrix...")
        self.build_term_doc_matrix()
        
        # Print the results
        self.print_results()
    
    def print_results(self):
        """
        Print the results after each processing step
        """
        print("\n" + "="*50)
        print("NORMALIZED DOCUMENTS (after removing punctuation):")
        print("="*50)
        for doc_name, content in self.normalized_docs.items():
            print(f"\n{doc_name}:")
            print(content[:500] + "..." if len(content) > 500 else content)
        
        print("\n" + "="*50)
        print("DOCUMENTS AFTER STOPWORDS REMOVAL:")
        print("="*50)
        for doc_name, content in self.no_stopwords_docs.items():
            print(f"\n{doc_name}:")
            print(content[:500] + "..." if len(content) > 500 else content)
        
        print("\n" + "="*50)
        print("DOCUMENTS AFTER STEMMING:")
        print("="*50)
        for doc_name, content in self.stemmed_docs.items():
            print(f"\n{doc_name}:")
            print(content[:500] + "..." if len(content) > 500 else content)
        
        print("\n" + "="*50)
        print("TERM-DOCUMENT MATRIX:")
        print("="*50)
        print(self.term_doc_matrix)
        
        # Print some statistics
        print("\n" + "="*50)
        print("STATISTICS:")
        print("="*50)
        print(f"Number of documents: {len(self.documents)}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Term-document matrix shape: {self.term_doc_matrix.shape}")


# Usage example
if __name__ == "__main__":
    # Make sure to install required packages:
    # pip install nltk pandas numpy
    
    # Download NLTK resources (only needed once)
    import nltk
    nltk.download('stopwords')
    
    # Create and process the IR system
    ir_system = IRSystem("myCorpus")
    ir_system.process_all()