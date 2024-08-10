import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import spacy
import re

# Load pre-trained word vectors (you'll need to download this file)
# word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# Alternatively, you can use a smaller model like:
word_vectors = KeyedVectors.load_word2vec_format('glove-wiki-gigaword-100.txt', binary=False)

# Load spaCy model for concept extraction
nlp = spacy.load("en_core_web_sm")

def preprocess(term):
    # Convert to lowercase and split into words
    return re.findall(r'\w+', term.lower())

def get_word_vector(word):
    try:
        return word_vectors[word]
    except KeyError:
        return np.zeros(word_vectors.vector_size)

def semantic_similarity(term1, term2):
    words1 = preprocess(term1)
    words2 = preprocess(term2)
    
    vec1 = np.mean([get_word_vector(word) for word in words1], axis=0)
    vec2 = np.mean([get_word_vector(word) for word in words2], axis=0)
    
    return cosine_similarity([vec1], [vec2])[0][0]

def structural_similarity(term1, term2):
    words1 = set(preprocess(term1))
    words2 = set(preprocess(term2))
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def concept_similarity(term1, term2):
    doc1 = nlp(term1)
    doc2 = nlp(term2)
    
    concepts1 = set([token.lemma_ for token in doc1 if token.pos_ in ['NOUN', 'VERB']])
    concepts2 = set([token.lemma_ for token in doc2 if token.pos_ in ['NOUN', 'VERB']])
    
    intersection = concepts1.intersection(concepts2)
    union = concepts1.union(concepts2)
    
    return len(intersection) / len(union) if union else 0

def are_terms_similar(term1, term2, threshold=0.7):
    sem_sim = semantic_similarity(term1, term2)
    struct_sim = structural_similarity(term1, term2)
    conc_sim = concept_similarity(term1, term2)
    
    # You can adjust the weights of these similarities based on their effectiveness
    combined_sim = 0.5 * sem_sim + 0.3 * struct_sim + 0.2 * conc_sim
    
    return combined_sim > threshold, combined_sim

# Example usage
term1 = 'smoke_age_start'
term2 = 'tobacco_smoking_onset_year'

is_similar, similarity_score = are_terms_similar(term1, term2)

print(f"Terms '{term1}' and '{term2}':")
print(f"Are similar: {is_similar}")
print(f"Similarity score: {similarity_score:.4f}")