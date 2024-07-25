
import math
from fuzzywuzzy import fuzz
import jellyfish
import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util


def compute_term_similarity(term1, term2, embedding_model=None):
    def preprocess(text):
        # Insert space before uppercase letters in PascalCase and camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Convert to lowercase
        text = text.lower()
        # Replace underscores and dots with spaces
        text = text.replace('_', ' ').replace('.', ' ')
        # Remove any remaining non-alphanumeric characters except spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Remove extra whitespace
        return ' '.join(text.split())

    def sound_similarity(s1, s2):
        tokens1 = s1.split()
        tokens2 = s2.split()

        soundex_sim = jellyfish.jaro_winkler_similarity(
            ' '.join(jellyfish.soundex(t) for t in tokens1),
            ' '.join(jellyfish.soundex(t) for t in tokens2)
        )
        # metaphone_sim = jellyfish.jaro_winkler_similarity(
        #     ' '.join(jellyfish.metaphone(t) for t in tokens1),
        #     ' '.join(jellyfish.metaphone(t) for t in tokens2)
        # )
        # return (soundex_sim + metaphone_sim) / 2
        return soundex_sim

    # def embedding_similarity(s1, s2, embedding_model=None):
    #     if embedding_model is None:
    #         embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    #     embedding1 = embedding_model.encode(s1, convert_to_tensor=True)
    #     embedding2 = embedding_model.encode(s2, convert_to_tensor=True)
    #     cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
    #     return cosine_similarity.item()

    def affix_similarity(s1, s2):
        def get_affixes(s, n=3):
            return set(s[:i] for i in range(1, min(n+1, len(s)+1))) | set(s[-i:] for i in range(1, min(n+1, len(s)+1)))

        affixes1 = set.union(*[get_affixes(word) for word in s1.split()])
        affixes2 = set.union(*[get_affixes(word) for word in s2.split()])

        return len(affixes1 & affixes2) / len(affixes1 | affixes2)

    def ngram_similarity(s1, s2, n=2):
        def get_ngrams(s, n):
            return [''.join(gram) for gram in zip(*[s[i:] for i in range(n)])]

        ngrams1 = Counter(get_ngrams(s1, n))
        ngrams2 = Counter(get_ngrams(s2, n))

        intersection = sum((ngrams1 & ngrams2).values())
        union = sum((ngrams1 | ngrams2).values())

        return intersection / union if union > 0 else 0

    processed_term1 = preprocess(term1)
    processed_term2 = preprocess(term2)

    similarity_dict = {
        'edit_distance': fuzz.ratio(processed_term1, processed_term2) / 100,
        'partial_ratio': fuzz.partial_ratio(processed_term1, processed_term2) / 100,
        'token_sort_ratio': fuzz.token_sort_ratio(processed_term1, processed_term2) / 100,
        'token_set_ratio': fuzz.token_set_ratio(processed_term1, processed_term2) / 100,
        'sound_similarity': sound_similarity(processed_term1, processed_term2),
        'jaro_winkler': jellyfish.jaro_winkler_similarity(processed_term1, processed_term2),
        'affix': affix_similarity(processed_term1, processed_term2),
        'digram': ngram_similarity(processed_term1, processed_term2, n=2),
        'trigram': ngram_similarity(processed_term1, processed_term2, n=3)
        # 'embedding': embedding_similarity(processed_term1, processed_term2, embedding_model)
    }

    return similarity_dict