import re
import hashlib
import math
import pandas as pd
from random import randint

# 1. Shingler class to create shingles from documents
class Shingler:
    def __init__(self, k=3):
        if k > 0:
            self.k = int(k)
        else:
            self.k = 3  # default to k=3
    
    def process_doc(self, document):
        """Remove extra spaces and lowercase the document."""
        return re.sub("( )+|(\n)+", " ", document).lower()
    
    def get_shingles(self, document):
        """Generate k-shingles for a document."""
        shingles = set()
        document = self.process_doc(document)
        for i in range(0, len(document) - self.k + 1):
            shingles.add(document[i:i+self.k])
        return shingles

# 2. HashFamily for consistent hash generation
class HashFamily:
    def __init__(self, i):
        self.result_size = 8  # bytes to return
        self.max_len = 20  # max salt length
        self.salt = str(i).zfill(self.max_len)[-self.max_len:]

    def get_hash_value(self, el_to_hash):
        return int(hashlib.sha1(str(el_to_hash).encode('utf-8') + self.salt.encode('utf-8')).hexdigest()[-self.result_size:], 16)

# 3. MinhashSigner to compute minhash signatures
class MinhashSigner:
    def __init__(self, sig_size):
        self.sig_size = sig_size
        self.hash_functions = [HashFamily(randint(0, 10000000000)) for _ in range(sig_size)]

    def compute_set_signature(self, shingles_set):
        """Compute minhash signature for a set of shingles."""
        signature = []
        for hash_funct in self.hash_functions:
            min_hash = math.inf
            for shingle in shingles_set:
                h = hash_funct.get_hash_value(shingle)
                if h < min_hash:
                    min_hash = h
            signature.append(min_hash)
        return signature

    def compute_signature_matrix(self, sets):
        """Compute signature matrix for a list of sets."""
        return [self.compute_set_signature(s) for s in sets]

# 4. Locality-Sensitive Hashing (LSH) class
class LSH:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def get_signature_matrix_bands(self, sig_matrix, bands_nr, sign_len):
        """Divide signature matrix into bands for LSH."""
        r = sign_len // bands_nr  # rows per band
        bands = {i: [] for i in range(bands_nr)}
        for signature in sig_matrix:
            for i in range(bands_nr):
                idx = i * r
                bands[i].append(' '.join(str(x) for x in signature[idx:idx+r]))
        return bands

    def get_band_buckets(self, band, hash_funct):
        """Hash band columns into buckets."""
        buckets = {}
        for doc_id, value in enumerate(band):
            hashed_value = hash_funct.get_hash_value(value)
            if hashed_value not in buckets:
                buckets[hashed_value] = [doc_id]
            else:
                buckets[hashed_value].append(doc_id)
        return buckets

    def get_candidates_list(self, buckets):
        """Return list of candidate document pairs."""
        candidates = set()
        for bucket, doc_ids in buckets.items():
            if len(doc_ids) > 1:
                for i in range(len(doc_ids) - 1):
                    for j in range(i + 1, len(doc_ids)):
                        candidates.add(tuple(sorted((doc_ids[i], doc_ids[j]))))
        return candidates

    def check_candidates(self, candidates_list, sig_matrix):
        """Check the Jaccard similarity of candidate pairs."""
        similar_docs = set()
        for doc1, doc2 in candidates_list:
            signature1 = set(sig_matrix[doc1])
            signature2 = set(sig_matrix[doc2])
            jaccard_sim = len(signature1.intersection(signature2)) / len(signature1.union(signature2))
            if jaccard_sim >= self.threshold:
                similar_docs.add((doc1, doc2))
        return similar_docs

    def get_similar_items(self, sig_matrix, bands_nr, sign_len):
        """Find similar items using LSH."""
        similar_items = set()
        bands = self.get_signature_matrix_bands(sig_matrix, bands_nr, sign_len)
        for band_id, band in bands.items():
            buckets = self.get_band_buckets(band, HashFamily(randint(0, 10000000000)))
            candidates = self.get_candidates_list(buckets)
            similar_items.update(self.check_candidates(candidates, sig_matrix))
        return similar_items

# 5. Main function to compare columns from source and target dataframes
def find_similar_columns(source_df, target_df, threshold=0.8, k=3, sig_size=100, bands_nr=10):
    shingler = Shingler(k)
    signer = MinhashSigner(sig_size)
    lsh_instance = LSH(threshold)

    # Create shingles for each column in both source and target
    source_shingles = [shingler.get_shingles(col) for col in source_df.apply(lambda col: ' '.join(col.astype(str)), axis=0)]
    target_shingles = [shingler.get_shingles(col) for col in target_df.apply(lambda col: ' '.join(col.astype(str)), axis=0)]

    # Compute MinHash signatures for each set of shingles
    source_signatures = signer.compute_signature_matrix(source_shingles)
    target_signatures = signer.compute_signature_matrix(target_shingles)

    # Combine source and target signatures
    combined_signatures = source_signatures + target_signatures

    # Apply LSH to find similar columns
    similar_pairs = lsh_instance.get_similar_items(combined_signatures, bands_nr, sig_size)

    # Map similar pairs back to original dataframe columns
    result_pairs = []
    num_source_cols = len(source_df.columns)
    for pair in similar_pairs:
        col1, col2 = pair
        if col1 < num_source_cols and col2 >= num_source_cols:
            result_pairs.append((source_df.columns[col1], target_df.columns[col2 - num_source_cols]))

    return result_pairs

# Example usage:
source = pd.DataFrame({'A': ['apple', 'banana', 'cherry'], 'B': ['dog', 'elephant', 'fox']})
target = pd.DataFrame({'X': ['apple', 'banana', 'kiwi'], 'Y': ['dog', 'giraffe', 'hippo']})

print("Source DataFrame:")
print(source)
print("\nTarget DataFrame:")
print(target)

similar_cols = find_similar_columns(source, target, threshold=0.2)
print("Similar column pairs:", similar_cols)
