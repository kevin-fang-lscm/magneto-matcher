from sentence_transformers import SentenceTransformer
import numpy as np
from .harmonizer import Harmonizer
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.match import Match
import pandas as pd
from typing import Dict, Tuple
from .utils import get_samples

import openai
from dotenv import load_dotenv
import os

import openai
from dotenv import load_dotenv
import os


class OpenAIClient:
    def __init__(self, model='gpt-4o-mini'):
        load_dotenv(os.path.expanduser('~/config.env'))
        self.api_key = os.getenv("API_KEY")
        openai.api_key = self.api_key
        self.model = model

    def get_completion(self, prompt, max_tokens=1000, temperature=0.2, n=1):
        response_text = ""
        representations_chunks = self.chunk_representations(
            prompt, chunk_size=1000)  # Adjust `chunk_size` based on needs

        for chunk in representations_chunks:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in database schema design and data integration who can decide whether two columns are semantically similar or not for a scenario of data integration between different data sources."},
                    {"role": "user", "content": chunk}
                ],
                max_tokens=max_tokens,
                n=n,
                temperature=temperature
            )

            response_text += response.choices[0].message.content.strip() + "\n"

        return response_text.strip()

    def chunk_representations(self, representations, chunk_size):
        """
        Splits the representations into chunks small enough to fit within max token constraints.
        """
        chunks = []
        current_chunk = ""
        for representation in representations.splitlines():
            # If adding the next line exceeds chunk_size, start a new chunk
            if len(current_chunk) + len(representation) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += representation + "\n"
        # Add any remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks


class MatchReranker(Harmonizer):
    def __init__(self, columnheader_model_name=None, value_model_name=None,
                 topk=10, use_instances=False, llm_model='gpt-4o-mini'):
        super().__init__(columnheader_model_name, value_model_name, topk, use_instances)

        self.llm_model = llm_model

    def rerank_and_filter_matches(self, source_table, target_table, initial_matches, top_k=3):
        # Prepare data in a DataFrame for easier manipulation
        matches_data = {
            'Source Column': [],
            'Target Column': [],
            'Score': []
        }

    # Populate the data dictionary with initial matches
        for match, score in initial_matches.items():
            source_col, target_col = match[0][1], match[1][1]
            matches_data['Source Column'].append(source_col)
            matches_data['Target Column'].append(target_col)
            matches_data['Score'].append(score)

        # Create a DataFrame from the data
        df = pd.DataFrame(matches_data)

    # List to store the results after normalization and sorting
        ranked_matches = []

        # Process each source column group separately
        for source_col, group_df in df.groupby('Source Column'):
            # Normalize scores within this group using min-max normalization
            min_score, max_score = group_df['Score'].min(
            ), group_df['Score'].max()
            group_df['Normalized Score'] = (
                group_df['Score'] - min_score) / (max_score - min_score)

            # Sort by the normalized score in descending order
            group_df = group_df.sort_values(
                by='Normalized Score', ascending=False)

            # Select top-k matches for the current source column
            top_k_matches = group_df.head(top_k)
            ranked_matches.append(top_k_matches)

        # Combine the top-k matches from each group into a single DataFrame
        final_reranked_df = pd.concat(ranked_matches).reset_index(drop=True)

        # Filter the initial matches based on the top-k reranked matches
        filtered_matches = {
            ((("source_table", row['Source Column']), ("target_table", row['Target Column']))): row['Score']
            for _, row in final_reranked_df.iterrows()
        }

        matches = {}

        for _, row in final_reranked_df.iterrows():
            match = Match(target_table.name, row['Target Column'],
                          source_table.name, row['Source Column'], row['Score']).to_dict
            matches.update(match)

        return matches

    def get_matches(self, source_table: BaseTable, target_table: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        # Get initial matches using parent class
        initial_matches = super().get_matches(source_table, target_table)

        filtered_top_k_matches = self.rerank_and_filter_matches(source_table, target_table,
                                                                initial_matches, top_k=2)
        
        

        return filtered_top_k_matches
