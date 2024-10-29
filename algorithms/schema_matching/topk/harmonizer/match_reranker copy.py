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




class OpenAIClient:
    def __init__(self, model='gpt-4o-mini'):
        load_dotenv(os.path.expanduser('~/config.env'))
        self.api_key = os.getenv("API_KEY")
        openai.api_key = self.api_key
        self.model = model

    def get_completion(self, prompt, max_tokens=1000, temperature=0.2, n=1):
        response_text = ""
        representations_chunks = self.chunk_representations(prompt, chunk_size=1000)  # Adjust `chunk_size` based on needs

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

    def load_openai_key(self):
        load_dotenv(os.path.expanduser('~/config.env'))
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def _get_representations(self, initial_matches, source_df, target_df):
        representations = []
        for column_pair, score in initial_matches.items():
            # Extract column names
            column_1, column_2 = column_pair[0][1], column_pair[1][1]
            
            # Format each column representation with "Column" and "Sample values"
            column_1_repr = (
                "Column: " + column_1 +
                ", Sample values: [" + ", ".join(get_samples(source_df[column_1])) + "]"
            )
            column_2_repr = (
                "Column: " + column_2 +
                ", Sample values: [" + ", ".join(get_samples(target_df[column_2])) + "]"
            )

            # Combine into a question structure
            question_repr = "Does " + column_1_repr + " match " + column_2_repr + "?"
            representations.append(question_repr)

        # Join all pairs into a single string with each question on a new line
        representations = "\n".join(representations)
        return representations

    def _get_prompt(self, representations):
        prompt = (
            "You are an expert in database schema design and data integration. "
            "Your task is to analyze column pairs based on their sample values and indicate 'Yes' if they are a clear match or 'Maybe' if they have some similarity but are not a definite match. "
            "Each pair is presented as a question, including the column name and a few sample values to help you assess the similarity. "
            "Note that some columns may not have sample values; evaluate the match based on the information provided without assuming missing values indicate a non-match. "
            "Only return pairs labeled 'Yes' or 'Maybe' in the output. Filter out any pairs that do not meet these criteria."
            "\n\nFor example:\n"
            "Given:\n"
            "Does Column: Name, Sample values: [John, Mary, Alice] match Column: Nome, Sample values: [Joao, Maria]?\n"
            "Does Column: Name, Sample values: [John, Mary, Alice] match Column: Alias, Sample values: [JJ, Lice]?\n"
            "Does Column: Name, Sample values: [John, Mary, Alice] match Column: Age, Sample values: [30, 25, 40]?\n"
            "Does Column: Country, Sample values: [USA, Canada, Mexico] match Column: City, Sample values: [New York, Toronto, Mexico City]?\n"
            "Your response should be:\n"
            "Name , Nome: Yes\n"
            "Name , Alias: Maybe\n\n"
            "The following pairs are non-matches and should be filtered out (do not include them in the output):\n"
            "Name , Age\n"
            "Country , City\n\n"
            "Now, respond to each of the following questions:\n"
            + representations + "\n"
            "Respond with each pair followed by 'Yes' or 'Maybe' on a new line, in this format:\n(column_name_1 , column_name_2: Yes/Maybe)\n"
            "Exclude any non-matching pairs from your response."
        )
        return prompt





    def clean_response(self, text):
        """Clean the response to ensure it returns a valid list of column names."""
        # Split by new lines and remove empty entries or unwanted characters

        return [item.strip().replace('"', '').replace("'", "") for item in text.split('\n') if item.strip()]

    def get_matches(self, source_table: BaseTable, target_table: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        # Get initial matches using parent class
        initial_matches = super().get_matches(source_table, target_table)

        source_df = source_table.get_df()
        target_df = target_table.get_df()

        # print(initial_matches)

        representations = self._get_representations(
            initial_matches, source_df, target_df)
        # print(representations)
        prompt = self._get_prompt(representations)
        # print('\n')
        # print(prompt)

        self.load_openai_key()

        # Call GPT model to process the prompt and get the matches
        print("Calling OpenAI API...")

        openai_client = OpenAIClient(model='gpt-4o-mini')

        response = openai_client.get_completion(prompt)

        print(f"Generated matches:\n{response}")

        
        
        # cleaned_names = self.clean_response(response)
        # print(f"Generated matches:\n{cleaned_names}")

        return initial_matches
