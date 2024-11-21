import openai
import tiktoken
import os
from dotenv import load_dotenv
import pandas as pd
import re
from typing import Dict, Tuple
from valentine.algorithms.match import Match
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.base_matcher import BaseMatcher


class GPTMatcher(BaseMatcher):
    def __init__(self, llm_model="gpt-4o-mini",  sample_size=10, include_example=False):
        self.llm_model = llm_model
        self.client = self._load_client()
        self.sample_size = sample_size
        self.include_example = include_example

    def _load_client(self):
        print("Loading OpenAI client")
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("API key not found in environment variables.")

        openai.api_key = api_key
        return openai

    def num_tokens_from_string(self, string, encoding_name="gpt-4"):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_matches(self, source_table: BaseTable, target_table: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        source_name = source_table.name
        target_name = target_table.name

        source_table = source_table.get_df()
        target_table = target_table.get_df()
        
        refined_matches = []

        source_columns = source_table.columns
        target_columns = target_table.columns

        source_samples = {col: source_table[col].dropna().sample(
            n=min(self.sample_size, len(source_table))).tolist() for col in source_columns}
        target_samples = {col: target_table[col].dropna().sample(
            n=min(self.sample_size, len(target_table))).tolist() for col in target_columns}

        for source_col in source_columns:
            source_col_info = f"{source_col}, Sample values: {
                ', '.join(map(str, source_samples[source_col]))}"

            target_cols_info = [
                f"{target_col}, Sample values: {
                    ', '.join(map(str, target_samples[target_col]))}"
                for target_col in target_columns
            ]

            target_columns_str = "\n".join(target_cols_info)

            prompt = self._get_prompt(
                source_col_info, target_columns_str, self.include_example)

            candidate_matches = self._get_matches_w_score(prompt)

            ranked_matches = self._parse_scored_matches(candidate_matches)
            if ranked_matches:
                refined_matches.extend(ranked_matches)

        final_matches = {}
        for ref_match in refined_matches:
            match = Match(target_name, ref_match[1],
                              source_name, ref_match[0], ref_match[2]).to_dict
            final_matches.update(match)
                
        return final_matches

    def _get_prompt(self, source_col_info, target_columns_str, include_example=False):
        prompt = (
            "On a scale from 0.00 to 1.00, rate the similarity of the candidate column from the source table to each target column from the target table. "
            "Each column is identified by its name and a sample of its respective values if available. "
            "Your response should only contain in each line the source and target column names followed by their similarity scores in parentheses, formatted to two decimal places, and separated by semi-colon. Like:\n"
            "ColumnSource1;ColumnTarget1;0.95\n"
            "ColumnSource1;ColumnTarget1;0.95\n"
        )

        if include_example:
            example = (
                "\n\nExample:\n"
                "Candidate Column:\n"
                "Column: EmployeeID, Sample values: [100, 101, 102]\n"
                "Target Columns:\n"
                "Column: WorkerID, Sample values: [100, 101, 102]\n"
                "Column: EmpCode, Sample values: [001, 002, 003]\n"
                "Column: StaffName, Sample values: ['Alice', 'Bob', 'Charlie']\n"
                "Response:\n"
                "EmployeeID;WorkerID;0.95\n"
                "EmployeeID;EmpCode;0.80\n"
                "EmployeeID;StaffName;0.20\n"
            )
            prompt += example

        final = (
            "\n\nRank the column-score pairs in descending order of similarity. Do not include any additional information, explanations, or quotation marks.\n"
            f"Candidate Column:\n{source_col_info}\n\nTarget Columns:\n{
                target_columns_str}\n\nResponse: "
        )

        prompt += final
        return prompt

    def _get_matches_w_score(self, prompt):
        messages = [
            {
                "role": "system",
                "content": "You are an AI trained to perform schema matching by providing similarity scores.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.3,
        )

        matches = response.choices[0].message.content
        return matches

    def _parse_scored_matches(self, candidate_matches):
        matched_columns = []

        try:
            candidate_matches = candidate_matches.split("\n")

            for entry in candidate_matches:
                if entry.strip():  # Skip empty lines
                    col1, col2, score = entry.split(";")
                    matched_columns.append((col1, col2, float(score)))

        except Exception as e:
            # Log the exception (if you have logging setup) and return an empty list
            print(f"Error parsing scored matches: {e}")
            return []

        return matched_columns



# if __name__ == "__main__":

#     source_data = {'Name': ['Alice', 'Bob'], 'Age': [
#         25, 30], 'City': ['New York', 'Los Angeles']}
#     target_data = {'Full_Name': ['Alice Johnson', 'Bob Smith'], 'Years': [
#         25, 30], 'Location': ['New York', 'Los Angeles']}

#     source_table = pd.DataFrame(source_data)
#     target_table = pd.DataFrame(target_data)

#     matcher = GPTMatcher(llm_model="gpt-4o-mini")

#     matches = matcher.match(source_table, target_table,  include_example=True)

#     for match in matches:
#         print(match)
