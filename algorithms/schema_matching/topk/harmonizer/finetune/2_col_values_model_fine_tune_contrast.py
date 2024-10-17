from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import json
import random
import itertools

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fine_tune_sentence_transformer(
    model_name='sentence-transformers/all-mpnet-base-v2',
    train_pairs=None,
    output_path='./models/fine_tuned_model',
    num_epochs=10,
    batch_size=32,
    learning_rate=2e-5
):

    if train_pairs is None:
        return

    # Create training examples
    train_examples = []

    for col1, col2, label in train_pairs:
        # Create InputExample with the pair and their similarity label
        train_examples.append(InputExample(
            texts=[col1, col2],
            label=float(label)  # Convert label to float for loss function
        ))

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )

    model = SentenceTransformer(model_name)

    train_loss = losses.ContrastiveLoss(
        model=model,
        margin=0.5
    )

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.02)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        output_path=output_path,
        show_progress_bar=True
    )

    print(f"Model fine-tuned and saved to {output_path}")
    return model


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def generate_column_name_pairs(value_data, columnname_data, num_samples=3):

    result = {}

    positive_pairs = []
    # semi_positive_pairs = []
    negative_pairs = []

    for column, column_values in value_data.items():

        # Find the corresponding column in columnname_data

        variations = []
        if column in columnname_data:
            col_variations = columnname_data[column]["Equivalents"] + \
                columnname_data[column]["Acronyms"] + \
                columnname_data[column]["Patterns"]
            if len(col_variations) >= 2:
                sampled_variations = random.sample(col_variations, 2)
                col_variations = sampled_variations

        else:

            continue

        all_column_values = column_values.keys()

        for value, variations in column_values.items():

            all_variations = variations['Equivalents'] + \
                variations['Acronyms'] + variations['Patterns']

            for value_variations in all_variations:
                example = (column+': ' + value, column +
                           ': '+value_variations, 1.0)
                positive_pairs.append(example)

                for col_variation in col_variations:
                    example = (column+': ' + value, col_variation +
                               ': '+value_variations, 1.0)
                    positive_pairs.append(example)
                    example = (col_variation+': ' + value,
                               col_variation+': '+value_variations, 1.0)
                    positive_pairs.append(example)

            # # assume the values of the same domain as similar
            # for other_values in all_column_values:
            #     if value != other_values:
            #         # semi_positive_pairs.append((value, other_values, 0.5))
            #         positive_pairs.append((value, other_values, 1.0))

    negative_pairs = []
    for current_tuple in positive_pairs:

        different_first = [
            t for t in positive_pairs if t[0] != current_tuple[0]]

        num_to_sample = min(num_samples, len(different_first))
        if num_to_sample > 0:
            sampled_tuples = random.sample(different_first, num_to_sample)

            for sampled_tuple in sampled_tuples:
                # Randomly choose whether to use first or second element from each tuple
                current_element = random.choice(
                    [current_tuple[0], current_tuple[1]])
                other_element = random.choice(
                    [sampled_tuple[0], sampled_tuple[1]])

                negative_pairs.append((current_element, other_element, 0))

    pairs = positive_pairs + negative_pairs

    return list(set(pairs))


def load_pairs(value_file_path, columnname_file_path):
    value_data = load_json_file(value_file_path)
    columnname_data = load_json_file(columnname_file_path)
    pairs = generate_column_name_pairs(value_data, columnname_data)

    return pairs


# Example usage
if __name__ == "__main__":

    current_path = os.path.dirname(__file__)
    value_file_path = os.path.join(
        current_path, 'gdc_value_variations_cont_full.json')

    columnname_file_path = os.path.join(
        current_path, 'gdc_column_variations.json')

    print("Loading column name pairs and generating negatives...")
    pairs = load_pairs(value_file_path, columnname_file_path)
    print(f"Generated {len(pairs)} pairs")

    # # Sample 1000 pairs from the generated pairs
    sampled_pairs = random.sample(pairs, min(20000, len(pairs)))
    pairs = sampled_pairs

    # for pair in pairs:
    #     print(pair)

    fine_tuned_model = fine_tune_sentence_transformer(
        train_pairs=pairs,
        output_path='./models/gdc_values_fine_tuned_model'
    )

    # # # Test the model
    # test_sentences = ["EmployeeName", "WorkerID", "Salary"]
    # embeddings = fine_tuned_model.encode(test_sentences)

    # print("Test embeddings shape:", embeddings.shape)
    # print("Sample embedding:", embeddings[0][:5])
