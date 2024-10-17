from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import json
import random

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


def generate_column_name_pairs(data, num_samples=10):

    positive_pairs = []
    for column_name, variations in data.items():
        for variation_type in variations:
            for variation in variations[variation_type]:
                positive_pairs.append((column_name, variation, 1))

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


def load_pairs(file_path):
    data = load_json_file(file_path)
    pairs = generate_column_name_pairs(data)

    return pairs


# Example usage
if __name__ == "__main__":

    current_path = os.path.dirname(__file__)
    file_path = os.path.join(current_path, 'gdc_column_variations.json')

    print("Loading column name pairs and generating negatives...")
    pairs = load_pairs(file_path)
    print(f"Generated {len(pairs)} pairs")
    # for pair in pairs:
    #     print(pair)

    # Sample 1000 pairs from the generated pairs
    sampled_pairs = random.sample(pairs, min(4000, len(pairs)))
    pairs = sampled_pairs

    fine_tuned_model = fine_tune_sentence_transformer(
        train_pairs=pairs,
        output_path='./models/gdc_column_header_fine_tuned_model'
    )

    # # Test the model
    test_sentences = ["EmployeeName", "WorkerID", "Salary"]
    embeddings = fine_tuned_model.encode(test_sentences)

    print("Test embeddings shape:", embeddings.shape)
    print("Sample embedding:", embeddings[0][:5])
