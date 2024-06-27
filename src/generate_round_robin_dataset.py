from datasets import concatenate_datasets, load_from_disk, DatasetDict
import argparse
import os



def create_round_robbin_dataset(datasets, take_samples=-1, seed=42):
    """
    Create a round robbin dataset from the given datasets
    """

    if take_samples > 0:
        datasets = [dataset.select(range(int(take_samples))) for dataset in datasets]

    dataset = concatenate_datasets(datasets)
    dataset = dataset.shuffle(seed=seed)
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, help="Directory to save the model and logs", default="fake_true_datasets")
    parser.add_argument("--nb_samples_per_dataset", type=int, help="Number of samples to take from each dataset in round robin training", default=2500)
    args = parser.parse_args()

    nb_samples_per_dataset = args.nb_samples_per_dataset

    base_dataset_path = args.save_dir
    datasets_names = ["fake_true_dataset_phi_10k", "fake_true_dataset_gemma_10k", "fake_true_dataset_mistral_10k"]
    datasets = [load_from_disk(f"{base_dataset_path}/{dataset_name}") for dataset_name in datasets_names]
    
    nb_samples_per_dataset = nb_samples_per_dataset
    datasets_train = [dataset["train"] for dataset in datasets]
    dataset_train = create_round_robbin_dataset(datasets_train, take_samples=nb_samples_per_dataset, seed=42)

    nb_samples_per_dataset = nb_samples_per_dataset / 10
    datasets_valid = [dataset["valid"] for dataset in datasets]
    dataset_valid = create_round_robbin_dataset(datasets_valid, take_samples=nb_samples_per_dataset, seed=42)

    nb_samples_per_dataset = nb_samples_per_dataset / 10
    datasets_test = [dataset["test"] for dataset in datasets]
    dataset_test = create_round_robbin_dataset(datasets_test, take_samples=nb_samples_per_dataset, seed=42)

    dataset = DatasetDict({"train": dataset_train, "valid": dataset_valid, "test": dataset_test})
    dataset.save_to_disk("./fake_true_datasets/fake_true_dataset_round_robin_10k")

