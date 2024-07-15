import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", required=True)
    args = parser.parse_args()
    
    
    TextQualityPipeline