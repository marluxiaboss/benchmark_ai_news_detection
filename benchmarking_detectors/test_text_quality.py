import argparse
from text_quality_evalution import Scorer, RefScorer, BertScoreScorer, SemScoreScorer, IDFScorer
from pipeline import TextQualityPipeline
import numpy as np




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset", required=True)
    parser.add_argument("--data_experiment_name", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--generator_name", type=str, help="Name of the generator", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
    args = parser.parse_args()
    
    scorer = BertScoreScorer("bert_score")
    
    dataset_name = args.dataset_name
    data_experiment_name = args.data_experiment_name
    generator_name = args.generator_name
    
    non_watermarked_dataset_path = f"data/generated_datasets/{dataset_name}/no_attack/no_watermark/{generator_name}_{data_experiment_name}"
    pipeline = TextQualityPipeline(scorer, non_watermarked_dataset_path, batch_size=args.batch_size)
    
    scores = pipeline.run_pipeline()
    print("Mean score: ", np.mean(scores))
    
    
    
    