import argparse
from text_quality_evalution import Scorer, RefScorer, BertScoreScorer, SemScoreScorer, IDFScorer
from pipeline import TextQualityPipeline
import numpy as np




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", required=True)
    args = parser.parse_args()
    
    scorer = BertScoreScorer("bert_score")
    pipeline = TextQualityPipeline(scorer, args.dataset_path)
    
    scores = pipeline.run_pipeline()
    
    print("Mean score: ", np.mean(scores))
    
    