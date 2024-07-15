from .experiment_pipeline import ExperimentPipeline
from .pipeline_utils import *
from text_quality_evalution import Scorer, RefScorer, BertScoreScorer, SemScoreScorer, IDFScorer

class TextQualityPipeline(ExperimentPipeline):
    
    def __init__(self, scorer: Scorer, dataset_path: str):
        self.scorer = scorer
        self.dataset = load_from_disk(dataset_path)
    
    def run_pipeline(self):
        
        dataset_test = self.dataset["test"]
        
        scorer = self.scorer
        
        if isinstance(scorer, RefScorer):
            dataset_test_df = dataset_test.to_pandas()
            dataset_test_grouped = dataset_test_df.groupby("prefix")

            human_ai_pairs = []

            for prefix, group in dataset_test_grouped:

                if group.shape[0] != 2:
                    continue
                
                ai_text = group[group["label"] == 1]["text"].values[0]
                human_text = group[group["label"] == 0]["text"].values[0]
                human_ai_pairs.append((human_text, ai_text))

            scores = []
            for human_text, ai_text in human_ai_pairs:
                score = scorer.score(ai_text, human_text)
                scores.append(score)
            
            return scores
        
        else:      
            ai_dataset_test = dataset_test.filter(lambda sample: sample["label"] == 1)
            #human_dataset_test = dataset_test["test"].filter(lambda sample: sample["label"] == 0)
            
            ai_texts = ai_dataset_test["text"][:]
            #human_texts = human_dataset_test["text"][:]
            
            scores = []
            
            for ai_text in ai_texts:
                score = scorer.score(ai_text)
                scores.append(score)
                
            return scores
        