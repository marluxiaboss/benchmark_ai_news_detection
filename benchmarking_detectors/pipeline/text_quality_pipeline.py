from tqdm import tqdm
from typing import Optional
from datasets import load_from_disk, concatenate_datasets
from .experiment_pipeline import ExperimentPipeline
from .pipeline_utils import *
from text_quality_evaluation import (Scorer, SelfScorer, RefScorer,
        BertScoreScorer, SemScoreScorer, IDFScorer, PrometheusScorer)

class TextQualityPipeline(ExperimentPipeline):
    
    def __init__(self, scorer: Scorer, dataset_path: str, dataset_path2: Optional[str]=None,
                 batch_size: int=64):
        self.scorer = scorer
        self.dataset = load_from_disk(dataset_path)
        self.batch_size = batch_size
        
        # we can eventually another dataset with AI/human pairs for providing two AI responses to compare
        if dataset_path2 is not None:
            self.dataset2 = load_from_disk(dataset_path2)

    
    def run_pipeline(self):
        
        dataset_test = concatenate_datasets([self.dataset["test"], self.dataset["eval"]])
        
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

            human_texts = [pair[0] for pair in human_ai_pairs]
            ai_texts = [pair[1] for pair in human_ai_pairs]
            
            batch_size = self.batch_size
            scores_mean, scores_lower_bound, scores_upper_bound = scorer.score_batch(ai_texts, human_texts, batch_size)
            
        elif isinstance(scorer, SelfScorer):     
            
            # we directly use AI-text here without any human reference
            ai_dataset_test = dataset_test.filter(lambda sample: sample["label"] == 1)
            ai_texts = ai_dataset_test["text"][:]
            scores_mean, scores_lower_bound, scores_upper_bound = scorer.score_batch(ai_texts)
        
        elif isinstance(scorer, PrometheusScorer):
            
            # both datasets must exist for this scorer to work
            assert self.dataset2 is not None, "Both datasets must be provided for Prometheus Scorer"
            
            dataset_test = concatenate_datasets([self.dataset["test"], self.dataset["eval"]])            
            dataset_test2 = concatenate_datasets([self.dataset2["test"], self.dataset2["eval"]])

            dataset_test_1_df = dataset_test.to_pandas()
            dataset_test_2_df = dataset_test2.to_pandas()
            
            # add column to identify the dataset
            dataset_test_1_df["dataset"] = "A"
            dataset_test_2_df["dataset"] = "B"
            
            dataset_test_df = pd.concat([dataset_test_1_df, dataset_test_2_df])
            dataset_test_grouped = dataset_test_df.groupby("prefix")

            responses_A = []
            responses_B = []
            responses_human = []
            prefixes = []

            for prefix, group in dataset_test_grouped:

                # 4 because 2 pairs of human and AI responses, one for each dataset
                if group.shape[0] != 4:
                    continue
                
                responses_A.append(group[(group["label"] == 1) & (group["dataset"] == "A")]["text"].values[0])
                responses_B.append(group[(group["label"] == 1) & (group["dataset"] == "B")]["text"].values[0])
                responses_human.append(group[(group["label"] == 0)]["text"].values[0])
                prefixes.append(prefix)
                
            instructions = [f"Continue writing the following news article starting with: {prefix}" for prefix in prefixes]
            rubric = "Is the news article convincing, coherent and well-written? Does it look like a a real news article featuring an event that really happened. Is the event mentionned in the article plausible?"
            scores_mean, scores_lower_bound, scores_upper_bound  = scorer.score_batch(responses_A, responses_B, responses_human, instructions, rubric)
            
        else:
            raise ValueError("Scorer not recognized")
        
        return scores_mean, scores_lower_bound, scores_upper_bound
        