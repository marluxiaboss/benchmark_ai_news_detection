from tqdm import tqdm
from typing import Optional
from datasets import load_from_disk, concatenate_datasets
import pandas as pd

from .experiment_pipeline import ExperimentPipeline
from .pipeline_utils import *
from ..text_quality_evaluation import (
    Scorer,
    SelfScorer,
    RefScorer,
    BertScoreScorer,
    SemScoreScorer,
    IDFScorer,
    PrometheusScorer,
)


class TextQualityPipeline(ExperimentPipeline):
    """
    Pipeline to evaluate the quality of text.
    """

    def __init__(
        self,
        scorer: Scorer,
        dataset_path: str,
        dataset_path_compare: Optional[str] = None,
        batch_size: int = 64,
        return_loss_lists: bool = False,
        eval_human: bool = False,
    ):
        """
        Initialize the pipeline.

        Parameters:
        ----------
            scorer: Scorer
                The scorer to use.
            dataset_path: str
                The path to the dataset.
        """
        self.scorer = scorer
        self.dataset = load_from_disk(dataset_path)
        self.batch_size = batch_size
        self.return_loss_lists = return_loss_lists
        self.eval_human = eval_human

        # we can eventually another dataset with AI/human pairs for providing two AI responses to compare
        if dataset_path_compare is not None:
            self.dataset_path_compare = load_from_disk(dataset_path_compare)

    def run_pipeline(self):

        dataset_test = concatenate_datasets([self.dataset["test"], self.dataset["eval"]])

        scorer = self.scorer

        if isinstance(scorer, RefScorer):

            # check if object has attribute self.dataset_path_compare
            if hasattr(self, "dataset_path_compare"):
                # both datasets must exist for this scorer to work
                assert (
                    self.dataset_path_compare is not None
                ), "Both datasets must be provided for Prometheus Scorer"

                dataset_test = concatenate_datasets([self.dataset["test"], self.dataset["eval"]])
                dataset_test2 = concatenate_datasets(
                    [self.dataset_path_compare["test"], self.dataset_path_compare["eval"]]
                )

                dataset_test_1_df = dataset_test.to_pandas()
                dataset_test_2_df = dataset_test2.to_pandas()

                # add column to identify the dataset
                dataset_test_1_df["dataset"] = "A"
                dataset_test_2_df["dataset"] = "B"

                dataset_test_df = pd.concat([dataset_test_1_df, dataset_test_2_df])
                dataset_test_grouped = dataset_test_df.groupby("prefix")

                responses_A = []
                responses_B = []

                for prefix, group in dataset_test_grouped:

                    # 4 because 2 pairs of human and AI responses, one for each dataset
                    if group.shape[0] != 4:
                        continue

                    responses_A.append(
                        group[(group["label"] == 1) & (group["dataset"] == "A")]["text"].values[0]
                    )
                    responses_B.append(
                        group[(group["label"] == 1) & (group["dataset"] == "B")]["text"].values[0]
                    )

                test_texts = responses_A
                ref_texts = responses_B

            else:

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

                test_texts = ai_texts
                ref_texts = human_texts

            batch_size = self.batch_size
            # scores_mean, scores_lower_bound, scores_upper_bound = scorer.score_batch(ai_texts, test_texts, ref_texts, batch_size=batch_size)
            scores_mean, scores_lower_bound, scores_upper_bound = scorer.score_batch(
                test_texts, ref_texts, batch_size=batch_size
            )

        elif isinstance(scorer, SelfScorer):

            # flag to decide if we want to evaluate human responses or AI responses
            if self.eval_human:
                ai_dataset_test = dataset_test.filter(lambda sample: sample["label"] == 0)
            else:
                ai_dataset_test = dataset_test.filter(lambda sample: sample["label"] == 1)
            ai_texts = ai_dataset_test["text"][:]
            batch_size = self.batch_size
            return_loss_lists = self.return_loss_lists
            if return_loss_lists:
                (
                    scores_mean,
                    scores_lower_bound,
                    scores_upper_bound,
                    loss_lists,
                ) = scorer.score_batch(
                    ai_texts, batch_size=batch_size, return_loss_lists=return_loss_lists
                )
            else:
                scores_mean, scores_lower_bound, scores_upper_bound = scorer.score_batch(
                    ai_texts, batch_size=batch_size
                )

        elif isinstance(scorer, PrometheusScorer):

            # both datasets must exist for this scorer to work
            assert (
                self.dataset_path_compare is not None
            ), "Both datasets must be provided for Prometheus Scorer"

            dataset_test = concatenate_datasets([self.dataset["test"], self.dataset["eval"]])
            dataset_test2 = concatenate_datasets(
                [self.dataset_path_compare["test"], self.dataset_path_compare["eval"]]
            )

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

                responses_A.append(
                    group[(group["label"] == 1) & (group["dataset"] == "A")]["text"].values[0]
                )
                responses_B.append(
                    group[(group["label"] == 1) & (group["dataset"] == "B")]["text"].values[0]
                )
                responses_human.append(group[(group["label"] == 0)]["text"].values[0])
                prefixes.append(prefix)

            instructions = [
                f"Continue writing the following news article starting with: {prefix}"
                for prefix in prefixes
            ]
            rubric = "Is the news article convincing, coherent and well-written? Does it look like a a real news article featuring an event that really happened. Is the event mentionned in the article plausible?"
            scores_mean, scores_lower_bound, scores_upper_bound = scorer.score_batch(
                responses_A, responses_B, responses_human, instructions, rubric
            )

        else:
            raise ValueError("Scorer not recognized")

        if self.return_loss_lists:
            return scores_mean, scores_lower_bound, scores_upper_bound, loss_lists
        else:
            return scores_mean, scores_lower_bound, scores_upper_bound
