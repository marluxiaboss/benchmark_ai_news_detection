import bert_score
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# for prometheus 
from .scorer_utils import bootstrap_score
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import RELATIVE_PROMPT

from abc import ABC, abstractmethod


class Scorer(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod 
    def score(self) -> float:
        pass

class RefScorer(Scorer):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod 
    def score(self, eval_texts: str, ref_text: str) -> float:
        pass
    
class SelfScorer(Scorer):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod 
    def score(self, eval_texts: str) -> float:
        pass
    
class CompareScorer(Scorer):
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def score(self, eval_text1: str, eval_text2: str, ref_text: Optional[str]=None) -> float:
        pass
    

class BertScoreScorer(RefScorer):
    def __init__(self, name):
        super().__init__(name)
        
        self.model = "microsoft/deberta-xlarge-mnli"
        self.num_layers = 40
        
    def score(self, eval_text: str, ref_text: str) -> float:
        cands = [eval_text]
        refs = [ref_text]
        precision, recall, f1_score = bert_score.score(cands, refs, lang='en', model_type=self.model, num_layers=self.num_layers, rescale_with_baseline=True)
        return f1_score.item()
    
    def score_batch(self, eval_texts: list[str], ref_texts: list[str], batch_size) -> float:
        cands = eval_texts
        refs = ref_texts
        precision, recall, f1_scores = bert_score.score(cands, refs, lang='en', model_type=self.model, num_layers=self.num_layers, rescale_with_baseline=True, batch_size=batch_size, verbose=True)
        
        f1_score_mean, f1_score_lower_bound, f1_score_upper_bound = bootstrap_score(f1_scores)
        return f1_score_mean, f1_score_lower_bound, f1_score_upper_bound
    
"""
TODO: maybe use it later, ignore it for now since very similar to BERT score
class SemScoreScorer(RefScorer):
    def __init__(self, name):
        super().__init__(name)
        
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')        
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def score(self, eval_text: str, ref_text: str) -> float:
        tokenized_text = self.tokenizer([ref_text, eval_text], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**tokenized_text)
        embeds = self.mean_pooling(model_output, tokenized_text['attention_mask'])
        sentence_embeddings = F.normalize(embeds, p=2, dim=1)
        cosine_scores = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
        return cosine_scores.item()
"""
class IDFScorer(SelfScorer):
    
    def __init__(self, name, corpus: list[str]):
        super().__init__(name)
        self.corpus = corpus
        
        nltk.download('stopwords')
        
        # remove stopwords from the corpus
        filtered_corpus = self.remove_stopwords(self.corpus)
        self.filtered_corpus = filtered_corpus
        
        # Initialize and fit the TfidfVectorizer
        # Note: Sk learn's TF-IDF does log(N_doc / N_doc where term appear + 1) 
        # where N_doc and N_doc where term appear include the eval_sentence.
        vectorizer = TfidfVectorizer()
        vectorizer.fit(filtered_corpus)
        
        # Create a dictionary mapping words to their IDF values
        feature_names = vectorizer.get_feature_names_out()
        idf_values = vectorizer.idf_
        self.word_to_idf = dict(zip(feature_names, idf_values))
        
    def remove_stopwords(self, corpus: list[str]):
        
        filtered_corpus = []
        # Remove stopwords from the corpus
        for sentence in corpus:
            tokenized_sentence = sentence.split()
            filtered_sentence = [word for word in tokenized_sentence if word not in stopwords.words('english')]
            filtered_corpus.append(" ".join(filtered_sentence))
            
        # drop empty sentences
        filtered_corpus = [sentence for sentence in filtered_corpus if sentence]
        
        return filtered_corpus
        
    def score(self, eval_text: str) -> float:
        
        filtered_eval_text = self.remove_stopwords([eval_text])[0]
        
        # Compute the average IDF of the words in the sentence
        tokenized_sentence = filtered_eval_text.split()
        idfs = [self.word_to_idf.get(word, 0) for word in tokenized_sentence]
        average_idf = np.mean(idfs)
        median_idf = np.median(idfs)
        
        return median_idf
    
    def score_batch(self, eval_texts: list[str], batch_size=1) -> float:
        
        scores = []
        for text in tqdm(eval_texts, desc="Scoring..."):
            score = self.score(text)
            scores.append(score)
            
        scores_mean, scores_lower_bound, scores_upper_bound = bootstrap_score(scores)
        return scores_mean, scores_lower_bound, scores_upper_bound
        
class PrometheusScorer(CompareScorer):
    # use either judge LM or prometheus LM
    # Cavetat: Only works with VLLM!
        
    def __init__(self, name):
        self.name = name
        self.model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
        self.judge = PrometheusEval(model=self.model, relative_grade_template=RELATIVE_PROMPT)

        
    def score(self, eval_text1: str, eval_text2: str, ref_text: Optional[str]=None) -> float:
        pass
    
    def shuffle_positions(self, text_list1: list[str], text_list2: list[str]) -> tuple[list[str], list[str]]:
    
        assert len(text_list1) == len(text_list2), "The number of elements in text_list1 and text_list2 must be equal!"
        
        # create a new list with shuffled elements so that we avoid the bias for the first element
        text_list1_shuffled = []
        text_list2_shuffled = []
        
        for i in range(len(text_list1)):
            
            # draw a random number between 0 and 1
            random_number = np.random.randint(0, 2)
            
            if random_number == 0:
                text_list1_shuffled.append(text_list1[i])
                text_list2_shuffled.append(text_list2[i])
            else:
                text_list1_shuffled.append(text_list2[i])
                text_list2_shuffled.append(text_list1[i])
                
        return text_list1_shuffled, text_list2_shuffled


    def reassign_scores(self, scores: list[int], text_list1: list, text_list2: list) -> list[int]:
        
        true_scores = []
        
        for i in range(len(scores)):
            score = scores[i]
            
            if text_list1[i][1] == "A":
                true_scores.append(score)
            else:
                print("Flipping the score")
                # if A was in B, flip the score
                if score == "A":
                    true_scores.append("B")
                else:
                    true_scores.append("A")
                    
        return true_scores
    
    def score_batch(self, eval_texts1: list[str], eval_texts2: list[str], ref_texts: list[str],
        instructions: list[str], rubric: str,  batch_size=1, compare_human_to_ai: bool=False) -> float:
        
        # when we compare human text to AI text directly, set the text to compare to human text
        if compare_human_to_ai:
            eval_texts1 = ref_texts
            eval_texts2 = eval_texts1
        
        eval_texts1_with_index = [(text, "A") for text in eval_texts1]
        eval_texts2_with_index = [(text, "B") for text in eval_texts2]
        
        text_list1_shuffled, text_list2_shuffled = self.shuffle_positions(eval_texts1_with_index, eval_texts2_with_index)
            
        # use human text as reference only if we don't compare human to AI directly
        feedbacks, scores = self.judge.relative_grade(
            instructions=instructions,
            responses_A=text_list1_shuffled,
            responses_B=text_list2_shuffled,
            rubric=rubric,
            reference_answers=None if compare_human_to_ai else ref_texts,
        )
        
        scores = self.reassign_scores(scores, text_list1_shuffled, text_list2_shuffled)
        
        for feedback, score in zip(feedbacks, scores):
            print(f"Feedback: {feedback}")
            print(f"Score: {score}")
            print()
        
        score_array = [1 if score == "A" else 0 for score in scores]
        
        # compute_CI using bootstrap resampling
        score, lower_bound, upper_bound = bootstrap_score(score_array)
        
        return score, lower_bound, upper_bound