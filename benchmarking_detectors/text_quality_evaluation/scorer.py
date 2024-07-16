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
        return f1_scores.tolist()
    
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
            
        return scores
        
        
        
class PrometheusScorer(CompareScorer):
    # use either judge LM or prometheus LM
    # Cavetat: Only works with VLLM!
        
    def __init__(self, name):
        self.name = name
        self.model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
        self.judge = PrometheusEval(model=self.model, relative_grade_template=RELATIVE_PROMPT)

        
    def score(self, eval_text1: str, eval_text2: str, ref_text: Optional[str]=None) -> float:
        pass
    
    def score_batch(self, eval_texts1: list[str], eval_texts2: list[str], ref_texts: list[str],
        instructions: list[str], rubric: str,  batch_size=1, compare_human_to_ai: bool=False) -> float:
        
        
        eval_texts1_with_index = [(text, "A") for text in eval_texts1]
        eval_texts2_with_index = [(text, "B") for text in eval_texts2]
        
        # create a new list with shuffled elements so that we avoid the bias for the first element
        eval_text1_shuffled = []
        eval_texts2_shuffled = []
        
        if len(eval_texts1) != len(eval_texts2):
            raise ValueError("The number of eval_texts1 and eval_texts2 must be equal!")
        
        for i in range(len(eval_texts1)):
            
            # draw a random number between 0 and 1
            random_number = np.random.randint(0, 2)
            
            if random_number == 0:
                eval_text1_shuffled.append(eval_texts1_with_index[i])
                eval_texts2_shuffled.append(eval_texts2_with_index[i])
            else:
                eval_text1_shuffled.append(eval_texts2_with_index[i])
                eval_texts2_shuffled.append(eval_texts1_with_index[i])
            
        if compare_human_to_ai:
            
            eval_texts1_with_index = [(text, "A") for text in eval_texts1]
            ref_texts_with_index = [(text, "B") for text in ref_texts]
            
            eval_texts1_shuffled = []
            ref_texts_shuffled = []
            
            if len(eval_texts1) != len(ref_texts):
                raise ValueError("The number of eval_texts1 and ref_texts must be equal!")
            
            for i in range(len(eval_texts1)):
                
                # draw a random number between 0 and 1
                random_number = np.random.randint(0, 2)
                
                if random_number == 0:
                    eval_texts1_shuffled.append(eval_texts1_with_index[i])
                    ref_texts_shuffled.append(ref_texts_with_index[i])
                else:
                    eval_texts1_shuffled.append(ref_texts_with_index[i])
                    ref_texts_shuffled.append(eval_texts1_with_index[i])
                    
            
            feedbacks, scores = self.judge.relative_grade(
                instructions=instructions,
                responses_A=eval_texts1_shuffled,
                responses_B=ref_texts_shuffled,
                rubric=rubric
            )
            true_scores = []
            
            for i in range(len(scores)):
                score = scores[i]
                
                if eval_texts1_shuffled[i][1] == "A":
                    true_scores.append(score)
                else:
                    print("Flipping the score")
                    # if A was in B, flip the score
                    if score == "A":
                        true_scores.append("B")
                    else:
                        true_scores.append("A")
        else:
            feedbacks, scores = self.judge.relative_grade(
                instructions=instructions,
                responses_A=eval_text1_shuffled,
                responses_B=eval_texts2_shuffled,
                rubric=rubric,
                reference_answers=ref_texts
            )
            
            # retrieve the true feedback and scores
            true_scores = []
            for i in range(len(feedbacks)):
                score = scores[i]
                
                if eval_text1_shuffled[i][1] == "A":
                    true_scores.append(score)
                else:
                    print("Flipping the score")
                    # if A was in B, flip the score
                    if score == "A":
                        true_scores.append("B")
                    else:
                        true_scores.append("A")
                    
        scores = true_scores
        
        for feedback, score in zip(feedbacks, scores):
            print(f"Feedback: {feedback}")
            print(f"Score: {score}")
            print()
        
        # score in scores B are either "Score: A" or "Score: B"
        count_A = 0
        count_B = 0
        
        for score in scores:
            if score == "A":
                count_A += 1
            elif score == "B":
                count_B += 1
                
        # compute the ratio of A to B
        
        score = count_A / (count_A + count_B)
        
        
        score_array = [1 if score == "A" else 0 for score in scores]
        
        # compute CI with bootstrapp resampling
        
        n_bootstraps = 1000
        
        means = []
        for i in range(n_bootstraps):
            bootstrap_sample = np.random.choice(score_array, len(score_array), replace=True)
            winrate_A = np.mean(bootstrap_sample)
            means.append(winrate_A)
            
        lower_bound = np.percentile(means, 2.5)
        upper_bound = np.percentile(means, 97.5)
        

        
        return score, lower_bound, upper_bound