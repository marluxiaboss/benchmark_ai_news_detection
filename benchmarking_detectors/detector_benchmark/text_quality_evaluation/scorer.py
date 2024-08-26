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
import evaluate

# for prometheus 
from .scorer_utils import bootstrap_score

# uncomment to use prometheus
# commented for now because of vllm

#from prometheus_eval.vllm import VLLM
#from prometheus_eval import PrometheusEval
#from prometheus_eval.prompts import RELATIVE_PROMPT

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
    
class PPLScorer(SelfScorer):
    
    def __init__(self, name, model, tokenizer):
        self.name = name
        
        self.scorer_model = model
        self.scorer_tokenizer = tokenizer
        self.device = model.device
        self.metric = evaluate.load("perplexity", module_type="metric")
        
    def score(self, eval_text: str) -> float:
        pass
    
    def score_batch(self, eval_texts: list[str], batch_size=1, return_ppl_list) -> float:
        """
        See https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/main/perplexity.py
        """

        model = self.scorer_model
        tokenizer = self.scorer_tokenizer
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        max_length = None
        max_tokenized_len = max_length

        encodings = tokenizer(
            eval_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(model.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        
        per_token_ppl = []

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]
            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            # take the logits of all the tokens except the last one
            shift_logits = out_logits[..., :-1, :].contiguous()
            
            # take all the "next tokens" except the first one
            shift_labels = labels[..., 1:].contiguous()
            
            # logits and labels are always shifted by one
            # ie. when we want to compute the loss for the second token, we use the logits of the first token
            # when we want to compute the loss for the third token, we use the logits of the first and second token etc.
            
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )
            
            ppls += perplexity_batch.tolist()
            
            if return_ppl_list:
                
                # compute perplexity at each token to have a list of perplexities
                perplexity_batch = [[] for _ in range(batch_size)]
                for i in range(batch_size):
                    for j in range(shift_logits.size(1)):
                        perplexity_batch[i].append(torch.exp(loss_fct(shift_logits[i, j], shift_labels[i, j])))
                
                per_token_ppl.extend(perplexity_batch)
        
        mean_score, lower_bound, upper_bound = bootstrap_score(ppls)
        
        if return_ppl_list:
            return mean_score, lower_bound, upper_bound, per_token_ppl
        else:
            return mean_score, lower_bound, upper_bound

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


#TODO: maybe use it later, ignore it for now since very similar to BERT score
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
            
        scores_mean, scores_lower_bound, scores_upper_bound = bootstrap_score(scores)
        return scores_mean, scores_lower_bound, scores_upper_bound
        
class PrometheusScorer(CompareScorer):
    # use either judge LM or prometheus LM
    # Cavetat: Only works with VLLM!
        
    def __init__(self, name, compare_human_to_ai: bool=False):
        self.name = name
        self.model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
        self.judge = PrometheusEval(model=self.model, relative_grade_template=RELATIVE_PROMPT)
        self.compare_human_to_ai = compare_human_to_ai
        
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
                # if A was in B, flip the score
                if score == "A":
                    true_scores.append("B")
                else:
                    true_scores.append("A")
                    
        return true_scores
    
    def score_batch(self, eval_texts1: list[str], eval_texts2: list[str], ref_texts: list[str],
        instructions: list[str], rubric: str,  batch_size=1) -> float:
        
        compare_human_to_ai = self.compare_human_to_ai
        
        # when we compare human text to AI text directly, set the text to compare to human text
        if compare_human_to_ai:
            eval_texts1 = eval_texts1
            eval_texts2 = ref_texts
        
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