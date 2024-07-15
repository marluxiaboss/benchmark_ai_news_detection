import bert_score
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from abc import ABC, abstractmethod


class Scorer(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod 
    def score(self, eval_text: str) -> float:
        pass

class RefScorer(Scorer):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod 
    def score(self, eval_texts: str, ref_text: str) -> float:
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
    
    def score_batch(self, eval_texts: list[str], ref_texts: list[str]) -> float:
        cands = eval_texts
        refs = ref_texts
        precision, recall, f1_score = bert_score.score(cands, refs, lang='en', model_type=self.model, num_layers=self.num_layers, rescale_with_baseline=True)
        return f1_score.item()
    
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
    
class IDFScorer(Scorer):
    
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
        
        
        
        