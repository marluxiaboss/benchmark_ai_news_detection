import numpy as np
from torch.utils.data import DataLoader
import glob
import json
import torch
from tqdm import tqdm
from datasets import Dataset
import os

from .detector import Detector


class FastDetectGPT(Detector):
    def __init__(self, ref_model, scoring_model, ref_tokenizer, scoring_tokenizer, device, detection_threshold = 0.5):
        self.ref_model = ref_model
        self.scoring_model = scoring_model
        self.ref_tokenizer = ref_tokenizer
        self.scoring_tokenizer = scoring_tokenizer
        self.device = device
        self.detection_threshold = detection_threshold
        
    def get_samples(logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        nsamples = 10000
        lprobs = torch.log_softmax(logits, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs)
        samples = distrib.sample([nsamples]).permute([1, 2, 0])
        return samples

    def get_likelihood(logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        return log_likelihood.mean(dim=1)

    def get_sampling_discrepancy(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        samples = self.get_samples(logits_ref, labels)
        log_likelihood_x = (logits_score, labels)
        log_likelihood_x_tilde = self.get_likelihood(logits_score, samples)
        miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
        sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
        discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
        return discrepancy.item()
    
    def get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()
    
    class ProbEstimatorFastDetectGPT:
        def __init__(self, args=None, ref_path=None):
            if args is None:
                ref_path = ref_path
            else:
                ref_path = args.ref_path
            self.real_crits = []
            self.fake_crits = []
            for result_file in glob.glob(os.path.join(ref_path, '*.json')):
                with open(result_file, 'r') as fin:
                    res = json.load(fin)
                    self.real_crits.extend(res['predictions']['real'])
                    self.fake_crits.extend(res['predictions']['samples'])
            print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')

        def crit_to_prob(self, crit):
            offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
            cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
            cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
            return cnt_fake / (cnt_real + cnt_fake)

            
    def detect(self, texts: list, batch_size: int) -> list:
        reference_model_name = "gpt-neo-2.7B"
        scoring_model_name = "gpt-neo-2.7B"
        
        ref_path = "detector/local_infer_ref"
        device = self.device

        ref_model = self.ref_model
        ref_tokenizer = self.ref_tokenizer
        
        scoring_model = self.scoring_model
        scoring_tokenizer = self.scoring_tokenizer
        

                
        # evaluate criterion
        #name = "sampling_discrepancy_analytic"
        criterion_fn = self.get_sampling_discrepancy_analytic
        prob_estimator = self.ProbEstimatorFastDetectGPT(ref_path=ref_path)


        # iterate over the dataset and do detection on each sample
        dataset = Dataset.from_dict({"text": texts})

        # create dataloader
        batch_size = 1
        
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        preds = []
        probs = []
        preds_at_threshold = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Performing detection on dataset..."):
                text = batch["text"]
                print("text: ", text)
                tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
                labels = tokenized.input_ids[:, 1:]
                logits_score = scoring_model(**tokenized).logits[:, :-1]

                if reference_model_name == scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = ref_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = ref_model(**tokenized).logits[:, :-1]

                for i in range(batch_size):
                    crit = criterion_fn(logits_ref[i:i+1], logits_score[i:i+1], labels[i:i+1])
                    prob = prob_estimator.crit_to_prob(crit)
                    pred = 1 if prob > 0.5 else 0
                    pred_at_threshold = 1 if prob > self.detection_threshold else 0
                    
                    probs.append(prob)
                    preds.append(pred)
                    preds_at_threshold.append(pred_at_threshold)
                    

        preds = np.array(preds)
        probs = np.array(probs)
        
        logits_pos_class = probs
        
        return preds, logits_pos_class, preds_at_threshold
            
