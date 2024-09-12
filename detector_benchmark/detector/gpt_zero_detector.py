import os
import requests

from .detector import Detector


class GPTZero(Detector):
    def __init__(self, api_key, debug_mode=False) -> None:
        """
        Initialize the GPTZero detector.

        Parameters:
        ----------
            api_key: str
                The API key
            debug_mode: bool
                Whether to print the debug information
        """
        self.api_key = api_key
        if self.api_key is None:
            raise ValueError(
                "GPT Zero API key is missing, set it as environment variable GPT_ZERO_API_KEY"
            )

        self.debug_mode = debug_mode

    def predict_gpt_zero(self, text, api_key, debug_mode=False) -> dict:
        """
        Predict the GPT-Zero score for the text.

        Parameters:
        ----------
            text: str
                The text to predict
            api_key: str
                The API key
            debug_mode: bool
                Whether to print the debug information

        Returns:
        ----------
            dict
                The prediction result
        """
        url = "https://api.gptzero.me/v2/predict/text"
        payload = {"document": text, "version": "2024-04-04", "multilingual": False}
        headers = {
            "Accept": "application/json",
            "content-type": "application/json",
            "x-api-key": api_key,
        }

        while True:
            try:
                # 1 request per 10 minutes for free access

                # 0.06 should correspond to 1000 requests per 1 minute
                # time.sleep(0.06)
                response = requests.post(url, json=payload, headers=headers)

                if debug_mode:
                    print(response.json())

                # return response.json()['documents'][0]['completely_generated_prob']

                # try to access document
                # response_doc = response.json()["documents"][0]
                return response.json()
            except Exception as ex:
                print("Issue with prediction, skipping (see error below):")
                print("response: ", response)
                print(ex)

                # better to skip since no point in retrying
                return None

    def detect(
        self, texts: list, batch_size: int, detection_threshold: float = 0.5
    ) -> tuple[list[int], list[float], list[int]]:
        """
        Detect the GPT-Zero score for the texts.

        Parameters:
        ----------
            texts: list
                The texts to detect
            batch_size: int
                The batch size
            detection_threshold: float
                The threshold to use for the detection

        Returns:
        ----------
            tuple[list[int], list[float], list[int]]
                The predictions, the logits for the positive class, and the predictions at the threshold
        """
        api_key = self.api_key

        # iterate over the dataset
        pred_res_list = []

        for text in texts:

            pred_json = self.predict_gpt_zero(text, api_key=api_key, debug_mode=self.debug_mode)

            # if prediction failed
            if pred_json is None:
                pred_res = {}
                pred_res["text"] = text
                pred_res["pred"] = None
                pred_res["prob"] = None
                pred_res["pred_at_threshold"] = None

                pred_res_list.append(pred_res)

            else:

                pred_json_doc = pred_json["documents"][0]
                pred_class = pred_json_doc["predicted_class"]

                if pred_class == "human":
                    pred = 0

                elif pred_class == "ai":
                    pred = 1

                elif pred_class == "mixed":

                    pred_score_ai = pred_json_doc["class_probabilities"]["ai"]
                    pred_score_human = pred_json_doc["class_probabilities"]["human"]
                    pred = 1 if pred_score_ai > pred_score_human else 0

                    # if mixed is higher prob than human and ai, set to 1
                    if (
                        pred_json_doc["class_probabilities"]["mixed"] > pred_score_ai
                        and pred_json_doc["class_probabilities"]["mixed"] > pred_score_human
                    ):
                        pred = 1

                else:
                    raise ValueError("Unknown class")

                # record probability for positive class (mixed considered as positive class)
                prob = (
                    pred_json_doc["class_probabilities"]["ai"]
                    + pred_json_doc["class_probabilities"]["mixed"]
                )

                # compute the prediction at threshold given as argument
                pred_at_threshold = 1 if prob > detection_threshold else 0

                # create prediction res dict to tie results to text
                pred_res = {}

                pred_res["text"] = text
                pred_res["pred"] = pred
                pred_res["prob"] = prob
                pred_res["pred_at_threshold"] = pred_at_threshold

                pred_res_list.append(pred_res)

        preds = [elem["pred"] for elem in pred_res_list]
        logits_pos_class = [elem["prob"] for elem in pred_res_list]
        preds_at_threshold = [elem["pred_at_threshold"] for elem in pred_res_list]

        return preds, logits_pos_class, preds_at_threshold
