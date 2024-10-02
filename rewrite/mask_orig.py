import torch
from transformers import AutoTokenizer
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import display, HTML
from DecompX.src.decompx_utils import DecompXConfig
from DecompX.src.modeling_bert import BertForSequenceClassification
from DecompX.src.modeling_roberta import RobertaForSequenceClassification


class Masker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.toxicity_tokenizer = AutoTokenizer.from_pretrained(
            "s-nlp/roberta_toxicity_classifier"
        )
        self.toxicity_model = RobertaForSequenceClassification.from_pretrained(
            "s-nlp/roberta_toxicity_classifier"
        ).to(self.device)

        self.DecompXCONFIGS = {
            "DecompX": DecompXConfig(
                include_biases=True,
                bias_decomp_type="absdot",
                include_LN1=True,
                include_FFN=True,
                FFN_approx_type="GeLU_ZO",
                include_LN2=True,
                aggregation="vector",
                include_classifier_w_pooler=True,
                tanh_approx_type="ZO",
                output_all_layers=True,
                output_attention=None,
                output_res1=None,
                output_LN1=None,
                output_FFN=None,
                output_res2=None,
                output_encoder=None,
                output_aggregated="norm",
                output_pooler="norm",
                output_classifier=True,
            ),
        }

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def release_model(self):
        self.toxicity_model.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def classify_sentence(self, sentence):
        is_single_sentence = isinstance(sentence, str)

        if is_single_sentence:
            sentence = [sentence]

        if len(sentence) == 1:
            sentence = [sentence[0], sentence[0]]

        tokenized_sentence = self.toxicity_tokenizer(
            sentence, return_tensors="pt", padding=True
        ).to(self.device)
        batch_lengths = tokenized_sentence["attention_mask"].sum(dim=-1)

        with torch.no_grad():
            self.toxicity_model.eval()
            (
                logits,
                hidden_states,
                decompx_last_layer_outputs,
                decompx_all_layers_outputs,
            ) = self.toxicity_model(
                **tokenized_sentence,
                output_attentions=False,
                return_dict=False,
                output_hidden_states=True,
                decompx_config=self.DecompXCONFIGS["DecompX"],
            )

        decompx_outputs = {
            "tokens": [
                self.toxicity_tokenizer.convert_ids_to_tokens(
                    tokenized_sentence["input_ids"][i][: batch_lengths[i]]
                )
                for i in range(len(sentence))
            ],
            "logits": logits.cpu().detach().numpy().tolist(),  # (batch, classes)
            "cls": hidden_states[-1][:, 0, :]
            .cpu()
            .detach()
            .numpy()
            .tolist(),  # Last layer & only CLS -> (batch, emb_dim)
        }

        ### decompx_last_layer_outputs.aggregated ~ (1, 8, 55, 55) ###
        importance = np.array(
            [
                g.squeeze().cpu().detach().numpy()
                for g in decompx_last_layer_outputs.aggregated
            ]
        ).squeeze()  # (batch, seq_len, seq_len)

        importance = [
            importance[j][: batch_lengths[j], : batch_lengths[j]]
            for j in range(len(importance))
        ]
        decompx_outputs["importance_last_layer_aggregated"] = importance

        ### decompx_last_layer_outputs.pooler ~ (1, 8, 55) ###
        importance = np.array(
            [
                g.squeeze().cpu().detach().numpy()
                for g in decompx_last_layer_outputs.pooler
            ]
        ).squeeze()  # (batch, seq_len)
        importance = [importance[j][: batch_lengths[j]] for j in range(len(importance))]
        decompx_outputs["importance_last_layer_pooler"] = importance

        ### decompx_last_layer_outputs.classifier ~ (8, 55, 2) ###
        importance = np.array(
            [
                g.squeeze().cpu().detach().numpy()
                for g in decompx_last_layer_outputs.classifier
            ]
        ).squeeze()  # (batch, seq_len, classes)
        importance = [
            importance[j][: batch_lengths[j], :] for j in range(len(importance))
        ]
        decompx_outputs["importance_last_layer_classifier"] = importance

        ### decompx_all_layers_outputs.aggregated ~ (12, 8, 55, 55) ###
        importance = np.array(
            [
                g.squeeze().cpu().detach().numpy()
                for g in decompx_all_layers_outputs.aggregated
            ]
        )  # (layers, batch, seq_len, seq_len)
        importance = np.einsum(
            "lbij->blij", importance
        )  # (batch, layers, seq_len, seq_len)
        importance = [
            importance[j][:, : batch_lengths[j], : batch_lengths[j]]
            for j in range(len(importance))
        ]
        decompx_outputs["importance_all_layers_aggregated"] = importance

        decompx_outputs_df = pd.DataFrame(decompx_outputs)

        return decompx_outputs_df

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def mask_low_importance_words(
        self, importance, tokenized_text, tokenizer, threshold=-0.4
    ):
        output = []

        importance = importance / np.abs(importance).max() / 1.5  # Normalize
        masked_text = []
        i = 0

        while i < len(tokenized_text):
            end = i + 1
            while end < len(tokenized_text) and not tokenized_text[end].startswith("Ġ"):
                end += 1

            should_mask = any(importance[j] > threshold for j in range(i, end))

            if should_mask:
                masked_text.append(" " + tokenizer.mask_token)
            else:
                for j in range(i, end):
                    masked_text.append(tokenized_text[j].replace("Ġ", " "))

            i = end

        masked_sentence = "".join(masked_text).strip()
        output.append(masked_sentence)

        return output

    def process_text(self, sentence, threshold):
        NO_CLS_SEP = False
        df = self.classify_sentence(sentence)
        output = []

        for _, row in df.iterrows():
            for col in ["importance_last_layer_classifier"]:
                if col in df and row[col] is not None:
                    sentence_importance = row[col][:, 1]
                    if not NO_CLS_SEP:
                        sentence_importance = sentence_importance[1:-1]
                        tokenized_text = row["tokens"][1:-1]
                    else:
                        tokenized_text = row["tokens"]

                    masked_text = self.mask_low_importance_words(
                        sentence_importance,
                        tokenized_text,
                        self.toxicity_tokenizer,
                        threshold=threshold,
                    )
                    output.append(" ".join(masked_text))

        return output
