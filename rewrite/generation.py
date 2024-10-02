import argparse
from pathlib import Path
from typing import Union, List
import os
from numpy import ufunc
from transformers import BartForConditionalGeneration, BartTokenizer
from IPython import embed
from .infilling import *
from utils import *
import nltk.tokenize.casual
import torch
import torch.nn.functional as F
import sys
from . import gen_utils
from . import generation_logits_process
import pandas as pd
import functools
import operator
from tqdm import tqdm
from .mask_orig import Masker as Masker_single

"""
Infiller module
- Initialize with a base model, antiexpert (optional), expert (optional)
- If expert_type == "none", don't use an expert. Same for antiexpert
"""


class Infiller:
    def __init__(
        self,
        seed=0,
        base_path="facebook/bart-base",
        antiexpert_path="facebook/bart-base",
        expert_path="facebook/bart-base",
        base_type="base",
        antiexpert_type="antiexpert",
        expert_type="expert",
        tokenizer="facebook/bart-base",
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if not torch.cuda.is_available():
            print("No GPUs found!")
        else:
            print("Found", str(torch.cuda.device_count()), "GPUS!")

        self.seed = seed
        seed_everything(self.seed)

        # Initalize tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)

        # Save mask info
        self.mask = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

        model_map = {
            "base": base_path,
            "antiexpert": antiexpert_path,
            "expert": expert_path,
        }

        # Initialize models
        self.base_model = None
        if base_type != "none":
            self.base_model = BartForConditionalGeneration.from_pretrained(
                model_map[base_type], forced_bos_token_id=self.tokenizer.bos_token_id
            ).to(self.device)

        self.antiexpert = None
        if antiexpert_type != "none":
            self.antiexpert = BartForConditionalGeneration.from_pretrained(
                model_map[antiexpert_type],
                forced_bos_token_id=self.tokenizer.bos_token_id,
            ).to(self.device)

        self.expert = None
        if expert_type != "none":
            self.expert = BartForConditionalGeneration.from_pretrained(
                model_map[expert_type], forced_bos_token_id=self.tokenizer.bos_token_id
            ).to(self.device)

    """
    Public generate model
    Parameters
    * inputs - list of text inputs
    * inputs_masked - list of text inputs with potentially toxic tokens masked
    * max_length - maximum length to generate too
    * sample - whether to sample or not
    * filter_p - nucleus sampling parameter on base logits
    * k - top_k parameter
    * p - nucleus sampling parameter on ensembled logits
    * temperature - for sampling
    * alpha_a - weight on antiexpert for ensenmbling distributions during decoding
    * alpha_e - weight on expert for ensenmbling distributions during decoding
    * alpha_b - weight on base model for ensenmbling distributions during decoding
    * repetition_penalty - how much to penalize repetition
    * batch_size - how many seqs to generate at once
    * verbose - whether or not to print generations during generation time
    """

    def generate(
        self,
        inputs: Union[str, List[str]],
        inputs_masked: Union[str, List[str]],
        max_length: int = 128,
        sample: bool = False,
        filter_p: float = 1.0,
        k: int = 0,
        p: float = 1.0,
        temperature: float = 1.0,
        alpha_a: float = 0.0,
        alpha_e: float = 0.0,
        alpha_b: float = 1.0,
        repetition_penalty: float = 1.0,
        batch_size=50,
        verbose=False,
        ranking=False,
        ranking_eval_output=20,
    ):
        # Initialize repetition penalty processor
        rep_penalty_proc = (
            generation_logits_process.RepetitionPenaltyLogitsProcessor(
                penalty=repetition_penalty
            )
            if repetition_penalty != 1.0
            else None
        )

        # Set models to eval
        if self.base_model:
            self.base_model.eval()
        if self.expert:
            self.expert.eval()
        if self.antiexpert:
            self.antiexpert.eval()

        final_outputs = []
        sentence_outputs_dict = {}

        with torch.no_grad():
            if ranking:
                sample = True
                max_seq_lens = 0
                NO_CLS_SEP = False
                classifier = Masker_single()

                extended_inputs = [
                    (sentence, masked)
                    for sentence, masked in zip(inputs, inputs_masked)
                    for _ in range(ranking_eval_output)
                ]
                batched_inputs = [
                    extended_inputs[i : i + batch_size]
                    for i in range(0, len(extended_inputs), batch_size)
                ]

                for batch_idx, batch in tqdm(
                    enumerate(batched_inputs),
                    total=len(batched_inputs),
                    desc="Filling in masks",
                ):
                    batch_inputs, batch_inputs_masked = zip(*batch)
                    outputs = gen(
                        inputs=list(batch_inputs),
                        inputs_masked=list(batch_inputs_masked),
                        tokenizer=self.tokenizer,
                        model=self.base_model,
                        expert=self.expert,
                        antiexpert=self.antiexpert,
                        alpha_a=alpha_a,
                        alpha_e=alpha_e,
                        alpha_b=alpha_b,
                        max_length=max_length,
                        verbose=verbose,
                        temperature=temperature,
                        rep_proc=rep_penalty_proc,
                        device=self.device,
                        k=k,
                        p=p,
                        filter_p=filter_p,
                        sample=sample,
                    )
                    decoded_outputs = self.tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                    for i, (raw_output, decoded_output) in enumerate(
                        zip(outputs, decoded_outputs)
                    ):
                        original_sentence_idx = (
                            batch_idx * batch_size + i
                        ) // ranking_eval_output
                        if original_sentence_idx not in sentence_outputs_dict:
                            sentence_outputs_dict[original_sentence_idx] = {
                                "raw_outputs": [],
                                "decoded_outputs": [],
                            }
                        sentence_outputs_dict[original_sentence_idx][
                            "raw_outputs"
                        ].append(raw_output)
                        sentence_outputs_dict[original_sentence_idx][
                            "decoded_outputs"
                        ].append(decoded_output)

                for i in tqdm(range(len(inputs)), desc="Ranking"):
                    df = classifier.classify_sentence(
                        sentence_outputs_dict[i]["decoded_outputs"]
                    )
                    min_importance_sum = float("inf")
                    min_row_index = -1

                    for idx, row in df.iterrows():
                        for col in ["importance_last_layer_classifier"]:
                            if col in df and row[col] is not None:
                                sentence_importance = row[col][:, 1]
                                if not NO_CLS_SEP:
                                    sentence_importance = sentence_importance[1:-1]

                                sentence_importance = (
                                    sentence_importance
                                    / np.abs(sentence_importance).max()
                                    / 1.5
                                )
                                importance_sum = sentence_importance.sum()

                                if importance_sum < min_importance_sum:
                                    min_importance_sum = importance_sum
                                    min_row_index = idx

                    selected_outputs = []
                    selected_output = sentence_outputs_dict[i]["raw_outputs"][
                        min_row_index
                    ]
                    selected_outputs.append(selected_output)

                    concatenated_output = torch.cat(
                        [output.unsqueeze(0) for output in selected_outputs], dim=0
                    )

                    max_seq_lens = max(max_seq_lens, concatenated_output.shape[1])
                    final_outputs.append(concatenated_output)

                processed_outputs = []
                for f in final_outputs:
                    if f.dim() == 1:
                        f = f.unsqueeze(0)
                    processed_outputs.append(f)

                padded_outputs = [
                    torch.nn.functional.pad(
                        f,
                        pad=(0, max_seq_lens - f.shape[1]),
                        value=self.tokenizer.pad_token_id,
                    )
                    for f in processed_outputs
                ]
                final_outputs = torch.cat(padded_outputs, dim=0)
            else:
                if len(inputs) <= batch_size:
                    outputs = gen(
                        inputs=inputs,
                        inputs_masked=inputs_masked,
                        tokenizer=self.tokenizer,
                        model=self.base_model,
                        expert=self.expert,
                        antiexpert=self.antiexpert,
                        alpha_a=alpha_a,
                        alpha_e=alpha_e,
                        alpha_b=alpha_b,
                        max_length=max_length,
                        verbose=verbose,
                        temperature=temperature,
                        rep_proc=rep_penalty_proc,
                        device=self.device,
                        k=k,
                        p=p,
                        filter_p=filter_p,
                        sample=sample,
                    )
                    final_outputs = outputs
                else:
                    max_seq_lens = 0
                    for i in tqdm(
                        range(0, len(inputs), batch_size), desc="Filling in masks"
                    ):
                        cur_inputs = inputs[i : i + batch_size]
                        cur_inputs_masked = inputs_masked[i : i + batch_size]

                        outputs = gen(
                            inputs=cur_inputs,
                            inputs_masked=cur_inputs_masked,
                            tokenizer=self.tokenizer,
                            model=self.base_model,
                            expert=self.expert,
                            antiexpert=self.antiexpert,
                            alpha_a=alpha_a,
                            alpha_e=alpha_e,
                            alpha_b=alpha_b,
                            max_length=max_length,
                            verbose=verbose,
                            temperature=temperature,
                            rep_proc=rep_penalty_proc,
                            device=self.device,
                            k=k,
                            p=p,
                            filter_p=filter_p,
                            sample=sample,
                        )

                        max_seq_lens = max(max_seq_lens, outputs.shape[1])
                        final_outputs.append(outputs)

                    final_outputs = [
                        torch.nn.functional.pad(
                            f,
                            pad=(0, max_seq_lens - f.shape[1]),
                            value=self.tokenizer.pad_token_id,
                        )
                        for f in final_outputs
                    ]

                    final_outputs = torch.cat(final_outputs)
        # Return both the tokenized outputs and the decoded outputs
        return final_outputs, self.tokenizer.batch_decode(
            final_outputs, skip_special_tokens=True
        )


# Private method for generation that is called by generate()
def gen(
    inputs,
    inputs_masked,
    tokenizer,
    model,
    expert,
    antiexpert,
    alpha_a: float = 0.0,
    alpha_e: float = 0.0,
    alpha_b: float = 1.0,
    max_length: int = 128,
    device=torch.device("cuda"),
    verbose: bool = False,
    sample: bool = False,
    filter_p: float = 1.0,
    k: int = 0,
    p: float = 1.0,
    temperature: float = 1.0,
    rep_proc=None,
):
    # Convert inputs to list if they aren't already
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(inputs_masked, list):
        inputs = [inputs_masked]

    assert len(inputs) == len(inputs_masked)

    # Tokenize - the regular inputs, and the masked inputs
    batch = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
    batch_masked = tokenizer(inputs_masked, return_tensors="pt", padding=True).to(
        device
    )

    if verbose:
        print("ORIGINAL \t")
        print(tokenizer.batch_decode(batch.input_ids))
        print("\t")
        print(tokenizer.batch_decode(batch_masked.input_ids))

    # Keep track of which generations aren't finished yet
    unfinished_sents = torch.ones(len(inputs), dtype=torch.int32, device=device)

    # Start off our outputs with the eos token id, then the bos token id (match how BART generates)
    outputs = (
        torch.Tensor([tokenizer.eos_token_id, tokenizer.bos_token_id])
        .expand(len(inputs), -1)
        .long()
        .to(device)
    )
    start_length = 2

    loop_idx = 0
    # Substract start length from max length, since we start with 2 tokens
    while loop_idx < (max_length - start_length):
        # Compute the logits for base, antiexpert, and expert
        # Base model sees the nonmasked inputs, expert and antiexpert see the masked inputs
        base_logits = model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=outputs,
        ).logits
        antiexpert_logits = antiexpert.forward(
            input_ids=batch_masked["input_ids"],
            attention_mask=batch_masked["attention_mask"],
            decoder_input_ids=outputs,
        ).logits
        expert_logits = expert.forward(
            input_ids=batch_masked["input_ids"],
            attention_mask=batch_masked["attention_mask"],
            decoder_input_ids=outputs,
        ).logits

        if verbose:
            print("Current outputs\n\t", tokenizer.batch_decode(outputs))
            print("Base\n")
            for idxs in torch.topk(base_logits[:, -1, :], 5, dim=-1).indices:
                print("\t", tokenizer.batch_decode(idxs))
            # print("Base masked", tokenizer.batch_decode(torch.topk(base_logits2[:,-1,:], 10).indices[0]))
            print("Anti\n")
            for idxs in torch.topk(antiexpert_logits[:, -1, :], 5, dim=-1).indices:
                print("\t", tokenizer.batch_decode(idxs))
            print("Expert\n")
            for idxs in torch.topk(expert_logits[:, -1, :], 5, dim=-1).indices:
                print("\t", tokenizer.batch_decode(idxs))
            # print("Expert nonmasked", tokenizer.batch_decode(torch.topk(expert_logits2[:,-1,:], 10).indices[0]))

        # eos_predicted = torch.argmax(base_logits[:,-1,:], dim=-1) == tokenizer.eos_token_id

        # top_p filtering on the base logits
        if filter_p < 1.0:
            base_logits = gen_utils.top_k_top_p_filtering(base_logits, top_p=filter_p)

        # Change values of the logits with the temperature
        # Temperature (higher temperature => more likely to sample low probability tokens)
        if temperature != 1.0:
            base_logits = base_logits / temperature

        # Ensemble the logits and get the next token logits
        ensemble_logits = (
            alpha_b * base_logits
            + alpha_e * expert_logits
            - alpha_a * antiexpert_logits
        )
        next_token_logits = ensemble_logits[:, -1, :]

        # Add repetition penalty
        if rep_proc is not None:
            next_token_logits = rep_proc(outputs, next_token_logits)

        # Sample or greedily decode from the next_token_logits
        if sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            # if temperature != 1.0:
            #     next_token_logits = next_token_logits / temperature
            if k > 0 or p < 1.0:
                next_token_logits = gen_utils.top_k_top_p_filtering(
                    next_token_logits, top_k=k, top_p=p
                )
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        # Get the tokens to add and identify sentences that are done generating
        tokens_to_add = next_tokens * unfinished_sents + tokenizer.pad_token_id * (
            1 - unfinished_sents
        )
        eos_in_sents = tokens_to_add == tokenizer.eos_token_id
        unfinished_sents.mul_((~eos_in_sents).int())

        # Update the outputs and the loop index
        outputs = torch.cat((outputs, tokens_to_add.unsqueeze(-1)), dim=-1)
        loop_idx += 1

        if verbose:
            print("Ensemble\n")
            for idxs in torch.topk(ensemble_logits[:, -1, :], 5, dim=-1).indices:
                print("\t", tokenizer.batch_decode(idxs))
            print("Next token:", tokenizer.batch_decode(tokens_to_add))

        # Stop generation when there is an EOS in each sentence
        if unfinished_sents.max() == 0:
            break

    if verbose:
        decodes = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("MINE:")
        for d in decodes:
            print("\t", d)
        generated_ids = model.generate(
            batch_masked["input_ids"],
            max_length=max_length,
            num_beams=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        for step, scores in enumerate(generated_ids.scores):
            # softmax를 적용하여 logits를 확률로 변환합니다.
            probs = scores.softmax(dim=-1)
            top_probs, top_ids = probs[0].topk(10)

            print(f"Step {step + 1}:")
            for prob, token_id in zip(top_probs, top_ids):
                token = tokenizer.decode(token_id, skip_special_tokens=True)
                print(f"\tToken: {token}, Probability: {prob.item()}")

        generated_ids = model.generate(
            batch_masked["input_ids"],
            max_length=max_length,
            num_beams=1,
            do_sample=False,
        )

        output = "\n\t".join(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        )

        print("INPUT\n\t", "\n\t".join(inputs))
        print("\nbase OUTPUT\n\t", output)

        generated_ids = expert.generate(
            batch_masked["input_ids"],
            max_length=max_length,
            num_beams=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        for step, scores in enumerate(generated_ids.scores):
            probs = scores.softmax(dim=-1)
            top_probs, top_ids = probs[0].topk(10)

            print(f"Step {step + 1}:")
            for prob, token_id in zip(top_probs, top_ids):
                token = tokenizer.decode(token_id, skip_special_tokens=True)
                print(f"\tToken: {token}, Probability: {prob.item()}")

        generated_ids = expert.generate(
            batch_masked["input_ids"],
            max_length=max_length,
            num_beams=1,
            do_sample=False,
        )
        output = "\n\t".join(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        )
        print("\nexpert OUTPUT\n\t", output)

        generated_ids = antiexpert.generate(
            batch_masked["input_ids"],
            max_length=max_length,
            num_beams=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        for step, scores in enumerate(generated_ids.scores):
            # softmax를 적용하여 logits를 확률로 변환합니다.
            probs = scores.softmax(dim=-1)
            top_probs, top_ids = probs[0].topk(10)

            print(f"Step {step + 1}:")
            for prob, token_id in zip(top_probs, top_ids):
                token = tokenizer.decode(token_id, skip_special_tokens=True)
                print(f"\tToken: {token}, Probability: {prob.item()}")

        generated_ids = antiexpert.generate(
            batch_masked["input_ids"],
            max_length=max_length,
            num_beams=1,
            do_sample=False,
        )
        output = "\n\t".join(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        )
        print("\nanti expert OUTPUT\n\t", output)

    return outputs


if __name__ == "__main__":
    # Some examples of using the generate method on some masked inputs with alpha_a = 0.5, alpha_e = 2.5, and temperature = 1.25
    # If you want to run just the Infiller and generate method from the command line, you can modify the below to take in a list of masked inputs, process them, and feed them into the generate method
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_a", type=float, default=1.5)
    parser.add_argument("--alpha_e", type=float, default=4.75)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--alpha_b", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--ranking",
        action="store_true",
        help="Rewrite with ranking",
    )

    rewriter = Infiller(
        seed=0,
        base_path="facebook/bart-base",
        antiexpert_path="hallisky/bart-base-toxic-antiexpert",
        expert_path="hallisky/bart-base-nontoxic-expert",
        base_type="base",
        antiexpert_type="antiexpert",
        expert_type="expert",
        tokenizer="facebook/bart-base",
    )
    args = parser.parse_args()

    inputs = ["I absolutely hate trans people"]
    inputs_masked = [
        "I absolutely <mask> trans <mask>",
    ]

    outputs, decoded_outputs = rewriter.generate(
        inputs,
        inputs_masked,
        alpha_a=args.alpha_a,
        alpha_e=args.alpha_e,
        temperature=args.temperature,
        verbose=args.verbose,
        alpha_b=args.alpha_b,
        ranking=args.ranking,
    )

    print(
        "inputs:",
        inputs,
        "\nmasked inputs:",
        inputs_masked,
        "\noutputs:",
        decoded_outputs,
    )
