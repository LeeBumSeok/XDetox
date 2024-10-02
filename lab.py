import os
import re
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Run rewrite script with different data types."
)
parser.add_argument("--data_type", type=str, help="The type of data to process.")
parser.add_argument(
    "--output_folder", type=str, required=True, help="Output folder for the results."
)
parser.add_argument("--all", action="store_true", help="Process all data types.")
parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode.")
parser.add_argument(
    "--overwrite_gen", action="store_true", help="Overwrite existing generation files."
)

parser.add_argument(
    "--overwrite_eval", action="store_true", help="Overwrite existing evaluation files."
)

parser.add_argument(
    "--ranking",
    action="store_true",
    help="Rewrite with ranking",
)

args = parser.parse_args()

data_configs = {
    "microagressions_val": {
        "data_path": "./datasets/microagressions/val.csv",
        "rep_penalty": 1.0,
        "alpha_a": 1.5,
        "alpha_e": 4.25,
        "temperature": 2.5,
    },
    "microagressions_test": {
        "data_path": "./datasets/microagressions/test.csv",
        "rep_penalty": 1.0,
        "alpha_a": 1.5,
        "alpha_e": 4.25,
        "temperature": 2.5,
    },
    "sbf_val": {
        "data_path": "./datasets/sbf/sbfdev.csv",
        "rep_penalty": 1.5,
        "alpha_a": 1.5,
        "alpha_e": 5.0,
        "temperature": 2.9,
    },
    "sbf_test": {
        "data_path": "./datasets/sbf/sbftst.csv",
        "rep_penalty": 1.5,
        "alpha_a": 1.5,
        "alpha_e": 5.0,
        "temperature": 2.9,
    },
    "dynabench_val": {
        "data_path": "./datasets/dynabench/db_dev.csv",
        "rep_penalty": 1.0,
        "alpha_a": 1.5,
        "alpha_e": 4.75,
        "temperature": 2.5,
    },
    "dynabench_test": {
        "data_path": "./datasets/dynabench/db_test.csv",
        "rep_penalty": 1.0,
        "alpha_a": 1.5,
        "alpha_e": 4.75,
        "temperature": 2.5,
    },
    "jigsaw_toxic": {
        "data_path": "./datasets/jigsaw_full_30/test_10k_toxic.txt",
        "rep_penalty": 1.0,
        "alpha_a": 1.5,
        "alpha_e": 4.75,
        "temperature": 2.5,
    },
    "paradetox": {
        "data_path": "./datasets/paradetox/test_toxic_parallel.txt",
        "rep_penalty": 1.0,
        "alpha_a": 1.5,
        "alpha_e": 4.75,
        "temperature": 2.5,
    },
    "appdia_original": {
        "data_path": "./datasets/appdia/original-annotated-data/original-test.tsv",
        "rep_penalty": 1.0,
        "alpha_a": 1.5,
        "alpha_e": 4.75,
        "temperature": 2.5,
    },
    "appdia_discourse": {
        "data_path": "./datasets/appdia/discourse-augmented-data/discourse-test.tsv",
        "rep_penalty": 1.0,
        "alpha_a": 1.5,
        "alpha_e": 4.75,
        "temperature": 2.5,
    }
}


def parse_folder_name(folder_name):
    pattern = r"aa(\d+\.\d+)_ae(\d+\.\d+)_ab(\d+\.\d+)_base(.*?)_anti(.*?)_expert(.*?)_temp(\d+\.\d+)_sample(.*?)_topk(\d+)_reppenalty(\d+\.\d+)_filterp(\d+\.\d+)_maxlength(\d+)_topp(\d+\.\d+)"
    match = re.match(pattern, folder_name)
    if match:
        return {
            "alpha_a": match.group(1),
            "alpha_e": match.group(2),
            "alpha_b": match.group(3),
            "base_type": match.group(4),
            "antiexpert_type": match.group(5),
            "expert_type": match.group(6),
            "temperature": match.group(7),
            "sample": match.group(8),
            "top_k": match.group(9),
            "rep_penalty": match.group(10),
            "filter_p": match.group(11),
            "max_length": match.group(12),
            "top_p": match.group(13),
        }
    else:
        return None


def all_eval_threshold(output_folder, data_type):
    for i in np.arange(0.15, 0.3, 0.05, dtype=np.float64):
        if i == 0:
            base_path = f"data/dexp_outputs/{output_folder}/{data_type}/DecompX0.0"
        else:
            base_path = (
                f"data/dexp_outputs/{output_folder}/{data_type}/DecompX{abs(i):g}"
            )
        folders = os.listdir(base_path)

        for folder in tqdm(folders, desc="Processing folders", total=len(folders)):
            params = parse_folder_name(folder)
            if params:
                orig_path = os.path.join(base_path, folder, "orig.txt")
                gen_path = os.path.join(base_path, folder, "gen.txt")

                if not (
                    os.path.exists(os.path.join(base_path, folder, "gen_stats.txt"))
                    or args.overwrite_eval
                ):
                    command = (
                        f"python3 -m evaluation.evaluate_all "
                        f"--orig_path {orig_path} "
                        f"--gen_path {gen_path} "
                    )
                    print("Executing:", command)
                    subprocess.run(command, shell=True)
                else:
                    continue
            else:
                print(f"Could not parse folder: {folder}")


def save_eval_data(output_folder, data_type):
    data = []
    for i in np.arange(0.15, 0.3, 0.05, dtype=np.float64):
        if i == 0:
            base_path = f"data/dexp_outputs/{output_folder}/{data_type}/DecompX0.0"
        else:
            base_path = (
                f"data/dexp_outputs/{output_folder}/{data_type}/DecompX{abs(i):g}"
            )
        folders = os.listdir(base_path)

        for folder in tqdm(folders, desc="Processing folders", total=len(folders)):
            params = parse_folder_name(folder)
            if params:
                with open(os.path.join(base_path, folder, "gen_stats.txt"), "r") as f:
                    lines = f.readlines()
                    stats = {
                        "folder": i,
                        "toxicity_gen": float(lines[2].split(": ")[1].strip()),
                        "bertscore": float(lines[0].split(": ")[1].strip()),
                        "perplexity_gen": float(lines[3].split(": ")[1].strip()),
                        "bleu4": float(lines[1].split(": ")[1].strip()),
                        # "toxicity_orig": float(lines[4].split(": ")[1].strip()),
                        # "perplexity_orig": float(lines[5].split(": ")[1].strip()),
                        "percent_toxic_gen": float(lines[6].split(": ")[1].strip()),
                        # "percent_toxic_ref": float(lines[7].split(": ")[1].strip()),
                    }
                    data.append(stats)

    if data:
        df = pd.DataFrame(data)
        df.to_csv(f"data/dexp_outputs/{output_folder}/{data_type}/{data_type}.csv")


def run_for_data_type(output_folder, data_type):
    config = data_configs[data_type]
    command = f"python -m rewrite.rewrite_example --output_dir data/dexp_outputs/{args.output_folder} --data_type {data_type} --data_path {config['data_path']} --rep_penalty {config['rep_penalty']} --alpha_a {config['alpha_a']} --alpha_e {config['alpha_e']} --temperature {config['temperature']}"
    if args.ranking:
        command += " --ranking"
    if "jigsaw" in config["data_path"]:
        command += " --batch_size 10"
    if "paradetox" in config["data_path"]:
        command += " --batch_size 10"
    if "appdia" in config["data_path"]:
        command += " --batch_size 10"

    for i in np.arange(0.15, 0.3, 0.05, dtype=np.float64):
        if i == 0:
            base_path = f"data/dexp_outputs/{output_folder}/{data_type}/DecompX0.0"
        else:
            base_path = (
                f"data/dexp_outputs/{output_folder}/{data_type}/DecompX{abs(i):g}"
            )

        try:
            folders = os.listdir(base_path)
        except FileNotFoundError:
            full_command = command + f" --thresh {i:.2f}"
            print("Executing:", full_command)
            subprocess.run(full_command, shell=True)
            continue

        for folder in tqdm(folders, desc="Processing folders", total=len(folders)):
            params = parse_folder_name(folder)
            if params:
                if not (
                    os.path.exists(os.path.join(base_path, folder, "gen.txt"))
                    or args.overwrite_gen
                ):
                    full_command = command + f" --thresh {i:.2f}"
                    print("Executing:", full_command)
                    subprocess.run(full_command, shell=True)
                else:
                    print("Exiting, since generation already exists")
                    continue


if __name__ == "__main__":
    if args.all:
        for data_type in data_configs.keys():
            run_for_data_type(args.output_folder, data_type)
            if args.evaluate:
                all_eval_threshold(args.output_folder, data_type)
                save_eval_data(args.output_folder, data_type)
    elif args.data_type:
        if args.data_type in data_configs:
            run_for_data_type(args.output_folder, args.data_type)
            all_eval_threshold(args.output_folder, args.data_type)
            save_eval_data(args.output_folder, args.data_type)
        else:
            print(f"No configuration found for data type '{args.data_type}'.")
    else:
        print("No data type specified.")
