import numpy as np
from qaoa_pipeline import *
import pandas as pd
from metrics import *
import os
import re
from datetime import datetime
from collections import defaultdict
import pickle 

def run():

    np.random.seed(1234)

    folder_bf = 'FOLDER CONTAINING BRUTE FORCE SOLUTIONS'
    folder_params = 'FOLDER CONTAINING TRAINED QAOA ANGLES'
    folder_output = 'OUTPUT FOLDER'

    # -----------------------------
    # Filename patterns
    # -----------------------------
    filename_pattern_bf = re.compile(
        r'^candidate_space_(?P<candidate_space>\d+)_candidates_50_busses_'
        r'(?P<load>\d+\.\d+)_load_factor_'
        r'(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_'
        r'(?P<instance>\d+)\.csv$'
    )

    filename_pattern_params = re.compile(
        r'^candidate_space_(?P<candidate_space>\d+)_candidates_50_busses_'
        r'(?P<load>\d+\.\d+)_load_factor_'
        r'(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_'
        r'(?P<instance>\d+)_p0_(?P<p>\d+)\.pkl$'
    )

    # -----------------------------
    # Unified parser
    # -----------------------------
    def parse_filename(filename):
        for pattern, filetype in [
            (filename_pattern_bf, "bf"),
            (filename_pattern_params, "params"),
        ]:
            match = pattern.match(filename)
            if match:
                data = match.groupdict()
                return {
                    "type": filetype,
                    "candidate_space": int(data["candidate_space"]),
                    "load": float(data["load"]),
                    "timestamp": datetime.strptime(data["timestamp"], "%Y-%m-%d_%H-%M-%S"),
                    "instance": int(data["instance"]),
                    "p": int(data["p"]) if "p" in data and data["p"] else None,
                    "filename": filename,
                }
        return None

    # -----------------------------
    # New structure
    # -----------------------------
    files_dict = {
        "bf": defaultdict(lambda: defaultdict(list)),
        "params": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }

    # -----------------------------
    # Load BF files
    # -----------------------------
    for f in os.listdir(folder_bf):
        parsed = parse_filename(f)
        if parsed and parsed["type"] == "bf":
            cs = parsed["candidate_space"]
            load = parsed["load"]
            files_dict["bf"][cs][load].append(parsed)

    # -----------------------------
    # Load PARAM files
    # -----------------------------
    for f in os.listdir(folder_params):
        parsed = parse_filename(f)
        if parsed and parsed["type"] == "params":
            cs = parsed["candidate_space"]
            load = parsed["load"]
            p = parsed["p"]

            files_dict["params"][cs][load][p].append(parsed)

    # -----------------------------
    # Sorting
    # -----------------------------
    def sort_key(parsed):
        return (
            parsed["candidate_space"],
            parsed["load"],
            parsed["instance"],
            parsed["p"] if parsed["p"] is not None else -1,
        )

    # sort bf
    for cs in files_dict["bf"]:
        for load in files_dict["bf"][cs]:
            files_dict["bf"][cs][load] = sorted(
                files_dict["bf"][cs][load], key=sort_key
            )

    # sort params
    for cs in files_dict["params"]:
        for load in files_dict["params"][cs]:
            for p in files_dict["params"][cs][load]:
                files_dict["params"][cs][load][p] = sorted(
                    files_dict["params"][cs][load][p], key=sort_key
                )

    # -----------------------------
    # Example usage
    # -----------------------------
    # all BF instances
    # files_dict["bf"][2][0.2]

    # all params (flattened if needed)
    # [f for p in files_dict["params"][2][0.2].values() for f in p]

    # specific p
    # files_dict["params"][2][0.2][128]


    # Calculate all possibilities
    def collect_results(files_dict, folder_bf, folder_params, metric_fn):
        results = defaultdict(  # cs
            lambda: defaultdict(  # load
                lambda: defaultdict(  # p
                    dict  # instance -> value
                )
            )
        )

        for cs in files_dict["bf"]:
            if cs not in files_dict["params"]:
                continue

            common_loads = set(files_dict["bf"][cs]).intersection(files_dict["params"][cs])

            for load in common_loads:
                for p in files_dict["params"][cs][load]:

                    # choose ONE params file (as before)
                    params_entry = files_dict["params"][cs][load][p][0]
                    params_file = folder_params + params_entry["filename"]

                    for bf_entry in files_dict["bf"][cs][load]:
                        instance = bf_entry["instance"]
                        bf_file = folder_bf + bf_entry["filename"]

                        value = metric_fn(bf_file, params_file)

                        results[cs][load][p][instance] = value

        return results

    def to_regular_dict(d):
        if isinstance(d, defaultdict):
            d = {k: to_regular_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            d = {k: to_regular_dict(v) for k, v in d.items()}
        return d

    def save_results(results, filename="results.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(to_regular_dict(results), f)

    def load_results(filename="results.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def compute_tts_star(results_tts):
        results_star = defaultdict(  # cs
            lambda: defaultdict(  # load
                dict  # instance -> best TTS
            )
        )

        for cs in results_tts:
            for load in results_tts[cs]:
                
                # collect all p values
                p_dict = results_tts[cs][load]

                # get all instances (assume consistent across p)
                instances = set()
                for p in p_dict:
                    instances.update(p_dict[p].keys())

                for instance in instances:
                    best = np.inf

                    for p in p_dict:
                        if instance in p_dict[p]:
                            val = p_dict[p][instance]
                            if val < best:
                                best = val

                    results_star[cs][load][instance] = best

        return results_star
    print('Collecting RAAR results')
    results_raar = collect_results(files_dict, folder_bf, folder_params, raar_qaoa)
    save_results(results_raar, folder_output+'raar_qaoa.pkl')

    print('Collecting TTS results')
    results_tts = collect_results(files_dict, folder_bf, folder_params, tts_qaoa)
    save_results(results_tts, folder_output+'tts_qaoa.pkl')

    print('Collecting TTS* results')
    for cs, loads in results_tts.items():
        for load, p_dict in loads.items():
            if 256 in p_dict:
                del p_dict[256]

    results_tts_star = compute_tts_star(results_tts)
    save_results(results_tts_star, folder_output+'tts_star_qaoa.pkl')

    print('Collecting AR results')
    results_ar = collect_results(files_dict, folder_bf, folder_params, ar_qaoa)
    save_results(results_ar, folder_output+'ar_qaoa.pkl')

if __name__ == '__main__':
    run()