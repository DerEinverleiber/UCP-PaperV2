import numpy as np
from qaoa_pipeline import *
import os
import re
import time 
import pickle
from pathlib import Path
from datetime import datetime 

def run(seed=1234):

    # set parameters and load data
    np.random.seed(seed)
    p_vals = [16, 32, 64, 128]
    # repalce with path of brute force data
    folder = 'FOLDER CONTAINING BRUTE FORCE DATA' # brute force data is too large to be stored on github
    files = sorted(os.listdir(folder))
    # old:
    # candidate_space_{2**N}_candidates_50_busses_{load_factor}_load_factor_{instance}_seed_{timestamp}.csv
    # new:
    # candidate_space_{2**N}_candidates_50_busses_{load_factor}_load_factor_{timestamp}_{instance}.csv


    # hierarchical sort by candidate space and load factor
    def sort_key(filename): 
        size = int(re.search(r'candidate_space_(\d+)_candidates', filename).group(1))
        load = float(re.search(r'_busses_(\d+\.\d+)_load_factor', filename).group(1))
        #instance = int(re.search(r'load_factor_(\d+)_seed', filename).group(1))
        instance = int(re.search(r'_(\d+)\.csv$', filename).group(1))
        
        return (size, load, instance)

    files_sorted = sorted(files, key=sort_key)

    #for file in files_sorted:
    #    print(file)

    from collections import defaultdict 

    # group files by (num_generators, load)
    groups = defaultdict(list)

    for file in files:
        num_generators = int(re.search(r'candidate_space_(\d+)_candidates', file).group(1))
        load = float(re.search(r'_busses_(\d+\.\d+)_load_factor', file).group(1))  
        groups[(num_generators, load)].append(file)

    output_folder = Path('OUTPUT FOLDER')
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "errors.log"

    for num_generators in sorted(set(k[0] for k in groups)): # iterate through all num_generator_vals
        #num_generators = int(np.log(size))
        for load in sorted(set(k[1] for k in groups if k[0] == num_generators)): # iterate through all loads
            candidates = groups[(num_generators, load)] # load all candidates
            for p0 in p_vals: # for each number of layers
                chosen_file = np.random.choice(candidates) # randomly choose file
                print(datetime.now(), flush=True)
                print(f"p0={p0}, num_generators={num_generators}, load={load} -> {chosen_file}", flush=True)
                output_file = output_folder / f"{Path(chosen_file).stem}_p0_{p0}.pkl"
                if output_file.exists():
                    print(f"Skipping existing result: {output_file.name}")
                    continue
                try:
                    # optimize QAOA parameters
                    # data is sorted in computational basis {[0, 0, ...,0], [0, 0, ..., 1], ..., [1, 1, ...., 1]}
                    stepsize = 0.01
                    opt_steps = 25
                    p_max = p0*2
                    delta_p = p0//8
                    gammas = np.linspace(0, 1, p0) 
                    betas = np.linspace(1, 0, p0)
                    params = (gammas, betas) # initialize QAOA parameters (ramp)
                    C = p_max // 8 # number of Chebyshev coefficients
                    
                    file_path = os.path.join(folder, chosen_file)
                    costs = np.loadtxt(
                        file_path,
                        delimiter=",",
                        skiprows=1,   # skip header
                        usecols=1     # second column = "loss"
                    )
                    # Min-max normalization
                    costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
                
                    # Perform Iterative Interpolation 
                    qaoa = QAOA_circuit(n=int(np.log2(num_generators)), p=p0, costs=costs)
                    cheby_optimizer = ChebyshevOptimizer(qaoa, C, stepsize=stepsize)
                    cheby_optimizer.optimizer = qml.AdamOptimizer(stepsize) # used optimizer

                    II = IterativeInterpolation(
                        params, qaoa, cheby_optimizer,
                        p0=p0, p_max=p_max, C=C,
                        epsilon=1, tau=3,
                        delta_p=delta_p, target_AR=.80, # 80% AR is sufficient
                        opt_steps=opt_steps
                    )

                    start_time = time.perf_counter()
                    r_gammas, r_betas, _, r_AR = II.run()
                    end_time = time.perf_counter()
                    runtime = end_time - start_time
                    print(f'AR={r_AR}', flush=True)
                    # Save Results (Same file name, different folder)
            
                    tmp_file = output_file.with_suffix(".tmp")
                    with open(tmp_file, "wb") as f:
                        pickle.dump(
                            {
                                "AR": r_AR,
                                "gammas": r_gammas,
                                "betas": r_betas,
                                "runtimes": runtime,
                                "p0": p0,
                                "num_generators": num_generators,
                                "load": load,
                                "source_file": chosen_file,
                            },
                            f,
                        ) 
                    tmp_file.replace(output_file)
                
                except Exception as e:
                    print(
                        f"ERROR for p0={p0}, num_generators={num_generators}, "
                        f"load={load}, file={chosen_file}",
                        flush=True,
                    )
                    print(repr(e), flush=True)

                    with open(log_file, "a") as log:
                        log.write(
                            f"{datetime.now()} | "
                            f"p0={p0}, num_generators={num_generators}, "
                            f"load={load}, file={chosen_file} | {repr(e)}\n"
                        )

                    continue
                
            print('')

if __name__ == '__main__':
    run()