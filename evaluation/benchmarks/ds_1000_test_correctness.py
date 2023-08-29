# This file will be executed inside a separate virtual environment specifially for DS-1000.
# See evaluation/benchmarks/ds_1000.py where this file will be called.

import multiprocessing

# Fork is slightly faster, but spawn would also work here
if multiprocessing.get_start_method(allow_none=True) != 'fork':
    multiprocessing.set_start_method('fork')

def test_individual(args):
    lib, lib_index, item, model_reply = args
    return [lib, lib_index, item.test('\n' + model_reply + '\n')]

def main():
    import os
    import multiprocessing
    import sys
    import json

    import tqdm
    import ds1000

    with open(sys.argv[1]) as f:
        model_outputs = json.load(f)

    progress_bar_description = sys.argv[2]

    dataset = ds1000.DS1000Dataset('../ds1000_data/ds1000_data').data

    # Idk which ones of these are actually needed.
    # But some definitely are.
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

    libs = list(model_outputs.keys())
    multiprocessing_args = [[lib, i, dataset[lib][i], model_outputs[lib][i]] for lib in libs for i in range(len(model_outputs[lib]))]

    with multiprocessing.Pool(os.cpu_count()) as pool:
        results = list(tqdm.tqdm(
            pool.imap_unordered(test_individual, multiprocessing_args),
            total=sum([len(model_outputs[lib]) for lib in libs]),
            desc=progress_bar_description,
        ))

    scores = {}
    for lib in libs:
        scores[lib] = [None] * len(model_outputs[lib])

    for (lib, lib_index, is_correct) in results:
        scores[lib][lib_index] = is_correct

    print(json.dumps(scores))

if __name__ == '__main__':
    main()
