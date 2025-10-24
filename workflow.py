from cereeberus import Interleave
import itertools
from multiprocessing import Pool, cpu_count
import letters, process_letters

def get_input_data():
    letter_data = letters.gen_letters() # these are point clouds
      
    mappers = {}
    for key, samples in letter_data.items():
        mappers[key] = [process_letters.LetterMapper(sample) for sample in samples]
    return mappers

def pairwise_comparison(args):
    letter1, idx1, letter2, index2, mappers = args
    
    mapper1 = mappers[letter1][idx1]
    mapper2 = mappers[letter2][index2]

    myInt = Interleave(mapper1, mapper2)
    val = myInt.fit()

    return (f"{letter1}_{idx1}", f"{letter2}_{idx2}", val)

def run_all_comparisons(mappers, num_procs = None):
    if num_procs is None:
        num_procs = cpu_count()

    letters = sorted(mappers.keys())

    tasks = []

    for l1, l2 in itertools.combinations_with_replacement(letters, 2):
        for i, j in itertools.product(range(len(mappers[l1])), range(len(mappers[l2]))):
            tasks.append((l1, i, l2, j, mappers))

    with Pool(processes=num_procs) as pool:
        results = pool.map(pairwise_comparison, tasks)

    df = process_letters.build_results_dataframe(results)

if __name__ == "__main__":
    mappers = get_input_data()
    df = run_all_comparisons(mappers, n_jobs=32)
    df.to_csv("pairwise_results.csv", index=False)