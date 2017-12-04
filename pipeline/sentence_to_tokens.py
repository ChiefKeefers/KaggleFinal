import sys
import os
import json

from glob import glob
from functools import reduce
from collections import Counter
import concurrent.futures as cfs 

# Need to add to path for imports 
cwd = os.getcwd()
sys.path.append(cwd)
from cleaners.sentence_cleaner import SentenceCleaner

features = dict()   

def build_tokens(file_location):
    print("    Parsing ", file_location)
    fname = file_location.split('/')[-1]
    file_num = int(fname.split('.')[0])
    
    with open(file_location, 'r') as f:
        sentences = f.read()
    
    features[file_num] = Counter(cleaner.clean_sentence(sentences))
    return file_num

if __name__=="__main__":
    pipeline = os.environ['PIPELINE']
    import time;
    start = time.time()
    print("\nProgram Begin")
    print("=======================")
    print("Estimated time: 15 min. with 36 cores")
    cleaner = SentenceCleaner()
    rows = list()
    description_files = glob('data/descriptions_{}/*.txt'.format(pipeline))
    
    with cfs.ThreadPoolExecutor() as executor:
        futures = executor.map(build_tokens, description_files)
    
    with open('pipeline/data/tokens_{}.json'.format(pipeline), 'w') as f:
        json.dump(features, f)
    
    print("took", time.time() - start, "seconds")
