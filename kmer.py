#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
from Bio import SeqIO
import pandas as pd
import time

"""Quick kmer
   @author abhishake
   @github https://github.com/AbhishakeL

"""

def count_kmers(read, k):
    """Count kmer occurrences in a given read."""
    counts = {}
    num_kmers = len(read) - k + 1
    for i in range(num_kmers):
        kmer = read[i:i+k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts

def find_val_in_dict(dct, dicVal):
    """Simple value matching from a dictionary."""
    return dct.get(dicVal, 0)



if __name__ == '__main__':
    start = time.time()
    m = [''.join(i) for i in product('ATGC', repeat=8)]
    records = list(SeqIO.parse("train_seq2.fasta", "fasta"))
    
    df_data = {'Kmers': m}
    for record in records:
        t = record.id.split('|')[-2]
        inputString = record.seq
        q = count_kmers(inputString, 8)
        d = [find_val_in_dict(q, dicVal) for dicVal in m]
        df_data[t] = d
    
    df_0 = pd.DataFrame(df_data)
    df_0.to_csv('kmer-tab-23.csv', header=True, index=False)
    
    print(time.time() - start)

