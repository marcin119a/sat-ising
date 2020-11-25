import time
import csv

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(list(map(int, seq[int(last):int(last + avg)])))
        last += avg

    return out