from dwave.system import DWaveSample
from dwave.system.composites import EmbeddingComposite


from utilities import data_to_matrix, ising_params

def dw_solve(formulas, N, m):
    matrix = data_to_matrix(m, N, formulas)
    J, h = ising_params(m, N, matrix)
    token = ""

    h2  = { i: k for i, k in enumerate(h) }

    J2 = {}
    for i, x in enumerate(J):
        for j,m in enumerate(x):
            if m !=0:
                J2[(i,j)] = m


    sampler = EmbeddingComposite(DWaveSampler(token=token))
    sampleset = sampler.sample_ising(h2, J2)
    

    for sample in sampleset.samples():
        result = (sample)
    x = [1 if spin == 1 else 0 for key, spin in result.items()]

    return x
