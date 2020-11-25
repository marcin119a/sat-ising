from utilities import data_to_matrix, ising_params
import ising 


def solve_ising(formulas, N, m):
    matrix = data_to_matrix(m, N, formulas)
    J, h = ising_params(m, N, matrix)
  
    graph  = { (i,i): k for i, k in enumerate(h) }


    J2 = {}
    for i, x in enumerate(J):
        for j, m in enumerate(x):
            if m !=0:
                graph[(i,j)] = m


    result = ising.search(graph, num_states=N)
    return result