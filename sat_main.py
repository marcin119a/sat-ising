import itertools
import os
import tensorflow as tf
from helpers import chunkIt
import time
import pycosat
import mlflow
from utilities import ising_params
from ising_solver import solve_ising
from dwave.system import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dimodmock import StructuredMock

def predict(file_n, complexity, formulas_number):
    tfrecord_location = file_n
    alpha, bin_vector_dw = [], []
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_location)
    batch_size = formulas_number
    n = complexity
    test_set = {'cnf': list(), 'sat': list()}


    token = ""
    
    #sampler = EmbeddingComposite(DWaveSampler(token=token))
    sampler = EmbeddingComposite(StructuredMock.from_sampler(DWaveSampler(token=token)))

    with mlflow.start_run():
        
        list_out_spin = []
        list_out_energy = []
        for string_record in itertools.islice(record_iterator, batch_size):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            

            m = len(example.features.feature["inputs"].float_list.value) // 2
            
            cnf2 = chunkIt(example.features.feature["inputs"].float_list.value, m) #split examp: [0,1,2,3,4,5] into [0,1], [2,3], [4,5]

            J, h =  ising_params(m, n, cnf2)

            h2  = { i: k for i, k in enumerate(h) }

            J2 = {}
            for i, x in enumerate(J):
                for j, k in enumerate(x):
                    if k !=0:
                        J2[(i,j)] = k
            
            sampleset = sampler.sample_ising(h2, J2)
            best_vector = [sampleset.first.sample[i] for i in sorted(h2)]
            best_energy = sampleset.first.energy
            list_out_spin.append(best_vector)
            list_out_energy.append(best_energy)

            out = pycosat.solve(cnf2) != 'UNSAT'
            bin_vector_dw.append(out)
            alpha.append(n/m)

            test_set['cnf'].append(cnf2)
            targ = int(example.features.feature["sat"].float_list.value[0])
            test_set['sat'].append(targ)

        mlflow.log_param("file_name", file_n)
        mlflow.log_param("out_vector", bin_vector_dw)
        mlflow.log_param("out_alphas", alpha)
        mlflow.log_param("test_set", test_set)
        mlflow.log_param("list_out_spin", list_out_spin)
        mlflow.log_param("list_out_energy", list_out_energy)
                

    return (bin_vector_dw, test_set)

predict('sr_50/train_1000_sr_50.tfrecord', complexity=50, formulas_number=100)