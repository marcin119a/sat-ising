import itertools
import os
import tensorflow as tf
from helpers import chunkIt, save_to_csv
import time
import pycosat
import mlflow
from utilities import ising_params


def predict(file_n, complexity, formulas_number):
    tfrecord_location = file_n
    bin_vector_dw = []
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_location)
    batch_size = formulas_number
    n = complexity
    test_set = {'cnf': list(), 'sat': list()}

    with mlflow.start_run():

        for string_record in itertools.islice(record_iterator, batch_size):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            

            m = len(example.features.feature["inputs"].float_list.value) // 2
            
            cnf2 = chunkIt(example.features.feature["inputs"].float_list.value, m) #split examp: [0,1,2,3,4,5] into [0,1], [2,3], [4,5]

            J, h =  ising_params(m, n, cnf2)


            out = pycosat.solve(cnf2) != 'UNSAT'
            bin_vector_dw.append(out)

            test_set['cnf'].append(cnf2)
            targ = int(example.features.feature["sat"].float_list.value[0])
            test_set['sat'].append(targ)

        mlflow.log_param("file_name", file_n)
        mlflow.log_param("out_vector", bin_vector_dw)
        mlflow.low_param("out_alphas", alpha)
                

    return (bin_vector_dw, test_set)

predict('sr_50/train_1000_sr_50.tfrecord', complexity=50, formulas_number=100)