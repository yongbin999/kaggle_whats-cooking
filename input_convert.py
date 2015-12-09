from __future__ import division
from collections import defaultdict

import time
import os
import random
import ast

def scikit_learn_format(json_data):
    
    data = ast.literal_eval(json_data)
    ##print(data[0])

    y_terms = []
    x_terms = []
    for row in data:
        y_terms.append(row['cuisine'])
        x_terms.append(row['ingredients'])

    print x_terms[:2]
    print y_terms[:2]

    return x_terms,y_terms



if __name__=='__main__':

    import sys

    if len(sys.argv) > 1:
        json_data = open(sys.argv[1],'r').read()
    else:
        print 'no training files defined: python input_covert.py "___train.json___"'
        sys.exit()

    data = scikit_learn_format(json_data)

