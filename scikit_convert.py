from __future__ import division
from collections import defaultdict
from sklearn import preprocessing

import time
import os
import random
import ast

def scikit_y_format(json_data):
    
    data = ast.literal_eval(json_data)
    ##print(data[0])

    y_terms = []
    x_terms = []
    x_flat = []
    for row in data:
        y_terms.append(row['cuisine'])
        x_terms.append(row['ingredients'])
        x_flat +=row['ingredients']

    le_y = preprocessing.LabelEncoder()
    le_y.fit(y_terms)
    converted_y = le_y.transform(y_terms)
    """
    le_x = preprocessing.LabelEncoder()
    le_x.fit(x_flat)
    converted_x = []

    for index, cui_ingredients in enumerate(x_terms):
        converted_x.append(le_x.transform(cui_ingredients))
        if (index %1000==0):
            print index
    """

    print y_terms[:2]
    print converted_y[:2]

    return converted_y


def scikit_x_vectorize(json_data):
    from sklearn.feature_extraction import DictVectorizer
    v = DictVectorizer(sparse=False)
    """
    D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    X = v.fit_transform(D)
    X
    array([[ 2.,  0.,  1.],
        [ 0.,  1.,  3.]])
    """

    data = ast.literal_eval(json_data)
    ##print(data[0])
    x_terms = []
    for index,row in enumerate(data):
        temp = {}
        for ingre in row['ingredients']:
            if ingre not in temp:
                temp[ingre] =1

        if (index %10000==0):
            print index

        #print temp
        x_terms.append(temp) ## dict of each ingredient list
    
    #print x_terms
    X = v.fit_transform(x_terms) ## convert dict list to ints vects matrix

    print data[0]['ingredients']
    print X[0]

    return X

if __name__=='__main__':

    import sys
    import json

    if len(sys.argv) > 1:
        json_data = open(sys.argv[1],'r').read()
    else:
        print 'no training files defined: python input_covert.py "___train.json___"'
        sys.exit()

    data = scikit_x_vectorize(json_data), scikit_y_format(json_data)



    ## same output files so we dont have to reload later
    outfile = open( 'outputs/scikit_converted_data_x', 'w' )
    for item in data[0]:
        tempstring = ''
        for code in item:
            tempstring += "%s," % code
        outfile.writelines(tempstring[:-1]+'\n')

    outfile = open( 'outputs/scikit_converted_data_y', 'w' )
    tempstring =''
    for code in data[1]:
        tempstring += "%s," % code
    outfile.writelines(tempstring[:-1]+'\n')

    print "out data"
