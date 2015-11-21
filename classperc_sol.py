from __future__ import division
from collections import defaultdict
from matplotlib import pyplot as plt

import time
import os
import random

# Global class labels.


###################################
# Utilities

def dict_subtract(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    return max(dct.iterkeys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors, where each is
    represented as a dictionary of {index: weight} pairs, where indexes are any
    keys, potentially strings.  If a key does not exist in a dictionary, its
    value is assumed to be zero."""
    smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
    total = 0
    for key in smaller.iterkeys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

def tokenize_doc(json_data):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """

    words = {}
    for w in json_data['ingredients']:
        if w not in words:
            words[w] =0
        words[w] +=1
    
    return words

def get_stats_count(training_set):
    cuisine_type_count = defaultdict(float)
    for row in training_set:
        cuisine_type_count[row['cuisine']] +=1

    return cuisine_type_count

def get_ingredient_count(training_set):
    cuisine_type_count = defaultdict(float)
    for row in training_set:
        for ingredient in row['ingredients']:
            cuisine_type_count[ingredient] +=1

    return cuisine_type_count

def construct_dataset(training_set):
    """
    Build a dataset. If train is true, build training set otherwise build test set.
    The training set is a list of training examples. A training example is a tuple
    containing a dictionary (that represents features of the training example) and
    a label for that training instance (either 1 or -1).
    """

    dataset = []

    num_inst = 25000
    print "[constructing dataset...]"
    for row in training_set:
        dataset.append((tokenize_doc(row), row['cuisine']))
    print "[dataset constructed.]"
    return dataset


###############################
# YOUR CODE GOES BELOW

def fullseq_features(bow, stat_cuisine):
    """
    !! MAKE SURE YOU UNDERSTAND WHAT IS GOING ON HERE                     !!
    !! You will need to implement/use a function very similar to this for !!
    !! the structured perceptron code.                                    !!

    The full f(x,y) function. Pass in features (represented as a dictionary) and
    a string label. Returns one big feature vector (that is used for all classes).
    """
    all_feat_vec = defaultdict()
    for cuisine in stat_cuisine:
        feat_vec = defaultdict(float)
        for word, value in bow.iteritems():
            feat_vec["%s_%s" % (cuisine, word)] = value
        all_feat_vec[cuisine] = feat_vec
    return all_feat_vec

def predict_multiclass(bow, weights, all_feat_vec):  ##takes 1 vec
    """
    Takes a bag of words represented as a dictionary and a weight vector that contains
    weights for features of each label (represented as a dictionary) and
    performs perceptron multi-class classification (i.e., finds the class with the highest
    score under the model.

    You may find it peculiar that the name of this function has to do with multiclass
    classification or that we are making predictions in this relatively complicated way
    given the binary classification setting. You are correct; this is a bit weird. But,
    this code looks a lot like the code for multiclass perceptron and the structured
    perceptron (i.e., making the next part of this homework easier).
    """
    scores = defaultdict()
    for cuisine in all_feat_vec:
        scores[cuisine] = dict_dotprod(all_feat_vec[cuisine], weights)

    return dict_argmax(scores)

def train(examples, stat_cuisine, stepsize=1, numpasses=10, do_averaging=False, devdata=None):
    """
    Trains a perceptron.
      examples: list of (featvec, label) pairs; featvec is a dictionary and label is a string
      stepsize: hyperparameter; this affects how the weights are updated
      numpasses: the number of times to loop through the dataset during training
      do_averaging: boolean that determines whether to use averaged perceptron
      devdata: the test set of examples

    returns a dictionary containing the vanilla perceptron accuracy on the train set and test set
    and the averaged perceptron's accuracy on the test set.
    """

    weights = defaultdict(float)
    weightSums = defaultdict(float)
    t = 0

    train_acc = []
    test_acc = []
    avg_test_acc = []

    def get_averaged_weights():
        return {f: weights[f] - 1/t*weightSums[f] for f in weightSums}

    print "[training...]"
    for pass_iteration in range(numpasses):
        start = time.time()
        print "\tTraining iteration %d" % pass_iteration
        random.shuffle(examples)
        for bow, goldlabel in examples:
            t += 1

            all_feat_vec = fullseq_features(bow,stat_cuisine)

            predlabel = predict_multiclass(bow, weights, all_feat_vec)
            if predlabel != goldlabel:
                # compute gradient
                goldfeats = all_feat_vec[goldlabel]

                for cuisine in stat_cuisine:
                    if cuisine != goldlabel:
                        predfeats = all_feat_vec[cuisine]  ## each except the gold
                        
                        featdelta = dict_subtract(goldfeats, predfeats)

                        for feat_name, feat_value in featdelta.iteritems():
                            weights[feat_name] += stepsize * feat_value
                            weightSums[feat_name] += (t-1) * stepsize * feat_value
        end = time.time()
        print "time used: " + str(round(end - start,2))
        print "TR RAW EVAL:",
        train_acc.append(do_evaluation(examples, weights,all_feat_vec))

        if devdata:
            print "DEV RAW EVAL:",
            test_acc.append(do_evaluation(devdata, weights,all_feat_vec))

        if devdata and do_averaging:
            print "DEV AVG EVAL:",
            avg_test_acc.append(do_evaluation(devdata, get_averaged_weights(),all_feat_vec))

    print "[learned weights for %d features from %d examples.]" % (len(weights), len(examples))

    return { 'train_acc': train_acc,
             'test_acc': test_acc,
             'avg_test_acc': avg_test_acc,
             'weights': weights if not do_averaging else get_averaged_weights() }


def do_evaluation(examples, weights, stat_cuisine):
    """
    Compute the accuracy of a trained perceptron.
    """
    num_correct, num_total = 0, 0
    for feats, goldlabel in examples:
        all_feat_vec = fullseq_features(feats,stat_cuisine)
        predlabel = predict_multiclass(feats, weights,all_feat_vec)
        if predlabel == goldlabel:
            num_correct += 1.0
        num_total += 1.0
    print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
    return num_correct/num_total

def plot_accuracy_vs_iteration(train_acc, test_acc, avg_test_acc, naive_bayes_acc = 0.83):
    """
    Plot the vanilla perceptron accuracy on the trainning set and test set
    and the averaged perceptron accuracy on the test set.
    """

    plt.plot(range(len(train_acc)), train_acc)
    plt.plot(range(len(test_acc)), test_acc)
    plt.plot(range(len(avg_test_acc)), avg_test_acc)
    plt.plot(range(len(avg_test_acc)), [naive_bayes_acc] * len(avg_test_acc))
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Accuracy (%)')
    plt.title('Iterations  vs. Accuracy')
    plt.legend(('train','test','avg test', 'NB'),'upper left')
    plt.show()

if __name__=='__main__':
    import sys
    import ast

    if len(sys.argv) > 1:
        json_data = open(sys.argv[1],'r').read()
    else:
        print 'no training files defined: python classprec.py "___train.json___"'
        sys.exit()

    data = ast.literal_eval(json_data)
    print(data[0])

    ## split data into training and testing
    slicenum = int(len(data)*0.8)
    training = data[slicenum:]
    testing = data[:slicenum]

    training_set = construct_dataset(training)
    test_set = construct_dataset(testing)
    ##print training_set[0]

    stat_cuisine = get_stats_count(training)
    stat_ingredient = get_ingredient_count(training)

    ##print len(stat_ingredient)

    ##sol_dict = train(training_set, stat_cuisine,stepsize=1, numpasses=1, do_averaging=True, devdata=None)
    sol_dict = train(training_set, stat_cuisine,stepsize=1, numpasses=10, do_averaging=True, devdata=test_set)
    plot_accuracy_vs_iteration(sol_dict['train_acc'], sol_dict['test_acc'], sol_dict['avg_test_acc'])
