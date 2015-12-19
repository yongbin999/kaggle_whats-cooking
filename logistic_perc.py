from __future__ import division
from collections import defaultdict
from matplotlib import pyplot as plt
import math

import time
import os
import random

# Global class labels.


###################################
# Utilities
def select_weighted(d): 
##6180/7954 = 0.7770 accuracy +0.007
    if sum(d.itervalues()) > 1:
        offset = random.randint(0, int(sum(d.itervalues()))-1)
        for k, v in d.iteritems():
            if offset < v:
                return k
            offset -= value
    else:
        return max(d.iterkeys(), key=lambda k: d[k])

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    #print 
    #print max(dct.iterkeys(), key=lambda k: dct[k])
    #print dct.items()
    ##print select_weighted(dct)

    return select_weighted(dct)

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
    ## cuisine size counts
    cuisine_type_count = defaultdict(float)
    for row in training_set:
        cuisine_type_count[row['cuisine']] +=1

    return cuisine_type_count

def get_ingredient_count(training_set):
    ## ingre by cusine counts
    ing_count = defaultdict(float)
    for row in training_set:
        for ingredient in row['ingredients']:
            ing_count[ingredient] +=1

    return ing_count

def get_model(training_set):
    ##get base bow by cuisine 
    cuisine_type_bag = defaultdict()
    for row in training_set:
        for ingredient in row['ingredients']:
            if row['cuisine'] not in cuisine_type_bag:
                cuisine_type_bag[row['cuisine']] = defaultdict(float)

            if ingredient not in cuisine_type_bag[row['cuisine']]:
                cuisine_type_bag[row['cuisine']][ingredient] =1
            else:
                cuisine_type_bag[row['cuisine']][ingredient]+=1
    
    return cuisine_type_bag

def ingre_list_stats(training_set):
    ##get base bow by cuisine 
    ingstats = []
    lenstats =[]
    for row in training_set:
        lenstats.append(len(row['ingredients']))

        for ingredient in row['ingredients']:
            ingstats.append(len(ingredient.split(' ')))
            if len(ingredient.split(' ')) >=10:
                print ingredient

    print " ingr states" 
    print max(ingstats)
    print min(ingstats)
    print max(lenstats)
    print min(lenstats)


def get_ingredient_count_class(ing_count, cuisine_type_bag):
    ## get individual ingre and count how many cuisine it belongs to 
    ing_count_class =defaultdict(float)
    for each in ing_count:
        for cuisine,ingredients in cuisine_type_bag.items():
            #print ingredients
            if each in ingredients:
                ing_count_class[each] += 1 

    return ing_count_class

def construct_dataset(training_set):
    ##with training data
    dataset = []
    print "[constructing dataset...]"
    for row in training_set:
        dataset.append((tokenize_doc(row), row['cuisine']))
    print "[dataset constructed.]"
    return dataset


###############################


def fullseq_features(bow, stat_cuisine):
    """
    generate feature vectors from bow
    """
    all_feat_vec = defaultdict()
    for cuisine in stat_cuisine:
        feat_vec = defaultdict(float)

        ##split ingredient into individual words
        for ingredient, value in bow.iteritems() :
            if ingredient is not None:
                ##reg bow :0.7669
                feat_vec["%s_%s" % (cuisine, ingredient)] = value

                ##add feature strip adjectives n-gram approach:
                ing_words = ingredient.split(' ')
                ing_phrase = ingredient.split(',')
                of_phrase = ingredient.split('of')
                if len(ing_phrase)>=2:
                    ## replace instruction with just ingredient
                    ing_words =ing_phrase
                if len(of_phrase)>=2:
                    ## replace instruction with just of split
                    ing_words =of_phrase


                if len(ing_words)==2:
                    ##6137/7954 = 0.7716 accuracy
                    feat_vec["%s_%s" % (cuisine, ing_words[1])] +=1

                if len(ing_words)==3:
                    ##6155/7954 = 0.7738 accuracy
                    feat_vec["%s_%s" % (cuisine, (' ').join(ing_words[1:]) )] +=1

                """
                if len(ing_words)==4:
                    ##0.7737 accuracy
                    feat_vec["%s_%s" % (cuisine, (' ').join(ing_words[1:-1]) )] +=1
                    feat_vec["%s_%s" % (cuisine, (' ').join(ing_words[2:]) )] +=1

                if len(ing_words)>4:
                    ##6137/7954 = 0.7716 accuracy
                    feat_vec["%s_%s" % (cuisine, (' ').join(ing_words[len(ing_words)-1]) )] +=1
                    feat_vec["%s_%s" % (cuisine, (' ').join(ing_words[len(ing_words)-2:]) )] +=1
                    feat_vec["%s_%s" % (cuisine, (' ').join(ing_words[int(len(ing_words)/2)]) )] +=1
                    feat_vec["%s_%s" % (cuisine, (' ').join(ing_words[int(len(ing_words)/2)+1]) )] +=1
                    feat_vec["%s_%s" % (cuisine, (' ').join(ing_words[int(len(ing_words)/2):int(len(ing_words)/2)+1]) )] +=1


                ## added features ingredients:
                if ('fresh' in ingredient):
                    feat_vec["%s_%s"% (cuisine, 'fresh')] += 1
                if ('rice' in ingredient):
                    feat_vec["%s_%s"% (cuisine, 'rice')] += 1
                if ('creme' in ingredient):
                    feat_vec["%s_%s"% (cuisine, 'crene')] += 1
                if ('oil' in ingredient):
                    feat_vec["%s_%s"% (cuisine, 'oil')] += 1
                if ('olive' in ingredient):
                    feat_vec["%s_%s"% (cuisine, 'olive')] += 1
                if ('fat' in ingredient):
                    feat_vec["%s_%s"% (cuisine, 'fat')] += 1
                if ('sauce' in ingredient):
                    feat_vec["%s_%s"% (cuisine, 'sauce')] += 1
                if ('noodle' in ingredient):
                    feat_vec["%s_%s"% (cuisine, 'sauce')] += 1
                if ('Hunt\'s' in ingredient):
                    feat_vec["%s_%s"% (cuisine, 'tomato')] += 1
                """

        ## added features general:
        feat_vec["%s_%s"% (cuisine, len(bow))] = 1
        ## + 0.7678


        all_feat_vec[cuisine] = feat_vec

    return all_feat_vec



def predict_multiclass(bow, weights, all_feat_vec,ing_count_adj=None):  ##takes list of same vec but diff cuisine
    """
    current weightss from bow * the existing feature vec in 20 diff cuisine
    """
    scores = defaultdict()
    for cuisine in all_feat_vec:
        #print all_feat_vec[cuisine]
        #print ing_count_adj[cuisine]
        scores[cuisine] = dict_dotprod(all_feat_vec[cuisine],weights)
        #scores[cuisine] = dict_dotprod(dict_dotprod(all_feat_vec[cuisine],weights),ing_count_adj[cuisine])


    return dict_argmax(scores)

def train(examples, stat_cuisine, ing_count_adj=None,stepsize=1, numpasses=10, do_averaging=False, devdata=None, outputonly=False):
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


    ing_adj_vec = fullseq_features(ing_count_adj,stat_cuisine)
    
    print "[training...]"
    for pass_iteration in range(numpasses):
        start = time.time()
        print "\tTraining iteration %d" % pass_iteration
        random.shuffle(examples)
        select_sam = examples[int(len(examples)/5):]
        print "\t training size: "+ str(len(select_sam))


        #print ing_adj_vec['italian']

        for bow, goldlabel in select_sam:
            t += 1

            all_feat_vec = fullseq_features(bow,stat_cuisine)


            predlabel = predict_multiclass(bow, weights, all_feat_vec, ing_adj_vec)
            if predlabel != goldlabel:
                # compute gradient
                ##positive add for goldfeats
                goldfeats = all_feat_vec[goldlabel]
                for feat_name, feat_value in goldfeats.iteritems():
                    weights[feat_name] += stepsize * feat_value
                    weightSums[feat_name] += (t-1) * stepsize * feat_value

                ## negative for bad feats
                predfeats = all_feat_vec[predlabel]
                for feat_name, feat_value in predfeats.iteritems():
                    weights[feat_name] -= stepsize * feat_value
                    weightSums[feat_name] -= (t-1) * stepsize * feat_value

        end = time.time()
        print "\ttime used training: " + str(round(end - start,2))

        if outputonly==False:
            print "TR RAW EVAL:",
            select_sam = select_sam[:int(len(select_sam)/10)]
            train_acc.append(do_evaluation(select_sam, weights,stat_cuisine))

            if devdata:
                print "DEV RAW EVAL:",
                test_acc.append(do_evaluation(devdata, weights,stat_cuisine))

            if devdata and do_averaging:
                print "DEV AVG EVAL:",
                avg_test_acc.append(do_evaluation(devdata, get_averaged_weights(),stat_cuisine))
                testend = time.time()
                print "\ttotal time this round: " + str(round(testend - start,2))

            print "[learned weights for %d features from %d examples.]" % (len(weights), len(examples))

    print "final score:"
    select_sam = select_sam[:int(len(select_sam)/10)]
    final_evaluation(select_sam, weights,stat_cuisine)
    
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

def final_evaluation(examples, weights, stat_cuisine):
    """
    Compute the accuracy of a trained perceptron.
    """
    incorrects = []
    for feats, goldlabel in examples:
        all_feat_vec = fullseq_features(feats,stat_cuisine)
        predlabel = predict_multiclass(feats, weights,all_feat_vec)
        if predlabel != goldlabel:
            incorrects.append((predlabel,goldlabel,feats))

    outfile = open( './outputs/final_errors.csv', 'w' )
    outfile.write( 'pred' + ',' + 'gold' +',' + 'feats' + '\n')
    for each in incorrects:
        outfile.write( str(each[0]) + ',' + str(each[1]) + str(each[2]) +  '\n' )

    print "output errors"
    

def plot_accuracy_vs_iteration(train_acc, test_acc, avg_test_acc, naive_bayes_acc = 0.73):
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



def output_csv_submission(weights):

    outfile = open( 'outputs/submission_results.csv', 'w' )
    outfile.write( 'id' + ',' + 'cuisine' + '\n')


    json_data = open(sys.argv[2],'r').read()
    test_data = ast.literal_eval(json_data)

    output = defaultdict(float)
    for items in test_data:
        all_feat_vec = fullseq_features(tokenize_doc(items),stat_cuisine)
        output[items['id']] = predict_multiclass(tokenize_doc(items), weights, all_feat_vec)


    for key, value in sorted( output.items() ):
        outfile.write( str(key) + ',' + str(value) + '\n' )
    print "outfile"


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
    slicenum = int(len(data)*0.2)
    training = data[slicenum:]
    testing = data[:slicenum]

    if len(sys.argv) ==3:
        training_set = construct_dataset(data)  ## full dataset
    else:
        training_set = construct_dataset(training)  ## full dataset

    test_self = construct_dataset(testing)

    stat_cuisine = get_stats_count(data)
    stat_ingredient = get_ingredient_count(data)
    ##ingre_list_stats(data)


    cuisine_type_bag = get_model(data)
    ing_count_adj=get_ingredient_count_class(stat_ingredient,cuisine_type_bag)
    for key, value in ing_count_adj.items():
        ing_count_adj[key] = float(float(1)/ (value * value))

    print ing_count_adj[None]
    print "got stats by ingredient n classes"

    ##sol_dict = train(training_set, stat_cuisine,stepsize=1, numpasses=1, do_averaging=True, devdata=None)
    sol_dict = train(training_set, stat_cuisine, ing_count_adj, stepsize=1, numpasses=5, do_averaging=True, devdata=test_self,outputonly=False)

    if len(sys.argv) ==3:
        output_csv_submission(sol_dict['weights'])


    plot_accuracy_vs_iteration(sol_dict['train_acc'], sol_dict['test_acc'], sol_dict['avg_test_acc'])
