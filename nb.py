from __future__ import division

import math
import os, sys
import  json

from collections import defaultdict


def tokenize_doc(json_data):
    """
    IMPLEMENT ME!

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


def train_model(self, num_docs=None):
    """
    This function processes the entire training set using the global PATH
    variable above.  It makes use of the tokenize_doc and update_model
    functions you will implement.

    num_docs: set this to e.g. 10 to train on only 10 docs from each category.
    """

    if num_docs is not None:
        print "Limiting to only %s docs per clas" % num_docs

    pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
    neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
    print "Starting training with paths %s and %s" % (pos_path, neg_path)
    for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
        filenames = os.listdir(p)
        if num_docs is not None: filenames = filenames[:num_docs]
        for f in filenames:
            with open(os.path.join(p,f),'r') as doc:
                content = doc.read()
                self.tokenize_and_update_model(content, label)
    self.report_statistics_after_training()

def report_statistics_after_training(self):
    """
    Report a number of statistics after training.
    """

    print "REPORTING CORPUS STATISTICS"
    print "NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL]
    print "NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL]
    print "NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL]
    print "NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL]
    print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)

def get_model(training_set):

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


def tokenize_and_update_model(self, doc, label):
    p
def top_n(self, label, n):
    """
    Implement me!

    Returns the most frequent n tokens for documents with class 'label'.
    """
    return sorted(self.class_word_counts[label].items(), key=lambda (w,c): -c)[:n]

def p_word_given_label(word,training_types_count,training_types_bows_count, label):
    """
    Implement me!

    Returns the probability of word given label (i.e., P(word|label))
    according to this NB model.
    """
    return training_types_bows_count[label][word] / len(training_types_count[label])

def p_word_given_label_and_psuedocount(word,training_types_count,training_types_bows_count, label, alpha):
    """
    Implement me!

    Returns the probability of word given label wrt psuedo counts.
    alpha - psuedocount parameter
    """
    #print training_types_bows_count[label][word]
    #print training_types_count[label]
    #print sum(training_types_count.values())
    return float(training_types_bows_count[label][word] + alpha )/ float(training_types_count[label]+ alpha*sum(training_types_count.values()))

def log_likelihood(test_bow,training_types_count,training_types_bows_count, label, alpha):
    """
    Computes the log likelihood of a set of words give a label and psuedocount.
    bow - a bag of words (i.e., a tokenized document)
    label - either the positive or negative label
    alpha - float; psuedocount parameter
    """
    likelyhood = 0
    for word in test_bow:
        ##print word +" " +str(math.log(p_word_given_label_and_psuedocount(word, training_types_count,training_types_bows_count,label,alpha)))
        likelyhood += math.log(p_word_given_label_and_psuedocount(word, training_types_count,training_types_bows_count,label,alpha))

    return likelyhood

def log_prior(training_types_count, label):
    """
    Implement me!

    Returns a float representing the fraction of training documents
    that are of class 'label'.
    """
    prior_count = training_types_count[label]
    prior  = prior_count / sum(training_types_count.values())
    
    return math.log(prior)

def unnormalized_log_posterior(test_bow,training_types_count,training_types_bows_count, label, alpha):
    """
    Implement me!

    alpha - psuedocount parameter
    bow - a bag of words (i.e., a tokenized document)
    Computes the unnormalized log posterior (of doc being of class 'label').
    """
    
    return log_prior(training_types_count,label) + log_likelihood(test_bow, training_types_count,training_types_bows_count, label, alpha)

def classify(bow,training_types_count,training_types_bows_count, alpha):
    """
    for each label calculate the probability of label given bow
    """
    dict_oflargest = training_types_count.copy()
    for label in dict_oflargest:
        dict_oflargest[label] = unnormalized_log_posterior(bow, training_types_count,training_types_bows_count, label, alpha)

    return max(dict_oflargest, key=dict_oflargest.get)

def likelihood_ratio(self, word, alpha):
    """
    Implement me!

    alpha - psuedocount parameter.
    Returns the ratio of P(word|pos) to P(word|neg).
    """
    return self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha) / self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)

def evaluate_classifier_accuracy(testing_set,training_types_count,training_types_bows_count, alpha):
    """
    Implement me!

    alpha - psuedocount parameter.
    This function should go through the test data, classify each instance and
    compute the accuracy of the classifier (the fraction of classifications
    the classifier gets right.
    """
    numofcorrect = 0.0;
    totalnumdocs = len(testing_set);


    for row in testing_set:
        evualuated_label = classify(tokenize_doc(row),training_types_count,training_types_bows_count,alpha)
        if evualuated_label ==row['cuisine']:
            numofcorrect+=1.0


    return numofcorrect / totalnumdocs


def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuries. You may want to modify this function
    to enhance your plot.
    """

    import matplotlib.pyplot as plt

    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')

    plt.xscale('log')
    plt.show()














if __name__ == '__main__':
    import ast
    json_data = open(sys.argv[1],'r').read()
    data = ast.literal_eval(json_data)
    print(data[0])

    ## split data into training and testing
    slicenum = int(len(data)*0.8)
    training = data[slicenum:]
    testing = data[:slicenum]


    ## count the stats on cusine types 
    cuisine_type_count = defaultdict(float)
    for row in training:
        cuisine_type_count[row['cuisine']] +=1

    types = len(cuisine_type_count)
    print 'size of the training data' +str(len(training)) + ' size of testing' +str(len(testing))

    cuisine_type_count_percent = {w: round(c/len(training),3) for (w,c) in cuisine_type_count.items() }
    cuisine_type_count_percent = sorted(cuisine_type_count.items(), key=lambda x: x[1], reverse=1)

    print json.dumps(cuisine_type_count_percent)
    with open('training_types_count', 'w') as f:
        f.write(json.dumps(cuisine_type_count))




    ## create the training model
    cuisine_type_bag = get_model(training)


    print ""
    print "irish ingredients bow counts"
    ##print cuisine_type_bag['irish']
    top10 = sorted(cuisine_type_bag['irish'].items(), key=lambda x: x[1], reverse=1)
    print top10[:10]

    print ""
    print "italian ingredients bow counts"
    ##print cuisine_type_bag['irish']
    top10 = sorted(cuisine_type_bag['italian'].items(), key=lambda x: x[1], reverse=1)
    print top10[:10]

    ## bag of words by cusine types by 
    with open('training_types_bows_count', 'w') as f:
        f.write(json.dumps(cuisine_type_bag))


    ''' single item test accuracy
    ## test teh model accuracy
    print ""
    print "testing docs"
    print testing [0]
    bow = tokenize_doc(testing[0])
    results = classify(bow,cuisine_type_count,cuisine_type_bag, 0.01)
    print "gues results: " +results

    print ""
    print "full test accuracy using alpha =0.01"
    accuracy = evaluate_classifier_accuracy(testing, cuisine_type_count,cuisine_type_bag, 0.01)
    print accuracy
    '''


    ''' find alpha that max the accuracy
    ## graph some reulst using different alpha
    print "plot data"
    listofalpha = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]
    results = {}
    for n in listofalpha:
        results[n]=evaluate_classifier_accuracy(testing, cuisine_type_count,cuisine_type_bag,n)
        print "working on alpha: "+ str(n)
    keys = tuple(x[0] for x in sorted(results.items()))
    values = tuple(x[1] for x in sorted(results.items()))
    print plot_psuedocount_vs_accuracy(keys, values)
    print ""

    '''

