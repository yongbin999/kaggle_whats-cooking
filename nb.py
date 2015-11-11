from __future__ import division

import math
import os, sys
import  json

from collections import defaultdict


def tokenize_doc(doc):
    """
    IMPLEMENT ME!

    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """

    words = {}
    for w in doc.lower().split( ):
        if w not in words:
            words[w] =0
        words[w] +=1
    
    return words

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }


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

    def update_model(self, bow, label):
        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """

        if bow != None:
            self.vocab.update(bow.keys())

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
            self.class_total_doc_counts[label] += 1
        
        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
            self.class_total_word_counts[label] += len(bow)

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
            for w in bow:
                if w not in self.class_word_counts[label]:
                    self.class_word_counts[label][w] = bow[w]
                self.class_word_counts[label][w] += bow[w]


    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """
        Implement me!

        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.class_word_counts[label].items(), key=lambda (w,c): -c)[:n]

    def p_word_given_label(self, word, label):
        """
        Implement me!

        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        """
        return self.class_word_counts[label][word] / len(self.class_word_counts[label])

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        """
        Implement me!

        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        return (self.class_word_counts[label][word]+ alpha) / (len(self.class_word_counts[label]) + alpha*len(self.vocab))

    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """
        likelyhood = 0
        for w in bow:
            #print w + "   " + str(math.log(self.p_word_given_label_and_psuedocount(w,label,alpha)))
            likelyhood += math.log(self.p_word_given_label_and_psuedocount(w,label,alpha))

        return likelyhood

    def log_prior(self, label):
        """
        Implement me!

        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
        prior = self.class_total_doc_counts[label] / (self.class_total_doc_counts[POS_LABEL] + self.class_total_doc_counts[NEG_LABEL])
        
        return math.log(prior)

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Implement me!

        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        
        return self.log_prior(label) + self.log_likelihood(bow, label, alpha)

    def classify(self, bow, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """
        pos = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        neg = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)

        if pos > neg:
            return POS_LABEL
        else:
            return NEG_LABEL

    def likelihood_ratio(self, word, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        return self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha) / self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)

    def evaluate_classifier_accuracy(self, alpha, num_docs=None):
        """
        Implement me!

        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        numofcorrect = 0.0;
        totalnumdocs = 0.0;
        num_docs = num_docs;

        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)

        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    evualuated_label = self.classify(tokenize_doc(content),alpha)
                    totalnumdocs+=1.0
                    if evualuated_label ==label: numofcorrect+=1.0


        return numofcorrect / totalnumdocs


def produce_hw1_results():
    # PRELIMINARIES
    # uncomment the following 9 lines after you've implemented tokenize_doc
    d1 = "this sample doc has   words that  repeat repeat"
    bow = tokenize_doc(d1)

    assert bow['this'] == 1
    assert bow['sample'] == 1
    assert bow['doc'] == 1
    assert bow['has'] == 1
    assert bow['words'] == 1
    assert bow['that'] == 1
    assert bow['repeat'] == 2
    print ''

    # QUESTION 1.1
    # Implementation only

    # QUESTION 1.2
    # uncomment the next two lines when ready to answer question 1.2
    print "VOCABULARY SIZE: " + str(len(nb.vocab))
    print ''

    # QUESTION 1.3
    # uncomment the next set of lines when ready to answer qeuestion 1.2
    print "TOP 10 WORDS FOR CLASS " + POS_LABEL + " :"
    for tok, count in nb.top_n(POS_LABEL, 10):
        print '', tok, count
    print ''

    print "TOP 10 WORDS FOR CLASS " + NEG_LABEL + " :"
    for tok, count in nb.top_n(NEG_LABEL, 10):
        print '', tok, count
    print ''

    # hw 2.2
    print "P of fantastic given " + POS_LABEL + "|" +NEG_LABEL + " :"
    print str(nb.p_word_given_label("fantastic",POS_LABEL)) + "|" + str(nb.p_word_given_label("fantastic",NEG_LABEL))

    print "P of boring given " + POS_LABEL + "|" +NEG_LABEL + " :"
    print str(nb.p_word_given_label("boring",POS_LABEL)) + "|" + str(nb.p_word_given_label("boring",NEG_LABEL))
    print ''

    # hw2.4
    alpha = 1.0
    print "with alpha, P of fantastic given " + POS_LABEL + "|" +NEG_LABEL + " :"
    print str(nb.p_word_given_label_and_psuedocount("fantastic",POS_LABEL,alpha)) + "|" + str(nb.p_word_given_label_and_psuedocount("fantastic",NEG_LABEL,alpha))

    print "with alpha, P of boring given " + POS_LABEL + "|" +NEG_LABEL + " :"
    print str(nb.p_word_given_label_and_psuedocount("boring",POS_LABEL,alpha)) + "|" + str(nb.p_word_given_label_and_psuedocount("boring",NEG_LABEL,alpha))
    print ''

    #5.1
    print " evualuate classify with alpha"
    print nb.evaluate_classifier_accuracy(1.0, None)
    print ""

    #5.2

    print "plot data"
    listofalpha = [0.01,0.1,1,10,100,]
    results = {}
    for n in listofalpha:
        results[n]=nb.evaluate_classifier_accuracy(n, 1000)
    keys = tuple(x[0] for x in sorted(results.items()))
    values = tuple(x[1] for x in sorted(results.items()))
    print plot_psuedocount_vs_accuracy(keys, values)
    print ""



    #hw6.3
    print " LR ratios"
    print nb.likelihood_ratio("fantastic",1.0)
    print nb.likelihood_ratio("boring",1.0)
    print nb.likelihood_ratio("the",1.0)
    print nb.likelihood_ratio("to",1.0)
    print ""




    print '[done.]'

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


    ## count the stats on cusine types 
    cuisine_type_count = defaultdict(float)
    for row in data:
        cuisine_type_count[row['cuisine']] +=1

    types = len(cuisine_type_count)
    size = len(data)
    print 'size of the training data' +str(size)

    cuisine_type_count = {w: round(c/size,3) for (w,c) in cuisine_type_count.items() }
    cuisine_type_count = sorted(cuisine_type_count.items(), key=lambda x: x[1], reverse=1)

    print json.dumps(cuisine_type_count)
    with open(sys.argv[2], 'w') as f:
        f.write(json.dumps(cuisine_type_count))




    ## create the training model
    cuisine_type_bag = defaultdict()
    for row in data:
        for ingredient in row['ingredients']:
            if row['cuisine'] not in cuisine_type_bag:
                cuisine_type_bag[row['cuisine']] = {}

            if ingredient not in cuisine_type_bag[row['cuisine']]:
                cuisine_type_bag[row['cuisine']][ingredient] =1
            else:
                cuisine_type_bag[row['cuisine']][ingredient]+=1

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





    ##random.shuffle(examples)




