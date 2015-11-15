from __future__ import division
import sys,re,random
from collections import defaultdict
from pprint import pprint

##########################
# Important stuff you will use

# import vit
import vit_sol as vit
# OUTPUT_VOCAB = set(["B","I","O"])
OUTPUT_VOCAB = set(""" ! # $ & , @ A D E G L M N O P R S T U V X Y Z ^ """.split())

##########################
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

def read_tagging_file(filename):
    """Returns list of sentences from a two-column formatted file.
    Each returned sentence is the pair (tokens, tags) where each of those is a
    list of strings.
    """
    sentences = open(filename).read().strip().split("\n\n")
    ret = []
    for sent in sentences:
        lines = sent.split("\n")
        pairs = [L.split("\t") for L in lines]
        tokens = [tok for tok,tag in pairs]
        tags = [tag for tok,tag in pairs]
        ret.append( (tokens,tags) )
    return ret

###############################

def train(examples, stepsize=1, numpasses=10, do_averaging=False, devdata=None):
    # examples: list of  (label, featvec) pairs
    # the label is boolean
    # the featvecs are dictionaries

    weights = defaultdict(float)
    weightSums = defaultdict(float)
    t = 0

    def get_averaged_weights():
        return {f: weights[f] - 1/t*weightSums[f] for f in weightSums}

    for pass_iteration in range(numpasses):
        print "Training iteration %d" % pass_iteration
        for tokens,goldlabels in examples:
            t += 1
            predlabels = predict(tokens, weights)
            goldfeats = fullseq_features(tokens, goldlabels)
            predfeats = fullseq_features(tokens, predlabels)
            featdelta = dict_subtract(goldfeats, predfeats)
            for f,value in featdelta.items():
                weights[f] += stepsize * value
                weightSums[f] += (t-1) * stepsize * value

        # print "TR  RAW EVAL:",
        # do_evaluation(examples, weights)

        if devdata:
            print "DEV RAW EVAL:",
            do_evaluation(devdata, weights)

        if devdata and do_averaging:
            print "DEV AVG EVAL:",
            do_evaluation(devdata, get_averaged_weights())

    print "Learned weights for %d features from %d examples" % (len(weights), len(examples))

    return weights if not do_averaging else get_averaged_weights()

def do_evaluation(examples, weights):
    num_correct,num_total=0,0
    for tokens,goldlabels in examples:
        N = len(tokens); assert N==len(goldlabels)
        predlabels = predict(tokens, weights)
        num_correct += sum(predlabels[t]==goldlabels[t] for t in range(N))
        num_total += N
    print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
    return num_correct/num_total

def fancy_eval(examples, weights):
    confusion = defaultdict(float)
    bygold = defaultdict(lambda:{'total':0,'correct':0})
    for tokens,goldlabels in examples:
        predlabels = predict(tokens, weights)
        for pred,gold in zip(predlabels, goldlabels):
            confusion[gold,pred] += 1
            bygold[gold]['correct'] += int(pred==gold)
            bygold[gold]['total'] += 1
    goldaccs = {g: bygold[g]['correct']/bygold[g]['total'] for g in bygold}
    for gold in sorted(goldaccs, key=lambda g: -goldaccs[g]):
        print "gold %s acc %.4f (%d/%d)" % (gold,
                goldaccs[gold],
                bygold[gold]['correct'],bygold[gold]['total'],)

def predict(tokens, weights):
    Ascores,Bscores = calc_factor_scores(tokens, weights)
    predlabels = vit.viterbi(Ascores, Bscores, OUTPUT_VOCAB)
    # predlabels = greedy_decode(Ascores, Bscores, OUTPUT_VOCAB)
    # predlabels = indep_decode(Ascores, Bscores, OUTPUT_VOCAB)
    return predlabels

def greedy_decode(Ascores, Bscores, OUTPUT_VOCAB):
    """Left-to-right greedy decoding.  Uses transition feature for prevtag to curtag."""
    N=len(Bscores)
    if N==0: return []
    out = [None]*N
    out[0] = dict_argmax(Bscores[0])
    for t in range(1,N):
        tagscores = {tag: Bscores[t][tag] + Ascores[out[t-1], tag] for tag in OUTPUT_VOCAB}
        besttag = dict_argmax(tagscores)
        out[t] = besttag
    return out

def indep_decode(_, Bscores, OUTPUT_VOCAB):
    """Decode using only the Bscores.  Ignore transition features.  So each tag decision is independent."""
    N=len(Bscores)
    out = [None]*N
    for t in range(N):
        out[t] = dict_argmax(Bscores[t])
    return out

def local_emission_features_fancy(t, tag, tokens):
    """
    Feature vector for the B_t(y) function
    t: an integer, index for a particular position
    tag: a hypothesized tag to go at this position
    tokens: the list of strings of all the word tokens in the sentence.
    """
    curword = tokens[t]
    feats = {}
    feats["tag=%s_biasterm" % tag] = 1
    feats["tag=%s_curword=%s" % (tag, curword)] = 1

    # feats["tag=%s_lccurword=%s" % (tag, curword.lower())] = 1
    for delta in [-1,1]:
        if t+delta<0: continue
        if t+delta>=len(tokens): continue
        feats["tag=%s_lcword_at_%d=%s" % (tag, delta, tokens[t+delta].lower())] = 1
    feats["tag=%s_word_starts_with_@" % tag] = int(curword[0]=="@")
    feats["tag=%s_word_starts_with_#" % tag] = int(curword[0]=="#")
    # # feats["tag=%s_firstchar=%s" % (tag, curword[0])] = 1
    # feats["tag=%s_position=%d" % (tag,t)] = 1
    # feats["tag=%s_position_from_right=%d" % (tag,len(tokens)-t)] = 1
    feats["tag=%s_lastchar=%s" % (tag, curword[-1])] = 1
    if t>0:
        feats["tag=%s_prevword_lastchar=%s" % (tag, tokens[t-1][-1])] = 1
    return feats

def local_emission_features_starter(t,tag,tokens):
    curword = tokens[t]
    feats = {}
    feats["tag=%s_biasterm" % tag] = 1
    feats["tag=%s_curword=%s" % (tag, curword)] = 1
    return feats

local_emission_features = local_emission_features_starter

def fullseq_features(tokens, labelseq):
    """
    The full f(x,y) function. Returns one big feature vector. This is similar to
    fullseq_features in the classifier peceptron except here we aren't dealing
    with binary classification. Since this is structured prediction there are many
    tags to deal with.
    """
    feats = defaultdict(float)
    N = len(tokens)
    for t in range(N):
        if t>0:
            feats["trans_%s_%s" % (labelseq[t-1],labelseq[t])] += 1
        localfeats = local_emission_features(t, labelseq[t], tokens)
        for f,value in localfeats.iteritems():
            feats[f] += value
    return dict(feats)

def calc_factor_scores_dummy(tokens, weights):
    N = len(tokens)
    Ascores = { (tag1,tag2): 0 for tag1 in OUTPUT_VOCAB for tag2 in OUTPUT_VOCAB }
    Bscores = []
    for t in range(N):
        Bscores.append({tag: 0 for tag in OUTPUT_VOCAB})
    return Ascores, Bscores


def calc_factor_scores(tokens, weights):
    Ascores = {(a,b):weights.get("trans_%s_%s" % (a,b), 0) for a in OUTPUT_VOCAB for b in OUTPUT_VOCAB}
    Bscores = []
    N = len(tokens)
    for t in range(N):
        tagscores = {}
        for tag in OUTPUT_VOCAB:
            feats = local_emission_features(t, tag, tokens)
            tagscores[tag] = dict_dotprod(feats, weights)
        Bscores.append(tagscores)
    return Ascores,Bscores

def show_predictions(tokens, goldlabels, predlabels):
    print "%-20s %-4s %-4s" % ("word", "gold", "pred")
    for w, goldy, predy in zip(tokens, goldlabels, predlabels):
        out = "%-20s %-4s %-4s" % (w,goldy,predy)
        if goldy!=predy:
            out += "  *** Error"
        print out.encode("utf8")

# calc_factor_scores = calc_factor_scores_dummy

if __name__ == '__main__':
    # find most common tag
    dev_set = read_tagging_file('oct27.dev')
    training_set = read_tagging_file("oct27.train")
    tag_counts = defaultdict(float)
    for (tokens, tags) in dev_set:
        for t in tags:
            tag_counts[t] += 1

    print "[most common tag.]"
    print str(dict_argmax(tag_counts))

    # accuracy if predicting most common tag only
    def most_common_tag_eval(examples, weights):
        num_correct,num_total=0,0
        for tokens,goldlabels in examples:
            N = len(tokens); assert N==len(goldlabels)
            predlabels = ['V'] * N
            num_correct += sum(predlabels[t]==goldlabels[t] for t in range(N))
            num_total += N
        print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
        return num_correct/num_total

    print "Most Common Tag Eval (Train Set): %f" % most_common_tag_eval(training_set, 0)
    print "Most Common Tag Eval (Test Set): %f" % most_common_tag_eval(dev_set, 0)

    # testing local features
    test_sentence = "this is a sentence".split()
    tag = 'D'
    print "[local emission features.]"
    print local_emission_features(0, tag, test_sentence)
    print local_emission_features(1, tag, test_sentence)
    print local_emission_features(2, tag, test_sentence)
    print local_emission_features(3, tag, test_sentence)

    # testing fullseq_features
    print "[full seq features.]"
    print fullseq_features(test_sentence, ['V'] * len(test_sentence))

    # clas_factor_scores
    print "[calc factor scores.]"
    weights = {}
    for a in OUTPUT_VOCAB:
        for b in OUTPUT_VOCAB:
            weights["trans_%s_%s" % (a,b)] = random.random()
    # print calc_factor_scores(test_sentence, weights)

    # prediction
    print "[predict.]"
    print predict(test_sentence, weights)

    # training
    print "[training.]"
    weights = train(training_set, do_averaging=True, devdata=dev_set, numpasses=10)

    # eval
    print "[eval.]"
    print fancy_eval(dev_set, weights)

    i=0
    print show_predictions(dev_set[i][0], dev_set[i][1], predict(dev_set[i][0], weights))
