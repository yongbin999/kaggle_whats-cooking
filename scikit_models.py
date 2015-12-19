from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm

import sklearn
import scikit_convert

import sys
import os.path
import json


def scikit_readdata(filename):
	json_data = open(sys.argv[1],'r').read()

	x = open('./outputs/scikit_converted_data_x','r')
	xlines = x.read().split("\n")[:-1]
	for index,each in enumerate(xlines):
		xlines[index] = [float(x) for x in each.split(',')]

	y = open('./outputs/scikit_converted_data_y','r')
	ylines = [int(x) for x in y.read().split(',')][:-1]

	data = xlines,ylines
  	return data

def scikit_GDS(x,y, X_test,y_test=None, prevmodel="yes", output=False):
    from sklearn.linear_model import SGDClassifier
    from sklearn.externals import joblib

    clf = SGDClassifier(loss="hinge", penalty="l2")
    ##
    if prevmodel !="yes":
    	clf.fit(X, y)
    	joblib.dump(clf, 'trained_GDS_model.pkl') 
    else:
    	clf =joblib.load('trained_GDS_model.pkl')

    if output == False:
        predictions =  clf.predict(X_test)
        correctcount = 0
        totalcount = 0
        for index, each in enumerate(predictions):
        	if y_test[index] == each:
        		correctcount +=1
        	totalcount+=1

        print str(correctcount) +" / " + str(totalcount) +" = " + str(float(correctcount)/totalcount)
    else:
        predictions =  clf.predict(X_test)
        return predictions

def scikit_onevsrest(X,y, X_test,y_test=None):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC
    predictions = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X_test)
    correctcount = 0
    totalcount = 0
    for index, each in enumerate(predictions):
        if y_test[index] == each:
            correctcount +=1
        totalcount+=1

    print str(correctcount) +" / " + str(totalcount) +" = " + str(float(correctcount)/totalcount)


def scikit_onevsone(X,y, X_test,y_test=None):
    from sklearn.multiclass import OneVsOneClassifier
    from sklearn.svm import LinearSVC
    predictions = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X_test)
    correctcount = 0
    totalcount = 0
    for index, each in enumerate(predictions):
        if y_test[index] == each:
            correctcount +=1
        totalcount+=1

    print str(correctcount) +" / " + str(totalcount) +" = " + str(float(correctcount)/totalcount)

def scikit_outputcode(X,y, X_test,y_test=None):
    from sklearn.multiclass import OutputCodeClassifier
    from sklearn.svm import LinearSVC
    predictions = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0).fit(X, y).predict(X_test)
    correctcount = 0
    totalcount = 0
    for index, each in enumerate(predictions):
        if y_test[index] == each:
            correctcount +=1
        totalcount+=1

    print str(correctcount) +" / " + str(totalcount) +" = " + str(float(correctcount)/totalcount)



if __name__=='__main__':

    import sys
    import os.path
    import json

    if len(sys.argv) > 1:
        json_data = open(sys.argv[1],'r').read()
    else:
        print 'no training files defined: python input_covert.py "___train.json___"'
        sys.exit()


    data = scikit_convert.scikit_x_vectorize(json_data), scikit_convert.scikit_y_format(json_data)


    X= data[0][:30000]
    y= data[1][:30000]

    X_test = data[0][30000:]
    y_test = data[1][30000:]

    print X[:2]
    print y[:2]
    """
    X = [[2, 3], [1, 1]] ## matrix
    y = [0, 1]
    """

    test_data = None
    if len(sys.argv) > 2:
        json_test = open(sys.argv[2],'r').read()
        test_data = scikit_convert.scikit_x_vectorize(json_test)



## GDS 7552 / 9774 = 0.772662164927
    print "running gradient descent"
    scikit_GDS(X,y,X_test,y_test,"yes")


## one vs rest  multiclass 7515 / 9774 = 0.768876611418
    print "one vs rest"
    scikit_onevsrest(X,y,X_test,y_test)

## output code error corrtingmulticlass 7501 / 9774 = 0.76744423982
    print "out put code error corecting"
    scikit_outputcode(X,y,X_test,y_test)





## one vs one  memory error
    #print "one vs rest"
    ##scikit_onevsone(X,y,X_test,y_test)


##gausian nb memory error
"""
## naive bayes gussian
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(X, y)

    print y_pred.predict(X_test[:1])
    print y_test[:1]
"""
#### SVM memory error
"""
    clf = svm.SVC()
    clf.fit(X, y)  

    from sklearn.externals import joblib
    joblib.dump(clf, 'trained_SVM_model.pkl') 
    print "done svm"
"""
