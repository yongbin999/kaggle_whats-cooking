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

def scikit_GDS(x,y, X_test,y_test, prevmodel=None):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="hinge", penalty="l2")
    ##
    if prevmodel ==None:
    	clf.fit(X, y)
    	from sklearn.externals import joblib
    	joblib.dump(clf, 'trained_GDS_model.pkl') 
    	print "done"
    else:
    	clf =joblib.load('trained_GDS_model.pkl')


    predictions =  clf.predict(X_test)
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


    X= data[0][30000:]
    y= data[1][30000:]

    X_test = data[0][:10000]
    y_test = data[1][:10000]

    print X[:2]
    print y[:2]
    """
    X = [[2, 3], [1, 1]] ## matrix
    y = [0, 1]
    """


## GDS
    print "running gradient descent"
    scikit_GDS(X,y,X_test,y_test,"yes")




##gausian nb
"""
## naive bayes gussian
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(X, y)

    print y_pred.predict(X_test[:1])
    print y_test[:1]
"""
#### SVM
"""
    clf = svm.SVC()
    clf.fit(X, y)  

    from sklearn.externals import joblib
    joblib.dump(clf, 'trained_SVM_model.pkl') 
    print "done svm"
"""
