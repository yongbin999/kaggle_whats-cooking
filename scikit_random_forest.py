from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm

import sklearn

import scikit_convert



if __name__=='__main__':

    import sys
    import os.path
    import json

    if len(sys.argv) > 1:
        json_data = open(sys.argv[1],'r').read()
    else:
        print 'no training files defined: python input_covert.py "___train.json___"'
        sys.exit()

    data = None
    reloads = True

    if (reloads ==False):
    	x = open('./outputs/scikit_converted_data_x','r')
    	xlines = x.read().split("\n")[:-1]
    	for index,each in enumerate(xlines):
    		xlines[index] = [float(x) for x in each.split(',')]

    	y = open('./outputs/scikit_converted_data_y','r')
    	ylines = [int(x) for x in y.read().split(',')][:-1]

    	data = xlines,ylines

    else:
    	data = scikit_convert.scikit_x_vectorize(json_data), scikit_convert.scikit_y_format(json_data)





    X= data[0][30000:]
    y= data[1][30000:]

    X_test = data[0][:10000]
    y_test = data[1][:10000]

    print X[:2]
    print y[:2]
    
    """
    X = [[2, 3], [1, 1]] ## matrix
    X = sklearn.preprocessing.normalize(X)
    print X
    y = [0, 1]
    """

	
	clf = linear_model.BayesianRidge()
	clf.fit(X, y)
    from sklearn.externals import joblib
	joblib.dump(clf, 'trained_Bayes_model.pkl') 
	print "bayes done"

#### SVM

    clf = svm.SVC()
    clf.fit(X, y)  

    from sklearn.externals import joblib
	joblib.dump(clf, 'trained_SVM_model.pkl') 


    ##clf2 = pickle.loads(s)





    print clf.predict([[2., 2.]])


