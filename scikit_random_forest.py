from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model

import input_convert



if __name__=='__main__':

    import sys

    if len(sys.argv) > 1:
        json_data = open(sys.argv[1],'r').read()
    else:
        print 'no training files defined: python input_covert.py "___train.json___"'
        sys.exit()

    data = input_convert.scikit_learn_format(json_data)

    X= data[0][30000:]
    Y= data[1][30000:]

    X_test = data[0][:10000]
    y_test = data[1][:10000]

    clf = linear_model.BayesianRidge()
    clf.fit(X, Y)

    print clf

    BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
       normalize=False, tol=0.001, verbose=False)

    print lf.predict ([['salt']])