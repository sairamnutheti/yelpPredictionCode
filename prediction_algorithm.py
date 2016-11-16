__author__ = 'Data Freaks'

import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
import data_importer

def predict(trained_kernel, features):
    print "features:"
    print type(features)

    print 'Inside Prediction'
    print 'Calling Predict'

    return {'new_rating': trained_kernel.predict([int(features["data"]["item1"]),int(features["data"]["item2"]),int(features["data"]["item3"]),int(features["data"]["item4"]),int(features["data"]["item5"]),int(features["data"]["item6"]),int(features["data"]["item7"]),int(features["data"]["item8"]),int(features["data"]["item9"]),int(features["data"]["item10"]),int(features["data"]["item11"]),int(features["data"]["item12"]),int(features["data"]["item13"]),int(features["data"]["item14"]),int(features["data"]["item15"]),int(features["data"]["item16"]),int(features["data"]["item17"]),int(features["data"]["item18"]),int(features["data"]["item19"]),int(features["data"]["item20"]),int(features["data"]["item21"]),int(features["data"]["item22"]),int(features["data"]["item23"]),int(features["data"]["item24"]),int(features["data"]["item25"]),int(features["data"]["item26"]),int(features["data"]["item27"])])[0]};


def train_svm_rbf():
    yelp_data = data_importer.get_yelp_data()
    test_data = np.array(yelp_data)
    # print test_data
    yelp_x = yelp_data['data']
    yelp_y = yelp_data['target']
    #print yelp_x
    #print yelp_y
    yelp_x_train = yelp_x[:len(yelp_x)*85/100]
    yelp_y_train = yelp_y[:len(yelp_y)*85/100]

    yelp_x_test = yelp_x[(len(yelp_x)*85/100)+1:]
    yelp_y_test = yelp_y[(len(yelp_y)*85/100)+1:]

    # print yelp_x_test[:10]
    # print yelp_y_test[:10]
    svr_rbf=SVR(kernel='rbf', C=1e3, gamma=0.1)
    # -svr_poly = SVR(kernel='rbf', C=1e3, gamma=0.1)
    t0 = time.time()
    print('Training SVR algorithm with training data started.')
    print 'Training in Progress...... Please Wait'
    # - y_rbf = svr_rbf.fit(yelp_x, yelp_y)
    y_rbf = svr_rbf.fit(yelp_x_train, yelp_y_train)
    svr_fit = time.time() - t0
    print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit) 
    # # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # # The mean square error
    # print("Residual sum of squares: %.2f"% np.mean((regr.predict(yelp_x_test) - yelp_y_test) ** 2))
    # # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % svr_rbf.score(yelp_x_test, yelp_y_test))
    # print svr_rbf.predict([1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0])
    return y_rbf



# train()
