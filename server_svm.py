__author__ = 'Data Freaks'
# from bottle import route, run, template, request, post
from bottle import run, request, post, get
import json
import prediction_algorithm

#@route('/hello/<name>')
#def index(name):
#    return template('<b>Hello {{name}}</b>!', name=name)

@post('/getdata')
def index():
    print 'Got request, In post'
    # print request.forms.get('features')
    postdata = request.body.read()
    print type(postdata)
    # print json.loads(request.forms.get('features').replace("'", "\""))
    # print request.json('features')"0","item5":"0","item6":"0","item7":"1","item8":"0","item9":"0","item10":"0","item11":"0","item12":"1","item13":"1","item14":"1","item15":"1","item16":"1","item17":"1","item18":"1","item19":"0","item20":"0","item21":"0","item22":"0","item23":"0","item24":"1","item25":"1","item26":"1","item27":"1","__proto__":{}}}
    # {"data":{"item1":"1","item2":"2","item3":"0","item4":"0","item5":"0","item6":"0","item7":"1","item8":"0","item9":"0","item10":"0","item11":"0","item12":"1","item13":"1","item14":"1","item15":"1","item16":"1","item17":"1","item18":"1","item19":"0","item20":"0","item21":"0","item22":"0","item23":"0","item24":"1","item25":"1","item26":"1","item27":"1","__proto__":{}}}
    # Inside Prediction
    # features = json.loads(request.forms.get('features').replace("'", "\""))
    print json.loads(postdata.replace("'","\""))
    return prediction_algorithm.predict(trained_kernel, json.loads(postdata.replace("'","\"")))
    # return prediction_algorithm.predict(algorithm, business_id, json.loads(postdata.replace("'","\"")))
    # return prediction_algorithm.predict(algorithm, business_id, json.loads(postdata))

# algorithm = prediction_algorithm.train_svm()

trained_kernel = prediction_algorithm.train_svm_rbf()

run(host='localhost', port=9789)
