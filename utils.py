import pickle
import numpy as np
def load_model(data):
    global __model
    with open("./artifacts/diabetes.pickle",'rb') as f:
        print("file opend as f")
        __model = pickle.load(f)
    print("model load... done")

    print("predicting.... start")    
    pred = extimate_risk(data)
    return pred

def extimate_risk(data):
    lst = data[0]
    arr = np.asarray(lst)   
    risk = __model.predict([arr])
    return (risk[0])



if __name__=="__main__":
    load_model()