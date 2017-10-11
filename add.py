def f(x):
    return 1 if x>0 else 0

def get_training_dataset():
    input_vecs=[[1,1],[0,0],[1,0],[0,1]]

    lables=[1,0,0,0]
    return input_vecs,lables

def train_and_perceptron():
    p=Per