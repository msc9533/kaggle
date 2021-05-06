import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
using_labels = ['Fare','Age','Sex','Pclass']
weights = [10.0,1,50.0,10.0]

def make_csv():
    """
    csv by using pandas dataframe
    """
    pass

def min_list(l,x):
    for i in range(len(l)):
        if x < l[i]:
            return True, i
    return False, 0

#TODO: refactoring this func.
def knn_classifier(dataset, test, k = 3):
    nearest_neighbor = [0] * k
    nearest_distance = [9999999999] * k
    for ps in dataset:
        residual = get_residual(ps, test)
        is_min, index = min_list(nearest_distance, residual)
        if is_min:
            nearest_neighbor[index] = ps
            nearest_distance[index] = residual
    count = 0
    for ps in nearest_neighbor:
        if type(ps) ==  int:
            continue
        if ps.survived:
            count += 1
        else:
            count -= 1
    return count >= 0

def get_residual(a,b):
    result = 0
    for label,w in using_labels,weights:
        result += abs(a[label] - b[label]) * w
    return result

def main():
    data = pd.read_csv('titanic/train.csv')
    print(data.keys())
    for idx, row in data.iterrows():
        print(idx,row['PassengerId'])

def test():
    pass

def eval():
    pass
    
if __name__ == "__main__":
    main()