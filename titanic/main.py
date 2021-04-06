import numpy as np
import pandas as pd
from math import isnan

def get_residual(a,b):
    result = 0
    result += abs(a.fare - b.fare)
    result += abs(a.age - b.age)
    result += abs(a.sex - b.sex) * 30
    return result

class passenger():
    def __init__(self):
        self.passengerid = ""
        self.survived = ""
        self.pclass = ""
        self.name = ""
        self.sex = ""
        self.age = ""
        self.sibsp = ""
        self.parch = ""
        self.ticket = ""
        self.fare = ""
        self.cabin = ""
        self.embarked = ""

def csv2passengers(filename,is_test = False,gt_exist=False,gtfile=""):
    result = []
    csv = pd.read_csv(filename)
    if gt_exist:
        gt_csv = pd.read_csv(gtfile)
        gt_dict = {}
        for p_id,gt in zip(gt_csv['PassengerId'],gt_csv['Survived']):
            gt_dict[p_id] = int(gt)

    for i in range(len(csv['PassengerId'])):
        psg = passenger()
        psg.passengerid = csv['PassengerId'][i]
        if is_test and gt_exist:
            psg.survived = gt_dict[psg.passengerid]
        elif is_test:
            psg.survived = None
        else:
            psg.survived = int(csv['Survived'][i])
        if psg.survived:
            psg.color = 'blue'
        else:
            psg.color = 'red'
        psg.pclass = csv['Pclass'][i]
        psg.name = csv['Name'][i]
        psg.sex = csv['Sex'][i]
        if psg.sex == 'male':
            psg.sex = 1
        else:
            psg.sex = 0

        if isnan(csv['Age'][i]):
            psg.age = 0
        else:
            psg.age = int(csv['Age'][i])
        psg.sibsp = csv['SibSp'][i]
        psg.parch = csv['Parch'][i]
        psg.ticket = csv['Ticket'][i]
        psg.fare = float(csv['Fare'][i])
        psg.cabin = csv['Cabin'][i]
        psg.embarked = csv['Embarked'][i]
        result.append(psg)
    return result

def min_list(l,x):
    for i in range(len(l)):
        if x < l[i]:
            return True, i
    return False, 0

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

def read_gt(filename,testset):
    csv = pd.read_csv(filename)
    for i,ps in enumerate(testset):
        psg = passenger()
        psg.survived = csv['PassengerId'][i] #csv['Survived'][i]

def make_csv(data,filename):
    with open(filename,'w') as f:
        f.write('PassengerId,Survived\n')
        for ps in data:
            f.write(str(ps.passengerid))
            f.write(',')
            f.write(str(int(ps.survived)))
            f.write('\n')

def test():
    train = csv2passengers("titanic/train.csv")
    test = csv2passengers("titanic/test.csv",is_test=True,gt_exist=True,
                            gtfile="titanic/gender_submission.csv")

    count = 0
    total = len(test)
    correct = 0
    for ps in test:
        # if count >= 20:
            # break
        count += 1
        predict = knn_classifier(train,ps)
        if predict == ps.survived:
            correct += 1
        # print("pred :",predict, " gt : ", ps.survived)
    print(float(correct)/total)

def eval():
    train = csv2passengers("titanic/train.csv")
    test = csv2passengers("titanic/test.csv",is_test=True)
    for ps in test:
        predict = knn_classifier(train,ps)
        ps.survived = predict
    make_csv(test,'titanic/out.csv')


if __name__ == "__main__":
    # main()
    eval()