'''Nasit Uygun 240201012
    CENG463 hw1
    '''


import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random
import math



def plot_gaussian(mu,Sigma):
    x,y=np.mgrid[-50:50:300j,-4000:12000:300j]
    xy=np.column_stack([x.flat,y.flat])
    mean=np.array(mu)
    covariance=np.array(Sigma)
    z=multivariate_normal.pdf(xy,mean=mean,cov=covariance)
    z=z.reshape(x.shape)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    ax.plot_surface(x,y,z)
    plt.show()
    
def accuracy_score(prediction,actual):
    individual_accuracy=0
    for i in range(len(prediction)):
        if prediction[i]==actual[i]:
            individual_accuracy=individual_accuracy+1
    return individual_accuracy/len(prediction)
def splitData(percentage):
    dataDictionary={"class1":[[],[]],"class2":[[],[]]}
    for i in data:
        dataDictionary["class"+str(i[2])][0].append(i[0])
        dataDictionary["class"+str(i[2])][1].append(i[1])
        
    for i in range(int(percentage*len(data))):
        a=random.randint(0,100)
        if a<30:
            a=2
        else:
            a=1
        index=random.randint(0,10000)%len(dataDictionary["class"+str(a)][0])
        training["class"+str(a)][0].append(dataDictionary["class"+str(a)][0].pop(index))
        training["class"+str(a)][1].append(dataDictionary["class"+str(a)][1].pop(index))
    for i in range(len(dataDictionary["class"+str(1)][0])):
        if len(dataDictionary["class"+str(1)][0])>0:
            test["class"+str(1)][0].append(dataDictionary["class"+str(1)][0].pop())
            test["class"+str(1)][1].append(dataDictionary["class"+str(1)][1].pop())
        if len(dataDictionary["class"+str(2)][0])>0:
            test["class"+str(2)][0].append(dataDictionary["class"+str(2)][0].pop())
            test["class"+str(2)][1].append(dataDictionary["class"+str(2)][1].pop())

def calculateMLEParameters():
    global class1mean,class2mean,class2cov,class1cov
    class1mean=np.array([np.mean((training["class1"][0])),np.mean((training["class1"][1]))])
    class2mean=np.array([np.mean((training["class2"][0])),np.mean((training["class2"][1]))])
    class1cov=np.cov(training["class1"][0], training["class1"][1])
    class2cov=np.cov(training["class2"][0], training["class2"][1])
def log(y):
  return math.log(y) / math.log(10)
       

# to find maximum of the discriminant function results
def find_max_likelihood(class1,class2):
    if class1>class2:
        return "class1"
    else:
        return "class2" 

# to calculate un-normalized posteriors and do classification with respect to g1 and g2results
def do_classification():
    decision_list = []  # to store classification results to calculate accuracy later.
    checkList=[]
    class1PDF=multivariate_normal(mean=class1mean,cov=class1cov)
    class2PDF=multivariate_normal(mean=class2mean,cov=class2cov)
    for i in test.keys():
        for k in range(len(test[i][0])):
            class1LH=class1PDF.logpdf((test[i][0][k],test[i][1][k]))
            class2LH=class2PDF.logpdf((test[i][0][k],test[i][1][k]))
            class1prior=len(training["class"+str(1)][0])/(len(training["class"+str(1)][0])+len(training["class"+str(2)][0]))
            class2prior=len(training["class"+str(1)][0])/(len(training["class"+str(1)][0])+len(training["class"+str(2)][0]))

            class1post=(class1LH+log(class1prior))
            class2post=class2LH+log(class2prior)
            decision_list.append(find_max_likelihood(class1post,class2post))
            checkList.append(i)
            

    return [decision_list,checkList]



cols_names = ['Checking', 'Duration','Credit history','Purpose','Credit amount', 
              'Savings','Present employment since','Installment','Personal status and sex',
              'Other debtors / guarantors','Present residence since','Property',
              'age', 'Other installment','Housing','Number of existing credits at this bank',
              'Job','Number of people being liable to provide maintenance for',
              'Telephone','foreign worker','class']
data = pd.read_csv("german.data",sep=' ',names=cols_names)
data=np.array(data[['Duration','Credit amount','class']])



accuracy = 0  # total accuracy
# program will run for 500 times to find average accuracy
accuracies=[] # to store individual accuracies
for i in range(0, 500):
    training={"class1":[[],[]],"class2":[[],[]]}
    test={"class1":[[],[]],"class2":[[],[]]}
    splitData(0.67)
    
    calculateMLEParameters()

    classification_result = do_classification()

    # compare classification results with y_test
    individual_accuracy = accuracy_score(classification_result[0], classification_result[1])
    accuracy = accuracy + individual_accuracy
    accuracies.append(individual_accuracy)

print("Average Accuracy = %", (accuracy/500.0)*100)
print("Max Accuracy = %", max(accuracies)*100)

plot_gaussian(class1mean,class1cov)
plot_gaussian(class2mean,class2cov)


