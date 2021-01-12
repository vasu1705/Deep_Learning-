import numpy as np
import pandas as pd
import sys


sys.stdout=open('output.txt','w')


# Neural Network Class creation
#NOTE THAT Y IS OF [1,m] and X is [n,m]

class SingleLayer():
    def __init__(self,X,Y,nodes):
        self.n_x=X.shape[0]
        self.n_y=Y.shape[0]
        self.n_h=nodes
        self.W1=np.random.randn(self.n_h,self.n_x)*0.01
        self.b1=np.zeros((self.n_h,1))
        self.W2=np.random.randn(self.n_y,self.n_h)*0.01
        self.b2=np.zeros((self.n_y,1))
    def sigmoid ( self , a):
        return 1/(1+(np.exp(a)**-1))

    def cost(self,A2,Y):
        logprobs= np.multiply(np.log(A2),Y)+np.multiply((1-Y),np.log(1-A2))
        cost=- np.sum(logprobs)/Y.shape[1]
        return cost

    def propagation(self,X,Y,lr):
        # FORWARD PROPAGATION Also same as prediction  
        
        Z1=np.dot(self.W1,X)+self.b1
        A1=np.tanh(Z1)
        Z2=np.dot(self.W2,A1)+self.b2
        A2=self.sigmoid(Z2)
        cost=self.cost(A2,Y)

        # BACKWARD PROPAGATION 
        dZ2=A2-Y
        m=Y.shape[1]
        dW2=np.dot(dZ2,A1.T)/m
        db2=np.sum(dZ2,axis=1,keepdims=True)/m
        dZ1=np.dot(self.W2.T,dZ2)*(1-np.power(A1,2))
        dW1=np.dot(dZ1,X.T)/m
        db1=np.sum(dZ1,axis=1,keepdims=True)/m

        #UPDATING PARAMETERS
        self.W1-=lr*dW1
        self.W2-=lr*dW2
        self.b1-=lr*db1
        self.b2-=lr*db2
        return cost

    def predict(self,X):
        Z1=np.dot(self.W1,X)+self.b1
        A1=np.tanh(Z1)
        Z2=np.dot(self.W2,A1)+self.b2
        A2=self.sigmoid(Z2)
        # print(A2)
        return A2>0.5
    
def conversion(train_test_data):
    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    for dataset in train_test_data:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    for dataset in train_test_data:
        z=pd.get_dummies(dataset.Pclass,prefix='C')
        for x in z:
            dataset[x]=z[x]
    for dataset in train_test_data:
        z=pd.get_dummies(dataset.Embarked,prefix='E')
        for x in z:
            dataset[x]=z[x]
    for dataset in train_test_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
        dataset['AgeBand'] = pd.cut(dataset['Age'], 5)

    for dataset in train_test_data:
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    return train_test_data

def main():

#               **********   NOTE THAT Y IS OF [1,m] and X is [n,m] **********           #

    

    dataset=pd.read_csv('D:\Atcoder\preceptron_train.csv')
    datatrain=pd.read_csv('train.csv',index_col='PassengerId')
    datatest=pd.read_csv('test.csv',index_col='PassengerId')
    datatrain,datatest=conversion([datatrain,datatest])
    print(datatrain)

    # dataset.drop('Age',axis=1,inplace=True)
    # m=dataset.shape[0]
    # dataset['Sex_bool1']=np.where(dataset['Sex_bool']==0,100,50)
    # # print(dataset)
    # x_train=((dataset[['Sex_bool','C_1','C_2','C_3']]).to_numpy()).T
    # y_train=(np.array(dataset['Survived'])).reshape(1,m)
    
    # print(x_train)
    

    #              ****** This nuron implementation gives us Accuracy of 78 %  ****** 
    # nuron=SingleLayer(x_train,y_train,1)
    # for _ in range(100):
    #     nuron.propagation(x_train,y_train,1)
    # predictions=nuron.predict(x_train)
    # print ('Accuracy: %d' % float((np.dot(y_train,predictions.T) + np.dot(1-y_train,1-predictions.T))/float(y_train.size)*100) + '%')

    # nuron=SingleLayer(x_train,y_train,10)
    # for _ in range(500):
    #     nuron.propagation(x_train,y_train,1)
    # predictions=nuron.predict(x_train)
    # print ('Accuracy: %d' % float((np.dot(y_train,predictions.T) + np.dot(1-y_train,1-predictions.T))/float(y_train.size)*100) + '%')

    
    x_train=((datatrain[['Sex','C_1','C_2','C_3','Age','E_C','E_Q','E_S']]).to_numpy()).T
    m=dataset.shape[0]
    y_train=(np.array(datatrain['Survived'])).reshape(1,m)
    nuron=SingleLayer(x_train,y_train,5)
    for _ in range(2000):
        nuron.propagation(x_train,y_train,1)
    predictions=nuron.predict(x_train)
    print ('Accuracy: %d' % float((np.dot(y_train,predictions.T) + np.dot(1-y_train,1-predictions.T))/float(y_train.size)*100) + '%')
    x_test=((datatest[['Sex','C_1','C_2','C_3','Age','E_C','E_Q','E_S']]).to_numpy()).T
    predictions=np.where(nuron.predict(x_test)==True,1,0)
    an=pd.DataFrame({'PassengerId':datatest.index,'Survived':predictions[0]})
    an.to_csv( path_or_buf='perceptron_output_1',index=False)
    pass

if __name__=='__main__':
    main()

    



    
        
    
    


