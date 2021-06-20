

import numpy as np

#***************************************************************************************************************************************************
###  LOAD DATA 


def load_data(fname):

    """
    load data from dataset
    i/p : fname
    o/p : train_data,class_labels
    """
    data=[]     # empy lists to store data
    labels=[]   # empy lists to store labels
    with open(fname) as file:
        for line in file:
            features=[]

            # iteraing over each feature 
            for feature in line.strip().split(","):
                
                # filter out labels
                if(len(features)<4):
                    
                    feature=float(feature)
                    features.append(feature)
                else:
                    labels.append(feature)
            data.append(np.array(features))
    
    return np.array(data), np.array(labels)


#***************************************************************************************************************************************************
###  DATA PROCESSING

def encode_Ylabel(dataset):

    """
    encode y labels  as 1,2,3
    i/p : dataset
    o/p : y_temp
    """
    # temp list to store labels
    y_temp=[]
    for y in dataset:
        if('1' in str(y)):
            y_temp.append(1) # append class 1 code 
        elif('2' in str(y)):
            y_temp.append(2) # append class 2 code
        elif('3' in str(y)):
            y_temp.append(3) # append class 3 code


    return np.array(y_temp)


def shuffle_dataset(X, y):
    
    """
    Shuffle dataset
    i/p : X, y
    o/p : X, y
    """

    # checks for train objects and labels are of same size
    assert len(X) == len(y)
    
    #Reshuffle data and labels
    permutation =  np.random.permutation(len(X))
    X_shuffle   =  X[permutation]
    y_shuffle   =  y[permutation]
    
     
    return np.array(X_shuffle), np.array(y_shuffle)

def filterDataset(X,y,classA,classB):
    
    """
    Filter dataset by the two classes
    i/p : X,y,classA,classB
    o/p : X_filtered, y_filtered
    
    """
    # obtain class positions
    classA_pos = np.where(y == classA) 
    classB_pos = np.where(y == classB)
    
    # obtain class star adn end positions
    startA = classA_pos[0][0]
    endA = classA_pos[0][-1]
  
    startB = classB_pos[0][0]
    endB = classB_pos[0][-1]

    # filter using start and end positions
    X_a = X[startA:endA+1]
    X_b = X[startB:endB+1]
    y_A = y[startA:endA+1]
    y_B = y[startB:endB+1]

    # stack filtered data and labels to combine them
    X_filtered = np.vstack((X_a,X_b))
    y_filtered = np.hstack((y_A,y_B))

    return X_filtered,y_filtered
    
def GaussianNormalisation(dataset):
    
    """
    Apply Gaussian normalization on the features
    i/p : dataset
    o/p : (feature_mean, feature_std)
    """
    
    #Compute the number of features
    feature_count = len(dataset[0])
    
    feature_mean = np.empty(feature_count, float)
    feature_std = np.empty(feature_count, float)
    
    # iterating over  every feature
    for i in range(feature_count):
        
        # compute mean and std deviation
        feature_mean[i] = dataset[:,i].mean(axis=0) 
        feature_std[i] = dataset[:,i].std(axis=0)   
        #Apply Gaussian Noramlisation
        dataset[:,i] = (dataset[:,i] - feature_mean[i])/feature_std[i] 

    return (feature_mean, feature_std)


def normaliseTest(dataset, feature_mean, feature_std):
    
    """
    Apply Gaussian normalization on the features
    i/p : dataset
    o/p : (feature_mean, feature_std)
    """
    
    # calculate the total count of features
    feature_count = len(dataset[0])
    
    # iterating over  every feature
    for i in range(feature_count):
        # Applying Gaussian Noramlisation 
        dataset[:,i] = (dataset[:,i] - feature_mean[i])/feature_std[i]
        

#***************************************************************************************************************************************************
###  Perceptron model

class Perceptron:
    
    # perceptron model with default epcoh=20 and no L2 regularization
    def __init__(self,epoch=20,lamda2_coeff=0):
        """
        Perceptron initialisation 
        i/p : lr,epoch,lamda2_coeff
        """

        self.epoch=epoch 
        self.lamda2_coeff=lamda2_coeff
        
    def activation_score(self, X):
        """
        Calculate net input
        i/p : X
        o/p : summation
        """
        
        if(self.lamda2_coeff != 0):
            summation = (np.dot(X, self.weights) + self.bias)+self.lamda2_coeff*np.linalg.norm(self.weights) 
        else:
            summation = (np.dot(X, self.weights) + self.bias)
        
        return summation
    
    def predict(self, X):
        """
        Apply unit step function to predict using the activation score
        i/p : X
        o/p : prediction
        """
        
        prediction =  np.where(self.activation_score(X) > 0.0, 1, -1)    
        
        return prediction    
    
    def train_weights(self,X_train,y_train):
        """
        Train weights and bias
        i/p : X_train,y_train
        o/p : self.weights,self.bias
        """
        
        numFeatures = len(X_train[0]) # compute no of features
        
        #Initialize the bias term and the weights to zero
        self.weights = np.zeros(numFeatures)
        self.bias = 0
        self.errors = []
                
        
        for _ in range(self.epoch):
            
            X_train,y_train = shuffle_dataset(X_train,y_train) # shuffle dataset
            error = 0
            
            # iterating over entire dataset
            for x, target in zip(X_train, y_train):
               
                # compute activation score for the train object
                activation_score    = self.activation_score(x) 

                # check for misclassification
                if(target*activation_score<=0):

                    update        = target      # assign target as update
                    self.weights += update* x   # update weight
                    self.bias    += update      # update bias
                    error       += int(update != 0.0)
                
            self.errors.append(error)
        
        # return weights and bias
        return self.weights,self.bias
    
    
    def evaluateModel(self,X,y_test):
        """
            Evaluate model accuracy
            i/p : X,y_test
            o/p : accuracy
        """
        prediction=self.predict(X)  # predict o/p for x
        N = y_test.shape[0]         # compute no of labels

        accuracy = (y_test == prediction).sum() / N # compute accuracy
    
        # return model accuracy
        return accuracy
        

#***************************************************************************************************************************************************
###  Train Binary classification model 

def trainBinaryModel(X_train,y_train,classA,classB):
    """
        Train binary model
        i/p : X_train,y_train,classA,classB
        o/p : model
    """

    model = Perceptron(epoch=20) # Perceptron model called

    y = np.where(y_train == classA, 1, -1)  # 1 for classA and -1 for classB

    model.train_weights(X_train,y) # train weights
    
    return model


#***************************************************************************************************************************************************
###  Binary Classification 

def classifyBinary(Xtrain,ytrain,Xtest,ytest):

    """
        Binary classification  
        i/p : Xtrain,ytrain,Xtest,ytest
        o/p : train_accuracy_list,test_accuracy_list 
    """
    
    train_accuracy_list=[]  # empty list to train accuracies 
    test_accuracy_list=[]   # empty list to test accuracies

    for i in range(1,4):

        classA = 'class-'+str(i)
        if(i<3):
            classB = 'class-'+str(i+1)
        else:
            classB = 'class-'+str(abs(i-4))

        # filter train dataset based on classes
        X_train,y_train = filterDataset(Xtrain,ytrain,classA,classB) 

        # train model
        model = trainBinaryModel(X_train,y_train,classA,classB) 
    
        # encode y_train with 1 for class A and -1 for class B
        y_train  = np.where(y_train == classA, 1, -1)

        # evaluate test accuracy
        accuracy=model.evaluateModel(X_train,y_train)

        train_accuracy_list.append(accuracy)

        # filter train dataset based on classes
        X_test,y_test   = filterDataset(Xtest,ytest,classA,classB)

        # encode y_true with 1 for class A and -1 for class B
        y_test = np.where(y_test == classA, 1, -1)

        # evaluate test accuracy
        accuracy=model.evaluateModel(X_test,y_test)

        # append test accuracy
        test_accuracy_list.append(accuracy) 
        
        
    return train_accuracy_list,test_accuracy_list



#***************************************************************************************************************************************************
###  Multi Class Classification  model and Evaluation 

def oneVsRest_Prediction(dataset,models):

    """
        Multi class prediction  
        i/p : dataset,models
        o/p : y_predicted
    """
    # empty set to store predicted results 
    y_predicted=[]    
    
    # looping over every object in dataset
    for xi in dataset:
        scores=[]
        # looping over each one vs rest model
        for model in models:
            
            weights = model['weight']
            bias   = model['bias']
            # compute confidence score
            summation=np.dot(xi, weights) + bias  
            # append confidence score
            scores.append(summation)               
        
        # applying argmax function over confidence scores
        result = np.argmax(scores)  
        
        # append predicted label, 
        # +1 added to obtain label from list pos
        y_predicted.append(result+1)
        
    return  y_predicted

def oneVsRest_Evaluation(y_pred,y_true):

    """
        Multi class prediction with 1 vs Rest method  
        i/p : y_pred,y_true
        o/p : accuracy
    """
    # encode class labels as 1,2,3
    y_true = encode_Ylabel(y_true)
    # compute label count
    N = y_true.shape[0]
    # compute accuracy
    accuracy = (y_true == y_pred).sum() / N
    
    return accuracy

def multiClass_Classify(X_train,y_train,X_test,y_test,lamda2_coeff):

    """
        Multi class Classification with 1 vs Rest method 
        i/p : X_train,y_train,X_test,y_test,lamda2_coeff
        o/p : [train_Accuracy,test_Accuracy]
    """
    
    classes = np.unique(y_train)  # get unique classes 

    model = Perceptron(epoch=20,lamda2_coeff=lamda2_coeff) # Perceptron model called
    
    models = []  # empty list to store dict of model weights and bias 
    
    # iterating over the classes
    for label in classes:

        y = np.where(y_train == label, 1, -1)  # encode labels as 1 and -1

        weights,bias = model.train_weights(X_train,y) # train weights
        weight_dict = {'weight':weights,'bias':bias}  # store weights and bias in dictionary

        models.append(weight_dict)  # append weight dictionary to list
   
     
    y_train_predicted = oneVsRest_Prediction(X_train,models)          # train set prediction
    y_test_predicted  = oneVsRest_Prediction(X_test,models)           # test set prediction

    train_Accuracy =  oneVsRest_Evaluation(y_train_predicted,y_train) # train set accuracy
    test_Accuracy  =  oneVsRest_Evaluation(y_test_predicted,y_test)   # test set accuracy

    # return train and test accuracies 
    return [train_Accuracy,test_Accuracy]


#***************************************************************************************************************************************************
### CODE TESTING AGAINST CA1 QUESTIONS

def main():
    
    # load  train and test datasets
    Xtrain,ytrain = load_data(fname='data/train.data')
    Xtest,ytest   = load_data(fname='data/test.data')
    
    # Gaussian normalisation on Train Data
    (feature_mean, feature_std) = GaussianNormalisation(Xtrain)
    # Gaussian normalisation on test Data
    normaliseTest(Xtest, feature_mean, feature_std)
    
    
    """
       Question 2: Implement a binary perceptron
    """
    print('Question 2: Implement a binary perceptron')

    model = Perceptron(epoch=20)  # Perceptron Model Implemented
    y = np.where(ytrain == 'class-1', 1, -1)
    weights,bias = model.train_weights(Xtrain,y) # get weights and bias
    print("Perceptron Model implemented ")
    print('weights:',weights)
    print('bias:',bias)
    
    print('\n')
    """
     Question 3: Binary perceptron to train classifiers to discriminate between
        Class 1 and Class 2
        Class 2 and Class 3
        Class 1 and Class 3
    """
    print('Question 3: Binary perceptron to train classifiers to discriminate between')

    # obtain binary classification accuracies
    binary_train_accuracy,binary_test_accuracy = classifyBinary(Xtrain,ytrain,Xtest,ytest)

    print('Class 1 vs Class 2 Train & Test accuracies:',str(binary_train_accuracy[0]*100)+"% , "+str(binary_test_accuracy[0]*100)+"%")
    print('Class 2 vs Class 3 Train & Test accuracies:',str(binary_train_accuracy[1]*100)+"% , "+str(binary_test_accuracy[1]*100)+"%")
    print('Class 1 vs Class 3 Train & Test accuracies:',str(binary_train_accuracy[2]*100)+"% , "+str(binary_test_accuracy[2]*100)+"%")
    
    print('\n')
    """
     Question 4: Multi-class Classification using One vs Rest Approach
      
    """
    print('Question 4: Multi-class Classification using One vs Rest Approach')
    # obtain multi class accuracy
    accuracy = multiClass_Classify(Xtrain,ytrain,Xtest,ytest,lamda2_coeff=0)
    print('Multi Class Train Accuracy:',str(round(accuracy[0],3)*100),"%")
    print('Multi Class Test Accuracy:',str(round(accuracy[1],3)*100),"%")
    
   
    print('\n')
    
    """
     Question 5: Multi-class Classification using One vs Rest Approach with L2 Regularization
      
    """
    print('Question 5: Multi-class Classification using One vs Rest Approach with L2 Regularization')
    l2_coeff = [0.01, 0.1, 1.0, 10.0, 100.0]  # list contains L2 regularisation coefficients

    # iterating over the L2 coefficients
    for lamda2_coeff in l2_coeff:

        print('With L2 Regularization of ', lamda2_coeff )
        # obtain multi class accuracy with L2 coeff
        accuracy = multiClass_Classify(Xtrain,ytrain,Xtest,ytest,lamda2_coeff=lamda2_coeff)
        print('Multi Class Train Accuracy:',str(round(accuracy[0],3)*100),"%")
        print('Multi Class Test Accuracy:',str(round(accuracy[1],3)*100),"%")
        print('\n')


    

#***************************************************************************************************************************************************



if __name__ == "__main__":

    """
       main() runs only when this file is executed 
    """
    # main function contains testing for all questions
    main()  





