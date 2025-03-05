#-------------------------------------------------------------------------
# AUTHOR: Michelle Reyes
# FILENAME: 'decision_tree_2.py'
# SPECIFICATION: I trained the Decision tree model with varying datasets(different amount of instances). To observe the final accuracy,Itested the model on the test dataset over 10 iterations.Icounted the number of correct predictions based on the test data. Then find the average over the 10 iterations.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1:30 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
dictionary = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3,
                  'Hypermetrope': 1, 'Myope': 2, 'Yes': 2, 'No': 1, 'Normal': 1, 'Reduced': 2               
                   }



for ds in dataSets:


    dbTraining = []
    X = []
    Y = []


    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here

    for i in range(len(dbTraining)):
        array_helper = []
        for value in range(len(dbTraining[i])):  #0,1,2,3
            if dbTraining[i][value] in dictionary:
                if value == len(dbTraining[i]) - 1:  #get last element,class
                    Y.append(dictionary[dbTraining[i][value]]) 
                else:
                    dbTraining[i][value] = dictionary[dbTraining[i][value]] 
                    array_helper.append(dbTraining[i][value])  
        
        X.append(array_helper)

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Obtain Y through first loop
    average = []
    
    #Loop your training and test tasks 10 times here
    for i in range (10):
       

       #Fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest = []
       test_data = 'contact_lens_test.csv'
       with open(test_data, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)

       counter = 0
       count = 0


       for data in dbTest:
        
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            X_test = []
            Y_test = []

            array_helper = []
            for value in range(len(data)):  #0,1,2,3
                if value == len(data) - 1:  # Last element 
                        Y_test.append(dictionary[data[value]]) 
                else:
                        array_helper.append(dictionary[data[value]])  

            X_test.append(array_helper)               

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            for i in range(len(X_test)):
                class_predicted = clf.predict([X_test[i]])[0]  # Predict using the classifier
                if class_predicted == Y_test[i]:
                    counter += 1

  

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
     
       helper = counter/len(dbTest)
       average.append(helper)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    average = sum(average) /10
  
    print(f"final accuracy when training on {ds}: {average:.2f}")
    
