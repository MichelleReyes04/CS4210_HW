#-------------------------------------------------------------------------
# AUTHOR: Michelle Reyes
# FILENAME: 'knn.py'
# SPECIFICATION: To execute LOO-CV, at each row in the dataset, the current row is the test data(x_test,y_test) where as the rest of the data is the training the data. 
# I compare the predict y_test to the correct y_test and count the correct predictions, which will used  to  find the Error rate of the dataset.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 50 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

dictionary ={
   'ham': 1,
   'spam': 2
}

counter = 0
#Loop your data to allow each instance to be your test set
for i in range(len(db)):
    x_test = []
    y_test = []
    X = []
    Y = []



    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    for cross_val in range(len(db)):
       if i == cross_val:
          
          for float_value in db[cross_val][:-1]:    
             x_test.append(float(float_value))
          y_test.append( dictionary[db[cross_val][-1]])
       else:
          helper = []
          for float_value in db[cross_val][:-1]:
             helper.append(float(float_value))
          X.append(helper)
          Y.append(dictionary[db[cross_val][-1]])
             

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here


    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([x_test])[0]



    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != y_test:
       counter += 1

#Print the error rate
#--> add your Python code here
error_rate = counter /  len(db)
print(f"Error Rate for the -knn.py- : ", error_rate)
