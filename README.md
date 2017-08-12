# NaiveBayes-for-emailSpamTest
Use naive-bayes algorithm for testing spam emails.

## In spamTest.py:  
### createTrainMat():  
Load train data and calculate train Matrix for further training process.  
Parameter "rate" is used to control the percentage of the data to load. rate = 1 means all given data loaded.  
### trainModel():  
Train to get the naive bayes model.  
### spamTest():
Load test data and classify each email letter. Return the accuracy as well as other statistics.  
