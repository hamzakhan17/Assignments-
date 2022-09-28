

import pandas as pd
import numpy as np
import matplotlib.pyplot

fraud = pd.read_csv("Fraud_check(1).csv")

##Converting the Taxable income variable to bucketing. 
fraud["income"]="<=30000"
fraud.loc[fraud["Taxable.Income"]>=30000,"income"]="Good"
fraud.loc[fraud["Taxable.Income"]<=30000,"income"]="Risky"

##Droping the Taxable income variable
fraud.drop(["Taxable.Income"],axis=1,inplace=True)

fraud.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)
## As we are getting error as "ValueError: could not convert string to float: 'YES'".
## Model.fit doesnt not consider String. So, we encode

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in fraud.columns:
    if fraud[column_name].dtype == object:
        fraud[column_name] = le.fit_transform(fraud[column_name])
    else:
        pass
  
##Splitting the data into featuers and labels
features = fraud.iloc[:,0:5]
labels = fraud.iloc[:,5]

## Collecting the column names
colnames = list(fraud.columns)
predictors = colnames[0:5]
target = colnames[5]
##Splitting the data into train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)

##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)

model.estimators_
model.classes_
model.n_features_
model.n_classes_

model.n_outputs_

model.oob_score_
###74.7833%

##Predictions on train data
prediction = model.predict(x_train)

##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
##98.33%

np.mean(prediction == y_train)
##98.33%

##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)

##Prediction on test data
pred_test = model.predict(x_test)

##Accuracy
acc_test =accuracy_score(y_test,pred_test)
##78.333%

## In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO

tree = model.estimators_[5]

dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

## Creating pdf and png file the selected decision tree
graph.write_pdf('fraudrf.pdf')
graph.write_png('fraudrf.png')
