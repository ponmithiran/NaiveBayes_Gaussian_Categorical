# Testing

import numpy as np
from numpy.core.fromnumeric import ravel
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # for encoding categorical features from strings to number arrays
from sklearn.model_selection import cross_validate
# import matplotlib.pyplot as plt


# Differnt types of Naive Bayes Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB

# Prepare Training Data
data=pd.read_csv('adult_data.csv')
data['age'] = data['age'].replace(' ?',data['age'].mean())
data['fnlwgt'] = data['fnlwgt'].replace(' ?',data['fnlwgt'].mean())
data['capital-loss-abs']=-1*data['capital-loss']
data['capital_gain_total']= data['capital-gain']+data['capital-loss-abs']
data['capital_gain_total'] = data['capital_gain_total'].replace(' ?',data['capital_gain_total'].mean())

data.loc[data['capital_gain_total']==0,'Invest']="No"
data.loc[data['Invest']!="No",'Invest']="Yes"


# Prepare Test Data
data_test=pd.read_csv('adult_data_test.csv')
data_test['age'] = data_test['age'].replace(' ?',data_test['age'].mean())
data_test['fnlwgt'] = data_test['fnlwgt'].replace(' ?',data_test['fnlwgt'].mean())
data_test['capital-loss-abs']=-1*data_test['capital-loss']
data_test['capital_gain_total']= data_test['capital-gain']+data_test['capital-loss-abs']
data_test['capital_gain_total'] = data_test['capital_gain_total'].replace(' ?',data_test['capital_gain_total'].mean())

data_test.loc[data_test['capital_gain_total']==0,'Invest']="No"
data_test.loc[data_test['Invest']!="No",'Invest']="Yes"

# Select data for modeling

X_G=data[['age', 'fnlwgt','capital_gain_total']] # Gaussian, i.e. continuous
X_C=data[['workclass','education','education-num','marital-status','occupation','relationship','race','sex','native-country','Invest']] # Categorical, i.e. discrete
y_train=data.income

# Encode categorical variables
enc = OrdinalEncoder()
X_C = enc.fit_transform(X_C)

# Combine all four variables into one array
X_train=np.c_[X_G, X_C[:,0].ravel(), X_C[:,1].ravel(),X_C[:,2].ravel(), X_C[:,3].ravel(),X_C[:,4].ravel(), X_C[:,5].ravel(),X_C[:,6].ravel(), X_C[:,7].ravel(),X_C[:,8].ravel(),X_C[:,9].ravel()]



# ----- Fit the two models -----
# Now use the Gaussian model for continuous independent variable and 
model_G = GaussianNB()
clf_G = model_G.fit(X_train[:,0:3], y_train)
# Categorical model for discrete independent variable
model_C = CategoricalNB()
clf_C = model_C.fit(X_train[:,3:13], y_train)


#Select Test Data

X_G_2=data_test[['age', 'fnlwgt','capital_gain_total']] # Gaussian, i.e. continuous
X_C_2=data_test[['workclass','education','education-num','marital-status','occupation','relationship','race','sex','native-country','Invest']] # Categorical, i.e. discrete
y_test=data_test.income

X_C_2 = enc.fit_transform(X_C_2)
X_test=np.c_[X_G_2, X_C_2[:,0].ravel(), X_C_2[:,1].ravel(),X_C_2[:,2].ravel(), X_C_2[:,3].ravel(),X_C_2[:,4].ravel(), X_C_2[:,5].ravel(),X_C_2[:,6].ravel(), X_C_2[:,7].ravel(),X_C_2[:,8].ravel(),X_C_2[:,9].ravel()]

# ----- Get probability predictions from each model -----
# On training data
G_train_probas = model_G.predict_proba(X_train[:,0:3])
C_train_probas = model_C.predict_proba(X_train[:,3:13])
# And on testing data
G_test_probas = model_G.predict_proba(X_test[:,0:3])
C_test_probas = model_C.predict_proba(X_test[:,3:13])
# Combine probability prediction for class=1 from both models into a 2D array
X_new_train = np.c_[(G_train_probas[:,0], C_train_probas[:,0])] # Train
X_new_test = np.c_[(G_test_probas[:,0], C_test_probas[:,0])] # Test

# ----- Fit Gaussian model on the X_new -----
model = GaussianNB()
cv_results = cross_validate(model, X_new_train, y_train, cv=5)
sorted(cv_results.keys())
print('CV Test Score',cv_results['test_score'])
print('CV Fit Time',cv_results['fit_time'],'\n')

clf = model.fit(X_new_train, y_train)

# Predict class labels on a test data
pred_labels = model.predict(X_new_test)
#np.savetxt("predicted.csv", pred_labels, delimiter=",")

# ----- Print results -----
print('Classes: ', clf.classes_) # class labels known to the classifier
print('Class Priors: ',clf.class_prior_) # probability of each class.
# Use score method to get accuracy of model
print('--------------------------------------------------------')
score = model.score(X_new_test, y_test)
# score_G=model_G.score(X_test[:,0:3], y_test)
# score_C=model_C.score(X_test[:,3:13], y_test)
print('Accuracy Score (overall): ', score)
print('--------------------------------------------------------')
# Look at classification report to evaluate the model
print(classification_report(y_test, pred_labels))



