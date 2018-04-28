#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier




### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock',
                 'shared_receipt_with_poi', 'restricted_stock_deferred',
                 'total_stock_value', 'expenses', 'loan_advances',
                 'from_messages', 'from_this_person_to_poi', 'director_fees',
                 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#######################################################################
#Data overview
#######################################################################

#Identifying the size of the dataset and number of features
print "Size of dataset:",len(data_dict)
print "Number of features in the dictionary:",len(data_dict[data_dict.keys()[0]])

#Number of POIs (Persons of Interest) in the dataset
poi = 0
for user in data_dict:
    if data_dict[user]["poi"] == True:
        poi += 1
print "Number of POIs in the dictionary:" ,poi


#Number of missing features in the data set
user_keys = data_dict.keys()
features_keys = data_dict[data_dict.keys()[0]]
missing_features = {}
missing_values = 0

for feature in features_keys:
    missing_features[feature] = 0
for user in user_keys:
    for feature in features_keys:
        if data_dict[user][feature] == "NaN":
            missing_values += 1
            missing_features[feature] += 1
print "There are ", missing_values, "missing features in the data set"
print missing_features

#Removing NaN values

for user in user_keys:
    for feature in features_keys:
        if data_dict[user][feature] == "NaN":
            data_dict[user][feature] == 0


### Task 2: Remove outliers

#Scatterplot bonus vs salary
features = ["salary", "bonus"]
for value in ["TOTAL","THE TRAVEL AGENCY IN THE PARK"]:
    data_dict.pop(value, 0) #Removes outliers
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary/1000000, bonus/1000000)

plt.xlabel("salary (M $)")
plt.ylabel("bonus (M $)")
plt.title("Salary vs. Bonus")
plt.savefig('SalaryVsBonus.pdf')

#######################################################################
#Features engineering and selrction
#######################################################################

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#Function to calculate raport between to/from POI messages and total to/from messages
def scale_feature (poi_message, denominator):
    scale = 0
    if poi_message != "NaN" and denominator != "NaN":
        scale = poi_message / float(denominator)

    return scale

#Feature Engineering
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = scale_feature( from_poi_to_this_person, to_messages )
    data_dict[name]["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = scale_feature( from_this_person_to_poi, from_messages )
    data_dict[name]["fraction_to_poi"] = fraction_to_poi

features_list.append("fraction_from_poi")
features_list.append("fraction_to_poi")
my_dataset = data_dict

print features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
X = np.array(features)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=25,
                              random_state=0)

forest.fit(X, labels)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

selected_features = []
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, features_list[indices[f]], importances[indices[f]]))
    selected_features.append(features_list[indices[f]])

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.savefig('Features.pdf')

#Select top 10 features
selected_features = selected_features[:10]



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Selected feature list starting with poi
select_features = ['poi', 'total_payments', 'restricted_stock_deferred', 'fraction_from_poi',
                   'director_fees', 'total_stock_value', 'deferral_payments', 'exercised_stock_options',
                   'deferred_income', 'bonus', 'salary', 'expenses', 'from_poi_to_this_person', 'fraction_to_poi']


#######################################################################
#Testing classifiers and selecting the final one
#######################################################################

### Extract selected features and labels from dataset
from sklearn.cross_validation import train_test_split

data = featureFormat(my_dataset, select_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Spliting the dataset into test and training data
features_train, features_test, \
labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.5, random_state=42)



# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import accuracy_score

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred = clf.predict (features_test)
#accuracy = accuracy_score(pred, labels_test)
#print "Accuracy GaussianNB:", accuracy


#from sklearn import tree
#clf = tree.DecisionTreeRegressor(min_samples_split=12)
#clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)
#accuracy = accuracy_score(pred, labels_test)
#print "Accuracy DT:", accuracy

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()
#clf = clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#accuracy = accuracy_score(pred, labels_test)
#print "Accuracy RF:", accuracy

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "Accuracy AB:", accuracy

## Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = AdaBoostClassifier(n_estimators=600)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "Accuracy AB:", accuracy

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)