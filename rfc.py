#Load dataset as dataframe
import pandas as pd
d = pd.read_csv('dataset/graduate_admissions.csv')
len(d)
d.head()


#Convert the Chance of Admit col into binary with 1 - admit and 0 - Reject
d['admit'] = d.apply(lambda row: 1 if (row['Chance of Admit ']) >= 0.75 else 0, axis=1)

#Drop Unwanted columns
d = d.drop(['Serial No.'], axis=1)

#Drop Chance of Admit as we have already benefited from it 
d = d.drop(['Chance of Admit '], axis=1)


#Randomly shuffle all the rows for random sampling.
#Shuffling data before learning has many benefits though it is not necessary
d = d.sample(frac = 1)


#Split into training and testing data 
# 80/20 : train/test
d_train = d[:400]
d_test = d[400:]

#remove the admit col and save separately for cross validation from all of the datasets
d_train_att = d_train.drop(['admit'], axis=1)
d_train_admit = d_train['admit']

d_test_att = d_test.drop(['admit'], axis=1)
d_test_admit = d_test['admit']

d_att = d.drop(['admit'], axis=1)
d_admit = d['admit']


#Create Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Find the best number of features and number of trees required in Random Forest Classifier by looping through the parameters and immediately cross validating 
max_features_opts = range(1, 8)
n_estimators_opts = range(2, 20)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, d_train_att, d_train_admit, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" %(max_features, n_estimators, scores.mean(), scores.std() * 2))
        
# We find the best accuracy with 89%
print("Max features: 3, num estimators: 8, accuracy: 0.89 (+/- 0.06)")

#Regerate the tree with the best parameters
clf = RandomForestClassifier(max_features=3 , n_estimators=8)
clf.fit(d_train_att, d_train_admit)

#Create a confusion matrix 
from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(d_test_att)
cm = confusion_matrix(d_test_admit, pred_labels)

# Perform a prediction on the top 5 Training samples
print(clf.predict(d_train_att.head()))
