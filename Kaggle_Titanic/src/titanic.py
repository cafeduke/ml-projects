import numpy as np
import pandas as pd
import importlib
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
import pyduke.common.core_util as cu
import pyduke.common.data_util as du
import pyduke.mlutil.data_processor as dp
import re

# _________________________________________________________________________________________________
#
#                                         Data Processing
# _________________________________________________________________________________________________

X             = pd.read_csv(cu.PROJECT_ROOT + '/dataset/train.csv')
X_final_test  = pd.read_csv(cu.PROJECT_ROOT + '/dataset/test.csv')
y             = X.pop('Survived')
X_save = X.copy()

# Handlers
# --------
def convert_name_to_prefix (name):
    # Name format: <lastname> <lastname>, <prefix> <firstname> <middlename>
    # Note <prefix> may have '.' as in 'Mr.' or 'Miss.'
    token = re.sub('\s+', ' ', name).split(',')
    token = token[1].split() if len(token) >= 2 else ['Dear']
    token = token[0].replace('.', '') if len(token) >= 2 else 'Dear'
    return token.upper()

def process_empty_fare(df):
    # Group the entire dataset by column
    grouped = df.groupby('Pclass')    
    series_class_to_fare = grouped['Fare'].agg(np.mean)    
    series_fare = df['Pclass'].apply(lambda x: series_class_to_fare[x])
    return series_fare

def process_sibling_parent (df):
    return df['SibSp'] + df['Parch']   


tuple_age_range = (
    [-1,    3,      12,     19,      39,       59,       79,   120],
    ['INFANT', 'CHILD', 'TEEN', 'YOUTH', 'MIDDLE', 'SENIOR', 'OLD']
)

map_column_category_weight = {
    'Sex':{ 'FEMALE':0, 'MALE':1 },
    'Age':{ 'INFANT':1, 'CHILD':2, 'TEEN':3, 'YOUTH':4, 'MIDDLE':5, 'SENIOR':6, 'OLD':7  }
}

importlib.reload(dp)
X = X_save.copy()
pileline = Pipeline([
    ('rm_column'           , dp.RemoveColumn(column=['PassengerId', 'Cabin', 'Ticket'])),
    ('fill_nan_stats'      , dp.IndependentColumnImputer(column_median=['Age'], column_mode= ['Pclass', 'SibSp', 'Parch', 'Embarked'])),
    ('fill_empty_fare'     , dp.DependentColumnImputer({'Fare':process_empty_fare})),
    ('add_name_prefix'     , dp.Mapper({'Name':convert_name_to_prefix}, map_column_to_new={'Name':'NamePrefix'}, remove_original=True)),
    ('add_family_weight'   , dp.AddColumn({'FamilyTotal':process_sibling_parent})),
    ('age_to_category'     , dp.RangeToCategoryConverter({ 'Age':tuple_age_range })),
    ('string_to_category'  , dp.StringToCategoryConverter(['Sex', 'Embarked', 'NamePrefix'])),
    ('category_to_weight'  , dp.CategoryToWeightEncoder(map_column_category_weight)),
    ('category_to_onehot'  , dp.CategoryToOneHotEncoder(['Embarked', 'NamePrefix'])),
    ('scale'               , dp.Scaler())
])
X = pileline.fit_transform(X)
X.info()

X_final_test  = pd.read_csv(cu.PROJECT_ROOT + '/dataset/test.csv')
X_final_test = pileline.transform(X_final_test)
print ("X.shape={}, X_final_test.shape={}".format(X.shape, X_final_test.shape))

# Shuffle 
X_train, y_train, X_test, y_test = du.get_stratified_shuffle_split(X, y, test_size=141)

# Shapes
m, n = X_train.shape

# _________________________________________________________________________________________________
#
#                                         Random Forest Classifier
# _________________________________________________________________________________________________


# -------------------------------------------------------------------------------------------------
# Base Model
# -------------------------------------------------------------------------------------------------
cu.heading('[RandomForest][BaseModel]')

# Fit
from sklearn.ensemble import RandomForestClassifier
model_forest = RandomForestClassifier(random_state=du.SEED)
model_forest.fit(X_train, y_train)

# Predict Train
y_train_pred = model_forest.predict(X_train)
tuple_accuracy = du.get_scores(y_train, y_train_pred, title='[RandomForest][BaseModel][Train]')

# Train Accuracy = 0.9480
print ("Train Accuracy = {:2.4f}".format(tuple_accuracy[0]))

# Predict Test
y_test_pred = model_forest.predict(X_test)
tuple_accuracy = du.get_scores(y_test, y_test_pred, title='[RandomForest][BaseModel][Test]')

# Test Accuracy = 0.7943
print ("Test Accuracy = {:2.4f}".format(tuple_accuracy[0]))

# -------------------------------------------------------------------------------------------------
# Cross Validation
# -------------------------------------------------------------------------------------------------

##
# Cross validation across multiple folds
# ---------------------------------------
# In each fold
#    - The training set is divided into random Train/Val fold
#    - The model is fit with Train and predicted for Val
#    - The scores provided are for the Val. 
# This should be closer to what you will get when the model is actually tested    
##
from sklearn.model_selection import cross_val_score
model_forest = RandomForestClassifier(random_state=du.SEED)
list_score = cross_val_score(model_forest, X_train, y_train, cv=5, scoring='accuracy')

# Cross Validation Accuracy = 0.8148
print (list_score)
print ("Cross Validation Accuracy = {:2.4f}".format(list_score.mean()))

##
# Grid Search for best hyper parameters
# -------------------------------------
##
from sklearn.model_selection import GridSearchCV
param_grid = [{  
    'n_estimators': [70, 75, 77, 78, 80], 
    'max_depth': [None, 4, 5, 6, 7, 8] 
    }]
hyper_grid_search = GridSearchCV(model_forest, param_grid, cv=5, scoring='accuracy')
hyper_grid_search.fit(X_train, y_train)
result         = hyper_grid_search.cv_results_
best_param     = hyper_grid_search.best_params_
best_accuracy  = hyper_grid_search.best_score_ * 100
best_model     = hyper_grid_search.best_estimator_

##
# Best
# ----
# Best Accuracy  = 83.8667
# Best Param     = { 'max_depth':5, 'n_estimators':75}
##
print ("Best Accuracy  = {:2.4f}".format(best_accuracy))
print ("Best Param     = {}".format(best_param))

# -------------------------------------------------------------------------------------------------
# Train Model with best hyper param 
# -------------------------------------------------------------------------------------------------

# Train with best param
# ---------------------
model_forest = best_model
model_forest = RandomForestClassifier(random_state=42, n_estimators=75, max_depth=5)
model_forest.fit(X_train, y_train)    

# Train Accuracy with updated param
# ---------------------------------
# Train Accuracy = 0.8440
y_train_pred = model_forest.predict(X_train)
tuple_accuracy = du.get_scores(y_train, y_train_pred, title='[RandomForest][UpdatedModel][Train]')
print ("Train Accuracy = {:2.4f}".format(tuple_accuracy[0]))


# CV Accuracy with updated param
# ------------------------------
# CV Accuracy= 0.8387
from sklearn.model_selection import cross_val_score
list_score = cross_val_score(model_forest, X_train, y_train, cv=5, scoring='accuracy')
print (list_score)
print ("CV Accuracy= {:2.4f}".format(list_score.mean()))

# Test Accuracy with updated param
# --------------------------------
# model_forest = RandomForestClassifier(random_state=42, n_estimators=105, max_features=12)
y_test_pred = model_forest.predict(X_test)
tuple_accuracy = du.get_scores(y_test, y_test_pred, title='[RandomForest][UpdatedModel][Test]')

# Test Accuracy = 0.7943
print ("Test Accuracy = {:2.4f}".format(tuple_accuracy[0]))

# -------------------------------------------------------------------------------------------------
# Train Model combining dev test
# -------------------------------------------------------------------------------------------------
model_forest = RandomForestClassifier(random_state=42, n_estimators=105, max_features=12)
model_forest.fit(X, y)

# -------------------------------------------------------------------------------------------------
# Final Test
# -------------------------------------------------------------------------------------------------
df_final_test_orig  = pd.read_csv(cu.PROJECT_ROOT + '/dataset/test.csv')
y_final_test_pred = model_forest.predict(X_final_test)

# Scikit returns numpy after prediction!
# Convert to series
y_final_test_pred = pd.Series(name=y.name, data=y_final_test_pred)
df_final_togo     = pd.concat([df_final_test_orig.PassengerId, y_final_test_pred], axis=1)
df_final_togo.to_csv(cu.PROJECT_ROOT + '/result.csv', index=False)

# _________________________________________________________________________________________________
#
#                                 Support Vector Machine Classifier (SVM)
# _________________________________________________________________________________________________


# -------------------------------------------------------------------------------------------------
# Base Model
# -------------------------------------------------------------------------------------------------
cu.heading('[SVC][BaseModel]')

# Fit
from sklearn.svm import SVC
model_svc = SVC(random_state=du.SEED)
model_svc.fit(X_train, y_train)

# Predict Train
y_train_pred = model_svc.predict(X_train)
tuple_accuracy = du.get_scores(y_train, y_train_pred, title='[SVC][BaseModel][Train]')

# Train Accuracy = 0.8320
print ("Train Accuracy = {:2.4f}".format(tuple_accuracy[0]))

# Predict Test
y_test_pred = model_svc.predict(X_test)
tuple_accuracy = du.get_scores(y_test, y_test_pred, title='[SVC][BaseModel][Test]')

# Test Accuracy = 0.8085
print ("Test Accuracy = {:2.4f}".format(tuple_accuracy[0]))


# -------------------------------------------------------------------------------------------------
# Cross Validation
# -------------------------------------------------------------------------------------------------

##
# Cross validation across multiple folds
# ---------------------------------------
# In each fold
#    - The training set is divided into random Train/Val fold
#    - The model is fit with Train and predicted for Val
#    - The scores provided are for the Val. 
# This should be closer to what you will get when the model is actually tested    
##
from sklearn.model_selection import cross_val_score
model_svc = SVC(random_state=du.SEED, kernel='linear')
list_score = cross_val_score(model_svc, X_train, y_train, cv=5, scoring='accuracy')
print (list_score)

# Cross Validation Accuracy = 0.8107
print ("Cross Validation Accuracy = {:2.4f}".format(list_score.mean()))


##
# Grid Search for best hyper parameters
# -------------------------------------
##
from sklearn.model_selection import GridSearchCV
param_grid = [{  
    'kernel':['linear'],
    'C': [1, 1.25, 1.5, 1.75, 2], 
    'gamma': ['auto', 0.05, 0.01, 0.5, 0.1] 
    },{
    'kernel':['linear'],
    'shrinking' : [False],
    'C': [1, 1.25, 1.5, 1.75, 2], 
    'gamma': ['auto', 0.05, 0.01, 0.5, 0.1] 
    },
    
    ]
hyper_grid_search = GridSearchCV (model_svc, param_grid, cv=5, scoring='neg_mean_squared_error')
hyper_grid_search.fit(X_train, y_train)
result         = hyper_grid_search.cv_results_
best_param     = hyper_grid_search.best_params_
best_accuracy  = 100 - np.sqrt(-1 * hyper_grid_search.best_score_)
best_model     = hyper_grid_search.best_estimator_

# Error and Hyper-parameters that give least error
for score, param in zip(result["mean_test_score"], result["params"]):
    print(np.sqrt(-score), param)

##
# Best
# ----
# Best Accuracy  = 99.5757
# Best Param     = {'C': 2, 'gamma': 'auto', 'kernel': 'linear'}
# Least Error    = 0.4243
##
print ("Best Accuracy  = {:2.4f}".format(best_accuracy))
print ("Best Param     = {}".format(best_param))
print ("Least Error    = {:2.4f}".format(np.sqrt(-1 * hyper_grid_search.best_score_)))

# -------------------------------------------------------------------------------------------------
# Train Model with best hyper param 
# -------------------------------------------------------------------------------------------------

# Train with best param
# ---------------------
# model_svc = best_model
# model_svc = SVC(random_state=42, C=2, gamma='auto', kernel='linear')
model_svc = SVC(random_state=42, C=1, gamma=0.1, kernel='rbf')
model_svc.fit(X_train, y_train)    

# Train Accuracy with updated param
# ---------------------------------
# Train Accuracy = 0.8360
y_train_pred = model_svc.predict(X_train)
tuple_accuracy = du.get_scores(y_train, y_train_pred, title='[SVC][UpdatedModel][Train]')
print ("Train Accuracy = {:2.4f}".format(tuple_accuracy[0]))


# CV Accuracy with updated param
# ------------------------------
# CV Accuracy= 0.8267
from sklearn.model_selection import cross_val_score
list_score = cross_val_score(model_svc, X_train, y_train, cv=5, scoring='accuracy')
print (list_score)
print ("CV Accuracy= {:2.4f}".format(list_score.mean()))

# Test Accuracy with updated param
# --------------------------------
# model_forest = RandomForestClassifier(random_state=42, n_estimators=105, max_features=12)
y_test_pred = model_svc.predict(X_test)
tuple_accuracy = du.get_scores(y_test, y_test_pred, title='[SVC][UpdatedModel][Test]')

# Test Accuracy = 0.7943
print ("Test Accuracy = {:2.4f}".format(tuple_accuracy[0]))

# -------------------------------------------------------------------------------------------------
# Train Model combining dev test
# -------------------------------------------------------------------------------------------------
# model_svc = best_model
model_svc = SVC(random_state=du.SEED, C=2, gamma='auto', kernel='linear')
model_svc.fit(X, y)

# -------------------------------------------------------------------------------------------------
# Final Test
# -------------------------------------------------------------------------------------------------
df_final_test_orig  = pd.read_csv(cu.PROJECT_ROOT + '/dataset/test.csv')
y_final_test_pred = model_forest.predict(X_final_test)

# Scikit returns numpy after prediction!
# Convert to series
y_final_test_pred = pd.Series(name=y.name, data=y_final_test_pred)
df_final_togo     = pd.concat([df_final_test_orig.PassengerId, y_final_test_pred], axis=1)
df_final_togo.to_csv('result_svc.csv', index=False)


# _________________________________________________________________________________________________
#
#                                 Ensemble (RandomForest + SVM)
# _________________________________________________________________________________________________

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

model_forest  = RandomForestClassifier(random_state=42, n_estimators=75, max_depth=5)
model_svc_rbf = SVC(random_state=42, C=1, gamma=0.1, kernel='rbf', probability=True)

model_vote = VotingClassifier(
    estimators=[
        ('forest', model_forest), 
        ('svc_rbf', model_svc_rbf)
    ],
    voting='soft')
model_vote.fit(X_train, y_train)

y_test_pred = model_vote.predict(X_test)

for model in (model_forest, model_svc_rbf, model_vote):
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    model_name = model.__class__.__name__
    du.get_scores(y_test, y_test_pred, title='[{}][UpdatedModel][Test]'.format(model_name))
    
    
# -------------------------------------------------------------------------------------------------
# Train Model combining dev test
# -------------------------------------------------------------------------------------------------
model_vote = RandomForestClassifier(random_state=42, n_estimators=105, max_features=12)
model_vote.fit(X, y)

# -------------------------------------------------------------------------------------------------
# Final Test
# -------------------------------------------------------------------------------------------------
df_final_test_orig  = pd.read_csv(cu.PROJECT_ROOT + '/dataset/test.csv')
y_final_test_pred = model_vote.predict(X_final_test)

# Scikit returns numpy after prediction!
# Convert to series
y_final_test_pred = pd.Series(name=y.name, data=y_final_test_pred)
df_final_togo     = pd.concat([df_final_test_orig.PassengerId, y_final_test_pred], axis=1)
df_final_togo.to_csv(cu.PROJECT_ROOT + '/result.csv', index=False)    
