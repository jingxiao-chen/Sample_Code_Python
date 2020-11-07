#######################
###  Module Loading  ##
#######################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import sklearn
from sklearn.datasets import load_wine
from sklearn.datasets import load_diabetes
import statistics

#######################
###  Data Loading   ###
#######################
wine = load_wine()
diabetes = load_diabetes()

###########################
###  Data Exploration   ###
###########################
print("Keys of wine dataset: {}".format(wine.keys()))
print("Target names: {}".format(wine['target_names']))
print("Feature names: {}".format(wine['feature_names']))
print("Type of data: {}".format(type(wine['data'])))
print("Shape of data: {}".format(wine['data'].shape))
print("Type of target: {}".format(type(wine['target'])))
print("Shape of target: {}".format(wine['target'].shape))
print("Target:\n{}".format(wine['target']))
wine_dataframe = pd.DataFrame(wine.data, columns=wine.feature_names)
pd.plotting.scatter_matrix(wine_dataframe, c=wine['target'], figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=10, alpha=.8)
plt.savefig("./Homework1/scattermatrix_wine.png")

print("Keys of diabetes dataset: {}".format(diabetes.keys()))
print("Feature names: {}".format(diabetes['feature_names']))
print("Type of data: {}".format(type(diabetes['data'])))
print("Shape of data: {}".format(diabetes['data'].shape))
print("Type of target: {}".format(type(diabetes['target'])))
print("Shape of target: {}".format(diabetes['target'].shape))
print("Target:\n{}".format(diabetes['target']))
diabetes_dataframe = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
pd.plotting.scatter_matrix(diabetes_dataframe, c=diabetes['target'], figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=10, alpha=.8)
plt.savefig("./Homework1/scattermatrix_diabetes.png")
statistics.mean(diabetes['target'])
statistics.stdev(diabetes['target'])
#######################
# Split train and test
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(wine.data, wine.target,
                                                                        stratify=wine.target, random_state=1)
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = \
    train_test_split(diabetes.data, diabetes.target, random_state=2)

min_on_training = X_train_wine.min(axis=0)
range_on_training = (X_train_wine - min_on_training).max(axis=0)
X_train_wine_scaled = (X_train_wine - min_on_training) / range_on_training
X_test_wine_scaled = (X_test_wine - min_on_training) / range_on_training

min_on_training = X_train_diabetes.min(axis=0)
range_on_training = (X_train_diabetes - min_on_training).max(axis=0)
X_train_diabetes_scaled = (X_train_diabetes - min_on_training) / range_on_training
X_test_diabetes_scaled = (X_test_diabetes - min_on_training) / range_on_training
#######################
###    Methods      ###
#######################

#####################
# k-Nearest Neighbors

# k-neighbors classification

# try n_neighbors from 1 to 10
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    # fit to the model
    clf.fit(X_train_wine, y_train_wine)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train_wine, y_train_wine))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test_wine, y_test_wine))
# plot training vs. testing
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.title("K-Nearest Neighbor (Wine)")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig("./Homework1/KNN_wine.png")

# k-neighbors regression
training_accuracy = []
test_accuracy = []
for n_neighbors in neighbors_settings:
    # build the model
    reg = sklearn.neighbors.reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    # fit to the model
    reg.fit(X_train_diabetes, y_train_diabetes)
    # record training set accuracy
    training_accuracy.append(reg.score(X_train_diabetes, y_train_diabetes))
    # record generalization accuracy
    test_accuracy.append(reg.score(X_test_diabetes, y_test_diabetes))
# plot training vs. testing
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.title("K-Nearest Neighbor (Diabetes)")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig("./Homework/KNN_diabetes.png")

###############
# Linear Models

#####################
### Linear Regression

lr = LinearRegression().fit(X_train_diabetes, y_train_diabetes)
print("feature_names: {}".format(diabetes['feature_names']))
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X_train_diabetes, y_train_diabetes)))
print("Test set score: {:.2f}".format(lr.score(X_test_diabetes, y_test_diabetes)))

#####################
### Ridge Regression

ridge01 = Ridge(alpha=0.1).fit(X_train_diabetes, y_train_diabetes)
print("Training set score: {:.2f}".format(ridge01.score(X_train_diabetes, y_train_diabetes)))
print("Test set score: {:.2f}".format(ridge01.score(X_test_diabetes, y_test_diabetes)))

ridge1 = Ridge(alpha=1).fit(X_train_diabetes, y_train_diabetes)  # default alpha=1.0
print("Training set score: {:.2f}".format(ridge1.score(X_train_diabetes, y_train_diabetes)))
print("Test set score: {:.2f}".format(ridge1.score(X_test_diabetes, y_test_diabetes)))

ridge5 = Ridge(alpha=5).fit(X_train_diabetes, y_train_diabetes)
print("Training set score: {:.2f}".format(ridge5.score(X_train_diabetes, y_train_diabetes)))
print("Test set score: {:.2f}".format(ridge5.score(X_test_diabetes, y_test_diabetes)))

plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(ridge1.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge5.coef_, '^', label="Ridge alpha=5")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-1000, 1000)
plt.legend()
plt.savefig("./Homework1/Ridge_alphas_diabetes.png")

training_accuracy = []
test_accuracy = []
alpha_setting = range(0, 5)
for alpha in alpha_setting:
    # build the model
    ridge = Ridge(alpha=alpha).fit(X_train_diabetes, y_train_diabetes)
    # fit to the model
    ridge.fit(X_train_diabetes, y_train_diabetes)
    # record training set accuracy
    training_accuracy.append(ridge.score(X_train_diabetes, y_train_diabetes))
    # record generalization accuracy
    test_accuracy.append(ridge.score(X_test_diabetes, y_test_diabetes))
# plot training vs. testing
plt.plot(alpha_setting, training_accuracy, label="training accuracy")
plt.plot(alpha_setting, test_accuracy, label="test accuracy")
plt.title("Ridge Regression (Diabetes)")
plt.ylabel("Accuracy")
plt.xlabel("Alpha")
plt.legend()
plt.savefig("./Homework1/Ridge_diabetes.png")

#####################
### Lasso

lasso00001 = Lasso(alpha=0.0001).fit(X_train_diabetes, y_train_diabetes)
print("Training set score: {:.2f}".format(lasso00001.score(X_train_diabetes, y_train_diabetes)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test_diabetes, y_test_diabetes)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

lasso01 = Lasso(alpha=0.1).fit(X_train_diabetes, y_train_diabetes)  # default alpha=1.0
print("Training set score: {:.2f}".format(lasso01.score(X_train_diabetes, y_train_diabetes)))
print("Test set score: {:.2f}".format(lasso01.score(X_test_diabetes, y_test_diabetes)))
print("Number of features used: {}".format(np.sum(lasso01.coef_ != 0)))

lasso1 = Lasso(alpha=1).fit(X_train_diabetes, y_train_diabetes)
print("Training set score: {:.2f}".format(lasso1.score(X_train_diabetes, y_train_diabetes)))
print("Test set score: {:.2f}".format(lasso1.score(X_test_diabetes, y_test_diabetes)))
print("Number of features used: {}".format(np.sum(lasso1.coef_ != 0)))

plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(lasso01.coef_, 's', label="Lasso alpha=0.1")
plt.plot(lasso1.coef_, '^', label="Lasso alpha=1")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-1000, 1000)
plt.legend()
plt.savefig("./Homework1/Lasso_alphas_diabetes.png")

training_accuracy = []
test_accuracy = []
alpha_setting = [0.001, 0.1, 1, 2, 5]
for alpha in alpha_setting:
    # build the model
    lasso = Lasso(alpha=alpha).fit(X_train_diabetes, y_train_diabetes)
    # fit to the model
    lasso.fit(X_train_diabetes, y_train_diabetes)
    # record training set accuracy
    training_accuracy.append(lasso.score(X_train_diabetes, y_train_diabetes))
    # record generalization accuracy
    test_accuracy.append(lasso.score(X_test_diabetes, y_test_diabetes))
# plot training vs. testing
plt.plot(alpha_setting, training_accuracy, label="training accuracy")
plt.plot(alpha_setting, test_accuracy, label="test accuracy")
plt.title("Lasso Regression (Diabetes)")
plt.ylabel("Accuracy")
plt.xlabel("Alpha")
plt.legend()
plt.savefig("./Homework1/Lasso_diabetes.png")

#######################
### Logistic Regression

training_accuracy = []
test_accuracy = []
C_setting = [0.001, 0.1, 1, 2, 5]
for C in C_setting:
    # build the model
    logistic = LogisticRegression(C=C).fit(X_train_wine, y_train_wine)
    # fit to the model
    logistic.fit(X_train_wine, y_train_wine)
    # record training set accuracy
    training_accuracy.append(logistic.score(X_train_wine, y_train_wine))
    # record generalization accuracy
    test_accuracy.append(logistic.score(X_test_wine, y_test_wine))
# plot training vs. testing
plt.plot(C_setting, training_accuracy, label="training accuracy")
plt.plot(C_setting, test_accuracy, label="test accuracy")
plt.title("Logistic Regression (Wine)")
plt.ylabel("Accuracy")
plt.xlabel("Alpha")
plt.legend()
plt.savefig("./Homework1/Logistic_wine.png")

####################
# Tree-based Methods

####################
### Decision Trees

tree = DecisionTreeClassifier(random_state=2)
tree.fit(X_train_wine, y_train_wine)
print("Training set score: {:.2f}".format(tree.score(X_train_wine, y_train_wine)))
print("Test set score: {:.2f}".format(tree.score(X_test_wine, y_test_wine)))

tree = DecisionTreeRegressor(random_state=5)
tree.fit(X_train_diabetes, y_train_diabetes)
print("Training set score: {:.2f}".format(tree.score(X_train_diabetes, y_train_diabetes)))
print("Test set score: {:.2f}".format(tree.score(X_test_diabetes, y_test_diabetes)))

###############################
### Ensembles of Decision Trees

###############################
##### Random Forest

forest = RandomForestClassifier(n_estimators=10, random_state=2)
# n_estimators: number of trees
# max_features=“auto” (default) = sqrt(n_features)
forest.fit(X_train_wine, y_train_wine)
print("Training set score: {:.2f}".format(forest.score(X_train_wine, y_train_wine)))
print("Test set score: {:.2f}".format(forest.score(X_test_wine, y_test_wine)))

def plot_feature_importances_cancer(model):
    n_features = wine.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), wine.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(forest)
plt.savefig("./Homework1/forest_importance_wine.png")

forest = RandomForestRegressor(n_estimators=10, random_state=2)
# n_estimators: number of trees
# max_features=“auto” (default) = sqrt(n_features)
forest.fit(X_train_diabetes, y_train_diabetes)
print("Training set score: {:.2f}".format(forest.score(X_train_diabetes, y_train_diabetes)))
print("Test set score: {:.2f}".format(forest.score(X_test_diabetes, y_test_diabetes)))

###############################
##### Gradient boosted trees

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01, max_depth=1)
gbrt.fit(X_train_wine, y_train_wine)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train_wine, y_train_wine)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test_wine, y_test_wine)))

plot_feature_importances_cancer(gbrt)
plt.savefig("./Homework1/gbrt_importance_wine.png")

gbrt = GradientBoostingRegressor(random_state=0, learning_rate=0.01, max_depth=1)
gbrt.fit(X_train_diabetes, y_train_diabetes)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train_diabetes, y_train_diabetes)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test_diabetes, y_test_diabetes)))
####################
# Support Vector Machines

##############
### Linear SVM

linear_svm = LinearSVC().fit(X_train_wine_scaled, y_train_wine)
print("Training set score: {:.2f}".format(linear_svm.score(X_train_wine_scaled, y_train_wine)))
print("Test set score: {:.2f}".format(linear_svm.score(X_test_wine_scaled, y_test_wine)))

linear_svm = LinearSVR().fit(X_train_diabetes_scaled, y_train_diabetes)
print("Training set score: {:.2f}".format(linear_svm.score(X_train_diabetes_scaled, y_train_diabetes)))
print("Test set score: {:.2f}".format(linear_svm.score(X_test_diabetes_scaled, y_test_diabetes)))

##################
### Kernelized SVM

svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X_train_wine_scaled, y_train_wine)
print("Training set score: {:.2f}".format(svm.score(X_train_wine_scaled, y_train_wine)))
print("Test set score: {:.2f}".format(svm.score(X_test_wine_scaled, y_test_wine)))

svm = SVR(kernel='rbf', C=10, gamma=0.1).fit(X_train_diabetes_scaled, y_train_diabetes)
print("Training set score: {:.2f}".format(linear_svm.score(X_train_diabetes_scaled, y_train_diabetes)))
print("Test set score: {:.2f}".format(linear_svm.score(X_test_diabetes_scaled, y_test_diabetes)))

####################
# Neural Networks

mlp = MLPClassifier(max_iter=1000, alpha=1, hidden_layer_sizes=[10, 10], random_state=0)
mlp.fit(X_train_wine_scaled, y_train_wine)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_wine_scaled, y_train_wine)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_wine_scaled, y_test_wine)))

mlp = MLPRegressor(max_iter=1000, alpha=1, hidden_layer_sizes=[5, 5], random_state=0)
mlp.fit(X_train_diabetes_scaled, y_train_diabetes)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_diabetes_scaled, y_train_diabetes)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_diabetes_scaled, y_test_diabetes)))
