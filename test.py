import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
database = "iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(database, names=names)

#print the size of the data
#print(dataset.shape)

# Print first 30 data
#print(dataset.head(30))

print(dataset.describe())

print(dataset.groupby('class').size())
#for data in dataset:
#    print(data)

"""
Different types of plotting
1: Uinvariate plot
   A univariate plot shows the data and summarizes its distribution.
   A dot plot, also known as a strip plot, shows the individual observations.
   A box plot shows the five-number summary of the data – the minimum, 
   first quartile, median, third quartile, and maximum.
2: Multivariate plot
   Multivariate descriptive displays or plots are designed to reveal 
   the relationship among several variables simulataneously.. As was 
   the case when examining relationships among pairs of variables, there 
   are several basic characteristics of the relationship among sets of 
   variables that are of interest.
"""
# Univariate plot
dataset.plot(kind = 'box', subplots = True, 
             layout = (2,2), sharex = False, sharey = False)
plt.show()

# Plotting Histogram
dataset.hist()
plt.show()

# Multivariate plot
scatter_matrix(dataset)
plt.show()

array = dataset.values
X = array[:, 0:4]
#print("x", type(X))
Y = array[:, 4]
#print("y", type(Y))
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
                                                                    test_size = validation_size, 
                                                                    random_state = seed)

seed = 6
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each  model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, 
                                                 Y_train, cv = kfold,
                                                 scoring = scoring)
    results.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


