from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


# Naive Bayes
def nb(X_train_vect, y_train):
    nb = MultinomialNB()
    nb.fit(X_train_vect, y_train)

    return nb


# Random Forest
def rf(X_train_vect, y_train):
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train_vect, y_train)

    return rf


# Logistic Regression
def lr(X_train_vect, y_train):

    log = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=400)
    log.fit(X_train_vect, y_train)

    return log


# Linear Support Vector
def lsvc(X_train_vect, y_train):

    lsvc = LinearSVC(tol=1e-05)
    lsvc.fit(X_train_vect, y_train)

    return lsvc