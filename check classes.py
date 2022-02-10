import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 0, 0])

clf = SVC(gamma='auto', probability=True)

clf.fit(X,y)
print(clf.predict([[-0.8, -1]]))


rf = RandomForestClassifier()
rf.fit(X,y)
print(rf.predict([[-0.8, -1]]))

learner_list = [clf, rf]

known_classes = tuple(learner.classes_ for learner in learner_list)

classes_ = np.unique(
    np.concatenate(known_classes, axis=0),
    axis=0
)
n_classes_ = len(classes_)

def check_labels(*arg):


    try:
        classes_ = [estimator.classes_ for estimator in arg]
        print(classes_)
    except:
        print('Wrong)')

    for classifier_idx in range(len(arg) -1):
        if not np.array_equal(classes_[classifier_idx], classes_[classifier_idx+1]):
            print('False')
        else:
            print('True')
check_labels(*learner_list)

