import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score, precision_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# non-GUI backend, cannot show the figure, but save it to the specified path  (!)
import matplotlib; matplotlib.use('Agg')        # if want GUI backend, block this code -----------------< step 1


project_path = "F:/python_demo/mouse blood loss/"
save_path = "F:/python_demo/mouse blood loss/"
num_test_size = 0.25
num_random_state = 10



def check_data(Data):
    # --------------------------- Check data availability
    # print data dimensions
    print("DataSet shape：", Data.shape)

    # print the first 5 rows & the last 5 rows of the dataset
    print(Data.head(3))
    print(Data.tail(3))

    # print a concise summary of a Data Frame
    Data.info()

    # compute a summary of statistics pertaining to the Data Frame columns
    Data.describe()

    # checking for missing values
    print('Missing Values of DataSet:\n', Data.isnull().sum())



from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score

def class_score(y_true, y_pred):
    ## Calculate precision score
    print("macro_precision: %.2f" % precision_score(y_true, y_pred, average='macro'))
    print("micro_precision: %.2f" % precision_score(y_true, y_pred, average='micro'))

    ## Calculate reacall
    print("macro_recall: %.2f" % recall_score(y_true, y_pred, average='macro'))
    print("micro_recall: %.2f" % recall_score(y_true, y_pred, average='micro'))

    ## Calculate F1
    print("macro_f1: %.2f" % f1_score(y_true, y_pred, average='macro'))
    print("micro_f1: %.2f" % f1_score(y_true, y_pred, average='micro'))

def drow_score(score, i):
    score = np.array(score)
    plt.figure(figsize=(15, 10))
    sns.barplot(score, model_name)
    plt.title("Models' Comparison: %s" % classifier_name[i])
    plt.tight_layout(rect=(0.01, 0.01, 1, 1))
    plt.savefig("%sModel_Comparison_%s.png" % (save_path, classifier_name[i]))
    plt.show()

def Radarplot(data):
    labels = model_name
    data_length = len(model_name)
    angles = np.linspace(0, 2 * np.pi, data_length, endpoint=False)
    # turn RadarData off
    data1 = data[0];    data2 = data[1];    data3 = data[2]
    data1 = np.concatenate((data1, [data1[0]]))
    data2 = np.concatenate((data2, [data2[0]]))
    data3 = np.concatenate((data3, [data3[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    labels = np.concatenate((labels, [labels[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, data1, 'o-', linewidth=1, label='%s' % classifier_name[0])  # ('.-')
    ax.fill(angles, data1, alpha=0.1)       #np.concatenate((score[i], [score[i][0]]))
    ax.plot(angles, data2, 's-', linewidth=1, label='%s' % classifier_name[1])
    ax.fill(angles, data2, alpha=0.1)
    ax.plot(angles, data3, 's-', linewidth=1, label='%s' % classifier_name[2])
    ax.fill(angles, data3, alpha=0.1)
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    # plt.title("Model Comparison", pad=20)
    ax.set_ylim(0.3, 1.01)
    ax.grid(True)
    plt.legend(loc='best')
    plt.tight_layout(rect=(0.01, 0.01, 1, 1))
    plt.savefig("%sModel_Comparison_Radarplot.png" % save_path)
    plt.show()


#======================================== 3 Multiclass Model Algorithms Set==========================================
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier


### Create a Classifier List (includes 3 different multi-label / Multiclass Algorithms)
classifiers = [OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier]
classifier_name = np.array(["OneVsRestClassifier", "OneVsOneClassifier", "OutputCodeClassifier"])


# ================================================== 8 Classification Models ==============================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC; from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

KNN = KNeighborsClassifier()            # 1. K-Nearest neighbors
Dtree = tree.DecisionTreeClassifier()   # 2. Decision Trees
NB = GaussianNB()                       # 3. Naive Bayesian
LR = LogisticRegression()               # 4.Logistic Regression
SVM = CalibratedClassifierCV(SVC(gamma='auto', probability=True))   # 5.Support Vector Machine
RF = RandomForestClassifier(n_estimators=100)   # 6. Random Forest
NN = MLPClassifier()                    # 7. NN - Multi-layer Perceptron(MLP)
SGD = SGDClassifier(loss="log")         # 8.Stochastic Gradient Descent


#----- Create a Learner list (8 models)----------
models = [LR, KNN, Dtree, SVM, NB, RF, NN, SGD]
model_name = np.array(['Logistic Regression', 'K-Nearest Neighbor', 'Decision Tree', 'Support Vector Machine',
                       'Naive Bayes', 'Random Forest', 'Neural Networks', 'SGDClassifier'])


# ===================================================== main =========================================================

myData = pd.read_csv("%sFeatures_all.csv" % project_path)   #Features_all = Features 1+2+3
check_data(myData)

myData = pd.get_dummies(myData, drop_first=True)

X = myData.drop('keys', axis=1);    y = myData['keys']  # keys=(1,2,3) as label
X = (X - np.min(X)) / (np.max(X) - np.min(X))

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=num_test_size, random_state=num_random_state)

# Training & Testing 8 Machine Learning Models under Three Three-Label Classifiers Using Loop Nesting
classifiers_score = []
for i in range(len(classifiers)):
    print("\n(*)%s :" % classifier_name[i])
    score = []

    # polt 3 heat maps of confusion_matrix  --- #(for case 2) head of the confusion matrix
    class_names = [1, 2, 3]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    for j in range(len(models)):
        print("\n%s :" % model_name[j])
        #--------------run only one of the three cases-----------------

        ### (case 1)  only train set (not use train_test_split)
        # model = classifiers[i](models[j]).fit(X, y)
        # y_pred = model.predict(X)
        # score = accuracy_score(y, y_pred); print("accuracy: %.3f" % score)
        # class_score(y, y_pred)
        # score.append(score)


    #     ### (case 2)  split data into train & test set
    #     model = classifiers[i](models[j]).fit(train_x, train_y)
    #     pred1 = model.predict(train_x)
    #     score_train = accuracy_score(train_y, pred1)
    #     print('accuracy on TrainSet:: %.2f' % score_train)
    #     pred2 = model.predict(test_x)
    #     score_test = accuracy_score(test_y, pred2)
    #     print('accuracy on TestSet: %.2f' % score_test)
    #     score.append(score_test)
    #
    #     # confusion matrix
    #     conf_matrix = confusion_matrix(pred2, test_y)
    #     print("confusion_matrix:\n", conf_matrix)
    #     print(classification_report(test_y, pred2, labels=[1, 2, 3], target_names=['experiment 1', 'experiment 2', 'experiment 3']))
    #
    #     # create a heat map of confusion_matrix         # belly
    #     plt.subplot(3, 3, j+1)
    #     sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    #     ax.xaxis.set_label_position('top')
    #     plt.tight_layout()
    #     plt.title(model_name[j], y=1)
    #     plt.ylabel('Actual label')
    #     plt.xlabel('Predicted label')
    #
    # # plt.tight_layout(rect=(0.01, 0.01, 1, 1))         # tail
    # plt.savefig("%sconfusion_matrix%s.png" % (save_path, classifier_name[i]))
    # plt.show()


        ## (case 3)  split data into train & test set (+) Add cross validation
        model = classifiers[i](models[j]).fit(train_x, train_y)
        # Output the mean and confidence interval of the cross_val_score on training & test set
        scores_train = cross_val_score(model, train_x, train_y, cv=3, scoring='accuracy')
        print("accuracy on TrainSet: %0.2f (+/- %0.2f)" % (scores_train.mean(), scores_train.std() * 2))
        scores_test = cross_val_score(model, test_x, test_y, cv=3, scoring='accuracy')
        print("accuracy on the TestSet: %0.2f (+/- %0.2f)" % (scores_test.mean(), scores_test.std() * 2))
        score.append(scores_test.mean())


    # ---------------------------case over line------------------------------
    drow_score(score, i)
    classifiers_score.append(score)
Radarplot(classifiers_score)



