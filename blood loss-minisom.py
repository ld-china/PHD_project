'''
(*) SOM based on minison --> pip install minisom

The most important steps in the self-organizing map(SOM) learning process:
1.Weight initialization
2.The input vector is selected from the dataset and used as an input for the network
3.calculate BMU(The winning node is commonly known as the Best Matching Unit)
4.The radius of neighbors that will be updated is calculated
5.Each weight of the neurons within the radius are adjusted to make them more like the input vector
6.Steps from 2 to 5 are repeated for each input vector of the dataset
'''

import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from minisom import MiniSom


project_path = "F:/python_demo/mouse blood loss/"
save_path = "F:/python_demo/mouse blood loss/"


def check_data(Data):
    # --------------------------- Check data availability
    # print data dimensions
    print("DataSet shape：", Data.shape)

    # print the first 3 rows & the last 3 rows of the dataset
    # print(Data.head(3),Data.tail(3))
    print(Data.sample(5))   #5 samples are drawn at random

    # print a concise summary of a Data Frame
    Data.info()

    # compute a summary of statistics pertaining to the Data Frame columns
    Data.describe()

    # -checking for missing values
    print('Missing Values of DataSet:\n', Data.isnull().sum())

def classify(som, data):
    """Classifies each sample in data in one of the classes definited using the method labels_map.
    Returns a list of the same length of data where the i-th element is the class assigned to data[i]."""
    winmap = som.labels_map(X, y)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

# load data
df = pd.read_csv("%sFeatures_all.csv" % project_path)   #Features_all = Features 1+2+3
myData = pd.get_dummies(df, drop_first=True)
# check_data(myData)

# data normalization
X = myData.drop('keys', axis=1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))    # X = (X - np.min(X)) / (np.max(X) - np.min(X))
X = scaler.fit_transform(X)
# X = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, X)
y = myData['keys']  # keys=(1, 2, 3) as labels


# ---------------------------------MiniSOM--------------------------------------------------
# parameters
neighborhood_function = ['gaussian', 'mexican_hat', 'bubble', 'triangle']
    # ps: score=[0.93, 0.57, 0.86, 0.91] (when sigma=1, learning_rate=0.9, distance=manhattan)
learning_rate = [0.1*k for k in range(1, 10)]    #The larger the value of learning_rate, the greater the score
sigma = [0.1*k for k in range(1, 10)]            #The larger the value of sigma, the greater the score
activation_distance = ['euclidean', 'cosine', 'manhattan', 'chebyshev']
    # ps: score=[0.886, 0.864, 0.864, 0.864] (when neighborhood_func=gaussian, sigma=1, learning_rate=0.9);
    # score=[0.84, 0.82, 0.84, 0.89] (when neighborhood_func=gaussian, sigma=1, learning_rate=0.5)


plt.figure(figsize=(8, 6), num="class 1 - Red circle, class 2 - yellow square, class 3 - blue diamond")
for i in range(len(activation_distance)):
    print(" (*) activation_distance:", activation_distance[i])
    # init the model  <-- x*y>= 5*sqrt(samples)  --> 6*6 >= 5*sqrt(44 samples) ~= 33;  input_len=27 features
    som = MiniSom(x=6, y=6, input_len=27, sigma=1, learning_rate=0.5, neighborhood_function='gaussian', activation_distance=activation_distance[i], random_seed=50)
    # init the weight
    som.random_weights_init(X)  # som.pca_weights_init(X)

    # training the model
    t_start = time.time()
    som.train_random(data=X, num_iteration=500)  # som.train_batch(data=X, num_iteration=500, verbose=False)
    t_end = time.time()
    print("Execution time:", t_end - t_start, "s")
    print("Quantization error:", som.quantization_error(X))
    score = round(accuracy_score(y, classify(som, X)), 3); print("Model Accuracy Score: ", score)
    print("True labels:%s"%list(y), "\npredicted : %s\n"%classify(som, X))
    print(classification_report(y, classify(som, X)))


    # plotting distance matrix
    plt.subplot(2, 2, i+1)
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # cmap='Blues'
    plt.colorbar()
    # Transform labels into numeric codes for plotting
    label = np.zeros(len(y), dtype=int)
    label[y == 1] = 0;
    label[y == 2] = 1;
    label[y == 3] = 2
    markers = ['o', 's', 'd'];
    colors = ['r', 'y', 'b']  # y=1 : "Red circle"; y=2 : "yellow square"; y=3:"blue diamond"
    for k, m in enumerate(X):
        w = som.winner(m)  # getting the winner
        # plt.text(w[0]+.5, w[1]+.5, label[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.03, lw=0))
        plt.plot(w[0]+.5, w[1]+.5, markers[label[k]], markeredgecolor=colors[label[k]], markersize=15, markeredgewidth=2, markerfacecolor='None')
    plt.axis([0, 6, 0, 6])
    plt.title("activation_distance: %s\nMiniSom accuracy score: %s"%(activation_distance[i], score))
    plt.tight_layout()
    plt.pause(0.1)
plt.savefig("%sdistance matrix.png"%save_path)
plt.show()


