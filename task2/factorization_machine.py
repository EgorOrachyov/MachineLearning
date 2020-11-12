import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.utils import shuffle
from plotly import graph_objects
import scipy.sparse as sparse
import math
import csv


def pack_to_single_csv(output):
    input_mask = './dataset/combined_data_%s.txt'
    index_range = [1, 2, 3, 4]

    target_f = open(output, "w")

    for i in index_range:
        cur_movie_id = None

        with open(input_mask % i, "r") as csv_file:
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                if len(row) == 1:
                    cur_movie_id = row[0][:-1]
                else:
                    user_id = row[0]
                    rating = row[1]
                    target_f.write(user_id + "," + rating + ',' + cur_movie_id + "\n")

    target_f.close()


def get_data(path):
    df = pd.read_csv(path, header=None, names=['User_Id', 'Rating', 'Movie_Id'])
    print(df.iloc[::5000000, :])

    encoder = OneHotEncoder(categories='auto', sparse=True)

    # data to predict
    Y = np.asarray(df['Rating']).reshape(-1, 1)

    # (number_of_Y x number_of_users)
    one_hot_user_matrix = encoder.fit_transform(np.asarray(df['User_Id']).reshape(-1, 1))
    print("One-hot user matrix shape: " + str(one_hot_user_matrix.shape))

    # (number_of_Y x number_of_movie_ids)
    one_hot_movie_matrix = encoder.fit_transform(np.asarray(df['Movie_Id']).reshape(-1, 1))
    print("One-hot movie matrix shape: " + str(one_hot_movie_matrix.shape))

    # train data in CSR format
    X = hstack([one_hot_user_matrix, one_hot_movie_matrix]).tocsr()

    return shuffle(X, Y)


class FM(object):

    def __init__(self, X_train, Y_train, X_test, Y_test, k, learning_rate, n_epoch, batch_size):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.k = k
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.batch_size = batch_size

        self.rmse_train = -1
        self.rmse_test = -1
        self.r2_train = -1
        self.r2_test = -1

        self.w0 = 0
        self.w = sparse.csc_matrix(np.zeros(shape=(self.get_features_count(), 1), dtype=np.float))
        self.v = sparse.csc_matrix([[0.5 for _ in range(self.k)] for _ in range(self.get_features_count())])

    def get_features_count(self):
        return self.X_train.shape[1]

    def get_train_samples_count(self):
        return self.X_train.shape[0]

    def predict(self, x):
        x_dot_v = x.dot(self.v)
        x_dot_v_2 = x_dot_v.multiply(x_dot_v)

        return self.w0 + x.dot(self.w) + 0.5 * (x_dot_v_2 - x.dot(self.v.multiply(self.v))).sum(axis=1)

    def fit(self):
        batches_count = math.ceil(self.get_train_samples_count() / self.batch_size)

        for epoch in range(self.n_epoch):
            #X, Y = shuffle(self.X_train, self.Y_train)
            X, Y = self.X_train, self.Y_train

            print("Epoch=%s" % epoch)
            for i in range(batches_count):
                a = i * self.batch_size
                b = min(a + self.batch_size, self.get_train_samples_count())
                x = X[a:b]
                y = Y[a:b]
                m = b - a

                if ((i + 1) * self.batch_size) // 1000000 > (i * self.batch_size) // 1000000:
                    print("Process=%s" % (i * self.batch_size))

                prediction = self.predict(x)
                error = prediction - y
                x_t = x.transpose()
                x_dot_error = x_t.dot(error)

                self.w0 = self.w0 - self.learning_rate / (1 + epoch) /m * error.sum(axis=0)[0,0]
                self.w = self.w - self.learning_rate / (1 + epoch) / m * x_dot_error

                x_t_mul_error = x_t.multiply(error.transpose())
                x_dot_v = x.dot(self.v)
                sum_x_t_dot_error = sparse.csc_matrix(x_t_mul_error.sum(axis=1))
                sum_x_t_dot_error_mul_v = sum_x_t_dot_error.multiply(self.v)

                self.v = self.v - self.learning_rate / (1 + epoch) / \
                         m * (x_t_mul_error.dot(x_dot_v) - sum_x_t_dot_error_mul_v)

        self.rmse_train = self.rmse_metric(self.X_train, self.Y_train)
        self.rmse_test = self.rmse_metric(self.X_test, self.Y_test)
        self.r2_train = self.r2_metric(self.X_train, self.Y_train)
        self.r2_test = self.r2_metric(self.X_test, self.Y_test)

    def rmse_metric(self, x, y):
        prediction = self.predict(x)
        error = np.asarray(prediction) - y
        error_2 = error ** 2

        return (error_2.sum(axis=0)[0] / x.shape[0]) ** 0.5

    def r2_metric(self, x, y):
        m_exp = y.sum(axis=0)[0] / x.shape[0]
        ss_tot = ((y - m_exp) ** 2).sum(axis=0)[0]
        ss_res = ((y - np.asarray(self.predict(x))) ** 2).sum(axis=0)[0]

        return 1 - ss_res / ss_tot


def make_dummy(input, output):
    n = 1000000

    i = open(input, "r")
    o = open(output, "w")
    for l in range(n):
        o.write(next(i))

    i.close()
    o.close()


def get_m_exp(values):
    m_exp = 0.0
    n = len(values)
    for v in values:
        m_exp += v / n

    return m_exp


def get_std(values):
    m_exp = get_m_exp(values)
    std = 0.0
    n = len(values)
    for v in values:
        std += ((v - m_exp) ** 2) / n

    return math.sqrt(std)


preprocessed = "./dataset/preprocessed.csv"
preprocessed0 = "./dataset/preprocessed0.csv"
#pack_to_single_csv(preprocessed)
make_dummy(preprocessed, preprocessed0)
X, Y = get_data(preprocessed0)

learning_rate = 0.5
n_epoch = 5
batch_size = 8000
folds_count = 5
k = 2

m = X.shape[0]
fold_size = m / folds_count
a = [int(i * fold_size) for i in range(folds_count)]
b = [int(a[i] + fold_size) for i in range(folds_count)]
b[-1] = m

rmse_train = []
rmse_test = []
r2_train = []
r2_test = []

for fold in range(folds_count):
    X_test = X[a[fold]:b[fold]]
    Y_test = Y[a[fold]:b[fold]]
    X_train = vstack([X[a[i]:b[i]] for i in range(folds_count) if i != fold]).tocsr()
    Y_train = np.vstack([Y[a[i]:b[i]] for i in range(folds_count) if i != fold])

    print("Fold=%s" % fold)
    fm = FM(X_train, Y_train, X_test, Y_test, k, learning_rate, n_epoch, batch_size)
    fm.fit()

    rmse_train.append(fm.rmse_train)
    rmse_test.append(fm.rmse_test)
    r2_train.append(fm.r2_train)
    r2_test.append(fm.r2_test)

statistic_m_exp = [get_m_exp(data) for data in [rmse_train, r2_train, rmse_test, r2_test]]
statistic_std = [get_std(data) for data in [rmse_train, r2_train, rmse_test, r2_test]]

header = [" "] + ["T" + str(i + 1) for i in range(folds_count)] + ["E", "STD"]
cells = [["RMSE-train", "R^2-train", "RMSE-test", "R^2-test"]] + \
        [[rmse_train[i], r2_train[i], rmse_test[i], r2_test[i]] for i in range(folds_count)] + \
        [statistic_m_exp, statistic_std]

graph_objects.Figure([graph_objects.Table(header=dict(values=header), cells=dict(values=cells))]).show()