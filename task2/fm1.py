import numpy as np
import pandas as pd
import random
import array
import csv
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.utils import shuffle


class Model:

    def __init__(self, n, k):
        self.w0 = 0
        self.ws = array.array('f', [0 for _ in range(n)])
        self.v = [array.array('f', [random.uniform(-1, 1) for _ in range(k)]) for _ in range(n)]

    def get_k(self):
        return len(self.v[0])

    def get_n(self):
        return len(self.ws)


# Cool hack, since we know, that y belong to [1,5] heh
def remap_rating(r):
    max_rating = 5.0
    min_rating = 1.0
    return (float(r) - min_rating) / (max_rating - min_rating)


def prepare_data(inputs, output):
    target_f = open(output, "w")

    for i in inputs:
        print("Read file " + i)
        cur_movie_id = None

        with open(i, "r") as csv_file:
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                if len(row) == 1:
                    cur_movie_id = row[0][:-1]
                else:
                    user_id = row[0]
                    rating = row[1]
                    target_f.write(user_id + "," + cur_movie_id + "," + str(remap_rating(rating)) + "\n")

    target_f.close()


def load_prepared_and_shuffle():
    data_file_path = "./dataset/prepared.csv"

    df = pd.read_csv(data_file_path, header=None, names=['User_Id', 'Movie_Id', 'Rating'])
    print(df.iloc[::5000000, :])

    encoder = OneHotEncoder(categories='auto', sparse=True)

    # data to predict
    y = np.asarray(df['Rating']).reshape(-1, 1)

    # (number_of_Y x number_of_users)
    one_hot_user_matrix = encoder.fit_transform(np.asarray(df['User_Id']).reshape(-1, 1))
    print("One-hot user matrix shape: " + str(one_hot_user_matrix.shape))

    # (number_of_Y x number_of_movie_ids)
    one_hot_movie_matrix = encoder.fit_transform(np.asarray(df['Movie_Id']).reshape(-1, 1))
    print("One-hot movie matrix shape: " + str(one_hot_movie_matrix.shape))

    # train data in CSR format
    X = hstack([one_hot_user_matrix, one_hot_movie_matrix]).tocsr()

    # do shuffling so records will be evenly distributed over the matrix
    print("Shuffle data")

    return shuffle(X, y)


def get_row_iterator(matrix, r):
    for ind in range(matrix.indptr[r], matrix.indptr[r + 1]):
        yield matrix.indices[ind], matrix.data[ind]


def get_row(matrix, r):
    return [(j, x) for j, x in get_row_iterator(matrix, r)]


def compute_y(matrix, r, m: Model):
    y = m.w0

    for i, xi in get_row_iterator(matrix, r):
        y += m.ws[i] * xi

        for j, xj in get_row_iterator(matrix, r):
            if j > i:
                v = 0.0
                for f in range(m.get_k()):
                    v += m.v[i][f] * m.v[j][f]
                y += v * xi * xj

    return y


def run_mini_batch_sgd(X, y, iter_count: int, learning_rate: float, k: int, mini_batch_size: int):
    n = X.shape[1]  # attributes in data
    s = X.shape[0]  # samples count
    m, m0 = Model(n, k), Model(n, k)  # model to store weights

    batch_iter_count = int(s / mini_batch_size) + (1 if s % mini_batch_size != 0 else 0)

    for iter_num in range(iter_count):
        print("Iteration: " + str(iter_num) + "/" + str(iter_count))

        for bi in range(batch_iter_count):
            if (bi * mini_batch_size) % 1000000 == 0:
                print("Process: " + str(bi * mini_batch_size))

            bstart = bi * mini_batch_size
            bend = min((bi + 1) * mini_batch_size, s)

            Xb = X[bstart:bend]
            yb = y[bstart:bend]

            if ri % 100000 == 0:
                print("Process: " + str(ri))

            m, m0 = m0, m
            ldiff = learning_rate * (compute_y(X, ri, m) - y[ri])

            m.w0 = m0.W0 - ldiff
            for i, xi in get_row_iterator(X, ri):
                m.ws[i] = m0.W[i] - ldiff * xi

                for f in range(k):
                    s = 0
                    for j, xj in get_row_iterator(X, ri):
                        s += m0.v[j][f] * xj

                    m.V[i][f] = m0.v[i][f] - ldiff * (xi * s + m0.v[i][f] * (xi ** 2))

    return m


def compute_rmse(X, y, m: Model):
    mse = 0.0
    n = X.shape[0]

    for i in range(n):
        mse += ((compute_y(X, i, m) - y[i]) ** 2) / float(n)

    return mse ** 0.5


def compute_r2(X, y, m: Model):
    nominator = 0.0
    denominator = 0.0
    expect = 0.0
    n = X.shape[0]

    for i in range(n):
        nominator += (y[i] - compute_y(X, i, m)) ** 2
        expect += y[i] / float(n)

    for i in range(n):
        denominator += (y[i] - expect) ** 2

    return 1 - float(nominator) / float(denominator)


def run_learning(X, y, cv_count: int, iter_count: int, learning_rate: float, k: int, mini_batch_size):
    rmses = []
    r2s = []
    rmses_test = []
    r2s_test = []

    rows_count = X.shape[0]
    rows_per_fold = [int(rows_count / cv_count) for _ in range(cv_count)]
    rows_per_fold[cv_count - 1] = rows_count - sum(rows_per_fold[0:cv_count - 1])  # ensure proper division

    subsets = [(sum(rows_per_fold[0:cv_num], 0), sum(rows_per_fold[0:cv_num + 1])) for cv_num in range(cv_count)]
    X_tests = [X[a:b] for a, b in subsets]
    y_tests = [y[a:b] for a, b in subsets]

    for cv_num in range(cv_count):
        print("Prepare fold: " + str(cv_num))
        indices = [i for i in range(cv_count) if i != cv_num]
        X_test = X_tests[cv_num]
        _y_test = y_tests[cv_num]
        to_join_x = [X_tests[i] for i in indices]
        to_join_y = [y_tests[i] for i in indices]
        X_train = vstack(to_join_x).tocsr()
        _y_train = np.vstack(to_join_y)

        y_train = array.array('d')
        for i in _y_train:
            y_train.append(float(i[0]))
        y_test = array.array('d')
        for i in _y_test:
            y_test.append(float(i[0]))

        print("Run learning")
        model = run_mini_batch_sgd(X_train, y_train, iter_count, learning_rate, k, mini_batch_size)

        print("Eval stats for fold: " + str(cv_num))
        rmses.append(compute_rmse(X_train, y_train, model))
        rmses_test.append(compute_rmse(X_test, y_test, model))
        r2s.append(compute_r2(X_train, y_train, model))
        r2s_test.append(compute_r2(X_test, y_test, model))

    return rmses, rmses_test, r2s, r2s_test


# Prepares data, normalize y and merge into single file
pattern = "./dataset/combined_data_%s.txt"
inputs = [pattern % (i + 1) for i in range(4)]
prepared = "./dataset/prepared.csv"
# prepare_data(inputs, prepared)

X, y = load_prepared_and_shuffle()

# Split into CV folds into several data files
cv_count = 2
iterations = 1
learning_rate = 1 / 100
mini_batch_size = 32
k = 2

rmses, rmses_test, r2s, r2s_test = run_learning(X, y, cv_count, iterations, learning_rate, k, mini_batch_size)

print(rmses)
print(rmses_test)
print(r2s)
print(r2s_test)
