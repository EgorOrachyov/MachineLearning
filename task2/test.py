import numpy as np
import pandas as pd
import scipy.sparse as ss
import random
import csv
import threading
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.utils import shuffle
from plotly import graph_objects

class Model:

    def __init__(self, n, k):
        self.w0 = 0
        self.W = ss.csc_matrix(np.zeros(shape=(n, 1), dtype=np.float))
        self.V = ss.csc_matrix(np.asarray([[0.5 for _ in range(k)] for _ in range(n)]))

    def get_k(self):
        return len(self.V[0])

    def get_n(self):
        return len(self.W)


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


def load_prepared_and_shuffle(data_file_path):
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


# Return numpy csr_vector of computed y-hat for each row of the matrix X
def compute_yhat(X, m: Model):
    XV = X.dot(m.V)
    XV2 = XV.multiply(XV)

    X2 = X # lol, x is composed only from 1..1, X == X.multiply(X)
    V2 = m.V.multiply(m.V)

    Y = X.dot(m.W) + m.w0 + 0.5 * (XV2 - X2.dot(V2)).sum(axis=1)

    return np.asarray(Y)


def run_mini_batch_sgd(X, y, iter_count: int, learning_rate: float, k: int, mini_batch_size: int):
    n = X.shape[1] # attributes in data
    s = X.shape[0] # samples count
    m, m_next = Model(n, k), Model(n, k) # model to store weights

    batch_iter_count = int(s / mini_batch_size) + (1 if s % mini_batch_size != 0 else 0)

    for iter_num in range(iter_count):
        print("Iteration: " + str(iter_num+1) + "/" + str(iter_count))

        total = 0
        for bi in range(batch_iter_count):
            if (((bi + 1) * mini_batch_size) // 1000000) > (total // 1000000):
                print("Process: " + str((bi + 1) * mini_batch_size))

            total += mini_batch_size

            m_next, m = m, m_next

            bstart = bi * mini_batch_size
            bend = min((bi + 1) * mini_batch_size, s)
            bcount = bend - bstart

            Xb = X[bstart:bend].tocsr()
            yb = y[bstart:bend]
            yh = compute_yhat(Xb, m)

            diff = yh - yb
            diff_scalar = np.add.reduce(diff)[0]
            coef = learning_rate / bcount / (1 + iter_num)

            m_next.w0 = m.w0 - coef * diff_scalar
            m_next.W = m.W - coef * (Xb.transpose() * diff)

            XbD = Xb.multiply(diff)
            XbDt = XbD.transpose()

            XV = Xb.dot(m.V)
            XtXV = XbDt.dot(XV)
            Xt = XbDt.sum(axis=1)
            Xt2 = ss.csc_matrix(Xt)
            Xt2V = Xt2.multiply(m.V)
            XtXVmXt2V = XtXV - Xt2V

            m_next.V = ss.csc_matrix(m.V - coef * XtXVmXt2V)

    return m_next


def compute_rmse_r2(X, y, m: Model):
    n = X.shape[0]

    expect = 1/n * np.add.reduce(y)[0]
    nominator = 0
    denominator = 0
    mse = 0

    i = 0
    step = 2 ** 13
    while i * step < n:
        start = i * step
        end = min((i + 1) * step, n)
        i += 1

        slice_y = y[start:end]

        diff = compute_yhat(X[start:end], m) - slice_y
        diff2 = diff ** 2
        sum = np.add.reduce(diff2)[0]

        mse += sum
        nominator += sum

        diffe = slice_y - expect
        diffe2 = diffe ** 2
        denominator += np.add.reduce(diffe2)[0]

    return (1/n * mse) ** 0.5, 1 - nominator / denominator


def run_concurrent(X_train, y_train, iter_count, learning_rate, k, mini_batch_size, X_test, y_test, lock: threading.Lock, rmses, r2s, rmses_test, r2s_test):
    model = run_mini_batch_sgd(X_train, y_train, iter_count, learning_rate, k, mini_batch_size)

    rmse, r2 = compute_rmse_r2(X_train, y_train, model)
    rmse_test, r2_test = compute_rmse_r2(X_test, y_test, model)

    lock.acquire()
    rmses.append(rmse)
    rmses_test.append(rmse_test)
    r2s.append(r2)
    r2s_test.append(r2_test)
    lock.release()


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

    lock = threading.Lock()
    threads = []
    tcount = 2

    for cv_num in range(cv_count):
        print("Prepare fold: " + str(cv_num))
        indices = [i for i in range(cv_count) if i != cv_num]
        X_test = X_tests[cv_num]
        y_test = y_tests[cv_num]
        to_join_x = [X_tests[i] for i in indices]
        to_join_y = [y_tests[i] for i in indices]
        X_train = vstack(to_join_x).tocsr()
        y_train = np.vstack(to_join_y)

        print("Run learning")
        t = threading.Thread(target=run_concurrent, args=(X_train, y_train, iter_count, learning_rate, k, mini_batch_size, X_test, y_test, lock, rmses, r2s, rmses_test, r2s_test))
        t.start()
        threads.append(t)

        if len(threads) == tcount:
            for t in threads:
                t.join()
            threads.clear()

        # model = run_mini_batch_sgd(X_train, y_train, iter_count, learning_rate, k, mini_batch_size)
        #
        # print("Eval stats for fold: " + str(cv_num))
        # rmses.append(compute_rmse(X_train, y_train, model))
        # rmses_test.append(compute_rmse(X_test, y_test, model))
        # r2s.append(compute_r2(X_train, y_train, model))
        # r2s_test.append(compute_r2(X_test, y_test, model))

    for t in threads:
        t.join()

    return rmses, rmses_test, r2s, r2s_test


def make_dummy(input, output):
    n = 1000000

    i = open(input, "r")
    o = open(output, "w")
    for l in range(n):
        o.write(next(i))

    i.close()
    o.close()


def compute_stat(data):
    n = len(data)
    expectation = 0.0

    for d in data:
        expectation += d / float(n)

    sd = 0.0
    for d in data:
        sd += ((d - expectation) ** 2) / float(n)

    return expectation, sd ** 0.5


# Prepares data, normalize y and merge into single file
pattern = "./dataset/combined_data_%s.txt"
inputs = [pattern % (i + 1) for i in range(4)]
prepared = "./dataset/prepared.csv"
prepared0 = "./dataset/prepared0.csv"
# prepare_data(inputs, prepared)
# make_dummy(prepared, prepared0)

X, y = load_prepared_and_shuffle(prepared)

# Split into CV folds into several data files
cv_count = 4
iterations = 5
learning_rate = 0.85 # 0.85
mini_batch_size = 2 ** 13 # 11
k = 2

rmses, rmses_test, r2s, r2s_test = run_learning(X, y, cv_count, iterations, learning_rate, k, mini_batch_size)

stats = [compute_stat(data) for data in [rmses, r2s, rmses_test, r2s_test]]

values = ["X"] + ["Fold" + str(i) for i in range(cv_count)] + ["E","SD"]
cells = [ ["RMSE (train)", "R2 (train)", "RMSE (test)", "R2 (test)"] ] + \
        [ [rmses[i], r2s[i], rmses_test[i], r2s_test[i]] for i in range(cv_count) ] + \
        [ [stats[j][i] for j in range(len(stats))] for i in range(2) ]

table = graph_objects.Table(header=dict(values=values), cells=dict(values=cells))
figure = graph_objects.Figure(data=[table])
figure.show()

