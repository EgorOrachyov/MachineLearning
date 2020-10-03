from csv import reader
from sklearn import preprocessing
from plotly import graph_objects


def import_data(path):
    return [[float(f) for f in r] for r in reader(open(path, "r"))]


def normalize_data(dataset):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    normalized = scaler.fit_transform(dataset)
    return normalized.tolist()


def cross_validation_split(dataset, cross_validation_k):
    size = len(dataset)
    splits = [int(size / cross_validation_k) for _ in range(cross_validation_k)]
    splits[cross_validation_k - 1] = size - sum(splits[0:cross_validation_k-1])

    sets = list()
    offset = 0
    for s in splits:
        sets.append([dataset[i] for i in range(offset, offset + s)])
        offset += s

    return sets


def compute_y(model, row):
    m = len(row)
    y = 0.0

    for i in range(m):
        y += row[i] * model[i]

    return y + model[m]


def compute_grad_mse(model, row, actual):
    m = len(row)
    grad = [0.0 for _ in range(m + 1)]
    diff = compute_y(model, row) - actual

    for i in range(m):
        grad[i] += 2.0 * row[i] * diff

    grad[m] += 2.0 * diff
    return grad


def learn_stochastic_gradient_decent_mse(fold, actuals, learning_rate, iterations_count):
    m = len(fold[0])
    model = [0.0 for _ in range(m + 1)]

    for i in range(iterations_count):
        for j in range(len(fold)):
            row = fold[j]
            actual = actuals[j]
            grad = compute_grad_mse(model, row, actual)
            model = [model[k] - learning_rate / (1 + i) * grad[k] for k in range(m + 1)]

    return model


def compute_rmse(prediction, actual):
    mse = 0.0
    n = len(prediction)

    for i in range(n):
        mse += ((prediction[i] - actual[i]) ** 2) / float(n)

    return mse ** 0.5


def compute_r2(prediction, actual):
    nominator = 0.0
    denominator = 0.0
    expect = 0.0

    for i in range(len(actual)):
        nominator += (actual[i] - prediction[i]) ** 2

    for i in range(len(actual)):
        expect += actual[i] / float(len(actual))

    for i in range(len(actual)):
        denominator += (actual[i] - expect) ** 2

    return 1 - float(nominator) / float(denominator)


def run_learning(sets, learning_rate, iterations_count):
    models = []
    rmses = []
    r2s = []
    rmses_test = []
    r2s_test = []

    for set in sets:
        fold = list(sets)
        fold.remove(set)
        fold = sum(fold, [])
        test = set

        fold_actual = [r[-1] for r in fold]
        fold = [r[0:-1] for r in fold]

        test_actual = [r[-1] for r in test]
        test = [r[0:-1] for r in test]

        model = learn_stochastic_gradient_decent_mse(fold, fold_actual, learning_rate, iterations_count)

        fold_pred = [compute_y(model, row) for row in fold]
        test_pred = [compute_y(model, row) for row in test]

        rmses.append(compute_rmse(fold_pred, fold_actual))
        r2s.append(compute_r2(fold_pred, fold_actual))
        rmses_test.append(compute_rmse(test_pred, test_actual))
        r2s_test.append(compute_r2(test_pred, test_actual))

        models.append(model)

    return models, rmses, r2s, rmses_test, r2s_test


def compute_stat(data):
    n = len(data)
    expectation = 0.0

    for d in data:
        expectation += d / float(n)

    sd = 0.0
    for d in data:
        sd += ((d - expectation) ** 2) / float(n)

    return expectation, sd ** 0.5


filepath = "dataset/Training/Features_Variant_1.csv"
cv_k = 5
learn_rate = 0.01
iterations = 50
dataset = normalize_data(import_data(filepath))
sets = cross_validation_split(dataset, cv_k)

models, rmse_train, r2_train, rmse_test, r2_test = run_learning(sets, learn_rate, iterations)
models_tr = [[models[i][j] for i in range(cv_k)] for j in range(len(dataset[0]))]

stats = [compute_stat(data) for data in [rmse_train, r2_train, rmse_test, r2_test] + models_tr]

values = ["X"] + ["Fold" + str(i) for i in range(cv_k)] + ["E","SD"]
cells = [ ["RMSE (train)", "R2 (train)", "RMSE (test)", "R2 (test)"] + ["f" + str(i) for i in range(len(dataset[0]))] ] + \
        [ [rmse_train[i], r2_train[i], rmse_test[i], r2_test[i]] + models[i] for i in range(cv_k) ] + \
        [ [stats[j][i] for j in range(len(stats))] for i in range(2) ]

table = graph_objects.Table(header=dict(values=values), cells=dict(values=cells))
figure = graph_objects.Figure(data=[table])
figure.show()

