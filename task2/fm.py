from csv import reader
import random
import json
import array


class Model:

    def __init__(self):
        self.w0 = 0
        self.ws = []
        self.v = [[]]

    def get_k(self):
        return len(self.v[0])

    def get_n(self):
        return len(self.ws)

    def clone(self):
        m = Model()
        m.ws = [0.0 for _ in range(len(self.ws))]
        m.v = [[0.0 for _ in range(self.get_k())] for _ in range(self.get_n())]
        return m


# Returns iterator to file dataset as a normal container
class DataIter:

    def __init__(self, file_path):
        self.file_path = file_path
        self.reader = reader(open(file_path, "r"))

    @staticmethod
    def _parse_row(row):
        return [row[0], row[1], float(row[2])]

    def create(self):
        return DataIter(self.file_path)

    def __next__(self):
        return self._parse_row(next(self.reader))

    def __iter__(self):
        return self

    def reset(self):
        self.reader = reader(open(self.file_path, "r"))
        return self


# Custom csr matrix
class MatrixCsr:

    def __init__(self, dim, rowptr, columns, values):
        self.values = values
        self.rowptr = rowptr
        self.columns = columns
        self.dim = dim

        if self.dim[0] + 1 != len(self.rowptr):
            raise Exception("Invalid matrix size")

    @staticmethod
    def rows_subset(matrix, subsets):
        rowptr = array.array('i')
        colums = array.array('i')
        values = array.array('f')

        print("Start subset matrix: " + str(subsets))

        for (row_start, row_end) in subsets:
            for i in range(row_start, row_end):
                rowptr.append(len(colums))
                for j in range(matrix.rowptr[i], matrix.rowptr[i+1]):
                    colums.append(matrix.columns[j])
                    values.append(matrix.values[j])

        dim = (len(rowptr), matrix.get_columns_count())
        rowptr.append(len(values))

        print("Subset data to matrix: " + str(dim))

        return MatrixCsr(dim, rowptr, colums, values)

    def get_rows_count(self):
        return self.dim[0]

    def get_columns_count(self):
        return self.dim[1]

    def elements(self):
        result = []

        for i in range(self.get_rows_count()):
            for j in range(self.rowptr[i], self.rowptr[i + 1]):
                result.append((i, self.columns[j], self.values[j]))

        return result

    def get_row(self, i):
        result = []

        for j in range(self.rowptr[i], self.rowptr[i + 1]):
            result.append((self.columns[j], self.values[j]))

        return result

    def get_dim(self):
        return self.dim

    def toarray(self):
        r, c = self.dim()
        result = [[None for j in range(c)] for i in range(r)]

        for (i, j, v) in self.elements():
            result[i][j] = v

        return result

    def __str__(self):
        return str(self.elements())


class TransformerEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Transformer):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


# Transforms dataset iterator values into
class Transformer:

    def __init__(self):
        self.attribs_mappings = dict()
        self.attribs_offsets = dict()
        self.total_vectorized_attribs = 0
        self.rows_count = 0

    def save(self, file_path):
        print("Save transformer to " + file_path)
        with open(file_path, "w") as file:
            json.dump(self, file, cls=TransformerEncoder)

    @staticmethod
    def load(file_path):
        self = Transformer()

        with open(file_path, "r") as file:
            data = json.load(file)

            self.total_vectorized_attribs = int(data["total_vectorized_attribs"])
            self.rows_count = int(data["rows_count"])
            self.attribs_mappings = {int(k): v for k, v in data["attribs_mappings"].items()}
            self.attribs_offsets = {int(k): int(v) for k, v in data["attribs_offsets"].items()}

        print("Load transformer from " + file_path)
        print("Transformer: " + str(self.total_vectorized_attribs))
        print("Transformer: " + str(self.attribs_offsets))
        print("Transformer: " + str(self.rows_count))
        return self

    @staticmethod
    def create(data_iter):
        self = Transformer()
        self.rows_count = 1

        row0 = next(data_iter)

        print("Create transformer")

        for k in range(len(row0)):
            self.attribs_offsets.update({k: 0})
            self.attribs_mappings.update({k: dict()})

        self._fit_row(row0)
        for row in data_iter:
            self._fit_row(row)
            self.rows_count += 1

        print("Assign indices")

        current_base = 0
        for k in range(len(row0)):
            self.attribs_offsets[k] = current_base
            current_base += 1 if len(self.attribs_mappings[k]) == 0 else len(self.attribs_mappings[k])

        self.total_vectorized_attribs = current_base

        print("Transformer: " + str(self.total_vectorized_attribs))
        print("Transformer: " + str(self.attribs_offsets))
        print("Transformer: " + str(self.rows_count))

        return self

    def _fit_row(self, parsed_row):
        for k in range(len(parsed_row)):
            v = parsed_row[k]
            if not (type(v) == str):
                continue

            attribs = self.attribs_mappings[k]
            if not (v in attribs):
                idx = len(attribs)
                attribs.update({v: idx})

    def transform_row(self, row):
        vector = []

        for k in range(len(row)):
            v = row[k]
            if not (type(v) == str):
                vector.append((self.attribs_offsets[k], v))
            else:
                vector.append((self.attribs_offsets[k] + self.attribs_mappings[k][v], 1))

        return vector

    def transform(self, data_iter: DataIter):
        rowptr = array.array('i')
        colums = array.array('i')
        values = array.array('f')

        for row in data_iter.reset():
            rowptr.append(len(values))
            for k in range(len(row)):
                v = row[k]
                if not (type(v) == str):
                    values.append(v)
                    colums.append(self.attribs_offsets[k])
                else:
                    values.append(1)
                    colums.append(self.attribs_offsets[k] + self.attribs_mappings[k][v])

        dim = (len(rowptr), self.total_vectorized_attribs)
        rowptr.append(len(values))

        print("Transform data to matrix: " + str(dim))

        return MatrixCsr(dim, rowptr, colums, values)


# Cool hack, since we know, that ratings belong to [1,5] heh
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
            csv_reader = reader(csv_file)

            for row in csv_reader:
                if len(row) == 1:
                    cur_movie_id = row[0][:-1]
                else:
                    user_id = row[0]
                    rating = row[1]
                    target_f.write(user_id + "," + cur_movie_id + "," + str(remap_rating(rating)) + "\n")

    target_f.close()


def prepare_cv_folds(train_mask: str, test_mask: str, cv_count: int, t: Transformer, data_file: str):
    rows_count = t.rows_count
    rows_per_fold = [int(rows_count / cv_count) for _ in range(cv_count)]
    rows_per_fold[cv_count - 1] = rows_count - sum(rows_per_fold[0:cv_count - 1])  # ensure proper division

    indices = [i for i in range(cv_count)]
    data_iter = iter(open(data_file, "r"))

    # Create files with test data
    for f in range(cv_count):
        test_file_path = test_mask % f
        test_file = open(test_file_path, "w")
        print("Create file: " + test_file_path)
        i = 0
        while i < rows_per_fold[f]:
            i += 1
            row = next(data_iter)
            test_file.write(row)
        test_file.close()

    # Create train files
    for i in indices:
        to_merge = list(indices)
        to_merge.remove(i)

        train_file_path = train_mask % i
        train_file = open(train_file_path, "w")
        print("Create file: " + train_file_path)

        for j in to_merge:
            file_path = test_mask % j
            data_iter = iter(open(file_path, "r"))
            for row in data_iter:
                train_file.write(row)

        train_file.close()


def get_target_y(sparse_row):
    return sparse_row[-1][1]


def compute_y(sparse_row, m: Model):
    y = m.w0

    for i, x in sparse_row[:-1]:
        y += m.ws[i] * x

    fact = 0.0

    for f in range(m.get_k()):
        sum_squared = 0.0
        sum_squares = 0.0
        for i, x in sparse_row[:-1]:
            sum_squared += m.v[i][f] * x
            sum_squares += (m.v[i][f] * x) ** 2
        fact += (sum_squared ** 2) - sum_squares

    y += 0.5 * fact

    return y


def run_sgd(data: MatrixCsr, iter_count: int, learning_rate: float, k: int, n: int):
    m = Model()
    m.ws = [0.0 for _ in range(n)]
    m.v = [[random.uniform(0.0, 1.0) for _ in range(k)] for _ in range(n)]

    v = [[random.uniform(0.0, 1.0) for _ in range(k)] for _ in range(n)]

    for iter_num in range(iter_count):
        print("Iteration: " + str(iter_num) + "/" + str(iter_count))
        for ri in range(data.get_rows_count()):
            row = data.get_row(ri)
            diff = compute_y(row, m) - get_target_y(row)

            m.w0 -= learning_rate * diff
            X = row[:-1]

            for i, xi in X:
                m.ws[i] -= learning_rate * diff * xi

            for i, xi in X:
                for f in range(k):
                    s = 0
                    for j, xj in X:
                        s += m.v[j][f] * xj

                    m.v[i][f] -= learning_rate * diff * (xi * s + v[i][f] * (xi ** 2))

            if ri > 10000000:
                break

    return m


def compute_rmse(data: MatrixCsr, m: Model):
    mse = 0.0

    for i in range(data.get_rows_count()):
        row = data.get_row(i)
        mse += (compute_y(row, m) - get_target_y(row)) ** 2

        if i > 10000000:
            break

    return mse / data.get_rows_count()


def compute_r2(data: MatrixCsr, m: Model):
    nominator = 0.0
    denominator = 0.0
    expect = 0.0

    for i in range(data.get_rows_count()):
        sparse_row = data.get_row(i)
        actual = get_target_y(sparse_row)
        nominator += (actual - compute_y(sparse_row, m)) ** 2
        expect += actual

        if i > 10000000:
            break

    #expect /= data.get_rows_count()
    expect /= 10000000

    for i in range(data.get_rows_count()):
        sparse_row = data.get_row(i)
        actual = get_target_y(sparse_row)
        denominator += (actual - expect) ** 2

        if i > 10000000:
            break

    return 1 - float(nominator) / float(denominator)


def run_learning(prepared: str, cv_count: int, iter_count: int, learning_rate: float, t: Transformer, k: int):
    rmses = []
    r2s = []
    rmses_test = []
    r2s_test = []

    n = t.total_vectorized_attribs - 1

    data = t.transform(DataIter(prepared))
    rows_count = data.get_rows_count()
    rows_per_fold = [int(rows_count / cv_count) for _ in range(cv_count)]
    rows_per_fold[cv_count - 1] = rows_count - sum(rows_per_fold[0:cv_count - 1])  # ensure proper division

    subsets = [(sum(rows_per_fold[0:cv_num], 0), sum(rows_per_fold[0:cv_num + 1])) for cv_num in range(cv_count)]

    for cv_num in range(cv_count):
        print("Train fold: " + str(cv_num))

        test_mat = MatrixCsr.rows_subset(data, subsets[cv_num:cv_num+1])

        to_join = list(subsets)
        to_join.remove(subsets[cv_num])

        train_mat = MatrixCsr.rows_subset(data, to_join)

        model = run_sgd(train_mat, iter_count, learning_rate, k, n)

        print("Eval stats for fold: " + str(cv_num))
        rmses.append(compute_rmse(train_mat, model))
        rmses_test.append(compute_rmse(test_mat, model))
        r2s.append(compute_r2(train_mat, model))
        r2s_test.append(compute_r2(test_mat, model))

    return rmses, rmses_test, r2s, r2s_test


# Prepares data, normalize ratings and merge into single file
pat = "./dataset/combined_data_%s.txt"
inputs = [pat % (i + 1) for i in range(4)]
prepared = "./dataset/prepared.csv"
# prepare_data(inputs, prepared)

# Cook transformer mappings
transformer_path = "./dataset/transformer.json"
# data_iter = DataIter(prepared)
# t = Transformer.create(data_iter)
# t.save(transformer_path)

t = Transformer.load(transformer_path)

# Split into CV folds into several data files
cv_count = 5
iterations = 1
learning_rate = 1 / 100
k = 3

rmses, rmses_test, r2s, r2s_test = run_learning(prepared, cv_count, iterations, learning_rate, t, k)

print(rmses)
print(rmses_test)
print(r2s)
print(r2s_test)
