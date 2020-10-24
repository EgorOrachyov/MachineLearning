from csv import reader


# Returns iterator to file dataset as a normal container
class DataIter:

    def __init__(self, file_path):
        self.file_path = file_path
        self.reader = reader(open(file_path, "r"))

    def _parse_row(self, row):
        return [row[0], row[1], float(row[2])]

    def create(self):
        return DataIter(self.file_path)

    def __next__(self):
        return self._parse_row(next(self.reader))

    def __iter__(self):
        return self


# Custom csr matrix
class MatrixCsr:

    def __init__(self, dim, rowsptr, columns, values):
        self.values = values
        self.rowptr = rowsptr
        self.columns = columns
        self.dim = dim

        if self.dim[0] + 1 != len(self.rowptr):
            raise Exception("Invalid matrix size")

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


# Transforms dataset iterator values into
class Transformer:

    def __init__(self, data_iter):
        self.attribs_mappings = dict()
        self.attribs_offsets = dict()
        self.row0 = next(data_iter)
        self.total_vectorized_attribs = 0
        self.rows_count = 1

        print("Prepare mappings")

        for k in range(len(self.row0)):
            self.attribs_offsets.update({k: 0})
            self.attribs_mappings.update({k: dict()})

        self._fit_row(self.row0)
        for row in data_iter:
            self._fit_row(row)
            self.rows_count += 1

        print("Assign indices")

        current_base = 0
        for k in range(len(self.row0)):
            self.attribs_offsets[k] = current_base
            current_base += 1 if len(self.attribs_mappings[k]) == 0 else len(self.attribs_mappings[k])

        self.total_vectorized_attribs = current_base

        print("Transformer meta: " + str(self.attribs_offsets))
        print("Total rows: " + str(self.rows_count))

    def _fit_row(self, parsed_row):
        for k in range(len(parsed_row)):
            v = parsed_row[k]
            if not (type(v) == str):
                continue

            attribs = self.attribs_mappings[k]
            if not (v in attribs):
                idx = len(attribs)
                attribs.update({v: idx})

    def transform(self, data_iter):
        values = []
        rowsptr = []
        colums = []

        i = 0

        for row in data_iter:
            rowsptr.append(len(values))
            for k in range(len(row)):
                v = row[k]
                if not (type(v) == str):
                    values.append(v)
                    colums.append(self.attribs_offsets[k])
                else:
                    values.append(1)
                    colums.append(self.attribs_offsets[k] + self.attribs_mappings[k][v])

            i += 1
            if i > 3:
                break

        dim = (len(rowsptr), self.total_vectorized_attribs)
        rowsptr.append(len(values))

        print("Transform data to matrix: " + str(dim))

        return MatrixCsr(dim, rowsptr, colums, values)


def remap_rating(r):
    max_rating = 5.0
    min_rating = 1.0
    return (float(r) - min_rating) / (max_rating - min_rating)


def prepare_data(inputs, output):
    target_f = open(output, "w+")

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
    rows_per_fold = [int(rows_count / cv_count) for i in range(cv_count)]
    rows_per_fold[cv_count - 1] += rows_count - int(rows_count / cv_count) * cv_count # ensure proper division

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


##############################################################
# Prepares data, normalize ratings and merge into single file

# pat = "./dataset/combined_data_%s.txt"
# inputs = [pat % (i + 1) for i in range(4)]
# prepared = "./dataset/prepared.csv"
# prepare_data(inputs, prepared)

prepared = "./dataset/prepared.csv"
data_iter = DataIter(prepared)
t = Transformer(data_iter)

##############################################################
# Split into CV folds into several data files

cv_count = 5
train_pat = "./dataset/train_%s.csv"
test_pat = "./dataset/test_%s.csv"
prepare_cv_folds(train_pat, test_pat, cv_count, t, prepared)


# v = VectorizerDense(data_iter)
# d = v.transform()

# print(d.dim())
