from sklearn.feature_extraction import DictVectorizer
import csv

train = [
    {"user": "1", "item": "5", "age": 19},
    {"user": "2", "item": "43", "age": 33},
    {"user": "3", "item": "20", "age": 55},
    {"user": "4", "item": "10", "age": 20},
]

v = DictVectorizer()
X = v.fit_transform(train)

print(X)
print(X.toarray())
print(type(X))

#
# combined_data_file_name = "/netflix-prize-data/combined_data_%s.txt"
# target_f = open("./netflix-prize-data/processed_data.csv", "w+")
#
# for i in [1,2,3,4]:
#     print("Read file " + str(i))
#     cur_movie_id = None
#
#     with open(combined_data_file_name % i) as csv_file:
#         csv_reader = csv.reader(csv_file)
#
#         for row in csv_reader:
#             if len(row) == 1:
#                 cur_movie_id = row[0][:-1]
#             else:
#                 user_id = row[0]
#                 rating = row[1]
#                 target_f.write(user_id + "," + rating + "," + cur_movie_id)
#
# target_f.close()