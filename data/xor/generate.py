import random

n_train_sets = 1000
n_test_sets = 100

train_data = open("train_data.csv", "w")
train_labels = open("train_labels.csv", "w")

test_data = open("test_data.csv", "w")
test_labels = open("test_labels.csv", "w")

for i in range(n_train_sets):
    i1 = random.randint(0, 1)
    i2 = random.randint(0, 1)
    train_data.write("%d,%d\n" % (i1, i2))

    o1 = not (i1 ^ i2)
    o2 =     (i1 ^ i2)
    train_labels.write("%d,%d\n" % (o1, o2))

for i in range(n_test_sets):
    i1 = random.randint(0, 1)
    i2 = random.randint(0, 1)
    test_data.write("%d,%d\n" % (i1, i2))

    o1 = not (i1 ^ i2)
    o2 =     (i1 ^ i2)
    test_labels.write("%d,%d\n" % (o1, o2))
