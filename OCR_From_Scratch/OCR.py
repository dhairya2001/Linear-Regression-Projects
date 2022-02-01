# Copy Path Here
train_data_filename='train-images.idx3-ubyte'
train_labels_filename='train-labels.idx1-ubyte'
test_data_filename='t10k-images.idx3-ubyte'
test_labels_filename='t10k-labels.idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for labels_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels

def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

def extract_features(X):
    return [flatten_list(sample) for sample in X]

def dist(x,y):
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)
        ]
    ) ** (0.5)


def get_training_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]

def get_most_frequent_element(l):
    return max(l, key=l.count)

def knn(X_train,Y_train,X_test,k=3):
    y_pred=[]
    for test_sample_idx, test_sample in enumerate(X_test):
        training_distances=get_training_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates=[
            Y_train[idx]
            for idx in sorted_distance_indices[:k]
            ]
        # print(f'Point is {Y_test[test_sample_idx]} and we guessed {candidates}')
        # print(candidates)
        top_candidates=get_most_frequent_element(candidates)
        y_pred.append(top_candidates)
    return y_pred


def main():
    X_train = read_images(train_data_filename, 1000)
    Y_train = read_labels(train_labels_filename,1000)
    X_test = read_images(test_data_filename, 5)
    Y_test = read_labels(test_labels_filename,5)
    X_train= extract_features(X_train)
    X_test=extract_features(X_test)

    y_pred=knn(X_train,Y_train,X_test,3)
    print(y_pred)
    correct_predictions=sum([
        (y_test_i)==(y_pred_i)
        for y_test_i,y_pred_i
        in zip(Y_test,y_pred)
    ])/len(Y_test)
    print(correct_predictions)

if __name__ == '__main__':
    main()
