import kmeans
import os
from skimage import io
from imageio import imwrite
import numpy as np

files = ['Koala.jpg', 'Penguins.jpg']
num_centroids = (2, 5, 10, 15, 20)
max_iters = 50


def load_file(filename):
    image = io.imread(filename)
    image = image / 255.0
    rows, cols, _ = image.shape
    X = image.reshape(image.shape[0] * image.shape[1], 3)
    return X, rows, cols


if __name__ == '__main__':
    for read_filename in files:
        X, rows, cols = load_file(read_filename)
        compression_ratio = []
        for k in num_centroids:
            print("Processing {} for k = {}".format(read_filename, k))
            initial_centroids = kmeans.init_centroids(X, k)
            centroids, _ = kmeans.run_kMean(X, initial_centroids, max_iters)
            idx = kmeans.closest_centroids(X, centroids)
            X_recovered = centroids[idx].reshape((rows,  cols,  3))
            X_recovered *= 255.0
            X_recovered = X_recovered.astype(np.uint8)

            write_filename = read_filename[:-len(".jpg")] + "_{}.jpg".format(k)
            imwrite(write_filename, X_recovered)

            info = os.stat(read_filename)
            uncompressed_size = info.st_size/1024
            print("Size of image \"{}\" before K-mean for k={}: {}KB".format(read_filename, k, uncompressed_size))
            info = os.stat(write_filename)
            compressed_size = info.st_size / 1024
            print("Size of image \"{}\" after K-mean for k={}: {} KB".format(write_filename, k, compressed_size))
            compression_ratio.append(uncompressed_size / compressed_size)
        avg = np.average(compression_ratio)
        var = np.var(compression_ratio)
        print("Average compression ratio:", avg)
        print("Variance in compression ratios:", var)

