from sklearn.cluster import KMeans
import skimage.io as io
from skimage import img_as_float
import numpy as np
from math import log10
import matplotlib.pyplot as plt


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()


def mse_kof(I, K):
    m = len(I)
    n = len(I[0])
    c = 3

    s = 0

    for i in range(m):
        for j in range(n):
            for k in range(c):
                s += (I[i][j][k] - K[i][j][k]) ** 2

    return s / (m * n * c)


def psnr_kof(mse, maxi):
    return 10 * log10(maxi ** 2 / mse)

image = img_as_float(io.imread('parrots.jpg'))

matrix = list()

for i in range(len(image)):
    for j in range(len(image[i])):
        matrix.append(image[i][j])

for clusters_count in range(1, 21):
    kmeans = KMeans(n_clusters=clusters_count, init='k-means++', random_state=241)
    kmeans.fit(matrix)

    noisy_image = np.empty((len(image), len(image[0]), len(image[0][0])))

    for i in range(len(image)):
        for j in range(len(image[i])):
            label = kmeans.labels_[i * len(image[i]) + j]

            noisy_image[i][j] = kmeans.cluster_centers_[label]

    mse_k = mse_kof(image, noisy_image)
    psnr_k = psnr_kof(mse_k, 1)

    plt.imshow(noisy_image)
    plt.title('Clusters: %i, PSNR: %.2f' % (clusters_count, psnr_k))
    plt.savefig('parrots_%i_%.2f.png' % (clusters_count, psnr_k))
    plt.show()

    print 'Count of clusters: %i, PSNR: %.2f' % (clusters_count, psnr_k)

    if psnr_k > 20:
        print '\n1. Min count of clusters: %i' % clusters_count

        create_answer_file(1, clusters_count)

        break
