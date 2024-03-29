from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random

import sys
#np.set_printoptions(threshold=sys.maxsize)

def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    print("image shape", image.shape)
    print("image shape [-1]", image.shape[-1])
    centroids_init = []
    for i in range(num_clusters):
        w = random.randint(0, image.shape[0] -1)
        h = random.randint(0, image.shape[1] -1)
        centroids_init.append(image[w][h])

    centroids_init = np.array(centroids_init)
    print(centroids_init)
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    def assign_cluster(centroids, image):
        z = []
        for i in range(image.shape[0]):
            z.append([])
            for j in range(image.shape[1]):
                point = image[i][j]
                min_d = None
                assignment = None
                for index, centroid in enumerate(centroids):
                    delta = point.astype(np.float16) - centroid.astype(np.float16)
                    d = delta.T.dot(delta)
#                    print("point:",point, "centroid", centroid, "delta", delta, "d:", d)
                    if min_d is None or min_d > d:
                        min_d = d
                        assignment = index
#                print("assignment:", assignment)
                z[i].append(assignment)
        return np.array(z)

    def cal_centroids(centroids, z, image):
        n_centroids = np.zeros(centroids.shape)
        c_count = np.zeros(centroids.shape[0])
#        print("centroids", centroids.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                assignment = z[i][j]
                n_centroids[assignment] += np.array(image[i][j])
                c_count[assignment] += 1
        for i in range(len(n_centroids)):
            n_centroids[i] /= c_count[i]
        return n_centroids


    for i in range(max_iter):
        z = assign_cluster(centroids, image)
#        print("z", z.shape)
        new_centroids = cal_centroids(centroids, z, image)
        if not np.any(new_centroids - centroids):
            break;
        centroids = new_centroids
        if i % print_every == 0 or False:
            print("iteration:", i)
            print("z", z.shape, ":", z);
            print("centroids", centroids)

    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    def update(centroids, image):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                point = image[i][j]
                min_d = None
                assignment = None
                for index, centroid in enumerate(centroids):
                    delta = point.astype(np.float16) - centroid.astype(np.float16)
                    d = delta.T.dot(delta)
#                    print("point:",point, "centroid", centroid, "delta", delta, "d:", d)
                    if min_d is None or min_d > d:
                        min_d = d
                        assignment = index
#                print("assignment:", assignment)
                image[i][j] = centroids[assignment]
    update(centroids, image)
    print("result", image)
    return image
    # *** END YOUR CODE ***

def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
