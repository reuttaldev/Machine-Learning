import argparse
import numpy as np
import sklearn.datasets
import sys
sys.path.append(".")
from config import RANDOM_SEED
parser = argparse.ArgumentParser()
parser.add_argument("--clusters", default=3, type=int, help="Number of clusters")
parser.add_argument("--examples", default=200, type=int, help="Number of examples")
parser.add_argument("--init", default="random", choices=["random", "kmeans++"], help="Initialization")
parser.add_argument("--iterations", default=20, type=int, help="Number of kmeans iterations to perform")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")


def plot(args: argparse.Namespace, iteration: int,
         data: np.ndarray, centers: np.ndarray, clusters: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if args.plot is not True:
        plt.gcf().get_axes() or plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Iteration {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")


def main(args: argparse.Namespace) -> np.ndarray:
    generator = np.random.RandomState(RANDOM_SEED)

    # Generate an artificial dataset.
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=RANDOM_SEED)

    if args.init == "random":
        centers = data[generator.choice(len(data), size=args.clusters, replace=False)]
    if args.init == "kmeans++":
        # Pick the first center uniformly at random
        first_center_i =generator.randint(len(data))
        centers = [data[first_center_i]]
        #   and then iteratively sample the rest of the clusters proportionally to
        #   the square of their distances to their closest cluster using
        #     generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
        #   Use the `np.linalg.norm` to measure the distances.
        for i in range(args.clusters-1):
            # Find the distance of each point to its nearest current center
            distances = np.min([np.linalg.norm(data - center, axis=1) for center in centers], axis=0)
            distances_squared = distances ** 2
            p = distances_squared / np.sum(distances_squared)
            # Sample the next center index according to p
            choice = generator.choice(len(data), p = p)
            centers.append(data[choice])
    centers = np.array(centers)


    if args.plot:
        plot(args, 0, data, centers, clusters=None)

    for iteration in range(args.iterations):      
        distances =  np.array([np.linalg.norm(data - center,axis=1) for center in centers])
        # each data point is assigned to the nearest center. these indexes say that at position ith data[i] is in cluster of index cluster[i]
        clusters = np.argmin(distances, axis=0)
        # now, after the clusters have been updates, compute their new mean, and update the center to this mean.
        for k in range(args.clusters):
            points_in_cluster = data[clusters == k]
            if len(points_in_cluster) > 0:
                centers[k] = points_in_cluster.mean(axis=0)

        if args.plot:
            plot(args, 1 + iteration, data, centers, clusters)

    return clusters


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    clusters = main(main_args)
    print("Cluster assignments:", clusters, sep="\n")
