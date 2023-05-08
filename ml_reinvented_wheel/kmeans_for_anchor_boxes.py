import matplotlib.pyplot as plt
import os
import yolo_globals as yg
import numpy as np


def class_histogram():
    root = yg.ROOT_DATASET_PATH
    all_files = os.listdir(root)
    all_labels = [i for i in all_files if os.path.splitext(i)[1] == yg.LABEL_FILETYPE]
    class_counts = [0]*yg.NUM_CLASSES
    for lbl in all_labels:
        try:
            f = open(root + lbl)
            label_txt = f.readlines()
            f.close()
        except FileNotFoundError:
            print("Could not find label file", lbl)
            return
        for ann in label_txt:
            split_line = ann.strip().split(" ")
            class_counts[int(split_line[0])] += 1

    seen_amts = dict(yg.INVERSE_CLASS_MAP)
    for i in range(len(class_counts)):
        class_name = yg.CLASS_MAP[i]
        seen_amts[class_name] = class_counts[i]

    y_pos = np.arange(len(seen_amts))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(y_pos, list(seen_amts.values()))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(seen_amts.keys()))
    ax.set_title("Class Histogram")
    plt.show()


def base_wh_cluster_plot(plot=True):
    root = yg.ROOT_DATASET_PATH
    all_files = os.listdir(root)
    all_labels = [i for i in all_files if os.path.splitext(i)[1] == yg.LABEL_FILETYPE]
    widths = []
    for lbl in all_labels:
        try:
            f = open(root + lbl)
            label_txt = f.readlines()
            f.close()
        except FileNotFoundError:
            print("Could not find label file", lbl)
            return
        for ann in label_txt:
            # Format is class, x, y, w, h
            split_line = ann.strip().split(" ")
            w, h = float(split_line[3]), float(split_line[4])
            widths.append([w, h])
    widths = np.array(widths)
    if plot:
        plt.figure(figsize=(10, 10))
        plt.scatter(widths[:, 0], widths[:, 1], alpha=0.1)
        plt.title("W/H Clusters Normalized to % of Image", fontsize=20)
        plt.xlabel("Normalized Width", fontsize=20)
        plt.ylabel("Normalized Height", fontsize=20)
        plt.show()
    return widths


# I stole these functions from fairyonice's yolov2 implementation
def _iou(box, clusters):
    """
    :param box:      np.array of shape (2,) containing w and h
    :param clusters: np.array of shape (N cluster, 2)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    return intersection / (box_area + cluster_area - intersection)


def _kmeans(boxes, k, dist=np.mean, seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is num. total objects in the dataset
    :param k: number of clusters
    :param dist: distance function
    :param seed: RNG Seed
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))  # N row x N cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(k):
            distances[:, icluster] = 1 - _iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances


def kmeans_with_visual():
    root = yg.ROOT_DATASET_PATH
    all_files = os.listdir(root)
    all_labels = [i for i in all_files if os.path.splitext(i)[1] == yg.LABEL_FILETYPE]
    widths = []
    for lbl in all_labels:
        try:
            f = open(root + lbl)
            label_txt = f.readlines()
            f.close()
        except FileNotFoundError:
            print("Could not find label file", lbl)
            return
        for ann in label_txt:
            # Format is class, x, y, w, h
            split_line = ann.strip().split(" ")
            w, h = float(split_line[3]), float(split_line[4])
            widths.append([w, h])

    widths = np.array(widths)
    clusters, nearest_clusters, dists = _kmeans(widths, yg.NUM_ANCHOR_BOXES, seed=yg.GLOBAL_RNG_SEED)
    c_map = {0: "cyan", 1: "green", 2: "orange", 3: "purple", 4: "blue", 5: "pink", 6: "yellow"}
    plt.figure(figsize=(10, 10))
    for g in np.unique(nearest_clusters):
        ix = np.where(nearest_clusters == g)
        plt.scatter(widths[ix, 0], widths[ix, 1], c=c_map[g], label=g, s=20, alpha=0.1)
    plt.scatter(clusters[:, 0], clusters[:, 1], alpha=1, marker="x", color="r", sizes=[100]*yg.NUM_ANCHOR_BOXES)
    plt.title("W/H Clusters Normalized to % of Image", fontsize=20)
    plt.xlabel("Normalized Width", fontsize=20)
    plt.ylabel("Normalized Height", fontsize=20)
    plt.show()
    return clusters


# clust = kmeans_with_visual()
# print(clust)
