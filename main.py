import parselmouth
import glob
import os.path
import numpy as np
from sklearn import mixture
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg

# make sure to pip install nose tornado and scikit-learn

## TODO:
# load in all audio files
# figure out how to cut off last sentence for testing (or a small phrase)
# figure out how to plot mfcc using api
# visualize gaussians
    # is it fast enough
def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        covariances = gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

color_iter = ['navy', 'c', 'cornflowerblue', 'gold', 'darkorange', 'red', 'blue', 'green', 'magenta', 'black', 'white', 'darksalmon', 'olivedrab', 'slategray', 'plum', 'coral', 'saddlebrown', 'indigo', 'lime', 'aqua']

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

iterations = 1

for i in range(0, iterations):
    values = [[0 for i in range(0, 4)] for j in range(0, 1)]
    targets = []
    counter = 0

    for fp in glob.iglob("male/*.mp3"):
        for file in glob.glob(fp):
            s = parselmouth.Sound(file)

        mfcc = s.to_mfcc()
        features = mfcc.extract_features()
        values = np.concatenate((values,np.transpose(features.as_array())), axis=0)
        target = [counter for i in range(0, len(features))]
        targets.append(target)
        counter += 1
        #[[0 for i in range(0, 4)] for j in range(0, 1)]
        #print str(len(np.transpose(features.as_array()))) + " male"

    for fp in glob.iglob("female/*.mp3"):
        for file in glob.glob(fp):
            s = parselmouth.Sound(file)

        mfcc = s.to_mfcc()
        features = mfcc.extract_features()
        values = np.concatenate((values,np.transpose(features.as_array())), axis=0)
        target = [counter for i in range(0, len(features))]
        targets.append(target)
        counter += 1
        #print str(len(np.transpose(features.as_array()))) + " female"

    ## delete later
    #iris = datasets.load_iris()
    #print iris.target
    #print iris.data

    flat_list = list(itertools.chain.from_iterable(targets))
    targets = np.array(flat_list)

    values = np.delete(values,0,0)
    skf = StratifiedKFold(n_splits=4)
    train_index, test_index = next(iter(skf.split(values, targets)))

    X_train = values[train_index]
    y_train = targets[train_index]
    X_test = values[test_index]
    y_test = targets[test_index]

    # print len(values)
    # print str(len(test_index)) + " y_test " + str(len(y_test))
    # print str(len(train_index)) + " y_test " + str(len(y_train))


    n_classes = len(np.unique(y_train))

    gmm = mixture.GaussianMixture(n_components = n_classes, max_iter = 200)
    trained = gmm.fit(X_train)
    labels = trained.predict(X_test)

    percent = 0;
    for j in range(0, (len(labels))):
        #print str(labels[j]) + " hi " + str(y_test[j])
        if labels[j] == y_test[j]:
            percent += 1
    percent = float(percent) / (len(labels))
    print percent
#plot_results(X_train, labels, trained.means_, trained.covariances_, 0,'Gaussian Mixture')
#plt.figure()
#plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,left=.01, right=.99)

#plt.scatter(values[:, 0], values[:, 1], values[:,2], c=labels, cmap='viridis');
#plt.show()
