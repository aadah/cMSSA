import pickle
import numpy as np

from multiprocessing import Pool
from scipy import linalg
from sklearn.cluster import SpectralClustering


class CMSSA(object):
    def __init__(self, window=1, alpha=0.0, num_comp=2, standardize=False,
                 verbose=False):
        self.window = window
        self.alpha = alpha
        self.num_comp = num_comp
        self.standardize = standardize

        self._collapse = None
        self._flatten = None
        self._denormalize = None

        self._verbose = verbose

    def trajectory_matrix(self, X):
        # a window of 1 just recreates X,
        # i.e. no temporal analysis takes place
        if self.window == 1:
            return X

        n, d = X.shape
        new_n = n - self.window + 1
        lag_copies = [linalg.hankel(X[:new_n, j],
                                    X[new_n-1:, j]) for j in range(d)]
        return np.concatenate(lag_copies, axis=1)

    def normalize(self, Xs, mean, std):
        if not isinstance(Xs, list):
            return self._normalize((Xs, mean, std))
        triples = [(X, mean, std) for X in Xs]
        with Pool() as p:
            return p.map(self._normalize, triples)

    def _normalize(self, triple):
        X, mean, std = triple
        X -= mean
        if self.standardize:
            X /= std
        return X

    def stats(self, Xs):
        if not isinstance(Xs, list):
            Xs = [Xs]
        X_all = np.vstack(Xs)
        mean = np.mean(X_all, axis=0)
        std = np.std(X_all, axis=0)
        return mean, std

    def cov(self, Xs):
        if not isinstance(Xs, list):
            Xs = [Xs]
        with Pool() as p:
            trajs = p.map(self.trajectory_matrix, Xs)
        X_traj = np.vstack(trajs)
        new_n, _ = X_traj.shape
        return (X_traj.T @ X_traj) / new_n

    def project(self, Xs):
        if not isinstance(Xs, list):
            return self._project(Xs)
        with Pool() as p:
            return p.map(self._project, Xs)

    def _project(self, X):
        X = self.normalize(X, self.mean, self.std)
        X_traj = self.trajectory_matrix(X)
        return X_traj @ self.E

    def reconstruct(self, As, Xs,
                    collapse=True,
                    flatten=False,
                    denormalize=False):
        self._collapse = collapse
        self._flatten = flatten
        self._denormalize = denormalize
        if not isinstance(Xs, list):
            return self._reconstruct((As, Xs))
        with Pool() as p:
            return p.map(self._reconstruct, zip(As, Xs))

    def _reconstruct(self, pair):
        A, X = pair
        D = X.shape[1]
        K = self.E.shape[1]
        Rs = []
        for k in range(K):
            Hk = np.outer(A[:, k], self.E[:, k])
            Hks = np.split(Hk, D, axis=1)
            xs = np.array([self._average_anti_diagonal(hk) for hk in Hks])
            Rs.append(np.array(xs).T)
        R = np.stack(Rs, axis=0)
        if self._flatten:
            R = R.transpose(1, 0, 2).reshape((R.shape[1], -1))
        elif self._collapse:
            R = np.sum(R, axis=0)
            if self._denormalize:
                if self.standardize:
                    R *= self.std
                R += self.mean
        return R

    @staticmethod
    def _average_anti_diagonal(X):
        a, b = X.shape
        num = a+b-1
        X = np.flipud(X)
        x = [np.mean(np.diag(X, i)) for i in range(-a+1, b)]
        return x

    def fit(self, X_fg, X_bg):
        self.mean, self.std = self.stats(X_fg)
        X_fg = self.normalize(X_fg, self.mean, self.std)
        self.C_fg = self.cov(X_fg)

        mean_bg, std_bg = self.stats(X_bg)
        X_bg = self.normalize(X_bg, mean_bg, std_bg)
        self.C_bg = self.cov(X_bg)

    def compute_best_alphas(self,
                            return_size=10,
                            candidate_size=100,
                            min_log_alpha=-1,
                            max_log_alpha=3,
                            matrix_norm='nuc'):

        if self._verbose:
            print("Generating candidate alphas...")

        # generate candidates
        candidates = np.logspace(min_log_alpha, max_log_alpha, candidate_size)
        candidates = np.concatenate(([0], candidates))  # add zero as candidate

        if self._verbose:
            print("Computing each alpha's eigen matrix...")

        with Pool() as p:
            Es = p.map(self.compute_eigen, candidates)

        if self._verbose:
            print("Computing alpha affinity matix...")

        # create similarity matrix
        cs = len(candidates)
        S = np.empty((cs, cs))
        for i in range(cs):
            Ei = Es[i][0]
            for j in range(i, cs):
                Ej = Es[j][0]
                S[i, j] = S[j, i] = linalg.norm(Ei.T @ Ej, ord=matrix_norm)

        # cluster
        algo = SpectralClustering(n_clusters=return_size,
                                  affinity='precomputed',
                                  n_jobs=-1)

        if self._verbose:
            print("Clustering alphas...", end='')

        algo.fit(S)

        if self._verbose:
            print("DONE")

        # extract mediod alphas and their sum affinties
        best = []
        labels = algo.labels_
        for cluster in range(return_size):
            idxs = np.where(labels == cluster)[0]
            if 0 in idxs:
                # 0 idx maps to 0.0 alpha
                # pass on all alphas in its cluster
                continue
            S_sub = S[idxs][:, idxs]
            S_sum = np.sum(S_sub, axis=0) - S_sub.diagonal()
            am = np.argmax(S_sum)
            mediod_idx = idxs[am]
            mediod_mean = S_sum[am] / len(idxs)
            best.append((candidates[mediod_idx], mediod_mean))

        best.append((0.0, 0.0))  # include 0.0 always
        best.sort(key=lambda p: -p[1])

        return best

    def pick_best_alpha(self):
        return self.compute_best_alphas()[0][0]

    def compute_eigen(self, alpha):
        C = self.C_fg
        if alpha > 0:
            C = self.C_fg - alpha * self.C_bg
        self.eig_vals, self.eig_vecs = linalg.eig(C)
        idxs = np.argsort(-self.eig_vals)
        E = self.eig_vecs[:, idxs[:self.num_comp]]
        e = self.eig_vals[idxs[:self.num_comp]]
        return E, e

    def set_alpha(self, alpha=None):
        self.alpha = alpha if alpha is not None else self.pick_best_alpha()

    def set_eigen(self):
        self.E, self.e = self.compute_eigen(self.alpha)

    def transform(self, X, space='R',
                  collapse=True,
                  flatten=False,
                  denormalize=False):
        self.set_eigen()
        A = self.project(X)
        if space == 'A':
            return A
        elif space == 'R':
            R = self.reconstruct(A, X,
                                 collapse=collapse,
                                 flatten=flatten,
                                 denormalize=denormalize)
            return R
        else:
            raise Exception('unknown space: {}'.format(space))

    def fit_transform(self, X_fg, X_bg, collapse=True):
        self.fit(X_fg, X_bg)
        return self.transform(X_fg, collapse=collapse)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class MSSA(CMSSA):
    def __init__(self, window=1, num_comp=2):
        super().__init__(window=window, alpha=0.0, num_comp=num_comp)

    def fit(self, X):
        return super().fit(X, X)

    def fit_transform(self, X, collapse=True, denormalize=False):
        self.fit(X)
        return self.transform(X, collapse=collapse, denormalize=denormalize)
