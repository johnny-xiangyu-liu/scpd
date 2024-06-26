import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.stats import multivariate_normal

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)

DEBUG = False

def dprint(*args):
    if DEBUG:
        print(*args)

def pdf(x, mu, sigma):
    return multivariate_normal.pdf(x, mean = mu, cov = sigma)

    dim = mu.shape[0]
    return (2 * math.pi) ** (-dim/2) * np.linalg.det(sigma) ** (-1/2) \
        * math.exp(-1/2 * (x - mu).T.dot(np.linalg.inv(sigma)).dot(x-mu))

def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    rand_assignment = np.random.randint(K, size=x.shape[0])
    groups = [[] for k in range(K)]
#    dprint(groups)
#    dprint(rand_assignment)
    for i in range(x.shape[0]):
        groups[rand_assignment[i]].append(x[i])

    dprint("groups size:", len(groups))
    cluster_means = []
    covariances = []
    for idx, g in enumerate(groups):
        g = np.array(g)
        if len(g) == 0:
            cluster_means.append( np.empty(x.shape[1]))
            covariances.append( np.empty(x.shape[1]))
        else:
            dprint(">>>", g.shape)
            cluster_means.append(np.mean(g, axis=0))
            covariances.append(np.cov(g.T))
    dprint(cluster_means)
    mu = np.array(cluster_means)
    sigma = np.array(covariances)

    dprint(mu)
    dprint(sigma)
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(K, 1/K)
    dprint(phi)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full((x.shape[0], K), 1/K)

    dprint(w)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    n = x.shape[0]
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        it +=1
        prev_ll = ll
        print("it:", it)
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # responsibility should be of shape of (n_examples, k)
        def e_step(x, w, phi, mu, sigma):
            #r = np.empty((x.shape[0], K))
            for i in range(len(w)):
                for k in range(len(w[i])):
                    w[i][k] = phi[k] * pdf(x[i], mu[k], sigma[k])
                sum_cluster_i = np.sum(w[i])
                w[i] /= sum_cluster_i

        e_step(x, w, phi, mu, sigma)
        dprint("new weight:", w)

        # (2) M-step: Update the model parameters phi, mu, and sigma

        def m_step(x, w, phi, mu, sigma):
            dprint("sigma shape:", sigma.shape)

            N = x.shape[0]
            dprint("N:", N)

            w_sum = np.sum(w, axis=0)
            dprint("w:", w.shape)
            dprint("w_sum:", w_sum)
            phi = w_sum / N

            dprint("phi:", phi)
            dprint("before mu", mu, mu.shape)

            mu = w.T.dot(x)
            dprint("dot mu", mu, mu.shape)
            for idx in range(len(mu)):
                mu[idx] /= w_sum[idx]
            dprint("after mu", mu)

            dprint("x shape:", x.shape)
            dprint("before sigma", sigma)
            for k in range(K):
                s = None
                for i in range(x.shape[0]):
                    diff = np.array(x[i] - mu[k])[np.newaxis]
#                    dprint("diff:", diff)
                    si = w[i][k] * diff.T.dot(diff)
                    if s is None:
                        s = si;
                    else:
                        s += si;
                sigma[k] = s / w_sum[k]

        m_step(x,w,phi,mu,sigma)
        print("sigma:", sigma)
        print("mu:", mu)

        ll = 0
        for i, xi in enumerate(x):
            p = 0
            for k in range(K):
                prob = pdf(xi, mu[k], sigma[k]) * w[i][k] * phi[k]
                p+=prob
            ll += math.log(p)

        if ll is not None and prev_ll is not None:
            print("ll:", ll, "prev_ll:", prev_ll, ". diff:", (ll - prev_ll))
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        it += 1
        prev_ll = ll
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        def e_step(x, w, phi, mu, sigma):
            #r = np.empty((x.shape[0], K))
            for i in range(len(w)):
                for k in range(len(w[i])):
                    w[i][k] = phi[k] * pdf(x[i], mu[k], sigma[k])
                sum_cluster_i = np.sum(w[i])
                w[i] /= sum_cluster_i
        e_step(x, w, phi, mu, sigma)
        # (2) M-step: Update the model parameters phi, mu, and sigma

        def m_step(x, x_tilde, z_tilde, w, phi, mu, sigma):
            dprint("sigma shape:", sigma.shape)

            w_sum = np.sum(w, axis=0)

            mu_unsup = w.T.dot(x)
            def one_hot(z):
                a = z.astype(int)
                res = np.zeros((a.size, a.max() +1))
                res[np.arange(a.size), a] = 1
                return res.astype(int)
            dprint("z_t:", z_tilde.shape, z_tilde)
            z_1hot = one_hot(z_tilde.T)
            z_1hot_sum = alpha* np.sum(z_1hot, axis=0)
            dprint("z_1hot:", z_1hot.shape, z_1hot)
            # update phi:
            dprint("w:", w.shape)
            dprint("w_sum:", w_sum)
            N = x.shape[0]
            phi = (w_sum + z_1hot_sum)/ N
            dprint("phi:", phi.shape, phi)

            # update mu:
            mu_sup = alpha * z_1hot.T.dot(x_tilde)
            dprint("mu_sup:", mu_sup.shape, mu_sup)
            dprint("mu_unsup:", mu_unsup.shape, mu_unsup)
            mu = mu_unsup + mu_sup
            for idx in range(len(mu)):
                mu[idx] /= (w_sum[idx] + z_1hot_sum[idx])

            dprint("mu:", mu.shape, mu)

            # update sigma
            for k in range(K):
                s = None
                for i in range(x.shape[0]):
                    diff = np.array(x[i] - mu[k])[np.newaxis]
#                    dprint("diff:", diff)
                    si = w[i][k] * diff.T.dot(diff)
                    if s is None:
                        s = si;
                    else:
                        s += si;
                for i in range(x_tilde.shape[0]):
                    if z_1hot[i][k] != 0:
                        diff_tilde = np.array(x_tilde[i] - mu[k])[np.newaxis]
                        s += diff_tilde.T.dot(diff_tilde)

                sigma[k] = s / (w_sum[k] + z_1hot_sum[k])
            dprint("sigma shape:", sigma.shape)

        m_step(x, x_tilde, z_tilde, w,phi,mu,sigma)
        print("sigma:", sigma)
        print("mu:", mu)
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        ll = 0
        for i, xi in enumerate(x):
            p = 0
            for k in range(K):
                prob = pdf(xi, mu[k], sigma[k]) * w[i][k] * phi[k]
                p+=prob
            ll += math.log(p)


        for i, xi in enumerate(x_tilde):
            k = z_tilde.T.squeeze().astype(int)[i]
            p_sup = pdf(xi, mu[k], sigma[k])
            ll += alpha * math.log(p_sup)
        if ll is not None and prev_ll is not None:
            print("ll:", ll, "prev_ll:", prev_ll, ". diff:", (ll - prev_ll))

        # *** END CODE HERE ***

    return w


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***
