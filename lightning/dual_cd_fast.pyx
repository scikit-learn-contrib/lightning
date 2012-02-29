# Author: Mathieu Blondel
# License: BSD

import numpy as np

def _dual_cd(X, y, C, loss, max_iter, rs, tol=1e-4, precomputed_kernel=False,
             verbose=0):
    if precomputed_kernel:
        n_samples = X.shape[0]
    else:
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)

    alpha = np.zeros(n_samples, dtype=np.float64)
    A = np.arange(n_samples)
    active_size = n_samples

    if loss == "l1":
        U = C
        D_ii = 0
    elif loss == "l2":
        U = np.inf
        D_ii = 1.0 / (2 * C)

    if precomputed_kernel:
        Q_bar = X * np.outer(y, y)
        Q_bar += np.eye(n_samples) * D_ii

    M_bar = np.inf
    m_bar = -np.inf

    for it in xrange(max_iter):
        rs.shuffle(A[:active_size])

        M = -np.inf
        m = np.inf

        s = -1
        #for (s=0; s<active_size; s++)
        while s < active_size-1:
            s += 1
            i = A[s]
            y_i = y[i]
            alpha_i = alpha[i]

            if precomputed_kernel:
                # Need to be optimized in cython
                #G = -1
                #for j in xrange(n_samples):
                    #G += Q_bar[i, j] * alpha[j]
                G = np.dot(Q_bar, alpha)[i] - 1
            else:
                G = y_i * np.dot(w, X[i]) - 1 + D_ii * alpha_i

            PG = 0

            if alpha_i == 0:
                if G > M_bar:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    s -= 1
                    continue
                elif G < 0:
                    PG = G
            elif alpha_i == U:
                if G < m_bar:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    s -= 1
                    continue
                elif G > 0:
                    PG = G
            else:
                PG = G

            M = max(M, PG)
            m = min(m, PG)

            if np.abs(PG) > 1e-12:
               alpha_old = alpha_i

               if precomputed_kernel:
                   Q_bar_ii = Q_bar[i, i]
               else:
                # FIXME: can be pre-computed
                   Q_bar_ii = np.dot(X[i], X[i]) + D_ii

               alpha[i] = min(max(alpha_i - G / Q_bar_ii, 0.0), U)

               if not precomputed_kernel:
                   w += (alpha[i] - alpha_old) * y_i * X[i]

        if M - m <= tol:
            if active_size == n_samples:
                if verbose >= 1:
                    print "Stopped at iteration", it
                break
            else:
                active_size = n_samples
                M_bar = np.inf
                m_bar = -np.inf
                continue

        M_bar = M
        m_bar = m

        if M <= 0: M_bar = np.inf
        if m >= 0: m_bar = -np.inf

    if precomputed_kernel:
        return alpha
    else:
        return w
