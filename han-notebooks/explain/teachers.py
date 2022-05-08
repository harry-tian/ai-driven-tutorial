import numpy as np
from scipy.stats import entropy
from collections import defaultdict


def user_model(w, x, y, alpha):
    # multi-class user model - w is CxD and X is NxD
    # prob is probability that the hyp agrees with the datapoint
    z = alpha*np.dot(x, w.T)
    pred_class = np.argmax(z,1)
    z_norm = np.exp(z) / np.exp(z).sum(1)[..., np.newaxis]

    prob = z_norm[np.arange(x.shape[0]), pred_class]  # pred_class == y
    inds = np.where(pred_class != y)[0]
    prob[inds] = 1.0 - prob[inds]                     # pred_class != y
    return prob, pred_class


def compute_likelihood(hyps, X, Y, alpha, one_v_all=False):
    # compute P(y|h,x) - size HxN
    # is set to one where h(x) = y i.e. correct guess
    likelihood = np.ones((len(hyps), X.shape[0]))
    likelihood_opp = np.ones((len(hyps), X.shape[0]))

    for hh in range(len(hyps)):
        if one_v_all:
            # assumes that hyps[hh] is a D dim vector
            prob_agree, pred_class = user_model_binary(hyps[hh], X, Y, alpha)
        else:
            # assumes that hyps[hh] is a CxD dim maxtrix
            prob_agree, pred_class = user_model(hyps[hh], X, Y, alpha)
        inds = np.where(pred_class != Y)[0]
        likelihood[hh, inds] = prob_agree[inds]

    return likelihood


def more_teaching_stats(cur_post, pred, lnr_post, lnr_pred, err_hyp, err_hyp_test, selected_ind):

    cur_post_norm = cur_post/cur_post.sum()
    lnr_post_norm = lnr_post/lnr_post.sum()
    lnr_hyp = np.random.choice(len(lnr_post_norm), p=lnr_post_norm)
    exp_err = (cur_post_norm*err_hyp).sum()
    exp_err_test = err_hyp_test[lnr_hyp]
    ent = entropy(lnr_post_norm)

    z = (lnr_post_norm[:,np.newaxis]*pred).sum(0) + 0.0000000001  # add small noise
    # z = lnr_pred[lnr_hyp] + 0.0000000001  # add small noise
    difficulty = -(z*np.log2(z) + (1-z)*np.log2(1-z))
    diff_x = difficulty[selected_ind]
    diff_mean = np.mean(difficulty)
    diff_pctl = np.argsort(difficulty)[selected_ind] / len(difficulty)
    pred_ent = entropy(difficulty / difficulty.sum())

    stats = {
        'lnr_post_norm': lnr_post_norm,
        'lnr_hyp': lnr_hyp,
        'exp_err': exp_err,
        'exp_err_test': exp_err_test,
        'hyp_entropy': ent,
        'difficulty': diff_x,
        'diff_all': difficulty,
        'diff_mean': diff_mean,
        'diff_pctl': diff_pctl,
        'pred_ent': pred_ent,
    } 

    return stats


class StrictTeacher:

    def __init__(self, dataset, alpha, lnr_alpha, prior_h, hyps):
        X, Y = dataset['X'], dataset['Y']
        self.teaching_exs = []
        self.unseen_exs = np.arange(X.shape[0])
        self.hyps = hyps
        self.prior_h = prior_h
        self.cur_post = prior_h.copy()
        self.alpha = alpha
        self.pred = np.zeros((len(hyps), len(X)))
        self.likelihood = compute_likelihood(hyps, X, Y, alpha)
        self.lnr_post = prior_h.copy()
        self.lnr_alpha = lnr_alpha
        self.lnr_pred = np.zeros((len(hyps), len(X)))
        self.lnr_likelihood = compute_likelihood(hyps, X, Y, lnr_alpha)
        for hh in range(len(hyps)):
            for xx in range(len(X)):
                self.pred[hh, xx], _ = user_model(hyps[hh], X[xx,:][np.newaxis, ...], Y[xx], self.alpha)
                self.lnr_pred[hh, xx], _ = user_model(hyps[hh], X[xx,:][np.newaxis, ...], Y[xx], self.lnr_alpha)
        self.stats = defaultdict(list)

    def run_teaching(self, num_teaching_itrs, dataset, hyps, err_hyp, err_hyp_test):
        for tt in range(num_teaching_itrs):
            self.teaching_iteration(dataset['X'], dataset['Y'], hyps, err_hyp, err_hyp_test)

    def teaching_iteration(self, X, Y, hyps, err_hyp, err_hyp_test):

        # this is eqivalent to looping over h and x
        # comes from separating P(h|(A U x)) into P(h|A)P(h|x)
        err = -np.dot(self.cur_post*err_hyp, self.likelihood)
        selected_ind = self.unseen_exs[np.argmax(err[self.unseen_exs])]

        # update the posterior with the selected example
        self.cur_post *= self.likelihood[:, selected_ind]
        self.lnr_post *= self.lnr_likelihood[:, selected_ind]

        # bookkeeping and compute stats
        stats = more_teaching_stats(self.cur_post, self.pred, self.lnr_post, self.lnr_pred, err_hyp, err_hyp_test, selected_ind)
        self.stats['lnr_post_norm'].append(stats['lnr_post_norm'])
        self.stats['lnr_hyp'].append(stats['lnr_hyp'])
        self.stats['exp_err'].append(stats['exp_err'])
        self.stats['exp_err_test'].append(stats['exp_err_test'])
        self.stats['hyp_entropy'].append(stats['hyp_entropy'])
        self.stats['difficulty'].append(stats['difficulty'])
        self.stats['diff_all'].append(stats['diff_all'])
        self.stats['diff_mean'].append(stats['diff_mean'])
        self.stats['diff_pctl'].append(stats['diff_pctl'])
        self.stats['pred_ent'].append(stats['pred_ent'])
        self.teaching_exs.append(selected_ind)
        self.unseen_exs = np.setdiff1d(np.arange(X.shape[0]), self.teaching_exs)


class RandomImageTeacher:
    # assumes CxD hypotheses

    def __init__(self, dataset, lnr_alpha, prior_h, hyps):
        X, Y = dataset['X'], dataset['Y']
        self.teaching_exs = []
        self.hyps = hyps
        self.prior_h = prior_h
        self.lnr_post = prior_h.copy()
        self.lnr_alpha = lnr_alpha
        self.lnr_pred = np.zeros((len(hyps), len(X)))
        self.lnr_likelihood = compute_likelihood(hyps, X, Y, lnr_alpha)
        for hh in range(len(hyps)):
            for xx in range(len(X)):
                self.lnr_pred[hh, xx], _ = user_model(hyps[hh], X[xx,:][np.newaxis, ...], Y[xx], self.lnr_alpha)
        self.stats = defaultdict(list)

    def run_teaching(self, num_teaching_itrs, dataset, hyps, err_hyp, err_hyp_test):
        X = dataset['X']
        Y = dataset['Y']
        self.teaching_exs = np.random.choice(X.shape[0], num_teaching_itrs, replace=False)

        for teaching_ex in self.teaching_exs:

            self.lnr_post *= self.lnr_likelihood[:, teaching_ex]

            # bookkeeping and compute stats
            stats = more_teaching_stats(self.lnr_post, self.lnr_pred, self.lnr_post, self.lnr_pred, err_hyp, err_hyp_test, teaching_ex)
            self.stats['lnr_post_norm'].append(stats['lnr_post_norm'])
            self.stats['lnr_hyp'].append(stats['lnr_hyp'])
            self.stats['exp_err'].append(stats['exp_err'])
            self.stats['exp_err_test'].append(stats['exp_err_test'])
            self.stats['hyp_entropy'].append(stats['hyp_entropy'])
            self.stats['difficulty'].append(stats['difficulty'])
            self.stats['diff_all'].append(stats['diff_all'])
            self.stats['diff_mean'].append(stats['diff_mean'])
            self.stats['diff_pctl'].append(stats['diff_pctl'])
            self.stats['pred_ent'].append(stats['pred_ent'])
