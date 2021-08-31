import numpy as np
from scipy.stats import entropy
from collections import defaultdict

def user_model_binary(w, x, y, alpha):
    # binary user model - w is D and X is NxD
    # prob is probability that the hyp agrees with the datapoint
    # need to make prob = 1.0 / (1.0 + np.exp(-z*2*(2*y-1))) to be same as softmax
    if len(w.shape) == 2:
        z = alpha*np.dot(x, w[1,:])
    else:
        z = alpha*np.dot(x, w)
    pred_class = (z>0).astype(np.int)  # will be 0 or 1
    prob = 1.0 / (1.0 + np.exp(-z*(2*y-1)))  # make y={-1,1}
    return prob, pred_class


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


def teaching_stats(cur_post, pred, err_hyp, err_hyp_test):

    cur_post_norm = cur_post/cur_post.sum()
    exp_err = (cur_post_norm*err_hyp).sum()
    exp_err_test = (cur_post_norm*err_hyp_test).sum()
    ent = entropy(cur_post_norm)

    z = (cur_post_norm*pred).sum() + 0.0000000001  # add small noise
    difficulty = -(z*np.log2(z) + (1-z)*np.log2(1-z))

    return exp_err, exp_err_test, ent, difficulty


def teaching_stats_one_vs_all(cur_post, pred, err_hyp, err_hyp_test):
    
    exp_err = np.empty(cur_post.shape)
    exp_err_test = np.empty(cur_post.shape)
    entropy = np.empty(cur_post.shape)
    difficulty = np.empty(cur_post.shape)
    for cc in range(cur_post.shape[0]):
        exp_err[cc, :], exp_err_test[cc, :], entropy[cc, :], difficulty[cc, :] = teaching_stats(cur_post[cc, :], pred[cc, :], err_hyp[cc], err_hyp_test[cc])

    return exp_err.mean(), exp_err_test.mean(), entropy.mean(), difficulty.mean()


def more_teaching_stats(cur_post, pred, err_hyp, err_hyp_test, selected_ind):

    cur_post_norm = cur_post/cur_post.sum()
    exp_err = (cur_post_norm*err_hyp).sum()
    exp_err_test = (cur_post_norm*err_hyp_test).sum()
    ent = entropy(cur_post_norm)

    z = (cur_post_norm[:,np.newaxis]*pred).sum(0) + 0.0000000001  # add small noise
    difficulty = -(z*np.log2(z) + (1-z)*np.log2(1-z))
    diff_x = difficulty[selected_ind]
    diff_mean = np.mean(difficulty)
    diff_pctl = np.argsort(difficulty)[selected_ind] / len(difficulty)
    pred_ent = entropy(difficulty / difficulty.sum())

    stats = {
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


class EfficientStrictTeacher:

    def __init__(self, dataset, alpha, prior_h, hyps):
        self.initialize(dataset['X'], dataset['Y'], alpha, prior_h, hyps)

    def initialize(self, X, Y, alpha, prior_h, hyps):
        self.teaching_exs = []
        self.unseen_exs = np.arange(X.shape[0])
        self.hyps = hyps
        self.prior_h = prior_h
        self.cur_post = prior_h.copy()
        self.alpha = alpha
        self.pred = np.zeros((len(hyps), len(X)))
        for hh in range(len(hyps)):
            for xx in range(len(X)):
                self.pred[hh, xx], _ = user_model(hyps[hh], X[xx,:][np.newaxis, ...], Y[xx], self.alpha)
        self.stats = defaultdict(list)

    def posterior(self):
        return self.cur_post/self.cur_post.sum()

    def run_teaching(self, num_teaching_itrs, dataset, likelihood, hyps, err_hyp, err_hyp_test):
        for tt in range(num_teaching_itrs):
            self.teaching_iteration(dataset['X'], dataset['Y'], likelihood, hyps, err_hyp, err_hyp_test)

    def teaching_iteration(self, X, Y, likelihood, hyps, err_hyp, err_hyp_test):

        # this is eqivalent to looping over h and x
        # comes from separating P(h|(A U x)) into P(h|A)P(h|x)
        err = -np.dot(self.cur_post*err_hyp, likelihood)
        selected_ind = self.unseen_exs[np.argmax(err[self.unseen_exs])]

        # update the posterior with the selected example
        self.cur_post *= likelihood[:, selected_ind]

        # bookkeeping and compute stats
        # print(len(self.teaching_exs), '\t', Y[selected_ind], '\t', selected_ind, '\t', round(err[self.unseen_exs].max(),4))
        stats = more_teaching_stats(self.cur_post, self.pred, err_hyp, err_hyp_test, selected_ind)
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


class EfficientRandomImageTeacher:
    # assumes CxD hypotheses

    def __init__(self, dataset, alpha, prior_h, hyps):
        self.initialize(dataset['X'], dataset['Y'], alpha, prior_h, hyps)

    def initialize(self, X, Y, alpha, prior_h, hyps):
        self.teaching_exs = []
        self.hyps = hyps
        self.prior_h = prior_h
        self.cur_post = prior_h.copy()
        self.alpha = alpha
        self.pred = np.zeros((len(hyps), len(X)))
        for hh in range(len(hyps)):
            for xx in range(len(X)):
                self.pred[hh, xx], _ = user_model(hyps[hh], X[xx,:][np.newaxis, ...], Y[xx], self.alpha)
        self.stats = defaultdict(list)

    def posterior(self):
        return self.cur_post/self.cur_post.sum()

    def run_teaching(self, num_teaching_itrs, dataset, likelihood, hyps, err_hyp, err_hyp_test):
        X = dataset['X']
        Y = dataset['Y']
        self.teaching_exs = np.random.choice(X.shape[0], num_teaching_itrs, replace=False)

        for teaching_ex in self.teaching_exs:

            # compute the posterior of the selected example
            pred = np.zeros(len(hyps))
            for hh in range(len(hyps)):
                pred[hh], pred_class = user_model(hyps[hh], X[teaching_ex,:][np.newaxis, ...], Y[teaching_ex], self.alpha)
                if pred_class != Y[teaching_ex]:
                    self.cur_post[hh] *= pred[hh]

            # bookkeeping and compute stats
            stats = more_teaching_stats(self.cur_post, self.pred, err_hyp, err_hyp_test, teaching_ex)
            self.stats['exp_err'].append(stats['exp_err'])
            self.stats['exp_err_test'].append(stats['exp_err_test'])
            self.stats['hyp_entropy'].append(stats['hyp_entropy'])
            self.stats['difficulty'].append(stats['difficulty'])
            self.stats['diff_all'].append(stats['diff_all'])
            self.stats['diff_mean'].append(stats['diff_mean'])
            self.stats['diff_pctl'].append(stats['diff_pctl'])
            self.stats['pred_ent'].append(stats['pred_ent'])

    # @property
    # def exp_err():
    #     return self.stats['exp_err']

    # @property
    # def exp_err_test():
    #     return self.stats['exp_err_test']

    # @property
    # def hyp_entropy():
    #     return self.stats['hyp_entropy']

    # @property
    # def difficulty():
    #     return self.stats['difficulty']

    # @property
    # def diff_pctl():
    #     return self.stats['diff_pctl']

    # @property
    # def pred_ent():
    #     return self.stats['pred_ent']


class StrictTeacher:
    # Singla et al. Near-Optimally Teaching the Crowd to Classify
    # https://arxiv.org/abs/1402.2092

    def __init__(self, dataset, alpha, prior_h):
        self.initialize(dataset['X'], dataset['Y'], alpha, prior_h)

    def initialize(self, X, Y, alpha, prior_h):
        self.teaching_exs = []
        self.unseen_exs = np.arange(X.shape[0])
        self.prior_h = prior_h
        self.cur_post = prior_h.copy()
        self.alpha = alpha
        self.exp_err = []
        self.exp_err_test = []
        self.hyp_entropy = []
        self.difficulty = []

    def posterior(self):
        return self.cur_post/self.cur_post.sum()

    def run_teaching(self, num_teaching_itrs, dataset, likelihood, hyps, err_hyp, err_hyp_test):
        for tt in range(num_teaching_itrs):
            self.teaching_iteration(dataset['X'], dataset['Y'], likelihood, hyps, err_hyp, err_hyp_test)

    def teaching_iteration(self, X, Y, likelihood, hyps, err_hyp, err_hyp_test):

        # this is eqivalent to looping over h and x
        # comes from separating P(h|(A U x)) into P(h|A)P(h|x)
        err = -np.dot(self.cur_post*err_hyp, likelihood)
        selected_ind = self.unseen_exs[np.argmax(err[self.unseen_exs])]

        # update the posterior with the selected example
        self.cur_post *= likelihood[:, selected_ind]

        # get predictions for each hyp for selected example
        pred = np.zeros(len(hyps))
        for hh in range(len(hyps)):
            pred[hh], _ = user_model(hyps[hh], X[selected_ind,:][np.newaxis, ...], Y[selected_ind], self.alpha)

        # bookkeeping and compute stats
        print(len(self.teaching_exs), '\t', Y[selected_ind], '\t', selected_ind, '\t', round(err[self.unseen_exs].max(),4))
        ee, ee_test, ent, diff = teaching_stats(self.cur_post, pred, err_hyp, err_hyp_test)
        self.exp_err.append(ee)
        self.exp_err_test.append(ee_test)
        self.hyp_entropy.append(ent)
        self.difficulty.append(diff)
        self.teaching_exs.append(selected_ind)
        self.unseen_exs = np.setdiff1d(np.arange(X.shape[0]), self.teaching_exs)

    def teaching_iteration_slow(self, X, Y, hyps, err_hyp):
        eer = np.zeros(self.unseen_exs.shape[0])
        for ii, ex in enumerate(self.unseen_exs):
            cur_post_delta = np.ones(len(hyps))
            for hh in range(len(hyps)):

                # can store a H*X matrix where it will be 1 where hyp gets it correct and y_p else where
                y_p, pred_class = user_model(hyps[hh], X[ex,:][np.newaxis, ...], Y[ex], self.alpha)
                if pred_class != Y[ex]:
                    cur_post_delta[hh] *= y_p
                eer[ii] += (self.prior_h[hh] - (self.cur_post[hh]*cur_post_delta[hh]))*err_hyp[hh]
                #eer[ii] += -(self.cur_post[hh]*cur_post_delta[hh])*err_hyp[hh]  # dont need to subtract prior

        # recompute the posterior of the selected example
        selected_ind = self.unseen_exs[np.argmax(eer)]
        pred = np.zeros(len(hyps))
        for hh in range(len(hyps)):
            pred[hh], pred_class = user_model(hyps[hh], X[selected_ind,:][np.newaxis, ...], Y[selected_ind], self.alpha)
            if pred_class != Y[selected_ind]:
                self.cur_post[hh] *= pred[hh]

        # bookkeeping and compute stats
        print(len(self.teaching_exs), '\t', selected_ind, '\t', round(eer.max(),4))
        ee, ee_test, ent, diff = teaching_stats(self.cur_post, pred, err_hyp, err_hyp_test)
        self.exp_err.append(ee)
        self.exp_err_test.append(ee_test)
        self.hyp_entropy.append(ent)
        self.difficulty.append(diff)
        self.teaching_exs.append(selected_ind)
        self.unseen_exs = np.setdiff1d(np.arange(X.shape[0]), self.teaching_exs)


class StrictTeacherOneVsAll:
    # 1 vs all version of
    # Singla et al. Near-Optimally Teaching the Crowd to Classify
    # https://arxiv.org/abs/1402.2092

    def __init__(self, dataset, alpha, prior_h):
        self.initialize(dataset['X'], dataset['Y'], alpha, prior_h)

    def initialize(self, X, Y, alpha, prior_h):
        self.teaching_exs = []
        self.unseen_exs = np.arange(X.shape[0])
        self.classes = np.unique(Y)  # TODO need to update this for binary
        self.prior_h = np.tile(prior_h, (len(self.classes), 1))
        self.cur_post = np.tile(prior_h.copy(), (len(self.classes), 1))
        self.alpha = alpha
        self.exp_err = []
        self.exp_err_test = []
        self.hyp_entropy = []
        self.difficulty = []

    def posterior(self):
        return self.cur_post/self.cur_post.sum(1)[..., np.newaxis]

    def run_teaching(self, num_teaching_itrs, dataset, likelihood, hyps, err_hyp, err_hyp_test):
        for tt in range(num_teaching_itrs):
            self.teaching_iteration(dataset['X'], dataset['Y'], likelihood, hyps, err_hyp, err_hyp_test)

    def teaching_iteration(self, X, Y, likelihood, hyps, err_hyp, err_hyp_test):

        err = np.empty((len(self.classes), X.shape[0]))
        for cc in self.classes:
            # this is eqivalent to looping over h and x
            # comes from separating P(h|(A U x)) into P(h|A)P(h|x)
            err[cc, :] = -np.dot(self.cur_post[cc, :]*err_hyp[cc], likelihood[cc])

        if len(self.classes) > 2:
            err = err.sum(0)  # could try other methods for combining, min, max, ...
        selected_ind = self.unseen_exs[np.argmax(err[self.unseen_exs])]

        # update the posterior with the selected example
        for cc in self.classes:
            self.cur_post[cc, :] *= likelihood[cc][:, selected_ind]

        # get predictions for each hyp for selected example
        pred = np.zeros((len(self.classes), len(hyps)))
        for cc in self.classes:
            Y_bin = int(Y[selected_ind] == cc)
            for hh in range(len(hyps)):
                pred[cc, hh], _ = user_model_binary(hyps[hh], X[selected_ind,:][np.newaxis, ...], Y_bin, self.alpha)

        # bookkeeping and compute stats
        print(len(self.teaching_exs), '\t', Y[selected_ind], '\t', selected_ind, '\t', round(err[self.unseen_exs].max(),4))
        ee, ee_test, ent, diff = teaching_stats_one_vs_all(self.cur_post, pred, err_hyp, err_hyp_test)
        self.exp_err.append(ee)
        self.exp_err_test.append(ee_test)
        self.hyp_entropy.append(ent)
        self.difficulty.append(diff)
        self.teaching_exs.append(selected_ind)
        self.unseen_exs = np.setdiff1d(np.arange(X.shape[0]), self.teaching_exs)


class ExplainTeacher:
    def __init__(self, dataset, alpha, prior_h):
        self.initialize(dataset['X'], dataset['Y'], alpha, prior_h)

    def initialize(self, X, Y, alpha, prior_h):
        self.teaching_exs = []
        self.unseen_exs = np.arange(X.shape[0])
        self.prior_h = prior_h
        self.cur_post = prior_h.copy()
        self.alpha = alpha
        self.exp_err = []
        self.exp_err_test = []
        self.hyp_entropy = []
        self.difficulty = []

    def posterior(self):
        return self.cur_post/self.cur_post.sum()

    def run_teaching(self, num_teaching_itrs, dataset, likelihood, hyps, err_hyp, err_hyp_test):
        for tt in range(num_teaching_itrs):
            self.teaching_iteration(dataset['X'], dataset['Y'], dataset['X_density'], dataset['explain_interp'], likelihood, hyps, err_hyp, err_hyp_test)

    def teaching_iteration(self, X, Y, X_density, interpretability, likelihood, hyps, err_hyp, err_hyp_test):
        # X_density is how representative points are - dont want to select outliers
        # interpretability is how easy it is for user to make sense of explanation

        # this is eqivalent to looping over h and x
        # comes from separating P(h|(A U x)) into P(h|A)P(h|x)

        # err is negative, we want to find max. To increase it we multiply by smaller numbers
        # this has the effect of discounting less the relevant ones
        err = -np.dot(self.cur_post*err_hyp, likelihood)
        err = err*X_density*interpretability
        selected_ind = self.unseen_exs[np.argmax(err[self.unseen_exs])]

        # update the posterior with the selected example
        self.cur_post *= likelihood[:, selected_ind]*X_density[selected_ind]*interpretability[selected_ind]
        #self.cur_post = self.cur_post / self.cur_post.sum()  # don't need to renormalize

        # get predictions for each hyp for selected example
        pred = np.zeros(len(hyps))
        for hh in range(len(hyps)):
            pred[hh], _ = user_model(hyps[hh], X[selected_ind,:][np.newaxis, ...], Y[selected_ind], self.alpha)

        # bookkeeping and compute stats
        print(len(self.teaching_exs), '\t', Y[selected_ind], '\t', selected_ind, '\t', round(err[self.unseen_exs].max(),4))
        ee, ee_test, ent, diff = teaching_stats(self.cur_post, pred, err_hyp, err_hyp_test)
        self.exp_err.append(ee)
        self.exp_err_test.append(ee_test)
        self.hyp_entropy.append(ent)
        self.difficulty.append(diff)
        self.teaching_exs.append(selected_ind)
        self.unseen_exs = np.setdiff1d(np.arange(X.shape[0]), self.teaching_exs)


class ExplainTeacherOneVsAll:
    # 1 vs all version

    def __init__(self, dataset, alpha, prior_h):
        self.initialize(dataset['X'], dataset['Y'], alpha, prior_h)

    def initialize(self, X, Y, alpha, prior_h):
        self.teaching_exs = []
        self.unseen_exs = np.arange(X.shape[0])
        self.classes = np.unique(Y)
        self.prior_h = np.tile(prior_h, (len(self.classes), 1))
        self.cur_post = np.tile(prior_h.copy(), (len(self.classes), 1))
        self.alpha = alpha
        self.exp_err = []
        self.exp_err_test = []
        self.hyp_entropy = []
        self.difficulty = []

    def posterior(self):
        return self.cur_post/self.cur_post.sum(1)[..., np.newaxis]

    def run_teaching(self, num_teaching_itrs, dataset, likelihood, hyps, err_hyp, err_hyp_test):
        for tt in range(num_teaching_itrs):
            self.teaching_iteration(dataset['X'], dataset['Y'], dataset['X_density'], dataset['explain_interp'], likelihood, hyps, err_hyp, err_hyp_test)

    def teaching_iteration(self, X, Y, X_density, interpretability, likelihood, hyps, err_hyp, err_hyp_test):
        # X_density is how representative points are - dont want to select outliers
        # interpretability is how easy it is for user to make sense of explanation

        err = np.empty((len(self.classes), X.shape[0]))
        for cc in self.classes:
            # this is eqivalent to looping over h and x
            # comes from separating P(h|(A U x)) into P(h|A)P(h|x)
            err[cc, :] = -np.dot(self.cur_post[cc, :]*err_hyp[cc], likelihood[cc])
            # TODO should interpretability be per class or just for GT?
            err[cc, :] = err[cc, :]*X_density*interpretability

        if len(self.classes) > 2:
            err = err.sum(0)  # could try other methods for combining, min, max, ...
        selected_ind = self.unseen_exs[np.argmax(err[self.unseen_exs])]

        # update the posterior with the selected example
        for cc in self.classes:
            #self.cur_post[cc, :] *= likelihood[cc][:, selected_ind]
            self.cur_post[cc, :] *= likelihood[cc][:, selected_ind]*X_density[selected_ind]*interpretability[selected_ind]

        # get predictions for each hyp for selected example
        pred = np.zeros((len(self.classes), len(hyps)))
        for cc in self.classes:
            Y_bin = int(Y[selected_ind] == cc)
            for hh in range(len(hyps)):
                pred[cc, hh], _ = user_model_binary(hyps[hh], X[selected_ind,:][np.newaxis, ...], Y_bin, self.alpha)

        # bookkeeping and compute stats
        print(len(self.teaching_exs), '\t', Y[selected_ind], '\t', selected_ind, '\t', round(err[self.unseen_exs].max(),4))
        ee, ee_test, ent, diff = teaching_stats_one_vs_all(self.cur_post, pred, err_hyp, err_hyp_test)
        self.exp_err.append(ee)
        self.exp_err_test.append(ee_test)
        self.hyp_entropy.append(ent)
        self.difficulty.append(diff)
        self.teaching_exs.append(selected_ind)
        self.unseen_exs = np.setdiff1d(np.arange(X.shape[0]), self.teaching_exs)


class RandomImageTeacher:
    # assumes CxD hypotheses

    def __init__(self, dataset, alpha, prior_h):
        self.initialize(alpha, prior_h)

    def initialize(self, alpha, prior_h):
        self.teaching_exs = []
        self.alpha = alpha
        self.cur_post = prior_h.copy()
        self.exp_err = []
        self.exp_err_test = []
        self.hyp_entropy = []
        self.difficulty = []

    def posterior(self):
        return self.cur_post/self.cur_post.sum()

    def run_teaching(self, num_teaching_itrs, dataset, likelihood, hyps, err_hyp, err_hyp_test):
        X = dataset['X']
        Y = dataset['Y']
        self.teaching_exs = np.random.choice(X.shape[0], num_teaching_itrs, replace=False)

        for teaching_ex in self.teaching_exs:

            # compute the posterior of the selected example
            pred = np.zeros(len(hyps))
            for hh in range(len(hyps)):
                pred[hh], pred_class = user_model(hyps[hh], X[teaching_ex,:][np.newaxis, ...], Y[teaching_ex], self.alpha)
                if pred_class != Y[teaching_ex]:
                    self.cur_post[hh] *= pred[hh]

            # bookkeeping and compute stats
            ee, ee_test, ent, diff = teaching_stats(self.cur_post, pred, err_hyp, err_hyp_test)
            self.exp_err.append(ee)
            self.exp_err_test.append(ee_test)
            self.hyp_entropy.append(ent)
            self.difficulty.append(diff)


class RandomImageTeacherOneVsAll:
    # assumes 1xD hypotheses

    def __init__(self, dataset, alpha, prior_h):
        self.initialize(dataset['X'], dataset['Y'], alpha, prior_h)

    def initialize(self, X, Y, alpha, prior_h):
        self.teaching_exs = []
        self.unseen_exs = np.arange(X.shape[0])
        self.alpha = alpha
        self.classes = np.unique(Y)
        self.prior_h = np.tile(prior_h, (len(self.classes), 1))
        self.cur_post = np.tile(prior_h.copy(), (len(self.classes), 1))
        self.exp_err = []
        self.exp_err_test = []
        self.hyp_entropy = []
        self.difficulty = []

    def posterior(self):
        return self.cur_post/self.cur_post.sum(1)[..., np.newaxis]

    def run_teaching(self, num_teaching_itrs, dataset, likelihood, hyps, err_hyp, err_hyp_test):
        for tt in range(num_teaching_itrs):
            self.teaching_iteration(dataset['X'], dataset['Y'], likelihood, hyps, err_hyp, err_hyp_test)

    def teaching_iteration(self, X, Y, likelihood, hyps, err_hyp, err_hyp_test):

        selected_ind = np.random.choice(self.unseen_exs)

        # update the posterior with the selected example
        for cc in self.classes:
            self.cur_post[cc, :] *= likelihood[cc][:, selected_ind]

        # get predictions for each hyp for selected example
        pred = np.zeros((len(self.classes), len(hyps)))
        for cc in self.classes:
            Y_bin = int(Y[selected_ind] == cc)
            for hh in range(len(hyps)):
                pred[cc, hh], _ = user_model_binary(hyps[hh], X[selected_ind,:][np.newaxis, ...], Y_bin, self.alpha)

        # bookkeeping and compute stats
        print(len(self.teaching_exs), '\t', Y[selected_ind], '\t', selected_ind)
        ee, ee_test, ent, diff = teaching_stats_one_vs_all(self.cur_post, pred, err_hyp, err_hyp_test)
        self.exp_err.append(ee)
        self.exp_err_test.append(ee_test)
        self.hyp_entropy.append(ent)
        self.difficulty.append(diff)
        self.teaching_exs.append(selected_ind)
        self.unseen_exs = np.setdiff1d(np.arange(X.shape[0]), self.teaching_exs)
