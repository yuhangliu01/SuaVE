import json
import os
import fcntl
import errno
import time

import numpy as np
import torch
import networkx as nx
import math

import copy
import numpy as np
import pandas as pd

def make_dir(dir_name):
    if dir_name[-1] != '/':
        dir_name += '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def make_file(file_name):
    if not os.path.exists(file_name):
        open(file_name, 'a').close()
    return file_name


def get_exp_id(log_folder):
    log_folder = make_dir(log_folder)
    helper_id_file = log_folder + '.expid'
    if not os.path.exists(helper_id_file):
        with open(helper_id_file, 'w') as f:
            f.writelines('0')
    # helper_id_file = make_file(helper_id_file)
    with open(helper_id_file, 'r+') as file:
        st = time.time()
        while time.time() - st < 30:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # a = input()
                break
            except IOError as e:
                # raise on unrelated IOErrors
                if e.errno != errno.EAGAIN:
                    raise
                else:
                    print('sleeping')
                    time.sleep(0.1)
        else:
            raise TimeoutError('Timeout on accessing log helper file {}'.format(helper_id_file))
        prev_id = int(file.readline())
        curr_id = prev_id + 1

        file.seek(0)
        file.writelines(str(curr_id))
        fcntl.flock(file, fcntl.LOCK_UN)
    return curr_id


def from_log(args, argv, logpath):
    """
    read from log, and allow change of arguments
    assumes that arguments are assigned using an = sign
    assumes that the first argument is --from-log. so argv[1] is of the form --from-log=id
    everything that comes after --from-log in sys.argv will be resolved and its value substituted for the one in the log
    """
    i = args.from_log
    d = {}
    new_d = vars(args).copy()
    args_not_from_log = []
    add_to_log = False
    if len(argv) > 2:
        add_to_log = True
    for a in argv[1:]:  # start from 2 if the from-log value is to be overwritten by the one in the log
        sp = a.split('=')
        args_not_from_log.append(sp[0][2:].replace('-', '_'))
    file = open(logpath)
    for line in file:
        d = json.loads(line)
        if d['id'] == i:
            break
    file.close()
    for a in args_not_from_log:
        d.pop(a)
    del d['id'], d['train_perf'], d['test_perf']
    new_d.update(d)
    return new_d, add_to_log


def checkpoint(path, exp_id, iteration, model, optimizer, loss, perf):
    sub_path = make_dir(path + str(exp_id) + '/')
    weights_path = sub_path + str(exp_id) + '_ckpt_' + str(iteration) + '.pth'
    print('.. checkpoint at iteration {} ..'.format(iteration))
    torch.save({'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'perf': perf},
               weights_path)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.97):
        self.momentum = momentum
        self.val = None
        self.avg = 0

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class Averager:
    def __init__(self):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.sum = 0

    def reset(self):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.sum = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class Logger:
    """A logging helper that tracks training loss and metrics."""
    def __init__(self, logdir='log/', **metadata):
        self.logdir = make_dir(logdir)
        exp_id = get_exp_id(logdir)
        self.reset()
        self.metadata = metadata
        self.exp_id = exp_id
        self.log_dict = {}
        self.running_means = {}

    def get_id(self):
        return self.exp_id

    def add(self, key):
        self.running_means.update({key: Averager()})
        self.log_dict.update({key: []})

    def update(self, key, val):
        self.running_means[key].update(val)

    def _reset_means(self):
        for key in self.keys():
            self.running_means[key].reset()

    def reset(self):
        self.log_dict = {}
        self.running_means = {}

    def log(self):
        for key in self.keys():
            self.log_dict[key].append(self.running_means[key].avg)
        self._reset_means()

    def get_last(self, key):
        return self.log_dict[key][-1]

    def save_to_npz(self, path=None):
        if path is None:
            data_path = make_dir(self.logdir + 'data/')
            path = data_path + str(self.exp_id) + '.npz'
        else:
            if path[-4:] != '.npz':
                path += '.npz'
        for k, v in self.log_dict.items():
            self.log_dict[k] = np.array(v)
        np.savez_compressed(path, **self.log_dict)
        print('Log data saved to {}'.format(path))

    def save_to_json(self, path=None, method='last'):
        if path is None:
            path = make_file(self.logdir + 'log.json')
        with open(path, 'a') as file:
            log = {'id': self.exp_id}
            for k in self.keys():
                if method == 'last':
                    log.update({k: self.get_last(k)})
                elif method == 'full':
                    log.update({k: self.log_dict[k]})
                else:
                    raise ValueError('Incorrect method {}'.format(method))
            log.update({'metadata': self.metadata})
            json.dump(log, file)
            file.write('\n')
        print('Log saved to {}'.format(path))

    def add_metadata(self, **metadata):
        self.metadata.update(metadata)

    def __len__(self):
        return len(self.log_dict)

    def __get__(self, key):
        self.get_last(key)

    def keys(self):
        return self.log_dict.keys()


def kl_per_group(kl_all):
    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.abs(kl_all)
    kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals


def kl_balancer(kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
    if kl_balance and kl_coeff < 1.0:
        alpha_i = alpha_i.unsqueeze(0)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / alpha_i * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)
    else:
        kl_all = torch.stack(kl_all, dim=1)
        kl_vals = torch.mean(kl_all, dim=0)
        kl = torch.sum(kl_all, dim=1)
        kl_coeffs = torch.ones(size=(len(kl_vals),))

    return kl_coeff * kl, kl_coeffs, kl_vals


def kl_coeff(step, total_step, constant_step, min_kl_coeff):
    return max(min((step - constant_step) / total_step, 1.0), min_kl_coeff)


def kl_balancer_coeff(num_scales, groups_per_scale, fun):
    if fun == 'equal':
        coeff = torch.cat([torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'linear':
        coeff = torch.cat([(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'sqrt':
        coeff = torch.cat([np.sqrt(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'square':
        coeff = torch.cat([np.square(2 ** i) / groups_per_scale[num_scales - i - 1] * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    else:
        raise NotImplementedError
    # convert min to 1.
    coeff /= torch.min(coeff)
    return coeff


def groups_per_scale(num_scales=1, num_groups_per_scale=5, is_adaptive=False, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g




def make_symmetric(graph):
    # only valid for tabular graph
    if any([type(p[0]) in [list, tuple] for c,p in graph.items() if len(p)>0]): # if graph is time series, then return graph
        return graph
    g = {key: [] for key in graph.keys()}
    for key in graph.keys():
        for p in graph[key]:
            if p not in g[key]:
                g[key].append(p)
            if key not in g[p]:
                g[p].append(key)
    return g






class MetricsDAG(object):
    """
    Compute various accuracy metrics for B_est.
    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    reverse = an edge estimated with reversed direction.

    fdr: (reverse + FP) / (TP + FP)
    tpr: TP/(TP + FN)
    fpr: (reverse + FP) / (TN + FP)
    shd: undirected extra + undirected missing + reverse
    nnz: TP + FP
    precision: TP/(TP + FP)
    recall: TP/(TP + FN)
    F1: 2*(recall*precision)/(recall+precision)
    gscore: max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1

    Parameters
    ----------
    B_est: np.ndarray
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    B_true: np.ndarray
        [d, d] ground truth graph, {0, 1}.
    """

    def __init__(self, B_est, B_true):
        
        if not isinstance(B_est, np.ndarray):
            raise TypeError("Input B_est is not numpy.ndarray!")

        if not isinstance(B_true, np.ndarray):
            raise TypeError("Input B_true is not numpy.ndarray!")

        self.B_est = copy.deepcopy(B_est)
        self.B_true = copy.deepcopy(B_true)

        self.metrics = MetricsDAG._count_accuracy(self.B_est, self.B_true)

    @staticmethod
    def _count_accuracy(B_est, B_true, decimal_num=4):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.
        decimal_num: int
            Result decimal numbers.

        Return
        ------
        metrics: dict
            fdr: float
                (reverse + FP) / (TP + FP)
            tpr: float
                TP/(TP + FN)
            fpr: float
                (reverse + FP) / (TN + FP)
            shd: int
                undirected extra + undirected missing + reverse
            nnz: int
                TP + FP
            precision: float
                TP/(TP + FP)
            recall: float
                TP/(TP + FN)
            F1: float
                2*(recall*precision)/(recall+precision)
            gscore: float
                max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        # trans diagonal element into 0
        for i in range(len(B_est)):
            if B_est[i, i] == 1:
                B_est[i, i] = 0
            if B_true[i, i] == 1:
                B_true[i, i] = 0

        # trans cpdag [0, 1] to [-1, 0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == B_est[j, i] == 1:
                    B_est[i, j] = -1
                    B_est[j, i] = 0
        
        if (B_est == -1).any():  # cpdag
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        else:  # dag
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            # if not is_dag(B_est):
            #     raise ValueError('B_est should be a DAG')
        d = B_true.shape[0]
        
        # linear index of nonzeros
        pred_und = np.flatnonzero(B_est == -1)
        pred = np.flatnonzero(B_est == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        # trans cpdag [-1, 0, 1] to [0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == -1:
                    B_est[i, j] = 1
                    B_est[j, i] = 1

        W_p = pd.DataFrame(B_est)
        W_true = pd.DataFrame(B_true)

        gscore = MetricsDAG._cal_gscore(W_p, W_true)
        precision, recall, F1 = MetricsDAG._cal_precision_recall(W_p, W_true)

        mt = {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 
              'precision': precision, 'recall': recall, 'F1': F1, 'gscore': gscore}
        for i in mt:
            mt[i] = round(mt[i], decimal_num)
        
        return mt

    @staticmethod
    def _cal_gscore(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.
        
        Return
        ------
        score: float
            max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """
        
        num_true = W_true.sum(axis=1).sum()
        assert num_true!=0
        
        # true_positives
        num_tp =  (W_p + W_true).map(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
        # False Positives + Reversed Edges
        num_fn_r = (W_p - W_true).map(lambda elem:1 if elem==1 else 0).sum(axis=1).sum()
        score = np.max((num_tp-num_fn_r,0))/num_true
        
        return score

    @staticmethod
    def _cal_precision_recall(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.
        
        Return
        ------
        precision: float
            TP/(TP + FP)
        recall: float
            TP/(TP + FN)
        F1: float
            2*(recall*precision)/(recall+precision)
        """

        assert(W_p.shape==W_true.shape and W_p.shape[0]==W_p.shape[1])
        TP = (W_p + W_true).map(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
        TP_FP = W_p.sum(axis=1).sum()
        TP_FN = W_true.sum(axis=1).sum()
        precision = TP/(TP_FP + 1e-7)
        recall = TP/(TP_FN + 1e-7)
        F1 = 2*(recall*precision)/(recall+precision + 1e-7)
        
        return precision, recall, F1
    




if __name__ == '__main__':
    #assign_z=np.array([2,3,1,0])
    G = (np.array([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [1, 0, 1, 0]])) #nx.DiGraph
    #weights=np.random.rand(3,6)
    #graph_accuracy(weights,assign_z,G)
    G1 = (np.array([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [1, 0, 1, 0]]))
    
    Metrics = MetricsDAG(G1, G) #correct_w
    shd = Metrics.metrics["shd"]
    f1 = Metrics.metrics["F1"]
    print(shd, f1)
