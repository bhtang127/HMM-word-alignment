import numpy as np
from collections import defaultdict
from tqdm import tqdm
import sys
import logging
import argparse


def init_logging(level: str = 'INFO') -> None:
    logging.basicConfig(
        level=logging.getLevelName(level),
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%I:%M:%S'
    )

    
init_logging()
LOG = logging.getLogger()


def load_data(f_data, e_data, n):
    """load top n lines of data from file names"""
    with open(f_data) as f, open(e_data) as e:
        data_iter = list(zip(f, e))[:n]
        bitext = [[sentence.strip().split() for sentence in pair] for pair in data_iter]
    return bitext


def print_alignment(align):
    for a_n in align:
        for pair in a_n:
            sys.stdout.write("{}-{} ".format(*pair))
        sys.stdout.write("\n")


class IBM1(object):
    """Class that implemented IBM model 1"""
    def __init__(
        self,
        bitext,
        rev_text=False,
        rev_sentence=False
    ):
        """bitext for french and english
           theta for p(f | e)
        """
        self._bitext = bitext
        self._rev_text = False
        self._rev_sentence = False
        if rev_text:
            self.reverse_text()
        if rev_sentence:
            self.reverse_sentence()
        self._theta = self.init_prob(self._bitext)
    def reverse_text(self):
        LOG.info("Start Reversing sentence pairs")
        self._bitext = [[e, f] for f, e in self._bitext]
        self._rev_text = True
    def reverse_sentence(self):
        LOG.info("Start Reversing each sentence")
        self._bitext = [[f[::-1], e[::-1]] for f, e in self._bitext]
        self._rev_sentence = True
    @staticmethod
    def init_prob(bitext):
        """A heuristic intialization of alignment probability p(f|e)
           from counting co-occurrence
        """
        fe_count = defaultdict(int)
        e_count = defaultdict(int)
        prob = dict()
        LOG.info("Start initializing p(f|e) with sentence pairs")
        pbar = tqdm(total=len(bitext), position=0,
                    desc="Processing sentences")
        for n, (f, e) in enumerate(bitext):
            for fi in f:
                for ej in e + ["N-U-L-L"]:
                    fe_count[(fi, ej)] += 1
                    e_count[ej] += 1
            pbar.update()
        for (fi, ej), count in fe_count.items():
            prob[(fi, ej)] = count / e_count[ej]
        return prob
    def update_parameter(self, bitext):
        fe_count = defaultdict(float)
        e_count = defaultdict(float)
        pbar = tqdm(total=len(bitext), position=0,
                    desc="Processing sentences")
        for n, (f, e) in enumerate(self._bitext):
            for fi in f:
                Z = 0
                for ej in e:
                    Z += self._theta[(fi, ej)]
                for ej in e:
                    c = self._theta[(fi, ej)] / Z
                    fe_count[(fi, ej)] += c
                    e_count[ej] += c
            pbar.update()
        for (fi, ej), c in fe_count.items():
            self._theta[(fi, ej)] = c / e_count[ej]
    def train(self, max_iter=5):
        LOG.info("Training IBM model 1 with EM:")
        for i in range(max_iter):
            LOG.info("Start iteration {}:".format(i + 1))
            self.update_parameter(self._bitext)
    def predict(self, top_n=1):
        """Return top n pairs for each word
           if rev_text is true, pair (i, j) will be returned as (j, i)
           if rec_sentence is true, pair (i, j) will be returned as (I-i-1, J-j-1)
        """
        LOG.info("Generating alignment:")
        pbar = tqdm(total=len(self._bitext), position=0,
                    desc="Processing sentences")
        alignment = []
        for f, e in self._bitext:
            align_n = []
            for i, f_i in enumerate(f): 
                probs = []
                for j, e_j in enumerate(e):
                    probs.append((i, j, self._theta[(f_i, e_j)]))
                probs = sorted(probs, key=lambda x: x[-1], reverse=True)
                for k in range(min(top_n, j + 1)):
                    align_n.append(self.align_pair(probs[k][0], probs[k][1], len(f), len(e)))
            alignment.append(align_n)
            pbar.update()
        return alignment
    def align_pair(self, j, i, J, I):
        pair = (j, i)
        if self._rev_sentence:
            pair = (J - pair[0] - 1, I - pair[1] - 1)
        if self._rev_text:
            pair = (pair[1], pair[0])
        return pair


class HMM(object):
    """class that implemented HMM model with group null words"""
    def __init__(
        self,
        bitext,
        theta,
        eps=0.2,
        smooth=0.,
        rev_text=False,
        rev_sentence=False
    ):
        """Args:
              bitext: french and english sentences
              theta: initial value for p(f | e)
              eps: probability of jumping to null word
              smooth: smooth parameter of transition probability
              rev_text: whether to transpose French and English 
              rev_sentence: whether to reverse each sentence
        """
        self._bitext = bitext
        self._theta = {key:value for key, value in theta.items()}
        self._eps = eps
        self._smooth = smooth
        self._rev_text = False
        self._rev_sentence = False
        self.init_transition()
        if rev_text:
            self.reverse_text()
        if rev_sentence:
            self.reverse_sentence()
    def reverse_text(self):
        LOG.info("Start Reversing sentence pairs")
        self._bitext = [[e, f] for f, e in self._bitext]
        self._rev_text = True
    def reverse_sentence(self):
        LOG.info("Start Reversing each sentence")
        self._bitext = [[f[::-1], e[::-1]] for f, e in self._bitext]
        self._rev_sentence = True
    def init_transition(self):
        """p(i | i', I) \propto (1-eps)c(i-i') for I > i > 0 and eps for i = 0
           we bucket c(i-i') into bins c(<=-7), c(-6), c(-5), ... c(6), c(>=7)
           They will be initialized as 1/15
        """
        self._c = np.ones(15) / 15
    def get_bucket(self, delta):
        """for a array delta_i
           return c(delta_i)
        """
        delta = np.array(delta, dtype="int")
        delta[delta > 7] = 7
        delta[delta < -7] = -7
        return self._c[delta + 7]
    def get_transition_prob(self, I):
        """return matrix P 
           where P(i', i) = transition prob p(i | i', I)
           i, i' in [0, I]
           also, group of I empty words will be introduced
           therefore P(i' + I, i) = P(i', i)
           P(i' + I, i + I) = eps * 1(i' = i)
           P(i', i + I) = eps * 1(i' = i)
        """
        prob = np.zeros([2 * I, 2 * I])
        prob[:I, I:] = self._eps * np.eye(I)
        prob[I:, I:] = self._eps * np.eye(I)
        for i in range(I):
            delta = np.array([k - i for k in range(I)])
            cd = self.get_bucket(delta)
            prob[i, :I] = (1 - self._eps) * ((1 - self._smooth) * cd / np.sum(cd) +\
                                             self._smooth / I)
        prob[I:, :I] = prob[:I, :I]
        return prob
    def get_represent_prob(self, f, e):
        """get matrix of prob P
           P(i, j) = p(fj | ei)
           I empty words N-U-L-L will be introduced at e_i+I
        """
        null_words = ["N-U-L-L" for ei in e]
        null_e = e + null_words
        return np.array([[self._theta[(fj, ei)] for fj in f] for ei in null_e])
    def get_hmm_items(self, f, e):
        """get the items in HMM likelihood production
           which is H(i', i, j) = p(aj = i | aj-1 = i', I) p(fj | ei)
        """
        trans_prob = self.get_transition_prob(len(e))
        p_fj_ei = self.get_represent_prob(f, e)
        return np.einsum("ij, jk -> ijk", trans_prob, p_fj_ei)
    def fwd_bkw(self, f, e):
        """forward backward algorithm to calculate
           mui(j, i) = p(aj = i | f, e) and
           tao(k) = sum_i p(aj = i, aj+1 = i+k | f, e)
           this includes i -> i+k (<I) and i+I -> i+k (<I)
           and then k in [-I+1, I-1]
        """
        I, J = len(e), len(f)
        hmm_items = self.get_hmm_items(f, e)
        alpha = np.zeros([J, 2 * I])
        alpha[0, :] = hmm_items[0, :, 0]
        for j in range(1, J):
            alpha[j, :] = alpha[j - 1, :].dot(hmm_items[:, :, j])
        beta = np.zeros([J, 2 * I])
        beta[-1, :] = 1.
        for j in range(J - 2, -1, -1):
            beta[j, :] = beta[j + 1, :].dot(hmm_items[:, :, j + 1].T)
        Z = np.sum(alpha[-1, :])
        # math precision issue, return an average case
        if Z == 0:
            return np.ones([J, 2 * I]) / (2 * I), {0: 0.}
        Z_log = np.log(Z)
        with np.errstate(divide='ignore'):
            mui = np.exp(np.log(alpha) + np.log(beta) - Z_log)
        if J == 1:
            tao = {0: 0.}
        else:
            tao = {}
            for d in range(-I + 1, I):
                lb, ub = max(-d, 0), min(I - d, I)
                tao[d] = np.sum(alpha[:-1, lb:ub] * beta[1:, (lb + d):(ub + d)] * \
                                np.diagonal(hmm_items[:I, :I, 1:], d, axis1=0, axis2=1))
                tao[d] += np.sum(alpha[:-1, (lb + I):(ub + I)] * beta[1:, (lb + d):(ub + d)] * \
                                 np.diagonal(hmm_items[I:, :I, 1:], d, axis1=0, axis2=1))
                if tao[d] > 0:
                    tao[d] = np.exp(np.log(tao[d]) - Z_log)
        return mui, tao
    def update_parameter(self, bitext):
        fe_count = defaultdict(float)
        e_count = defaultdict(float)
        c_count = np.zeros(15)
        pbar = tqdm(total=len(bitext), position=0,
                    desc="Processing sentences")
        for n, (f, e) in enumerate(bitext):
            mui, tao = self.fwd_bkw(f, e)
            up_count, low_count = 0, 0
            for d, count in tao.items():
                if d >= 7:
                    up_count += 1
                    c_count[14] += count
                elif d <= -7:
                    low_count += 1
                    c_count[0] += count
                else:
                    c_count[d + 7] += count
            if up_count > 0:
                c_count[14] /= up_count
            if low_count > 0:
                c_count[0] /= low_count
            for j, fj in enumerate(f):
                null_count = np.sum(mui[j, len(e):])
                fe_count[(fj, "N-U-L-L")] += null_count
                e_count["N-U-L-L"] += null_count
                for i, ei in enumerate(e):
                    fe_count[(fj, ei)] += mui[j, i]
                    e_count[ei] += mui[j, i]
            pbar.update()
        for (f, e), count in fe_count.items():
            self._theta[(f, e)] = count / e_count[e]
        self._c = c_count / np.sum(c_count)
    def train(self, max_iter=5, train_smooth=None, eps=None):
        if train_smooth is not None:
            self._smooth = train_smooth
        if eps is not None:
            self._eps = eps
        LOG.info("Training HMM with EM:")
        for i in range(max_iter):
            LOG.info("Start iteration {}:".format(i + 1))
            self.update_parameter(self._bitext)
    def viterbi_decode(self, f, e):
        I, J = len(e), len(f)
        trans_prob = self.get_transition_prob(I)
        p_fj_ei = self.get_represent_prob(f, e)
        # max_prob[i, j] holds the max prob
        # for states a1, ..., aj-1, aj=i
        # with observation f1, ..., fj
        max_prob = np.zeros([2 * I, J])
        # max_position[i, j] holds the position aj-1
        # for maximum sequence of states a1, ..., aj-1, aj=i
        # with observation f1, ..., fj
        max_position = np.zeros([2 * I, J - 1], dtype="int")
        max_prob[:, 0] = trans_prob[0, :] * p_fj_ei[:, 0]
        for j in range(1, J):
            prod = np.einsum("k, ki, i -> ki",
                             max_prob[:, j - 1], trans_prob, p_fj_ei[:, j])
            max_prob[:, j] = np.max(prod, axis = 0)
            max_position[:, j - 1] = np.argmax(prod, axis = 0)
        i_j = np.argmax(max_prob[:, -1])
        align = []
        if i_j < I:
            align.append(self.align_pair(J - 1, i_j, J, I))
        for j in range(J - 2, -1, -1):
            i_j = max_position[i_j, j]
            if i_j < I:
                align.append(self.align_pair(j, i_j, J, I))
        return align
    def align_pair(self, j, i, J, I):
        pair = (j, i)
        if self._rev_sentence:
            pair = (J - pair[0] - 1, I - pair[1] - 1)
        if self._rev_text:
            pair = (pair[1], pair[0])
        return pair
    def predict(self, pred_smooth=None):
        """if rev_text is true, pair (i, j) will be returned as (j, i)
           if rec_sentence is true, pair (i, j) will be returned as (I-i-1, J-j-1)
        """
        if pred_smooth is not None:
            smooth_old = self._smooth
            self._smooth = pred_smooth
        LOG.info("Generate alignment with Viterbi decoding")
        pbar = tqdm(total=len(self._bitext), position=0,
                    desc="Processing sentences")
        align = []
        for n, (f, e) in enumerate(self._bitext):
            align.append(self.viterbi_decode(f, e))
            pbar.update()
        if pred_smooth is not None:
            self._smooth = smooth_old
        return align


def arg_parser(name: str = "Word Alignment") -> dict:
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument(
        "-d",
        metavar="data_prefix",
        dest="train",
        default="data/hansards",
        help="Data filename prefix (default=data/hansards)"
    )
    parser.add_argument(
        "-e",
        metavar="English_suffix",
        dest="english",
        default="e",
        help="Suffix of English filename (default=e)"
    )
    parser.add_argument(
        "-f",
        metavar="French_suffix",
        dest="french",
        default="f",
        help="Suffix of French filename (default=f)"
    )
    parser.add_argument(
        "-n",
        metavar="sentences_number",
        dest="num_sents",
        default=sys.maxsize,
        type=int,
        help="Number of sentences to use for training and alignment"
    )
    parser.add_argument(
        "-m", "--model",
        metavar="model",
        dest="model",
        default="HMM",
        help="Abbrev for models. Only support 'IBM1' for IBM model 1 and \
              'HMM' for HMM-based alignment model. HMM will be initialized using IBM1 \
              (default='HMM')"
    )
    parser.add_argument(
        "-i", "--iter",
        metavar="max_iter",
        dest="max_iter",
        type=int,
        default=5,
        help="Max iteration number for iterative models. (default=5)"
    )
    parser.add_argument(
        "-s", "--smooth",
        metavar="smooth_level",
        dest="smooth",
        type=float,
        default=0.4,
        help="Smooth level for prediction in HMM, 0.4 is a good value in practice (default=0.4)"
    )
    parser.add_argument(
        "--rt",
        action="store_true",
        dest="rev_text",
        help="Training models use transposed French and English text. \
              E-F will be trained instead of F-E if using this option"
    )
    parser.add_argument(
        "--rs",
        action="store_true",
        dest="rev_sentence",
        help="Training models while each sentence is in reverse order"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == "__main__":
    args = arg_parser()
    f_data = "{}.{}".format(args["train"], args["french"])
    e_data = "{}.{}".format(args["train"], args["english"])
    bitext = load_data(f_data, e_data, args["num_sents"])
    if args["model"].lower() == "ibm1":
        LOG.info("Start word alignment with IBM model 1:")
        ibm1 = IBM1(bitext, rev_text=args["rev_text"], rev_sentence=args["rev_sentence"])
        ibm1.train(args["max_iter"])
        align_ibm1 = ibm1.predict()
        print_alignment(align_ibm1)
        LOG.info("Success")
    elif args["model"].lower() == "hmm":
        LOG.info("Start HMM word alignment model with IBM model 1 initialization:")
        ibm1 = IBM1(bitext, rev_text=args["rev_text"], rev_sentence=args["rev_sentence"])
        ibm1.train(5)
        hmm = HMM(bitext, ibm1._theta, rev_text=args["rev_text"], rev_sentence=args["rev_sentence"])
        hmm.train(args["max_iter"])
        align_hmm = hmm.predict(pred_smooth=args["smooth"])
        print_alignment(align_hmm)
        LOG.info("Success")
    else:
        raise KeyError("model {} not supported yet".format(args["model"]))
    
