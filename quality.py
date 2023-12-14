import confmat
import utils


def quality_score(tp, tn, fp, fn):
    # computes the filter quality q from confusing matrix values
    """
    :param tp: number of true positives
    :param tn: number of trun negatives
    :param fp: number of false positives
    :param fn: number of false negatives
    :return: filter quality score as float
    """
    return (tp + tn) / (tp + tn + 10 * fp + fn)


def compute_quality_for_corpus(corpus_dir):
    # computes the quality score for specific corpus from !truth.txt and !prediction.txt files
    """
    :param corpus_dir: path to corpus directory
    :return: filter quality score as float
    """
    cm = confmat.BinaryConfusionMatrix(pos_tag="SPAM", neg_tag="OK")
    truth_dict = utils.read_classification_from_file(utils.get_full_path_to_file(corpus_dir, '!truth.txt'))
    prediction_dict = utils.read_classification_from_file(utils.get_full_path_to_file(corpus_dir, '!prediction.txt'))
    cm.compute_from_dicts(truth_dict=truth_dict, pred_dict=prediction_dict)
    return quality_score(tp=cm.tp, tn=cm.tn, fp=cm.fp, fn=cm.fn)


