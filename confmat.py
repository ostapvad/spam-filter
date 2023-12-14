class BinaryConfusionMatrix:
    """
        class representing the binary confusion matrix
    """

    def __init__(self, pos_tag, neg_tag):
        """
        :param pos_tag: positive tag
        :param neg_tag: negative tag
        """
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag
        self.tp = self.tn = self.fp = self.fn = 0

    def as_dict(self):
        # converts the binary confusion matrix values to dictionary
        """
        :return: dictionary of true positives, true negatives, false positives and false negatives values
        as dictionary keys
        """
        return {'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn}

    def update(self, truth, predicted):
        # update the binary confusion matrix values from truth  and predicted values
        """
        :param truth: truth value
        :param predicted: predicted value
        :return:
        """
        # checks if truth and predicted represent one of attribute tags
        try:
            if not (self.pos_tag == truth or self.neg_tag == truth) or \
                    not (self.pos_tag == predicted or self.neg_tag == predicted):
                raise ValueError

            if truth == predicted:
                if predicted == self.pos_tag:
                    self.tp += 1
                else:
                    self.tn += 1
            else:
                if truth == self.neg_tag:  # why
                    self.fp += 1
                else:
                    self.fn += 1
        except ValueError:
            print("Incorrect values for binary confusion matrix!")
            self.tp = self.tn = self.fp = self.fn = 0

    def compute_from_dicts(self, truth_dict, pred_dict):
        # updates the BCM from dictionaries
        """
        :param truth_dict: the dictionary of correct emails classifications
        :param pred_dict: the dictionary of predicted emails classifications
        :return:
        """
        for truth_key in truth_dict.keys():
            try:
                self.update(truth_dict[truth_key], pred_dict[truth_key])
            except KeyError:
                print("Prediction for key {} doesn't exist!".format(truth_key))
                continue
