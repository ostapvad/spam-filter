import utils
from trainingcorpus import TrainingCorpus
from corpus import Corpus


class MyFilter:
    """
    Implementation of Naive Bayess filter
    """

    def __init__(self):
        # emails labels
        self.spam_tag = 'SPAM'
        self.ham_tag = 'OK'
        # trained model from training corpus
        self.trained_model = TrainingCorpus(spam_tag=self.spam_tag, ham_tag=self.ham_tag)
        # classification dictionary for emails
        self.evaluated_emails = {}

    def train(self, path_to_emails):
        # trains NB filter, computes the statistics defined in training corpus
        """
        :param path_to_emails: path to training directory with !truth.txt file
        :return:
        """
        self.trained_model.compute_statistics(path_to_data_corpus=path_to_emails)

    def test(self, path_to_emails):
        """
        :param path_to_emails: path to testing directory for emails evaluation
        :return:
        """
        corpus = Corpus(path_to_emails)
        self.evaluated_emails = {}
        for file_name, content in corpus.emails():
            # get email class
            self.evaluated_emails[file_name] = self.trained_model.predict(file_content=content)
        # save to !prediction.txt
        utils.save_classification_to_file(path_to_emails, self.evaluated_emails)
