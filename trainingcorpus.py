import re
import utils
import email
import copy
import math
from email.parser import Parser
from corpus import Corpus
from collections import Counter


class TrainingCorpus:
    """
    Class for computing statistics from dataset and evaluating the new email based on the statistics,
    """
    def __init__(self, min_word_length=2, max_word_length=100, spam_tag="SPAM", ham_tag="OK"):
        """
        :param min_word_length: minimal word length for detecting feature words
        :param max_word_length: maximal word length for detecting feature words
        :param spam_tag: spam tag presented in classification
        :param ham_tag: ham tag presented in classification
        """
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.spam_tag = spam_tag
        self.ham_tag = ham_tag
        # stop words in English language in unicode format: don't consider them are as the feature ones in email
        self.stop_words = (u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she',
                           u"she's", u'her', u'hers', u'herself', u'it', u"it's", u'its', u'itself', u'they', u'them',
                           u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this',
                           u'that', u"that'll", u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be',
                           u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing',
                           u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until',
                           u'while', u'of', u'at', u'some')
        # data corpus of emails with !truth.txt file
        self.data_corpus = Corpus("")
        # smoothing constant for Laplace shifting
        self.smoothing_constant = 1
        # emails' labels read from !truth.txt file
        self.emails_labels = {self.spam_tag: [], self.ham_tag: []}
        # likelihoods of spamness and hamness for feature words
        self.classes_likelihoods = {self.spam_tag: Counter(), self.ham_tag: Counter()}
        # total number of specific word appears in spam and ham emails
        self.vocabulary = {self.spam_tag: Counter(), self.ham_tag: Counter()}
        # number of emails where the specific word appears, keys are ham or spam emails
        self.words_per_email = {self.spam_tag: Counter(), self.ham_tag: Counter()}

    def compute_statistics(self, path_to_data_corpus):
        # computes the statistics for data corpus
        """
        :param path_to_data_corpus: path to emails with !truth.txt file
        :return:
        """
        # update data corpus path
        self.data_corpus.emails_directory = path_to_data_corpus
        # update emails labels from !truth.txt file
        self.update_emails_classifications()
        # compute the vocabulary: number of feature words per email, feature words frequency relatively to all feature
        # words in every class
        self.update_vocabulary(emails_labels=self.emails_labels)
        # compute bayess likelihoods for every feature word
        self.compute_bayess_likelihoods(emails_labels=self.emails_labels)

    def hams(self, emails_labels):
        # yields the hams emails
        """
        :param emails_labels: dictionary {spam_tag: [email_1, ... email_i], ham_tag: [email_1, ... email_j]}
        :return:
        """
        for email_name in emails_labels[self.ham_tag]:
            # get full path to file from utils module
            path_to_file = utils.get_full_path_to_file(path_to_dir=self.data_corpus.emails_directory,
                                                       file_name=email_name)
            yield [email_name, utils.read_email_content(path_to_email=path_to_file)]

    def spams(self, emails_labels):
        # yields the spams emails
        """
        :param emails_labels: dictionary {spam_tag: email_1, ... email_i, ham_tag: [email_1, ... email_j]}
        :return:
        """
        for email_name in emails_labels[self.spam_tag]:
            # get full path to file from utils module
            path_to_file = utils.get_full_path_to_file(path_to_dir=self.data_corpus.emails_directory,
                                                       file_name=email_name)
            yield [email_name, utils.read_email_content(path_to_email=path_to_file)]

    def get_feature_words_from_file(self, file_content):
        # computes the feature words from file content
        """
        :param file_content: file content as string
        :return: Counter() with unique words number
        """
        # get header of email
        headers = Parser().parsestr(file_content)
        # get main content of email
        email_main_content = "".join([part for part in headers]) if headers.is_multipart() else headers.get_payload()
        # get all urls in file
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', email_main_content)
        # get all emails in file
        emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", email_main_content)
        # get the content types in email format
        content_types = [content.get_content_type() for content in email.message_from_string(file_content).walk()]
        # get all unique words: remove all symbols that are not in English alphabet with empty symbol, convert
        # all symbols to lowercase, separate them into words by empty symbol if the length of the word in the specific
        # range [min_length, max_length] and if this word is also not in stop words
        unique_words = [word for word in re.sub(r"[^a-zA-Z]", " ", email_main_content.lower()).split()
                        if self.max_word_length >= len(word) >= self.min_word_length
                        and word not in self.stop_words]
        # unite all lists and convert to Counter()
        return Counter(unique_words + content_types + emails + urls)

    def prob_from_likelihoods(self, email_class_unique_words, class_tag, class_probability):
        # computes the probability for the class from founded unique words in the email, which also presented in
        # likelihood class
        """
        :param email_class_unique_words:  [word_1, ... word_n]
        :param class_tag:  spam_tag or ham_tag
        :param class_probability: probability of class
        :return: predicted probability of email being to this class
        """
        # p_class * p_1(word)*...p_n(word)
        total_probability = math.log10(class_probability)
        for word in email_class_unique_words:
            # instead of p_1*p_2*..._p_n use the log10 prob as log10(p1)+log10(p2)+...+log(pn) to prevent
            # floating-point underflow
            total_probability += math.log10(self.classes_likelihoods[class_tag][word])
        return total_probability

    def predict(self, file_content, spam_probability=0.5, ham_probability=0.5):
        # makes the prediction to which class the file belongs
        """
        :param file_content: file content as string
        :param spam_probability: probability of spam
        :param ham_probability:  probability of ham
        :return: spam_tag or ham_tag
        """
        # get unique words from email
        unique_words = self.get_feature_words_from_file(file_content=file_content)
        # intersect unique email's words and classes' likelihoods unique words
        spam_words = unique_words.keys() & self.classes_likelihoods[self.spam_tag].keys()
        ham_words = unique_words.keys() & self.classes_likelihoods[self.ham_tag].keys()
        # calculate spam and ham probability
        spam_probability = self.prob_from_likelihoods(email_class_unique_words=spam_words, class_tag=self.spam_tag,
                                                      class_probability=spam_probability)
        ham_probability = self.prob_from_likelihoods(email_class_unique_words=ham_words, class_tag=self.ham_tag,
                                                     class_probability=ham_probability)
        # take argmax of probabilities
        return self.ham_tag if spam_probability < ham_probability else self.spam_tag

    def get_classes_probabilities(self):
        # computes the classes probabilities from training dataset()
        """
        :return: spam probability, ham probability
        """
        # use smoothing constant to prevent zero division, the default value is 1
        total_emails = len(self.emails_labels[self.spam_tag] + len(self.emails_labels[self.ham_tag])
                           + self.smoothing_constant * 2)
        spam_prob = (len(self.emails_labels[self.spam_tag]) + self.smoothing_constant) / total_emails
        ham_prob = 1.0 - spam_prob
        return spam_prob, ham_prob

    def update_emails_classifications(self):
        # updates emails labels from data corpus directory
        """
        :return:
        """
        # get path to !truth.txt file
        full_path_to_truth = utils.get_full_path_to_file(path_to_dir=self.data_corpus.emails_directory,
                                                         file_name="!truth.txt")
        # read classification
        emails_classification = utils.read_classification_from_file(path_to_file=full_path_to_truth)
        # save classification as  dictionary {SPAM: [email_1, ..., email_j], OK: [email_1, ... email_k]}
        for email_name, class_value in emails_classification.items():
            try:  # to prevent incorrect classes key
                self.emails_labels[class_value].append(email_name)
            except KeyError:
                continue

    def update_vocabulary(self, emails_labels):
        # update the vocabularies values
        """
        :param emails_labels: dictionary {SPAM: [email_1, ..., email_j], OK: [email_1, ... email_k]}
        :return:
        """

        for _, ham_content in self.hams(emails_labels=emails_labels):
            unique_words = self.get_feature_words_from_file(ham_content)
            # number of times word appears during learning
            self.vocabulary[self.ham_tag].update(unique_words)
            # number of emails where words appear
            self.words_per_email[self.ham_tag].update(Counter(dict.fromkeys(unique_words, 1)))

        # same computations for spam emails
        for _, spam_content in self.spams(emails_labels=emails_labels):
            unique_words = self.get_feature_words_from_file(spam_content)
            self.vocabulary[self.spam_tag].update(unique_words)
            self.words_per_email[self.spam_tag].update(Counter(dict.fromkeys(unique_words, 1)))

        print("HAMs: {}, SPAMs: {}, Total: {}".format(len(emails_labels[self.ham_tag]),
                                                      len(emails_labels[self.spam_tag]),
                                                      len(emails_labels[self.ham_tag]) +
                                                      len(emails_labels[self.spam_tag])))

    def compute_bayess_likelihoods(self, emails_labels={}):
        """
        :param emails_labels: dictionary {SPAM: [email_1, ..., email_j], OK: [email_1, ... email_k]}
        :return:
        """
        # to prevent negative values and zero of smoothing constant, apply laplace smoothing
        self.smoothing_constant = self.smoothing_constant if self.smoothing_constant >= 1 else 1
        self.laplace_smoothing(smoothing_constant=self.smoothing_constant)
        # copy likelihoods from vocabulary
        self.classes_likelihoods = copy.deepcopy(self.vocabulary)
        # compute statistics with smoothing constant
        total_ham_words = sum(self.vocabulary[self.ham_tag].values())
        total_spam_words = sum(self.vocabulary[self.spam_tag].values())
        # after Laplace smoothing the number of keys in Counters() spam and ham are same
        total_unique_words = len(self.vocabulary[self.ham_tag])
        # we suppose that there was smoothing constant number of extra emails for every class
        total_ham_emails = len(emails_labels[self.ham_tag]) + self.smoothing_constant
        total_spam_emails = len(emails_labels[self.spam_tag]) + self.smoothing_constant

        for spam_word, ham_word in zip(self.vocabulary[self.spam_tag], self.vocabulary[self.ham_tag]):
            # p_spam = p1*p2 # spamness of word
            # p1 - frequency of word relatively to spam words and unique words:
            # how many times words was met in spam emails/(total unique words in spam emails + total unique words)
            # p2 - in how many spam emails the word was met/ total spam emails
            self.classes_likelihoods[self.spam_tag][spam_word] = (self.vocabulary[self.spam_tag][spam_word]) / \
                                                                 (total_spam_words + total_unique_words)

            self.classes_likelihoods[self.spam_tag][spam_word] *= \
                (self.words_per_email[self.spam_tag][spam_word]) / total_spam_emails
            # same for ham emails, hamness of word
            self.classes_likelihoods[self.ham_tag][ham_word] = (self.vocabulary[self.ham_tag][ham_word]) / \
                                                               (total_ham_words + total_unique_words)

            self.classes_likelihoods[self.ham_tag][ham_word] *= \
                (self.words_per_email[self.ham_tag][ham_word]) / total_ham_emails

    def laplace_smoothing(self, smoothing_constant=1):
        # shift all emails by smoothing constant: we suppose that there was smoothing constant number of spam
        # emails where the only ham words was met smoothing constant number of times to prevent zero division
        # same for ham emalis
        """
        :param smoothing_constant: int number >= 1
        :return:
        """
        # get only ham and spam words
        only_spam_words = Counter(
            dict.fromkeys(Counter(
                self.vocabulary[self.spam_tag].keys() - self.vocabulary[self.ham_tag].keys()), smoothing_constant))
        only_ham_words = Counter(
            dict.fromkeys(Counter(
                self.vocabulary[self.ham_tag].keys() - self.vocabulary[self.spam_tag].keys()), smoothing_constant))
        # for example for smoothing constant 1(default): the only ham word was met once in spam emails + 1
        self.vocabulary[self.spam_tag] += Counter(
            dict.fromkeys(self.vocabulary[self.spam_tag], smoothing_constant)) | only_ham_words
        self.vocabulary[self.ham_tag] += Counter(
            dict.fromkeys(self.vocabulary[self.ham_tag], smoothing_constant)) | only_spam_words
        # same for words per email
        self.words_per_email[self.spam_tag] += Counter(
            dict.fromkeys(self.words_per_email[self.spam_tag], smoothing_constant)) | only_ham_words
        self.words_per_email[self.ham_tag] += Counter(
            dict.fromkeys(self.words_per_email[self.ham_tag], smoothing_constant)) | only_spam_words
