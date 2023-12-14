import os


def read_classification_from_file(path_to_file):
    # reads the emails' classification from file
    """
    :param path_to_file: full path to file including directory
    :return: dictionary of emails classifications or empty dictionary if path or file doesn't exist
    """
    emails_classification = {}
    if os.path.isfile(path_to_file):
        with open(path_to_file, 'r', encoding='utf-8') as file:
            for line in file:
                key, value = line.split()
                emails_classification[key] = value

    return emails_classification


def save_classification_to_file(path_to_corpus, class_tags):
    # saves the classification to !prediction.txt file
    """
    :param path_to_corpus: path to emails directory
    :param class_tags: {email1: class, ..., emailn:class} emails classification dictionary
    :return: boolean value if path exist and saving was successful
    """
    if os.path.exists(path=path_to_corpus):
        prediction_path = get_full_path_to_file(path_to_dir=path_to_corpus, file_name="!prediction.txt")
        with open(prediction_path, 'w', encoding='utf-8') as file:
            for email in class_tags:
                file.write("{} {}\n".format(email, class_tags[email]))
        return True
    return False


def read_email_content(path_to_email):
    # reads the email content into string
    """
    :param path_to_email: full path to email including directory
    :return: empty string if file doesn't exist or full file content as string
    """
    if os.path.isfile(path_to_email):
        with open(path_to_email, 'r', encoding='utf-8') as file:
            return file.read()
    return ""


def get_full_path_to_file(path_to_dir, file_name):
    # returns the path to file in the directory
    """
    :param path_to_dir: path to directory where the file is presented
    :param file_name: file name
    :return: full path to file as string
    """
    return os.path.join(path_to_dir, file_name)
