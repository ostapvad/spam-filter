import os
import utils


class Corpus:
    """
        Class for working with the directory where the dataset of emails is presented
    """

    def __init__(self, path_to_emails):
        """
        :param path_to_emails: path to corpus(emails directory)
        """
        self.emails_directory = path_to_emails

    def emails(self):
        # yields the email name and its content as list, if file or path doesn't exist returns the list of empty strings
        """
        :return: [email name, email content] or ["", ""]
        """
        try:
            if not os.path.isdir(self.emails_directory):
                raise OSError
            for filename in os.listdir(self.emails_directory):
                path_to_email = utils.get_full_path_to_file(path_to_dir=self.emails_directory, file_name=filename)
                if os.path.isfile(path_to_email) and not filename.startswith('!'):
                    yield [filename, utils.read_email_content(path_to_email=path_to_email)]
        except OSError as error:
            print("Directory or file doesn't exist {}!".format(error))
            yield ["", ""]


