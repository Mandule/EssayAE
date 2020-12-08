import os
import shutil

import requests


def upload_file(file_path, update_file_name='MF1933059.tsv'):
    """
    Update the designated file and get the kappa score. It will create new file with update_file_name at the current directory.
    :param file_path: The path of the file you want to update. (str)
    :param update_file_name: The valid file name, i.e. studentID.tsv. (str)
    :return: The returned kappa on test set. (float)
    """

    try:
        shutil.copyfile(file_path, update_file_name)
    except shutil.SameFileError:
        pass

    with open(update_file_name, 'rb') as f:
        file_dict = {'file': f}

        r = requests.post('http://47.110.235.226:12345/submission.html', files=file_dict).text

        pos = r.find('本次提交成绩为：')
        if pos == -1:
            print('File "%s" has not been evaluated correctly. Some problem happened.' % file_path)
            return file_path
        start = pos + 8
        end = r.find('<', start)

        result = r[start:end].strip()

        return eval(result)