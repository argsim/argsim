from os.path import expanduser, join
from util import Record
import json
import pickle


def pform(path, *names, sep= ''):
    """formats a path as `path` followed by `names` joined with `sep`."""
    return join(expanduser(path), sep.join(map(str, names)))


def load_txt(filename, encoding= None):
    """yields lines from text file."""
    with open(filename, encoding= encoding) as file:
        yield from (line[:-1] for line in file)


def save_txt(filename, lines):
    """writes lines to text file."""
    with open(filename, 'w') as file:
        for line in lines:
            print(line, file= file)


def load_pkl(filename):
    """loads pickle file."""
    with open(filename, 'rb') as dump:
        return pickle.load(dump)


def save_pkl(filename, obj):
    "saves to pickle file."
    with open(filename, 'wb') as dump:
        pickle.dump(obj, dump)


def load_json(filename):
    with open(filename) as file:
        return json.load(file)


def clean(post):
    """cleans a post for consistency

    - normalizes extra whitespaces, including newlines, to a single space
    - lowercases everything

    """
    return " ".join(post.split()).lower()
