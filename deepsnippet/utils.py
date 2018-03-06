"""
Defines various functions for processing the data.
"""
import logging
import re
from pathlib import PurePath

import numpy as np

from .char_map import char_map, index_map

RANDOM_SEED = 123

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
NOISE_TYPES = ['pink_noise',
               'running_tap',
               'dude_miaowing',
               'doing_the_dishes',
               'exercise_bike',
               'white_noise']

logger = logging.getLogger(__name__)


def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    text = []
    for c in int_sequence:
        ch = index_map[c]
        text.append(ch)
    return text


def to_label(path):
    label = np.zeros((len(LABELS)))

    word = to_word(path)

    label[to_label_index(word)] = 1

    return label


def to_word(path):
    if isinstance(path, PurePath):
        word = path.parent.name

    else:
        path_str = str(path)

        pattern = re.compile(".*/(_?\w+_?)/[^/]*\.wav$")
        match = pattern.match(path_str)

        if not match:
            raise Exception("invalid data path: {}".format(path))

        word = match.group(1)

    return word


def to_label_index(word):
    if word == "_background_noise_":
        return LABELS.index("silence")
    try:
        return LABELS.index(word)
    except ValueError:
        return LABELS.index("unknown")


def extract_hash(path):
    if isinstance(path, PurePath):
        file_name = path.name
    else:
        file_name = np.os.path.split(path)[-1]

    hash_pattern = re.compile(r"(.*)_nohash_.*$")
    matched = hash_pattern.match(file_name)
    if matched:
        return matched.group(1)
    noise_pattern = re.compile(r"^((?:"
                               + "|".join(NOISE_TYPES)
                               + r")_sampled_\d+).*$")
    matched = noise_pattern.match(file_name)
    if matched:
        return matched.group(1)
    raise Exception("invalid file name {}".format(file_name))


param_dicts = {}

# TODO extension of sklearn grid search to record all of the
# TODO extension of imbalanced learn for parameter
# TODO extension of
