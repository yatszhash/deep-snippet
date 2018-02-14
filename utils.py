"""
Defines various functions for processing the data.
"""
import json
import logging
import re
from pathlib import PurePath
from uuid import uuid4

import numpy as np
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.utils import resample

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


class RandomizedResamplingCVSearcher(object):
    def __init__(self, create_model_fn, compile_fn, score_fn, params_seed, n_iter, exp_root, n_fold=5):
        self.param_dicts = {}
        self.eval_results = {}
        self.params_sampler = ParameterSampler(params_seed, n_iter=n_iter)
        self.score_fn = score_fn
        self.create_model_fn = create_model_fn
        self.compile_fn = compile_fn

        self.exp_root = exp_root
        self.n_fold = n_fold

    def fit(self, X, Y, groups):
        for params in self.params_sampler:
            cv = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=RANDOM_SEED)
            param_id = str(uuid4())
            param_dicts[param_id] = params
            self.eval_results[param_id] = {"params": params}

            param_save_dir = self.exp_root.joinpath(param_id)
            param_save_dir.mkdir(exist_ok=True, parents=True)

            param_save_path = param_save_dir.joinpath("param.json")

            with param_save_path.open(encoding="utf-8", mode="w") as f:
                json.dump(params, fp=f)

            logger.info("[Params #{}]:\n{}".format(param_id, params))
            for n, (train_idx, test_idx) in enumerate(cv.split(X, Y, groups)):
                logger.info("[CV{}] training...".format(n))

                cv_save_dir = param_save_dir.joinpath("cv{}".format(n))
                cv_save_dir.mkdir(exist_ok=True, parents=True)

                cv_id = "cv{}".format(n)
                self.eval_results[param_id][cv_id] = {}
                wrapped_model = (WideResnetBuilder,
                                 log_filename=train_csv_path, save_model_path=model_path, params=params)
                model_path = cv_save_dir.joinpath("model.hdf5")

                self.eval_results[param_id][cv_id]["model_path"] = str(model_path)
                train_csv_path = cv_save_dir.joinpath("train_log.csv")

                cv_train_X, cv_valid_X = X[train_idx], X[test_idx]
                cv_train_Y, cv_valid_Y = Y[train_idx], Y[test_idx]

                cv_train_X, cv_train_Y = resample(cv_train_X, cv_train_Y, random_state=RANDOM_SEED)
                history, _ = wrapped_model.fit(cv_train_X, cv_train_Y, cv_valid_X, cv_valid_Y)

                self.eval_results[param_id][cv_id]["history"] = history
                self.eval_results[param_id][cv_id]["best_train_loss"] = np.max(history.history['train_loss'])
                self.eval_results[param_id][cv_id]["best_val_loss"] = np.max(history.history['val_loss'])

                wrapped_model.load(model_path)

                cv_train_Y_pred = wrapped_model.predict(cv_train_Y)
                train_score = self.score_fn(cv_train_Y, cv_train_Y_pred)
                self.eval_results[param_id][cv_id]["train_score"] = train_score

                cv_valid_Y_pred = wrapped_model.predict(cv_valid_Y)
                valid_score = self.score_fn(cv_valid_Y, cv_valid_Y_pred)
                self.eval_results[param_id][cv_id]["val_score"] = valid_score

            train_scores = [cv_result["train_score"] for _, cv_result in self.eval_results[param_id].items()]
            self.eval_results[param_id]["train_score_avg"] = np.mean(train_scores)
            self.eval_results[param_id]["train_score_std"] = np.std(train_scores)

            valid_scores = [cv_result["val_score"] for _, cv_result in self.eval_results[param_id].items()]
            self.eval_results[param_id]["val_score_avg"] = np.mean(valid_scores)
            self.eval_results[param_id]["val_score_std"] = np.std(valid_scores)

            result_save_path = self.exp_root.joinpath("result.json")

            with result_save_path.open(encoding="utf-8", mode="w+") as f:
                json.dump(self.eval_results, fp=f)

    def score(self, Y, Y_pred):
        return self.score_fn(Y, Y_pred)

# TODO extension of sklearn grid search to record all of the
# TODO extension of imbalanced learn for parameter
# TODO extension of
