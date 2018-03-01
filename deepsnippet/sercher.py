import inspect
import json
import logging
import pickle
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.model_selection import ParameterSampler, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)


class RandomizedResamplingCVSearcher(object):
    def __init__(self, create_model_fn, compile_fn, score_fn, fit_fn, preprocess_lists, params_seed, n_iter,
                 exp_root: Path,
                 n_fold=5, is_multilabel=False, random_seed=123):
        self.param_dicts = {}
        self.eval_results = {}
        self.failed_params = {}
        self.params_sampler = ParameterSampler(params_seed, n_iter=n_iter)
        self.score_fn = score_fn
        self.create_model_fn = create_model_fn
        self.compile_fn = compile_fn
        self.fit_fn = fit_fn
        # TODO replace with sklearn pipeline
        self.preprocess_lists = preprocess_lists

        self.exp_root = exp_root
        self.n_fold = n_fold
        self.is_multilabel = is_multilabel
        self.random_seed = random_seed
        self.model = None

    def fit(self, X, Y, groups=None, evaluate_w_best=True):
        self.X = X
        self.Y = Y
        self.evaluate_w_best = evaluate_w_best
        self.current_params = None
        self.current_param_id = None

        logger.info("*****************start parameter search*************************")
        for params in self.params_sampler:
            if self.is_multilabel and groups:
                cv = StratifiedKFold(n_splits=self.n_fold)

            elif self.is_multilabel:
                cv = KFold(n_splits=self.n_fold)

            else:
                cv = StratifiedKFold(n_splits=self.n_fold)

            if groups:
                train_set = shuffle(X, Y, groups, random_state=self.random_seed)
                cv_generator = cv.split(*train_set)
            else:
                train_set = shuffle(X, Y, random_state=self.random_seed)
                cv_generator = cv.split(*train_set)

            self.current_param_id = str(uuid4())
            self.current_params = params

            param_save_dir = self.exp_root.joinpath(self.current_param_id)
            param_save_dir.mkdir(exist_ok=True, parents=True)

            param_save_path = param_save_dir.joinpath("param.json")

            with param_save_path.open(encoding="utf-8", mode="w") as f:
                json.dump(params, fp=f)

            logger.info("===========[Params #{}]:\n{}======================".format(self.current_param_id, params))
            all_cv_result = self._fit_all_cv(cv_generator, param_save_dir)

            if all_cv_result:
                self.eval_results[self.current_param_id] = all_cv_result
                self._evaluate_param(param_save_dir)
                result_save_path = self.exp_root.joinpath("result.json")
                with result_save_path.open(encoding="utf-8", mode="w+") as f:
                    json.dump(self.eval_results, fp=f, cls=NumpyJsonEncoder)

            else:
                self.failed_params[self.current_param_id] = params
                failed_params_path = self.exp_root.joinpath("failed_params.json")
                with failed_params_path.open(encoding="utf-8", mode="w+") as f:
                    json.dump(self.failed_params, fp=f, cls=NumpyJsonEncoder)

        failed_params_path = self.exp_root.joinpath("failed_params.json")
        with failed_params_path.open(encoding="utf-8", mode="w+") as f:
            json.dump(self.failed_params, fp=f, cls=NumpyJsonEncoder)

        if not self.eval_results:
            logger.warning("There is no successful training with these params combinations.")
            return None, self.failed_params

        best_param_id, best_param_results = max(self.eval_results.items(), key=lambda x: x[1]["val_score_avg"])
        self.eval_results["best_id"] = best_param_id
        logger.info("Best parameter and its score:\n"
                    "avg train score {}, avg valid score {}\n"
                    "params #{}: {}\n".format(
            best_param_results["val_score_avg"],
            best_param_results["val_score_std"],
            best_param_id,
            best_param_results["params"]
        ))

        result_save_path = self.exp_root.joinpath("result.json")
        with result_save_path.open(encoding="utf-8", mode="w+") as f:
            json.dump(self.eval_results, fp=f, cls=NumpyJsonEncoder)

        self.load_with_id(best_param_id)
        logger.info("*****************end parameter search*************************")

        return deepcopy(self.eval_results), deepcopy(self.failed_params)

    #
    # def resume_fit(self, param_id, X, Y, groups=None, evaluate_w_best=True):
    #     param_dirs = list(self.exp_root.glob("*"))
    #     param_dirs = list(filter(lambda x: len(x.name), param_dirs))
    #
    #     if len(param_dirs) == 0:
    #         return self.fit(X, Y, groups, evaluate_w_best)
    #

    def _fit_all_cv(self, cv_generator, param_save_dir):
        all_cv_result = {"params": self.current_params, "cv_info": []}

        for n, (train_idx, test_idx) in enumerate(cv_generator):
            logger.info("-----------[CV{}] training...---------------".format(n))

            cv_save_dir = param_save_dir.joinpath("cv{}".format(n))
            cv_save_dir.mkdir(exist_ok=True, parents=True)

            cv_result = self._fit_cv(train_idx, test_idx, cv_save_dir)

            if cv_result:
                all_cv_result["cv_info"].append(cv_result)
            else:
                logger.warning("something wrong while training cv{}. Training were aborted with this params".format(n))
                return False

        return all_cv_result

    def _fit_cv(self, train_idx, test_idx, cv_save_dir):
        self.model = None

        cv_result = {}

        self.current_pipeline = self._create_preprocess_pipeline(self.current_params)
        preprocess_path = cv_save_dir.joinpath("preprocess.pickle")
        cv_result["preprocess_path"] = str(preprocess_path)

        processed_X = self.current_pipeline.fit_transform(self.X)
        cv_train_X, cv_valid_X = processed_X[train_idx], processed_X[test_idx]
        cv_train_Y, cv_valid_Y = self.Y[train_idx], self.Y[test_idx]

        self.model_params = self.filter_params(self.create_model_fn, self.current_params)
        self.model = self.create_model_fn(**self.model_params)

        model_path = cv_save_dir.joinpath("model.h5py")

        cv_result["model_path"] = str(model_path)
        train_csv_path = cv_save_dir.joinpath("train_log.csv")
        self.compile_params = self.filter_params(self.compile_fn, self.current_params)

        self.compile_params["model"] = self.model
        self.model = self.compile_fn(**self.compile_params)

        self.fit_params = self.filter_params(self.fit_fn, self.current_params)

        self.fit_params["model"] = self.model
        self.fit_params["x"] = cv_train_X
        self.fit_params["y"] = cv_train_Y
        self.fit_params["valid_x"] = cv_valid_X
        self.fit_params["valid_y"] = cv_valid_Y
        self.fit_params["log_filename"] = train_csv_path
        self.fit_params["save_model_path"] = model_path

        history = self.fit_fn(**self.fit_params)

        cv_result["history"] = history.history
        cv_result["best_train_loss"] = np.max(history.history['loss'])
        cv_result["best_val_loss"] = np.max(history.history['val_loss'])

        logger.info("finish training\nevaluating.....")

        if self.evaluate_w_best:
            # TODO pickle model and preprocesser by one pipilne
            if model_path.exists():
                with preprocess_path.open(mode="wb+") as f:
                    pickle.dump(self.current_pipeline, f)
                self.model.load_weights(str(model_path))
            else:
                logging.info("There is no model file. It may be caused by Nan loss value. "
                             "The train for this params are aborted.")
                return False

        cv_train_Y_pred = self.model.predict(cv_train_X)
        if np.any(np.isnan(cv_train_Y_pred)):
            logging.info("There are some nan in predicted values.\n "
                         "It may be caused by gradient vanishing. "
                         "The train for this params are aborted.")
            return False

        train_score = self.score_fn(cv_train_Y, cv_train_Y_pred)
        logger.info("train score: {}".format(train_score))
        cv_result["train_score"] = train_score

        cv_valid_Y_pred = self.model.predict(cv_valid_X)
        if np.any(np.isnan(cv_valid_Y_pred)):
            logging.info("There are some nan in predicted values.\n "
                         "It may be caused by gradient vanishing. "
                         "The train for this params are aborted.")
            return False
        valid_score = self.score_fn(cv_valid_Y, cv_valid_Y_pred)
        cv_result["val_score"] = valid_score
        logger.info("valid score: {}".format(valid_score))

        return cv_result

    def _evaluate_param(self, param_save_dir):
        train_scores = [cv_result["train_score"] for cv_result in self.eval_results[self.current_param_id]["cv_info"]]
        avg_train_score = np.mean(train_scores)
        std_train_score = np.std(train_scores)
        self.eval_results[self.current_param_id]["train_score_avg"] = avg_train_score
        self.eval_results[self.current_param_id]["train_score_std"] = std_train_score
        logger.info("train score for Param #{}: avg {}, std {}".format(self.current_param_id,
                                                                       avg_train_score, std_train_score))

        valid_scores = [cv_result["val_score"] for cv_result in self.eval_results[self.current_param_id]["cv_info"]]
        avg_valid_score = np.mean(valid_scores)
        std_valid_score = np.std(valid_scores)
        self.eval_results[self.current_param_id]["val_score_avg"] = avg_valid_score
        self.eval_results[self.current_param_id]["val_score_std"] = std_valid_score

        logger.info("valid score for Param #{}: avg {}, std {}".format(self.current_param_id,
                                                                       avg_valid_score, std_valid_score))

        param_result_save_path = param_save_dir.joinpath("result.json")
        with param_result_save_path.open(encoding="utf-8", mode="w+") as f:
            json.dump(self.eval_results[self.current_param_id], fp=f, cls=NumpyJsonEncoder)

        logger.info("===========finish [Params #{}]======================".format(self.current_param_id))

    def _create_preprocess_pipeline(self, params):
        estimaters = []

        for idx, _class in enumerate(self.preprocess_lists):
            init_params = self.filter_params(_class.__init__, params)
            estimaters.append(("#{}_{}".format(idx, _class.__name__), _class(**init_params)))

        return Pipeline(estimaters)

    @staticmethod
    def filter_params(fn, param_dics):
        arg_list = inspect.signature(fn).parameters
        fn_param_dics = {arg: param_dics[arg] for arg in arg_list if arg in param_dics}
        return fn_param_dics

    def filter_all_params(self, param_dics):
        self.fit_params = self.filter_params(self.fit_fn, param_dics)
        self.model_params = self.filter_params(self.create_model_fn, param_dics)
        self.compile_fn = self.filter_params(self.compile_fn, param_dics)

    def score(self, Y, Y_pred):
        return self.score_fn(Y, Y_pred)

    def load_with_id(self, param_id, cv_idx=None):
        result = self.eval_results[param_id]

        self.current_params = result["params"]
        self.filter_all_params(self.current_params)

        self.model = self.create_model_fn(**self.model_params)

        if cv_idx is None:
            # TODO more sophisticated algorithm to select best cv
            # val_score_avg = result["val_score_avg"]
            # cv_idx = np.argmin([abs(cv["val_score"] - val_score_avg) for cv in result["cv_info"]])
            cv_idx = np.argmax([cv["val_score"] for cv in result["cv_info"]])

        self.current_pipeline = pickle.load(Path(result["cv_info"][cv_idx]["preprocess_path"]).open(mode="rb"))
        self.model.load_weights(result["cv_info"][cv_idx]["save_model_path"])

    def transform(self, x):
        '''
        predict result
        :param x:
        :return:
        '''
        preprocessed_x = self.current_pipeline.transform(x)
        pred_y = self.model.predict(preprocessed_x)

        return pred_y


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        else:
            return super().default(o)
#
# class TemplateKerasClassifier(KerasClassifier):
#
#
#     def fit(self, x, y, **kwargs):
#     self.history_store = History()
#     self.epochs = params["epoch"]
#
#     log_filename.parent.mkdir(exist_ok=True, parents=True)
#     self.csvlogger = CSVLogger(log_filename, separator=",")
#
#     save_model_path.parent.mkdir(exist_ok=True, parents=True)
#     self.checkpointer = ModelCheckpoint(filepath=save_model_path,
#                                         verbose=1, save_best_only=True)
#
#     self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
#                                         patience=params["patience"], verbose=1, mode='auto')
#     self.lr_scheduler = LearningRateScheduler(learning_scheduler)
#     self.reduce_lr = ReduceLROnPlateau(
#         factor=params["plateau_factor"],
#         patience=params["plateau_patience"],
#         min_lr=1e-5)
#
#     optimizer = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
#
#     self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
#                        metrics=['categorical_accuracy'])
#
#     return super().fit(x, y, **kwargs)
