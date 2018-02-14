"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""
from __future__ import unicode_literals

import math
import os
import re
from pathlib import PureWindowsPath

import numpy as np
import soundfile
from common.utils import text_to_int_sequence
from python_speech_features import mfcc
from scipy import signal
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from tensorflow.python.lib.io import file_io

RANDOM_SEED = 123

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]


class MfccParams:
    def __init__(self, winlen, winstep, numcep, nfilt):
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt


class DataGenerator:
    def __init__(self, data_paths, input_feature,
                 minibatch_size=20, sample_rate=16000, fft_length=256,
                 stride_length=128, use_channel=True,
                 mfcc_params=None):
        self.ftt_length = fft_length
        self.sample_rate = sample_rate
        self.stride_length = stride_length
        self.minibatch_size = minibatch_size
        self.use_channel = use_channel

        self.input_feature = input_feature  # type: InputFeature

        self.mfcc_params = mfcc_params  # type: MfccParams
        if isinstance(self.input_feature, MfccFeature):
            self.mfcc_params = self.input_feature.mfcc_params
            self.ftt_length = self.input_feature.fft_length

        self.n_frames = input_feature.shape()[0]

        self.data_paths = shuffle(data_paths, random_state=RANDOM_SEED)
        self.num_datas = len(self.data_paths)

        self.steps_per_epoch = self.num_datas // self.minibatch_size
        self.steps_per_epoch += 1 if self.num_datas % self.minibatch_size else 0

        # state
        self.cur_index = 0

    def get_batch(self):

        if self.cur_index + self.minibatch_size > self.num_datas:
            current_paths = self.data_paths[self.cur_index:]

            self.data_paths = shuffle(self.data_paths)

            next_cur_index = self.cur_index + self.minibatch_size - self.num_datas
            current_paths.extend(self.data_paths[:next_cur_index])

            features = self.featurize(current_paths, self.minibatch_size)
            labels = self.to_labels(current_paths)

            self.cur_index = next_cur_index

        else:
            next_cur_index = self.cur_index + self.minibatch_size
            features = self.featurize(self.data_paths[self.cur_index:next_cur_index], self.minibatch_size)
            labels = self.to_labels(self.data_paths[self.cur_index:next_cur_index])

            self.cur_index = next_cur_index

        return (features, labels)

    def next(self):
        while True:
            yield self.get_batch()

    @staticmethod
    def to_label(path):
        label = np.zeros((len(LABELS)))

        word = DataGenerator.to_word(path)

        label[DataGenerator.to_label_index(word)] = 1

        return label

    @staticmethod
    def to_word(path):
        if isinstance(path, PureWindowsPath):
            path_str = str(path.as_posix())
        else:
            path_str = str(path)

        pattern = re.compile(".*/(_?\w+_?)/[^/]*\.wav$")
        match = pattern.match(path_str)

        if not match:
            raise Exception("invalid data path: {}".format(path))

        word = match.group(1)

        return word

    @staticmethod
    def to_label_index(word):
        if word == "_background_noise_":
            return LABELS.index("silence")
        try:
            return LABELS.index(word)
        except ValueError:
            return LABELS.index("unknown")

    def to_labels(self, paths):
        return np.vstack([self.to_label(path) for path in paths])

    #
    # def obtain_scaler(self):
    #     scaler = StandardScaler()
    #
    #     features = [self.to_spectrogram(self.load_wave_data(path)) for path in self.data_paths]
    #     scaler.fit(np.asarray(features))  #TODO partial fit
    #     return scaler

    def featurize(self, paths, batch_size):
        features = [self.to_feature(path) for path in paths]

        if type(self.input_feature) == InputFeature:
            padded_features = np.zeros((batch_size, self.n_frames))
            for idx, feature in enumerate(features):
                padded_features[idx, :feature.shape[0]] = feature

        else:
            padded_features = np.zeros((batch_size, self.n_frames, features[0].shape[-1]))
            for idx, feature in enumerate(features):
                padded_features[idx, :feature.shape[0], :] = feature

        if self.use_channel:
            padded_features = padded_features.reshape(padded_features.shape + (1,))

        return padded_features

    def to_feature(self, path):
        audio = self.load_wave_data(path)
        if self.mfcc_params:
            return scale(
                mfcc(audio, samplerate=self.sample_rate,
                     winlen=self.mfcc_params.winlen,
                     winstep=self.mfcc_params.winstep,
                     nfft=self.ftt_length,
                     numcep=self.mfcc_params.numcep,
                     nfilt=self.mfcc_params.nfilt),
                axis=0
            )
        elif type(self.input_feature) == InputFeature:
            return scale(audio)

        return scale(self.to_spectrogram(audio), axis=0)

    @staticmethod
    def load_wave_data(filepath, temp_dir="./temp_dir/data"):
        if filepath.startswith("gs:"):
            temp_path = os.path.join(temp_dir, filepath.replace("gs://", ""))
            file_io.recursive_create_dir(
                os.path.split(temp_path)[0])

            # cache after 1 epoch
            if not file_io.file_exists(temp_path):
                file_io.copy(filepath, temp_path, overwrite=True)
            with soundfile.SoundFile(temp_path) as f:
                audio = f.read(dtype='float32')
        else:
            with soundfile.SoundFile(str(filepath)) as f:
                audio = f.read(dtype='float32')
        return audio

    def to_spectrogram(self, array):
        return to_spectrogram(array, self.sample_rate, self.ftt_length, self.stride_length)


class CharDataGenerator(DataGenerator):
    max_label_length = max([len(label) for label in LABELS])

    def __init__(self, data_paths, input_feature, minibatch_size=20, sample_rate=16000, fft_length=256,
                 stride_length=128,
                 use_channel=True, mfcc_params=None):
        super().__init__(data_paths, input_feature,
                         minibatch_size, sample_rate,
                         fft_length, stride_length, use_channel, mfcc_params)

    @staticmethod
    def to_label(path):
        word = DataGenerator.to_word(path)

        if word == "_background_noise_":
            return np.array([0])

        return np.array(text_to_int_sequence(word))

    def to_labels(self, paths):
        labels = np.zeros((len(paths), self.max_label_length))
        label_length = np.zeros((len(paths), 1))
        for i, path in enumerate(paths):
            label = CharDataGenerator.to_label(path)
            labels[i, :len(label)] = label
            label_length[i] = len(label)

        return labels, label_length

    def get_batch(self):

        if self.cur_index + self.minibatch_size > self.num_datas:
            current_paths = self.data_paths[self.cur_index:]

            self.data_paths = shuffle(self.data_paths)

            next_cur_index = self.cur_index + self.minibatch_size - self.num_datas
            current_paths.extend(self.data_paths[:next_cur_index])

            features = self.featurize(current_paths, self.minibatch_size)
            labels, label_length = self.to_labels(current_paths)

            self.cur_index = next_cur_index

        else:
            next_cur_index = self.cur_index + self.minibatch_size
            features = self.featurize(self.data_paths[self.cur_index:next_cur_index], self.minibatch_size)
            labels, label_length = self.to_labels(self.data_paths[self.cur_index:next_cur_index])

            self.cur_index = next_cur_index

        inputs = {
            'input_1': features,
            'the_labels': labels,
            'input_length': np.ones((self.minibatch_size, 1)) * self.n_frames,
            'label_length': label_length
        }
        outputs = {'ctc': np.zeros([self.minibatch_size])}
        return (inputs, outputs)


class DataSetGenerator:
    def __init__(self, train_data_paths,
                 valid_data_paths, test_data_paths, input_feature,
                 minibatch_size=20, sample_rate=16000, fft_length=256, stride_length=128,
                 use_channel=True, mfcc_params=None):

        self.fft_length = fft_length
        self.sample_rate = sample_rate
        self.stride_length = stride_length
        self.minibatch_size = minibatch_size
        self.input_feature = input_feature

        self.use_channel = use_channel

        self.train_data_gen = None
        self.valid_data_gen = None
        self.test_data_paths = None

        self.create_data_generator(use_channel, train_data_paths, valid_data_paths,
                                   test_data_paths, mfcc_params)

    def create_data_generator(self, use_channel, train_data_paths, valid_data_paths, test_data_paths,
                              mfcc_params):
        self.train_data_gen = DataGenerator(train_data_paths, minibatch_size=self.minibatch_size,
                                            fft_length=self.fft_length, sample_rate=self.sample_rate,
                                            stride_length=self.stride_length, use_channel=use_channel,
                                            mfcc_params=mfcc_params,
                                            input_feature=self.input_feature)  # type: DataGenerator

        self.valid_data_gen = DataGenerator(valid_data_paths, minibatch_size=self.minibatch_size,
                                            fft_length=self.fft_length, sample_rate=self.sample_rate,
                                            stride_length=self.stride_length,
                                            use_channel=use_channel, mfcc_params=mfcc_params,
                                            input_feature=self.input_feature)  # type: DataGenerator

        self.test_data_paths = test_data_paths

    def next(self, partition):
        if partition == 'train':
            return self.train_data_gen.next()
        elif partition == 'valid':
            return self.valid_data_gen.next()

        else:
            raise Exception("Invalid partition. "
                            "Must be train/validation")

    def next_train(self):
        return self.next('train')

    def next_valid(self):
        return self.next('valid')


class CtcDataSetGenerator(DataSetGenerator):
    def __init__(self, train_data_paths,
                 valid_data_paths, test_data_paths, input_feature,
                 minibatch_size=20, sample_rate=16000, fft_length=256, stride_length=128,
                 use_channel=True, mfcc_params=None):
        super().__init__(train_data_paths,
                         valid_data_paths, test_data_paths,
                         minibatch_size=minibatch_size, sample_rate=sample_rate,
                         fft_length=fft_length, stride_length=stride_length,
                         use_channel=use_channel, mfcc_params=mfcc_params, input_feature=input_feature)

    def create_data_generator(self, use_channel, train_data_paths, valid_data_paths, test_data_paths, mfcc_params):
        self.train_data_gen = CharDataGenerator(train_data_paths, minibatch_size=self.minibatch_size,
                                                fft_length=self.fft_length, sample_rate=self.sample_rate,
                                                stride_length=self.stride_length, use_channel=use_channel,
                                                mfcc_params=mfcc_params,
                                                input_feature=self.input_feature)  # type: DataGenerator

        self.valid_data_gen = CharDataGenerator(valid_data_paths, minibatch_size=self.minibatch_size,
                                                fft_length=self.fft_length, sample_rate=self.sample_rate,
                                                stride_length=self.stride_length,
                                                use_channel=use_channel, mfcc_params=mfcc_params,
                                                input_feature=self.input_feature)  # type: DataGenerator

        self.test_data_paths = test_data_paths


def to_n_frames(sample_rate, fft_length, stride_length):
    return (sample_rate - fft_length) // stride_length + 1


def to_spectrogram(series, sample_rate, fft_length, stride_length):
    f, t, Sxx = signal.spectrogram(series, fs=sample_rate, nperseg=fft_length,
                                   noverlap=fft_length - stride_length, window="hanning", axis=0,
                                   return_onesided=True, mode="magnitude", scaling="density")
    return (Sxx).transpose()


#
# if __name__ == '__main__':
#     minibatch_size = 5
#
#     data_root = Path("data/train/audio")
#     with Path("data/train/train_list.txt").open(encoding="utf-8") as f:
#         train_data_paths = [data_root.joinpath(line.replace("\n", "" )) for line in f.readlines()]
#
#     with Path("data/train/testing_list.txt").open(encoding="utf-8") as f:
#         valid_data_paths = [data_root.joinpath(line.replace("\n", "" )) for line in f.readlines()]
#
#     data_gen = DataSetGenerator(train_data_paths=train_data_paths,
#                                 valid_data_paths=valid_data_paths,
#                                 test_data_paths=None)
#
#     for i, b in enumerate(data_gen.next("train")):
#         print("{} batch\n".format(i))
#         print(b[0].shape, b[1].shape)
#         print(b[0][:2, :, :])
#         print(b[1])
#
#         if i == 5:
#             break
#
#     for i, b in enumerate(data_gen.next("valid")):
#         print(b[0].shape, b[1].shape)
#         print(b[0][:2, :, :])
#         print(b[1])
#
#         if i == 5:
#             break
#     for i,b  in enumerate(data_gen.next_train()):
#         if i >= data_gen.train_data_gen.steps_per_epoch:
#             break
#         print("{} batch\n".format(i))
#         print(b[0].shape, b[1].shape)
#
#     for i,b  in enumerate(data_gen.next_valid()):
#         if i >= data_gen.valid_data_gen.steps_per_epoch:
#             break
#         print("{} batch\n".format(i))
#         print(b[0].shape, b[1].shape)
class InputFeature(object):
    def __init__(self, sample_rate, duration, use_channel=True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.use_channel = use_channel

    def shape(self):
        return (int(self.sample_rate * self.duration), 1)


class SpectrogramFeature(InputFeature):
    def __init__(self, sample_rate, duration, fft_length, stride_length):
        super(SpectrogramFeature, self).__init__(sample_rate, duration)
        self.fft_length = fft_length
        self.stride_length = stride_length

    def shape(self):
        input_shape = (to_n_frames(16000, self.fft_length, self.stride_length),
                       self.fft_length // 2 + 1, 1)
        return input_shape


class MfccFeature(InputFeature):
    def __init__(self, sample_rate, duration, winlen, winstep, numcep, nfilt):
        super(MfccFeature, self).__init__(sample_rate, duration)
        self.mfcc_params = MfccParams(
            #    winlen=0.04, winstep=0.02, numcep=40, nfilt=40
            winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt
        )  # type: MfccParams
        self.fft_length = int(math.ceil(self.mfcc_params.winlen * sample_rate))

    def shape(self):
        feature_dim = mfcc(np.array((range(self.sample_rate))),
                           samplerate=self.sample_rate, winstep=self.mfcc_params.winstep,
                           winlen=self.mfcc_params.winlen, numcep=self.mfcc_params.numcep,
                           nfft=self.fft_length,
                           nfilt=self.mfcc_params.nfilt).shape
        return feature_dim + (1,)
