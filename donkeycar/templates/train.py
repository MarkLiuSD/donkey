#!/usr/bin/env python3
'''
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: python train_v2.py --model models/mypilot

Usage:
    train.py [--tub=tub] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help              Show this screen.
'''

import os
import random
from pathlib import Path

import cv2
import numpy as np
from docopt import docopt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils.data_utils import Sequence

import donkeycar
from donkeycar.parts.keras import KerasInferred
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import get_model_by_type, load_scaled_image_arr


class TubDataset(object):
    '''
    Loads the dataset, and creates a train/test split.
    '''
    def __init__(self, tub_path, test_size=0.2, shuffle=True):
        self.tub_path = tub_path
        self.test_size = test_size
        self.shuffle = shuffle
        self.tub = Tub(self.tub_path)
        self.records = list()

    def train_test_split(self):
        print('Loading tub from path %s' % (self.tub_path))
        self.records.extend(self.tub)
        return train_test_split(self.records, test_size=self.test_size, shuffle=self.shuffle)


class TubSequence(Sequence):
    def __init__(self, keras_model, images_base_path, config, records=list()):
        self.keras_model = keras_model
        self.images_base_path = images_base_path
        self.config = config
        self.records = records
        self.batch_size = self.config.BATCH_SIZE

    def __len__(self):
        return len(self.records) // self.batch_size

    def __getitem__(self, index):
        count = 0
        records = []
        images = []
        angles = []
        throttles = []

        is_inferred = type(self.keras_model) is KerasInferred

        while count < self.batch_size:
            i = (index * self.batch_size) + count
            if i >= len(self.records):
                break

            record = self.records[i]
            record = self._transform_record(record)
            records.append(record)
            count += 1

        for record in records:
            image = record['cam/image_array']
            angle = record['user/angle']
            throttle = record['user/throttle']
        
            images.append(image)
            angles.append(angle)
            throttles.append(throttle)

        X = np.array(images)

        if is_inferred:
            Y = np.array(angles)
        else:
            Y = [np.array(angles), np.array(throttles)]

        return X, Y

    def _transform_record(self, record):
        for key, value in record.items():
            if key == 'cam/image_array' and isinstance(value, str):
                image_path = os.path.join(self.images_base_path, value)
                image = load_scaled_image_arr(image_path, self.config)
                record[key] = image

        return record


class ImagePreprocessing(Sequence):
    '''
    A Sequence which wraps another Sequence with an Image Augumentation.
    '''
    def __init__(self, sequence, augmentation):
        self.sequence = sequence
        self.augumentation = augmentation

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        X, Y = self.sequence[index]
        return self.augumentation.augment_images(X), Y

    @classmethod
    def region_of_interest(cls, image, x, y, width, height):
        '''
        Selects a region of interest from the original image.
        '''
        copy = image[y: y + height, x : x + width]
        return copy

    @classmethod
    def crop(cls, left, right, top, bottom, keep_size=False):
        '''
        The image augumentation sequence.
        Crops based on a region of interest among other things.
        '''
        import imgaug as ia
        import imgaug.augmenters as iaa

        augmentation = iaa.Sequential([
            iaa.Crop(
                px=(left, right, top, bottom),
                keep_size=keep_size
            ),
        ])
        return augmentation

    @classmethod
    def trapezoidal_mask(cls, lower_left, lower_right, upper_left, upper_right, min_y, max_y):
        '''
        Uses a binary mask to generate a trapezoidal region of interest.
        Especially useful in filtering out uninteresting features from an
        input image.
        '''
        import imgaug as ia
        import imgaug.augmenters as iaa

        def _transform_images(images, random_state, parents, hooks):
            # Transform a batch of images
            transformed = []
            mask = None
            for image in images:
                if mask is None:
                    mask = np.zeros(image.shape, dtype='bool')
                    # # # # # # # # # # # # #
                    #       ul     ur          min_y
                    #
                    #
                    #
                    #    ll             lr     max_y
                    points = [
                        [upper_left, min_y],
                        [upper_right, min_y],
                        [lower_right, max_y],
                        [lower_left, max_y]
                    ]
                    cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), 1)

                masked = cv2.bitwise_and(image, mask)
                transformed.append(masked)

            return transformed

        def _transform_keypoints(keypoints_on_images, random_state, parents, hooks):
            # No-op
            return keypoints_on_images

        augmentation = iaa.Sequential([
            iaa.Lambda(func_images=_transform_images, func_keypoints=_transform_keypoints)
        ])

        return augmentation


def train(cfg, tub_path, output_path, model_type):
    '''
    Train the model
    '''
    if 'linear' in model_type:
        train_type = 'linear'
    else:
        train_type = model_type

    kl = get_model_by_type(train_type, cfg)
    kl.compile()

    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

    batch_size = cfg.BATCH_SIZE
    dataset = TubDataset(tub_path, test_size=(1. - cfg.TRAIN_TEST_SPLIT))
    images_base_path = dataset.tub.images_base_path
    training_records, validation_records = dataset.train_test_split()
    print('Records # Training %s' % (len(training_records)))
    print('Records # Validation %s' % (len(validation_records)))

    training = TubSequence(kl, images_base_path, cfg, training_records)
    validation = TubSequence(kl, images_base_path, cfg, validation_records)

    # Setup early stoppage callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=cfg.EARLY_STOP_PATIENCE),
        ModelCheckpoint(
            filepath=output_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        )
    ]

    kl.model.fit_generator(
        generator=training,
        steps_per_epoch=len(training),
        callbacks=callbacks,
        validation_data=validation,
        validation_steps=len(validation),
        epochs=cfg.MAX_EPOCHS,
        verbose=cfg.VERBOSE_TRAIN,
        workers=1,
        use_multiprocessing=False
    )


def main():
    args = docopt(__doc__)
    cfg = donkeycar.load_config()
    tub = args['--tub']
    model = args['--model']
    model_type = args['--type']

    if not model_type:
        model_type = cfg.DEFAULT_MODEL_TYPE

    data_path = Path(os.path.expanduser(tub)).absolute().as_posix()
    output_path = os.path.expanduser(model)
    train(cfg, data_path, output_path, model_type)

if __name__ == "__main__":
    main()
