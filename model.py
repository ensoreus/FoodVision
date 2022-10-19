import tensorflow as tf
import tensorflow_datasets as tfds
import os
from helper_functions import create_tensorboard_callback


class ModelTrainer:
    _train_data = None
    _test_data = None
    _class_name = None
    _model_checkpoint_callback = None
    _tensorboard_callback = None
    _checkpoint_path = None
    _initial_epoch = 0
    _total_epochs = 5
    _compiled_model = None
    __load_model = False
    __model_name = "FV"

    def __init__(self, dataset_name, model_name, load_model=False):
        self._prepare_dataset(dataset_name)
        self._checkpoint_path = "model_checkpoint/checkpoint.ckpt"
        self.__load_model = load_model
        self.__model_name = model_name

    @staticmethod
    def _preprocess_image(input_img, title, img_shape=224):
        """
        :param input_img: image to prepare
        :param title: image name
        :param img_shape: input shape to fit the model
        :return: prepared resized image
        """
        x = tf.image.resize(input_img, [img_shape, img_shape])
        return tf.cast(x, tf.float32), title

    def _prepare_dataset(self, dataset_name):
        (train_data, test_data), ds_info = tfds.load(name=dataset_name,
                                                     split=["train", "validation"],
                                                     shuffle_files=True,
                                                     as_supervised=True,  # data returns as tuples (data, label)
                                                     with_info=True)
        self._train_data = train_data.map(map_func=self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        self._train_data = self._train_data.shuffle(buffer_size=1000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

        self._test_data = test_data.map(map_func=self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        self._test_data = self._test_data.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
        self._class_names = ds_info.features["label"].names

    def _make_model(self, model_class, load_from_checkpoint=False):
        self._base_model = model_class(include_top=False)
        self._base_model.trainable = False

        input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
        x = self._base_model(input_layer, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAveragePooling2D")(x)
        x = tf.keras.layers.Dense(len(self._class_names))(x)
        output = tf.keras.layers.Activation(tf.keras.activations.softmax, dtype=tf.float32, name="softmax_float32")(x)
        model = tf.keras.Model(input_layer, output)
        if load_from_checkpoint:
            model.load_weights(self._checkpoint_path)
        return model

    def _prepare_callbacks(self, experimant_name, save_best_only=False):
        self._model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(self._checkpoint_path,
                                                                             monitor="val_accuracy",
                                                                             save_best_only=save_best_only,
                                                                             save_weights_only=True,
                                                                             verbose=1)
        self._tensorboard_callback = create_tensorboard_callback(dir_name="learning_log",
                                                                 experiment_name=experimant_name)

    def feature_extract(self, model_class):
        if self.__load_model:
            model = tf.keras.models.load_model(self.__model_name)
            self._compiled_model = model
        else:
            model = self._make_model(model_class, load_from_checkpoint=True)
            self._compiled_model = model

        self._prepare_callbacks("FeatureExtraction")
        model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])
        self._compiled_model = model
        fe_history = model.fit(self._train_data, epochs=self._total_epochs,
                               steps_per_epoch=len(self._train_data),
                               validation_data=self._test_data,
                               use_multiprocessing=True,
                               validation_steps=int(len(self._test_data) * 0.15),
                               callbacks=[self._tensorboard_callback, self._model_checkpoint_callback]
                               )
        return fe_history

    def _unfreeze_top_layers(self, layers_to_unfreeze=10):
        for layer in self._base_model.layers[:-layers_to_unfreeze]:
            layer.trainable = True

    def fine_tune(self, model_class):
        if self.__load_model:
            model = tf.keras.models.load_model(self.__model_name)
            self._compiled_model = model
        else:
            model = self._make_model(model_class, load_from_checkpoint=True)
            self._compiled_model = model

        self._base_model.trainable = False
        self._unfreeze_top_layers()
        self._prepare_callbacks("FineTune", save_best_only=True)
        self._initial_epoch = 5
        self._compiled_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                                     metrics=["accuracy"])

        ft_history = self._compiled_model.fit(self._train_data, epochs=self._total_epochs,
                                              steps_per_epoch=len(self._train_data),
                                              validation_data=self._test_data,
                                              use_multiprocessing=True,
                                              initial_epoch=self._initial_epoch,
                                              validation_steps=int(len(self._test_data) * 0.15),
                                              callbacks=[self._tensorboard_callback, self._model_checkpoint_callback])
        return ft_history

    def save_as(self, filename):
        self.compiled_model.save(filename)
