# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

import matplotlib.pyplot as plt
import coremltools as ct
import tensorflow as tf
import model as mt
from helper_functions import plot_loss_curves

model_trainer = mt.ModelTrainer("Food101", model_name="feature_extract")
# history = model_trainer.feature_extract(tf.keras.applications.ResNet50)
# plot_loss_curves(history)
# history = model_trainer.fine_tune(tf.keras.applications.ResNet50)
# plot_loss_curves(history)

# model = model_trainer.pick_model_by_checkpoint(tf.keras.applications.ResNet50)
# model_trainer.save_as("FineTuned")

model = tf.keras.models.load_model("FineTuned")
image_input = ct.ImageType(name="input_2", shape=(1, 224, 224, 3,))
classifier_config = ct.ClassifierConfig(model_trainer.class_names)

coreml_model = ct.convert(model, source="tensorflow", convert_to='neuralnetwork',
                          inputs=[image_input],
                          classifier_config=classifier_config)
coreml_model.author = "Pylyp Maliuta"
coreml_model.short_description = "Classify 101 kinds of food from input image"
coreml_model.save("FoodVisionBig")
