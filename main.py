# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import coremltools as ct
import model as mt
from helper_functions import plot_loss_curves

model_trainer = mt.ModelTrainer("Food101", model_name="feature_extract", load_model=True)
#history = model_trainer.feature_extract()
#plot_loss_curves(history)
history = model_trainer.fine_tune()
plot_loss_curves(history)
model_trainer.save_as("FineTuned")
