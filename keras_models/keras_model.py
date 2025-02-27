#!/usr/bin/env python
# coding: utf-8

# # Transfer learning & fine-tuning
# 
# **Author:** [fchollet](https://twitter.com/fchollet)<br>
# **Date created:** 2020/04/15<br>
# **Last modified:** 2023/06/25<br>
# **Description:** Complete guide to transfer learning & fine-tuning in Keras.
# 
# Modified to train a model for CIFAR-100 


import numpy as np

import keras
from tensorflow import data as tf_data
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

batch_size = 8
base_arch = "EfficientNet"
epochs = 50
ft_epochs =  10
fine_tune = True
short_test_run = False # shortens datasets if True

input_shape = (224, 224, 3)
train_model = True

train_ds, test_ds = tfds.load("cifar100")
# or  keras.utils.image_dataset_from_directory

tfds.disable_progress_bar()

train_ds, validation_ds, test_ds = tfds.load(
    "cifar100",
    # Reserve 10% for validation
    split=["train[:90%]", "train[90%:]", "test[0:]"],
    as_supervised=True,  # Include labels
)

# trim the dataset for quick tests
if short_test_run:
  train_ds = train_ds.take(200)
  validation_ds = validation_ds.take(100)
  test_ds = test_ds.take(100)

print(f"Number of training samples: {train_ds.cardinality()}")
print(f"Number of validation samples: {validation_ds.cardinality()}")
print(f"Number of test samples: {test_ds.cardinality()}")


resize_fn = keras.layers.Resizing(*input_shape[:2])

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))


# ### Using random data augmentation
# 
# When you don't have a large image dataset, it's a good practice to artificially
# introduce sample diversity by applying random yet realistic transformations to
# the training images, such as random horizontal flipping or small random rotations. This
# helps expose the model to different aspects of the training data while slowing down
# overfitting.

# In[7]:


augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x


train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()


if base_arch == "Xception":
  base_model = keras.applications.Xception(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=input_shape,
      include_top=False,
  )  # Do not include the ImageNet classifier at the top.
  
  # Freeze the base_model
  base_model.trainable = False
  
  # Create new model on top
  inputs = keras.Input(shape=input_shape)
  
  # Pre-trained Xception weights requires that input be scaled
  # from (0, 255) to a range of (-1., +1.), the rescaling layer
  # outputs: `(inputs * scale) + offset`
  scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
  x = scale_layer(inputs)
  
  # The base model contains batchnorm layers. We want to keep them in inference mode
  # when we unfreeze the base model for fine-tuning, so we make sure that the
  # base_model is running in inference mode here.
  x = base_model(x, training=False)
  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
  outputs = keras.layers.Dense(100)(x)
  model = keras.Model(inputs, outputs)
  
elif base_arch == "EfficientNet":
  # base_model = keras.applications.EfficientNetV2B0(
  # base_model = keras.applications.ResNet152V2(
  base_model = keras.applications.EfficientNetV2L(
      classifier_activation="softmax",
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=input_shape,
      include_top=False,
      pooling="avg"
  )  # Do not include the ImageNet classifier at the top.
  
  # Freeze the base_model
  base_model.trainable = False
  
  # Create new model on top
  inputs = keras.Input(shape=input_shape)
  x = inputs
  # Pre-trained EfficientNet weights requires that input be in [0, 255]
  
  # The base model contains batchnorm layers. We want to keep them in inference mode
  # when we unfreeze the base model for fine-tuning, so we make sure that the
  # base_model is running in inference mode here.
  x = base_model(x, training=False)
  # x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
  # x = keras.layers.Dense(256)(x)    # if there's going to be a 2nd dense layer, it should have an activation
  x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
  outputs = keras.layers.Dense(100)(x)
  model = keras.Model(inputs, outputs)

else:
  raise RuntimeError("Right now only EfficientNet and Xception are supported")

model.summary(show_trainable=True)


num_batches = train_ds.cardinality()
lr_sched   = keras.optimizers.schedules.CosineDecay(1e-3, epochs, alpha=0.01)
# lr_callback = keras.callbacks.LearningRateScheduler(lr_sched)
# callbacks = [lr_callback]
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

optimizer = keras.optimizers.Adam(learning_rate=lr_sched)
# lr_metric = get_lr_metric(optimizer)



if train_model:
  model.compile(
      optimizer=optimizer,
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )
  
  print("Fitting the top layer of the model")
  xfer_hist = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
  model.save("cifar100_effnet_xfer.keras")
else:
  print("Loading model from disk")
  model = keras.saving.load_model("cifar100_effnet_xfer.keras")

# weights_pre_ft = model.layers[1].get_weights()
# model_pre_ft = keras.models.clone_model(model)
# model_pre_ft.set_weights(model.get_weights())
# model_pre_ft.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )

if train_model:
  plt.subplot(1,2,1)
  plt.plot(xfer_hist.epoch, xfer_hist.history['loss'], label="Loss")
  plt.plot(xfer_hist.epoch, xfer_hist.history['val_loss'], label="Val Loss")
  plt.legend()
  plt.subplot(1,2,2)
  plt.plot(xfer_hist.epoch, xfer_hist.history['sparse_categorical_accuracy'], label="Acc")
  plt.plot(xfer_hist.epoch, xfer_hist.history['val_sparse_categorical_accuracy'], label="Val Acc")
  plt.legend()
  plt.savefig("training_curve.png")


# ## Do a round of fine-tuning of the entire model
# 
# Finally, let's unfreeze the base model and train the entire model end-to-end with a low
#  learning rate.
# 
# Importantly, although the base model becomes trainable, it is still running in
# inference mode since we passed `training=False` when calling it when we built the
# model. This means that the batch normalization layers inside won't update their batch
# statistics. If they did, they would wreck havoc on the representations learned by the
#  model so far.

# In[14]:


if fine_tune:
  model.layers[1].trainable = True

  bn_layers_frozen = 0
  for i,l in enumerate(model.layers[1].layers):
    if l.__class__.__name__ == "BatchNormalization":
      # print(f"Layer {i} is BN, setting trainable=False")    
      l.trainable = False
      bn_layers_frozen += 1
  
  print(f"{bn_layers_frozen} BatchNorm layers set with trainable=False")
  
  model.compile(
      optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )
  
  
  print("Fitting the end-to-end model")
  ft_hist = model.fit(train_ds, epochs=ft_epochs, validation_data=validation_ds)
  
  model.save("cifar100_effnet_ft.keras")
  
  plt.subplot(1,2,1)
  plt.plot(ft_hist.epoch, ft_hist.history['loss'], label="Loss")
  plt.plot(ft_hist.epoch, ft_hist.history['val_loss'], label="Val Loss")
  plt.legend()
  plt.subplot(1,2,2)
  plt.plot(ft_hist.epoch, ft_hist.history['sparse_categorical_accuracy'], label="Acc")
  plt.plot(ft_hist.epoch, ft_hist.history['val_sparse_categorical_accuracy'], label="Val Acc")
  plt.legend()
  plt.savefig("finetuning_curve.png")
  
  print("Test dataset evaluation after fine tuning")
  model.evaluate(test_ds)

print("Done")