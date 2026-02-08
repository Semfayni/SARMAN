import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 30
MODEL_SAVE_PATH = "SAR_classifier_model_v2.keras"


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

normalization = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

model = models.Sequential([
    layers.Conv2D(
        32, (3, 3), activation="relu",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=30,
    validation_data=test_ds
)
model.save(MODEL_SAVE_PATH)