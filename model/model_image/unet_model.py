import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import load_data

layers = tf.keras.layers
models = tf.keras.models

dataset_path = r"D:\Entrepreneur ship project\model\model_image\Dataset_BUSI_with_GT"

X, y_masks, y_labels = load_data(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_masks, test_size=0.2, random_state=42
)

def unet_model(input_shape=(128,128,3)):
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c4)
    m5 = layers.concatenate([c3, u5])
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(m5)

    u6 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c5)
    m6 = layers.concatenate([c2, u6])
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(m6)

    u7 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c6)
    m7 = layers.concatenate([c1, u7])
    c7 = layers.Conv2D(32, 3, activation='relu', padding='same')(m7)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

    model = models.Model(inputs, outputs)
    return model

model = unet_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "unet_best_model.keras",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=8,
    callbacks=[checkpoint, early_stop]
)

model.save("unet_final_model.keras")