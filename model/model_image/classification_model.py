import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import load_data

layers = tf.keras.layers
models = tf.keras.models

dataset_path = r"D:\Entrepreneur ship project\model\model_image\Dataset_BUSI_with_GT"

X, y_masks, y_labels = load_data(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, test_size=0.2, random_state=42
)

def classification_model(input_shape=(128,128,3)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(3, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = classification_model()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'classification_model.keras',
    save_best_only=True,
    verbose=1
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=8,
    callbacks=[checkpoint]
)

model.save("classification_model_final.keras")