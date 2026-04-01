import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

classification_model = tf.keras.models.load_model("classification_model_final.keras")
unet_model = tf.keras.models.load_model("unet_best_model.keras")

class_labels = ["benign", "malignant", "normal"]

def preprocess_image(img_path, target_size=(128,128)):
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_path):
    img_input = preprocess_image(img_path)

    cls_pred = classification_model.predict(img_input)
    cls_index = np.argmax(cls_pred)
    cls_label = class_labels[cls_index]

    mask_pred = unet_model.predict(img_input)[0]
    mask_pred = (mask_pred > 0.5).astype(np.uint8)

    return cls_label, mask_pred

def show_results(img_path, mask):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))

    mask_resized = cv2.resize(mask, (128,128))

    overlay = img.copy()
    overlay[mask_resized.squeeze() == 1] = [255, 0, 0]

    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Mask")
    plt.imshow(mask_resized.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()

img_path = r"D:\Entrepreneur ship project\model\model_image\Dataset_BUSI_with_GT\benign\benign (1).png"

label, mask = predict(img_path)

print("Prediction:", label)
print("Mask shape:", mask.shape)

show_results(img_path, mask)