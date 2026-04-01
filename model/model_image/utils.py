import os
import cv2
import numpy as np
import re
import tensorflow as tf

def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else None

def load_data(dataset_path, img_size=(128, 128)):
    X = []
    y_masks = []
    y_labels = []
    
    classes = ['benign', 'malignant', 'normal']
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"Folder not found: {class_path}")
            continue
        
        images = {}
        masks = {}
        
        for file in os.listdir(class_path):
            if file.lower().endswith('.png'):
                file_path = os.path.join(class_path, file)
                number = extract_number(file)
                
                if number is None:
                    continue
                
                if '_mask' in file.lower():
                    masks[number] = file_path
                else:
                    images[number] = file_path
        
        common_numbers = sorted(set(images.keys()) & set(masks.keys()))
        
        for number in common_numbers:
            img = cv2.imread(images[number])
            mask = cv2.imread(masks[number], cv2.IMREAD_GRAYSCALE)
            
            if img is not None and mask is not None:
                img = cv2.resize(img, img_size)
                mask = cv2.resize(mask, img_size)
                
                mask = (mask > 127).astype(np.float32)
                
                X.append(img)
                y_masks.append(mask)
                y_labels.append(class_idx)
        
        print(f"Loaded {len(common_numbers)} samples from {class_name}")
    
    X = np.array(X, dtype=np.float32) / 255.0
    y_masks = np.array(y_masks, dtype=np.float32)
    y_masks = np.expand_dims(y_masks, axis=-1)
    
    y_labels = np.array(y_labels)
    y_labels_cat = tf.keras.utils.to_categorical(y_labels, num_classes=3)
    
    print(f"Total samples: {len(X)}")
    
    return X, y_masks, y_labels_cat

def load_single_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.resize(img, img_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img