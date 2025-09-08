'''
This file will rotate, flip, color jittering, random cropping, zooming , addoing noise, elastic transformation, brightness and contrast, denosing , sharpening, grey scale, brighten and smoothen images from the dataset
'''

import pandas as pd 
import cv2 as cv 
import numpy as np
import os 
import tensorflow as tf
import random
from tqdm import tqdm


BASE_DIR = "Datasets/acne/"

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
LABEL_COLUMNS = list(pd.read_csv(os.path.join(BASE_DIR, "train/train_labels.csv")).columns)[1:]
AUGMENTED_OUTPUT_DIR = os.path.join(BASE_DIR, "augmented_images")

print("Label columns found: ", LABEL_COLUMNS)


class Preprocessing():
    
    def __init__(self, base_dir=BASE_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE):
        self.base_dir = base_dir
        self.target_size = target_size
        self.batch_size = batch_size

    def get_label_columns(self):
        return LABEL_COLUMNS
        

    def load_and_preprocess_image(image_path, target_size=TARGET_SIZE, augment=False, save_augmented=False):
        """
        Load an image from the given path and preprocess it.
        """
        img = cv.imread(image_path)
        
        if img is None:
            raise ValueError(f"Image not found at {image_path}")
            return None
        
        # Resize the image to the target size
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)
        
        if augment:
            
            # Rotate the image by a random angle
            angle = random.randint(-10, 10)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
            img = cv.warpAffine(img, rotation_matrix, (w, h))
            
            
            # Color jittering
            hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv_img)
                    
            brihtness_factor = random.uniform(0.95, 1.05)
            v = np.clip(v * brihtness_factor, 0, 255).astype(np.uint8)
            
            saturation_factor = random.uniform(0.95, 1.05)
            s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8) 
            
            hsv_img = cv.merge((h, s, v))
            img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
            
            # Random cropping
            if random.random() < 0.1:
                h, w, _ = img.shape
                crop_size_ratio = random.uniform(0.9,0.95)
                crop_h = int(h * crop_size_ratio)
                crop_w = int(w * crop_size_ratio)
                
                start_x = random.randint(0, w - crop_w)
                start_y = random.randint(0, h - crop_h)
                img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
                img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)
                
                
            # Adding noise
            if random.random() > 0.95:
                gauss = np.random.normal(0, 0.5, img.size)
                gauss = gauss.reshape(img.shape[0], img.shape[1], 3).astype('uint8')
                img = cv.add(img, gauss)
                img = np.clip(img, 0, 255).astype('uint8')
                
                
            # Blurring
            if random.random() > 0.9:
                blur_kernel_size = random.choice([3,5])
                img = cv.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
                
            # Denoising 
            if random.random() > 0.98:
                img = cv.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
                
            # Sharpening
            if random.random() > 0.8:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                img = cv.filter2D(img, -1, kernel)
                
            # save augmented image
            if save_augmented:
                if not os.path.exists(AUGMENTED_OUTPUT_DIR):
                    os.makedirs(AUGMENTED_OUTPUT_DIR)
                augmented_image_path = os.path.join(AUGMENTED_OUTPUT_DIR, os.path.basename(image_path))
                cv.imwrite(augmented_image_path, img)
        
        # Normalize the image
        img = img.astype(np.float32) / 255.0
        
        return img


    def create_tf_dataset_from_csv(split_name, base_data_dir, target_size=TARGET_SIZE, augment=False, save_augmented=False):
        """
        Create a TensorFlow dataset from a CSV file.
        """
        csv_path = os.path.join(base_data_dir, split_name, f"{split_name}_labels.csv")
        image_folder_path = os.path.join(base_data_dir, split_name)
        
        
        df = pd.read_csv(csv_path)
        
        images = []
        labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            image_id = row['filename']
            image_path = os.path.join(image_folder_path, f'{image_id}.jpg')
            
            
            img = load_and_preprocess_image(image_path, target_size, augment, save_augmented=save_augmented)
            if img is not None:
                images.append(img)
                current_labels = row[LABEL_COLUMNS].values.astype(np.float32)
                labels.append(current_labels)
                
            
        if not images:
            raise ValueError(f"Warning: No images found or loaded in {split_name} split in {image_folder_path}")
            return tf.data.Dataset.from_tensor_slices((np.array([]), np.array([])))
        
        
        return tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))


    # print("Creating training dataset...")
    # train_dataset = create_tf_dataset_from_csv("train", BASE_DIR, TARGET_SIZE, augment=True , save_augmented=True)
    # # print("Creating validation dataset...")
    # # val_dataset = create_tf_dataset_from_csv("val", BASE_DIR, TARGET_SIZE, augment=False)
    # # print("Creating test dataset...")
    # # test_dataset = create_tf_dataset_from_csv("test", BASE_DIR, TARGET_SIZE, augment=False)

    # # Further preprocess for performance
    AUTOTUNE = tf.data.AUTOTUNE

    # Check if datasets are empty before caching and prefetching
    if tf.data.experimental.cardinality(train_dataset).numpy() > 0:
        train_dataset = train_dataset.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    else:
        print("Warning: Training dataset is empty. Skipping caching and prefetching.")
        train_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros((0, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=tf.float32), 
                                                            tf.zeros((0, len(LABEL_COLUMNS)), dtype=tf.float32)))


    if tf.data.experimental.cardinality(val_dataset).numpy() > 0:
        val_dataset = val_dataset.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    else:
        print("Warning: Validation dataset is empty. Skipping caching and prefetching.")
        val_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros((0, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=tf.float32), 
                                                        tf.zeros((0, len(LABEL_COLUMNS)), dtype=tf.float32)))
        
    if tf.data.experimental.cardinality(test_dataset).numpy() > 0:
        test_dataset = test_dataset.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    else:
        print("Warning: Test dataset is empty. Skipping caching and prefetching.")
        test_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros((0, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=tf.float32), 
                                                        tf.zeros((0, len(LABEL_COLUMNS)), dtype=tf.float32)))
    print("Datasets created successfully!")


    # # Example: Get the number of output classes
    # num_classes = len(LABEL_COLUMNS)
    # print(f"Number of output classes: {num_classes}")

    # # Example: Iterate through a batch of the training dataset to verify
    # for images, labels in train_dataset.take(1):
    #     print(f"Batch images shape: {images.shape}")
    #     print(f"Batch labels shape: {labels.shape}")
    #     print(f"Example labels (first 5 images): {labels[:5].numpy()}")
        
        
