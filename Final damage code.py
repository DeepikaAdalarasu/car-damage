
# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# import matplotlib.pyplot as plt
# import numpy as np

# train_dir = r"C:\Users\DEEPIKA\Downloads\car damage\training"
# val_dir = r"C:\Users\DEEPIKA\Downloads\car damage\validation"

# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator(rescale=1.0/255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'
# )


# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=20
# )


# model.save(r'C:\Users\DEEPIKA\Downloads\car damage\car_damage_detector.keras')

# def predict_damage(image_path, model_path=r'C:\Users\DEEPIKA\Downloads\car damage\car_damage_detector.keras'):

#     model = tf.keras.models.load_model(model_path)
    

#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    

#     prediction = model.predict(img_array)
#     class_label = 'Damaged' if prediction[0][0] > 0.5 else 'Undamaged'
    
#     plt.imshow(img)
#     plt.title(f"Prediction: {class_label}")
#     plt.axis('off')
#     plt.show()
    
#     return class_label

# image_path = r"C:\Users\DEEPIKA\Downloads\damaged-car7.jpg"
# result = predict_damage(image_path)
# print(f"The given image is: {result}")

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

def predict_damage(image_path, model_path=r'C:\Users\DEEPIKA\Downloads\car damage\car_damage_detector_mobilenetv2.keras'):
    model = tf.keras.models.load_model(model_path)
    
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    class_label = 'Undamaged' if confidence > 0.5 else 'Damaged'
    
    plt.imshow(img)
    plt.title(f"Prediction: {class_label} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()
    
    return class_label, confidence

image_path = r"C:\Users\DEEPIKA\Downloads\damage car6.jpg"  
result, confidence = predict_damage(image_path)
print(f"The given image is: {result} (Confidence: {confidence:.2f})")


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# import matplotlib.pyplot as plt
# import numpy as np


# train_dir = r"C:\Users\DEEPIKA\Downloads\car damage\training"
# val_dir = r"C:\Users\DEEPIKA\Downloads\car damage\validation"


# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator(rescale=1.0/255)


# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'
# )


# base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
#                                                include_top=False,
#                                                weights='imagenet')


# base_model.trainable = False


# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),  # Prevent overfitting
#     tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
# ])


# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=20
# )


# model.save(r'C:\Users\DEEPIKA\Downloads\car damage\car_damage_detector_mobilenetv2.keras')

# def predict_damage(image_path, model_path=r'C:\Users\DEEPIKA\Downloads\car damage\car_damage_detector_mobilenetv2.keras'):
#     # Load the model
#     model = tf.keras.models.load_model(model_path)
    
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
#     img_array = np.expand_dims(img_array, axis=0)  
    

#     prediction = model.predict(img_array)
#     confidence = prediction[0][0]
#     class_label = 'Damaged' if confidence > 0.5 else 'Undamaged'
    
#     plt.imshow(img)
#     plt.title(f"Prediction: {class_label} (Confidence: {confidence:.2f})")
#     plt.axis('off')
#     plt.show()
    
#     return class_label

# image_path = r"C:\Users\DEEPIKA\Downloads\damaged-car7.jpg"
# result = predict_damage(image_path)
# print(f"The given image is: {result}")
