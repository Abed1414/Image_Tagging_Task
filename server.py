import os
from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras._tf_keras.keras.applications import MobileNetV2
# from keras._tf_keras.keras.applications import EfficientNetB0
from keras._tf_keras.keras.applications.mobilenet_v2 import preprocess_input
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.callbacks import ModelCheckpoint

CATEGORIES = {
    "TYPE": [  
        "Velvet", "Curtain", "Double Purpose", "Upholstery", "Wallcovering",
        "Embroidery", "Faux Fur", "Faux Leather", "Jacquard", "Microfiber",
        "Organic", "Print & Embossed", "Satin", "Sheer", "Suede", "Sunscreen",
        "Wallpanel", "Wallpaper", "Weave"
    ],
    "COLOR": [  
        "Black", "Blue", "Brown", "Dark Beige", "Dark Grey", "Green", "Light Beige",
        "Light Grey", "Metallic", "Multicolor", "Orange", "Pink", "Purple", "Red",
        "White", "Yellow"
    ],
    "STYLE": [  
        "Children", "Classical", "Contemporary & Modern", "Ethnic & Oriental", "Floral",
        "Geometric", "Illustrative", "Stripes; Checks; And Zigzags", "Plain", "Textured"
    ],
    "USAGE": [
        "Curtain", "Double Purpose", "Upholstery", "Wallcovering"
    ]
}

app = Flask(__name__)

source_directory = "source_images"
output_directory = "categorized_images"

os.makedirs(source_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

def extract_dominant_colors(image_path, num_colors=3):
    img = Image.open(image_path)
    img = img.convert("RGB")
    pixels = np.array(img)
    
    pixels = pixels.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors).fit(pixels)
    
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    return [tuple(color) for color in dominant_colors]

def get_closest_colors(dominant_rgb_colors, max_matches=3, threshold=100):
    color_mapping = {
        "Black": (0, 0, 0), "Blue": (0, 0, 255), "Brown": (139, 69, 19),
        "Dark Beige": (139, 137, 112), "Dark Grey": (169, 169, 169),
        "Green": (0, 255, 0), "Light Beige": (245, 245, 220), "Light Grey": (211, 211, 211),
        "Metallic": (192, 192, 192), "Orange": (255, 165, 0), "Pink": (255, 192, 203),
        "Purple": (128, 0, 128), "Red": (255, 0, 0), "White": (255, 255, 255), "Yellow": (255, 255, 0)
    }
    
    matched_colors = []
    
    for rgb in dominant_rgb_colors:
        closest_color = None
        min_distance = float('inf')
        
        for color_name, color_rgb in color_mapping.items():
            distance = np.linalg.norm(np.array(rgb) - np.array(color_rgb))
            if distance < min_distance:
                closest_color = color_name
                min_distance = distance
        
        if closest_color and min_distance <= threshold:
            matched_colors.append((closest_color, min_distance))
    
    matched_colors.sort(key=lambda x: x[1])
    best_matches = [color for color, _ in matched_colors[:max_matches]]
    
    return best_matches

def build_custom_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False) 
    # base_model = EfficientNetB0(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    num_classes = len(CATEGORIES["TYPE"]) + len(CATEGORIES["STYLE"]) + len(CATEGORIES["USAGE"])
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def fine_tune_model(model, train_data_dir, validation_data_dir, model_save_path, epochs=10, batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=callbacks_list
    )

def classify_image_with_model(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    return predictions

def categorize_image(image_path, model, type_threshold=0.1, style_threshold=0.1, color_threshold=100):
    matched_tags = []
    
    dominant_colors = extract_dominant_colors(image_path, num_colors=3)
    closest_colors = get_closest_colors(dominant_colors, max_matches=3, threshold=color_threshold)
    
    for i, color in enumerate(closest_colors):
        matched_tags.append(f"COLOR{i+1}: {color}")
    
    predictions = classify_image_with_model(image_path, model)
    
    type_labels = CATEGORIES["TYPE"]
    style_labels = CATEGORIES["STYLE"]
    usage_labels = CATEGORIES["USAGE"]
    
    type_predictions = predictions[0][:len(type_labels)]
    style_predictions = predictions[0][len(type_labels):len(type_labels) + len(style_labels)]
    usage_predictions = predictions[0][len(type_labels) + len(style_labels):]
    
    top_types = np.argsort(type_predictions)[-3:][::-1]
    top_styles = np.argsort(style_predictions)[-3:][::-1]
    top_usage = np.argmax(usage_predictions)
    
    filtered_types = [type_labels[i] for i in top_types if type_predictions[i] >= type_threshold]
    filtered_styles = [style_labels[i] for i in top_styles if style_predictions[i] >= style_threshold]
    
    if not filtered_types:
        filtered_types = [type_labels[top_types[0]]]
    if not filtered_styles:
        filtered_styles = [style_labels[top_styles[0]]]
    
    filtered_types = filtered_types[:3]
    filtered_styles = filtered_styles[:3]
    
    for i, type_label in enumerate(filtered_types):
        matched_tags.append(f"TYPE{i+1}: {type_label}")
    
    for i, style_label in enumerate(filtered_styles):
        matched_tags.append(f"STYLE{i+1}: {style_label}")
    
    matched_tags.append(f"USAGE: {usage_labels[top_usage]}")
    
    matched_tags = matched_tags[:10]
    
    return matched_tags

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    type_threshold = 0.0001
    style_threshold = 0.0001
    color_threshold = 100
    
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    file_path = os.path.join(source_directory, file.filename)
    file.save(file_path)
    
    model = build_custom_model()
    result = categorize_image(file_path, model, type_threshold, style_threshold, color_threshold)
    
    return jsonify({"status": result})

if __name__ == '__main__':
    app.run(debug=True)
