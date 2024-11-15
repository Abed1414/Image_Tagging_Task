import os # keras._tf_keras.keras
from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.applications.mobilenet_v2 import preprocess_input
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D

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

def get_closest_colors(dominant_rgb_colors, max_matches=3):
    color_mapping = {
        "Black": (0, 0, 0), "Blue": (0, 0, 255), "Brown": (139, 69, 19),
        "Dark Beige": (139, 137, 112), "Dark Grey": (169, 169, 169),
        "Green": (0, 255, 0), "Light Beige": (245, 245, 220), "Light Grey": (211, 211, 211),
        "Metallic": (192, 192, 192), "Orange": (255, 165, 0), "Pink": (255, 192, 203),
        "Purple": (128, 0, 128), "Red": (255, 0, 0), "White": (255, 255, 255), "Yellow": (255, 255, 0)
    }
    
    matched_colors = set()
    
    for rgb in dominant_rgb_colors:
        closest_color = None
        min_distance = float('inf')
        
        for color_name, color_rgb in color_mapping.items():
            distance = np.linalg.norm(np.array(rgb) - np.array(color_rgb))
            if distance < min_distance:
                closest_color = color_name
                min_distance = distance
        
        if closest_color and closest_color not in matched_colors:
            matched_colors.add(closest_color)
            if len(matched_colors) >= max_matches:
                break
    
    return list(matched_colors)

def build_custom_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False)
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

def classify_image_with_model(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    return predictions

def categorize_image(image_path, model):
    matched_tags = []
    
    dominant_colors = extract_dominant_colors(image_path, num_colors=3)
    closest_colors = get_closest_colors(dominant_colors, max_matches=3)
    
    for i, color in enumerate(closest_colors):
        matched_tags.append(f"COLOR{i+1}: {color}")
    
    predictions = classify_image_with_model(image_path, model)
    
    type_labels = CATEGORIES["TYPE"]
    style_labels = CATEGORIES["STYLE"]
    usage_labels = CATEGORIES["USAGE"]
    
    type_predictions = predictions[0][:len(type_labels)]
    style_predictions = predictions[0][len(type_labels):len(type_labels) + len(style_labels)]
    usage_predictions = predictions[0][len(type_labels) + len(style_labels):]
    
    matched_tags.append(f"TYPE: {type_labels[np.argmax(type_predictions)]}")
    matched_tags.append(f"STYLE: {style_labels[np.argmax(style_predictions)]}")
    matched_tags.append(f"USAGE: {usage_labels[np.argmax(usage_predictions)]}")
    
    matched_tags = matched_tags[:10]
    
    return matched_tags

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']  
    
    if file:
        file_path = os.path.join(source_directory, file.filename)
        file.save(file_path)
        
        model = build_custom_model()
        result = categorize_image(file_path, model)
        return jsonify({"status": result})

    return jsonify({"error": "No file provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
