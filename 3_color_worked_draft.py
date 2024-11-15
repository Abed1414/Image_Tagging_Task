import os
from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import KMeans
import colorsys

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

def categorize_image(image_path):
    matched_tags = []
    
    dominant_colors = extract_dominant_colors(image_path, num_colors=3)
    closest_colors = get_closest_colors(dominant_colors, max_matches=3)
    
    for i, color in enumerate(closest_colors):
        matched_tags.append(f"COLOR{i+1}: {color}")
    
    for i, type_ in enumerate(CATEGORIES["TYPE"][:3]):
        matched_tags.append(f"TYPE{i+1}: {type_}")
    
    for i, style in enumerate(CATEGORIES["STYLE"][:3]):
        matched_tags.append(f"STYLE{i+1}: {style}")
    
    matched_tags.append(f"USAGE: {CATEGORIES['USAGE'][0]}")  

    matched_tags = matched_tags[:10]
    
    return matched_tags

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']  
    
    if file:
        file_path = os.path.join(source_directory, file.filename)
        file.save(file_path)
        
        result = categorize_image(file_path)
        return jsonify({"status": result})

    return jsonify({"error": "No file provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
