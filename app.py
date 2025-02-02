import os
import cv2
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import base64
from io import BytesIO
import numpy as np


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './outputs'

# Increase request size limit (e.g., 100MB)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 100MB

app.config['MAX_FORM_MEMORY_SIZE'] = 50 * (2 ** 10) ** 2

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load the model
# MODEL_PATH = './pretrained_model/RealESRGAN_x4plus_anime_6B.pth'
# MODEL_PATH = './pretrained_model/RealESRGAN_x4plus.pth'

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# MODEL = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
# UPSAMPLER = RealESRGANer(scale=4, model_path=MODEL_PATH, model=MODEL, tile=0, half=False, gpu_id=None)

# Define RRDBNet with correct parameters matching RealESRGAN_x4plus.pth
# MODEL = RRDBNet(
#     num_in_ch=3, 
#     num_out_ch=3, 
#     num_feat=64, 
#     num_block=23,  # Change from 6 to 23
#     num_grow_ch=32, 
#     scale=4
# )

# UPSAMPLER = RealESRGANer(
#     scale=4, 
#     model_path=MODEL_PATH, 
#     model=MODEL, 
#     tile=0, 
#     half=False, 
#     gpu_id=None
# )

@app.route('/enhance', methods=['POST'])
def enhance_image():
    try:
        print(f"Received content length: {request.content_length} bytes")
        if 'image' not in request.form:
            return jsonify({'error': 'No image provided'}), 400
        
        data_uri = request.form['image']
        file_name_with_ext = request.form["fileNameWithExt"]
        file_ext = request.form["fileExt"]

        # Check which model to load
        MODEL_PATH = None
        if request.form["isAnime"] == "y":
            MODEL_PATH = './pretrained_model/RealESRGAN_x4plus_anime_6B.pth'
        else:
            MODEL_PATH = './pretrained_model/RealESRGAN_x4plus.pth'

        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': f"Model file not found at {MODEL_PATH}"}), 400
        
        # set the configuration of selected model
        MODEL = None
        UPSAMPLER = None

        if request.form["isAnime"] == "y":
            MODEL = RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=6, 
                num_grow_ch=32, 
                scale=4)
            
            UPSAMPLER = RealESRGANer(
                scale=4, 
                model_path=MODEL_PATH, 
                model=MODEL, 
                tile=0, 
                half=False, 
                gpu_id=None)
        else:
            MODEL = RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=23,
                num_grow_ch=32, 
                scale=4
            )

            UPSAMPLER = RealESRGANer(
                scale=4, 
                model_path=MODEL_PATH, 
                model=MODEL, 
                tile=0, 
                half=False, 
                gpu_id=None
            )

        
        # Strip out the header part (e.g., "data:image/jpeg;base64,")
        if ',' in data_uri:
            data_uri = data_uri.split(',')[1]

        # Decode the base64 string into bytes
        image_data = base64.b64decode(data_uri)

        # Convert byte data into a numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Save image temporarily to process it
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name_with_ext)
        cv2.imwrite(input_path, img)
        
        # Process the image (enhance with ESRGAN, for example)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'enhanced_image.jpg')
        
        # Here, you would perform the enhancement using your model
        # For example:
        output, _ = UPSAMPLER.enhance(img, outscale=4)
        cv2.imwrite(output_path, output)

        # Encode the processed image back into Base64 format
        _, buffer = cv2.imencode(f'.{file_ext}', output)
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"data:image is ready!!!")
        # Return the enhanced image as a Base64 string
        return jsonify({'status': 'success', 'message': 'Image enhanced successfully', 'enhanced_image': f"data:image/{file_ext};base64,{enhanced_base64}"}), 200
    
    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'message': 'Error occurred!!'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
