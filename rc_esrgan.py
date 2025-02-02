import os
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Choose the correct model
model_path = './pretrained_model/RealESRGAN_x4plus_anime_6B.pth'

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

def rc_enhance_image(input_path, output_path):
    print(f"Processing: {input_path}")

    # Use correct architecture for the model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

    # Let RealESRGANer handle model loading
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        half=False,
        gpu_id=None
    )

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    output, _ = upsampler.enhance(img, outscale=4)
    cv2.imwrite(output_path, output)

    if os.path.exists(output_path):
        print(f"Image saved at: {os.path.abspath(output_path)}")
    else:
        print("Error: Image was not saved!")
