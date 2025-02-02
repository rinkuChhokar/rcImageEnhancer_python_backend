# Image Enhancement API using Real-ESRGAN

This project provides a Flask-based API to enhance images using Real-ESRGAN. The API allows users to upload an image, enhance it using a pre-trained Real-ESRGAN model, and receive the enhanced image as a Base64-encoded string.

## Features

- Image enhancement using Real-ESRGAN models.
- Supports both general and anime-specific enhancement models.
- Accepts images in Base64 format.
- Returns the enhanced image in Base64 format.
- Cross-Origin Resource Sharing (CORS) enabled.

## Technologies Used

- Python
- Flask
- OpenCV (cv2)
- Torch
- Real-ESRGAN
- NumPy
- Flask-CORS

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- pip

### Clone the Repository

```sh
 git clone https://github.com/rinkuChhokar/rcImageEnhancer_python_backend.git
 cd rcImageEnhancer_python_backend
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Download Pretrained Models

Download the required pre-trained models from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and place them in the `pretrained_model/` directory:

- `RealESRGAN_x4plus.pth` (for general images)
- `RealESRGAN_x4plus_anime_6B.pth` (for anime-style images)

## Usage

### Start the Server

Run the Flask application:

```sh
python app.py
```

The server will start on `http://127.0.0.1:5000/`.

### API Endpoint

#### `POST /enhance`

Enhances an input image using the selected Real-ESRGAN model.

##### Request Parameters (Form Data)

| Parameter         | Type   | Description                               |
| ----------------- | ------ | ----------------------------------------- |
| `image`           | string | Base64-encoded image                      |
| `fileNameWithExt` | string | Original filename with extension          |
| `fileExt`         | string | Image file extension (e.g., `jpg`)        |
| `isAnime`         | string | Use anime model (`y` for yes, `n` for no) |

##### Example Request (cURL)

```sh
curl -X POST "http://127.0.0.1:5000/enhance" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     --data-urlencode "image=BASE64_ENCODED_IMAGE" \
     --data-urlencode "fileNameWithExt=example.jpg" \
     --data-urlencode "fileExt=jpg" \
     --data-urlencode "isAnime=n"
```

##### Response Format (JSON)

```json
{
  "status": "success",
  "message": "Image enhanced successfully",
  "enhanced_image": "data:image/jpg;base64,..."
}
```

## Directory Structure

```
image-enhancement-api/
│── pretrained_model/   # Directory for storing model files
│── uploads/            # Stores uploaded images
│── outputs/            # Stores enhanced images
│── app.py              # Main Flask application
│── requirements.txt    # Python dependencies
│── README.md           # Documentation
```

## Error Handling

| Error Message          | Possible Cause                   |
| ---------------------- | -------------------------------- |
| `No image provided`    | No image was sent in the request |
| `Model file not found` | Missing or incorrect model path  |
| `Invalid image file`   | Image decoding failed            |
| `Error occurred!!`     | General exception handling       |

## Contributing

Feel free to submit issues and pull requests to improve the project.

## License

This project is licensed under the MIT License.

## Author

Developed by [Rinku](https://github.com/rinkuChhokar)
