from flask import Flask, request, jsonify, send_file, make_response
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import keras
from flask_cors import CORS
from io import BytesIO
from flask import send_file
from werkzeug.utils import secure_filename
import random
from metrics import categorical_dice_loss, categorical_dice_coef

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app



class Insulator:
    def __init__(self, leakage_current, applied_voltage, temperature=None, humidity=None):
        self._leakage_current = leakage_current
        self._applied_voltage = applied_voltage
        self._temperature = temperature if temperature is not None else 25
        self._humidity = humidity if humidity is not None else 75
    @property
    def leakage_current(self):
        return self._leakage_current

    @property
    def applied_voltage(self):
        return self._applied_voltage

    @property
    def temperature(self):
        return self._temperature

    @property
    def humidity(self):
        return self._humidity

    def calculate_life_left(self, max_leakage_current=10, max_applied_voltage=1000,
                            life_reduction_factor_per_degree=0.02,
                            life_reduction_factor_per_percent_humidity=0.075, reference_temperature=random.randint(25, 35)):
        print(reference_temperature)
        leakage_ratio = min(1, self.leakage_current / max_leakage_current)
        voltage_ratio = min(1, self.applied_voltage / max_applied_voltage)

        life_left_percentage = (leakage_ratio + voltage_ratio) * 50
        life_left_percentage -= (self.temperature - reference_temperature) * life_reduction_factor_per_degree
        life_left_percentage -= self.humidity * life_reduction_factor_per_percent_humidity

        return max(0, life_left_percentage)

# Define the directory to save processed video frames
output_folder = r"D:\Division\type1-Data\temp_result"
os.makedirs(output_folder, exist_ok=True)

# Define function to process video
def process_video(video_path, output_folder):
    # Open the video
    video = cv2.VideoCapture(video_path)

    # Get video fps
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval for every 2 seconds
    frame_interval = int(fps * 2)

    # Initialize frame counter
    frame_counter = 0
    image_counter = 0

    while True:
        # Read next frame
        ret, frame = video.read()

        # Break the loop if the video is over
        if not ret:
            break

        # Extract frame every 2 seconds
        if frame_counter % frame_interval == 0:
            # Resize the frame to 640x640
            resized_frame = cv2.resize(frame, (640, 640))

            # Save the resized frame as an image with sequential numbering
            output_path = os.path.join(output_folder, f"{image_counter}.jpg")
            cv2.imwrite(output_path, resized_frame)

            image_counter += 1

        # Increment frame counter
        frame_counter += 1

    # Release resources
    video.release()
    cv2.destroyAllWindows()
processed_images_dir = r"D:\Division\type1-Data\testimageessed_images"

# Adjust the size to match your U-Net training input
H, W = 640, 640

# Directory to save processed images
results_dir = r"D:\Division\type1-Data\testimage"
os.makedirs(results_dir, exist_ok=True)

# Load your U-Net model
# model_path = r"D:\zip256\Model complete\files\model.h5"
model_path = r"C:\Users\abhad\OneDrive\Pictures\files\model.h5"
# model_path = r"E:\FYP\files\model.h5"
model = load_model(model_path, custom_objects={'categorical_dice_loss': categorical_dice_loss, 'categorical_dice_coef': categorical_dice_coef})

def read_and_preprocess_image(file):
    img_array = np.frombuffer(file.read(), np.uint8)
    original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    original_image = cv2.resize(original_image, (W, H))

    # Preprocessing: Histogram Equalization in the YUV color space
    img_yuv = cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    original_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    preprocessed_image = original_image / 255.0
    preprocessed_image = preprocessed_image.astype(np.float32)
    return original_image, preprocessed_image

def save_results(original_image, y_pred, save_image_path):
    # Predicted mask with argmax to convert categorical to integer mask
    y_pred_argmax = np.argmax(y_pred, axis=-1)

    # Resize the predicted mask to match the original image size
    y_pred_argmax = cv2.resize(y_pred_argmax, (original_image.shape[1], original_image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    # Create an overlay image where the prediction mask will be overlaid on the original image
    overlay_image = original_image.copy()

    # Define the colors for your classes (assuming 3 classes including the background)
    # Class 1: Insulator, Class 2: Unhealthy part of the insulator
    class_colors = {1: [0, 0, 255], 2: [255, 0, 0]}  # Class index to BGR color mapping

    # Loop through your classes and apply colors
    for class_value, color in class_colors.items():
        overlay_image[y_pred_argmax == class_value] = color

    # Where there's no prediction (class 0), we keep the original image
    overlay_image[y_pred_argmax == 0] = original_image[y_pred_argmax == 0]

    # Concatenate the original image and the overlay image side-by-side
    concatenated_image = np.concatenate((original_image, overlay_image), axis=1)

    # Save the concatenated image to the specified path
    with open(save_image_path, 'wb') as f:
        f.write(cv2.imencode('.png', concatenated_image)[1])

def get_valid_float_input(value):
    try:
        return float(value)
    except ValueError:
        return None

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    original_image, image = read_and_preprocess_image(file)

    x = np.expand_dims(image, axis=0)  # Add batch dimension
    y_pred = model.predict(x)[0]  # Remove batch dimension with [0]

    # Generate a unique filename for the processed image
    filename = secure_filename(file.filename)
    save_image_path = os.path.join(results_dir, filename)

    # Save the processed image
    save_results(original_image, y_pred, save_image_path)

    # Return the filename of the processed image
    return jsonify({"processed_image_path": filename})

# Define route to process video
@app.route('/process_video', methods=['POST'])
def process_video_route():
    # Check if a video file is present in the request
    if 'file' not in request.files:
        return jsonify({"message": "No video file found"}), 400

    file = request.files['file']

    # Save the uploaded video to a temporary location
    video_path = os.path.join(output_folder, "temp_video.mp4")
    file.save(video_path)

    # Process the video
    process_video(video_path, output_folder)

    # Return success message
    return jsonify({"message": "Video processed successfully"})

# Endpoint to handle file upload and predict life left
@app.route('/predict_life', methods=['POST'])
def predict_life():
    # Get the input values from the user
    data = request.get_json()
    leakage = get_valid_float_input(data.get('leakage_current'))*100
    voltage = get_valid_float_input(data.get('applied_voltage'))

    if not all([leakage, voltage]):
        return jsonify({"error": "Invalid input. Please enter valid floating-point numbers."}), 400

    # Create an instance of the Insulator class
    insulator = Insulator(leakage_current=leakage, applied_voltage=voltage)

    # Calculate life left
    life_left = (insulator.calculate_life_left()/100)*50

    # Return the predicted life left
    return jsonify({"life_left": life_left})
    # return jsonify({"The Expected life_left based on these parameters is:": life_left})
if __name__ == '__main__':
    app.run(debug=True)
