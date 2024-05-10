# # Import required libraries
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import os
#
# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)
# # Define the directory to save processed video frames
# output_folder = r"D:\Division\type1-Data\temp_result"
# os.makedirs(output_folder, exist_ok=True)
#
# # Define function to process video
# def process_video(video_path, output_folder):
#     # Open the video
#     video = cv2.VideoCapture(video_path)
#
#     # Get video fps
#     fps = video.get(cv2.CAP_PROP_FPS)
#
#     # Calculate frame interval for every 2 seconds
#     frame_interval = int(fps * 2)
#
#     # Initialize frame counter
#     frame_counter = 0
#
#     while True:
#         # Read next frame
#         ret, frame = video.read()
#
#         # Break the loop if the video is over
#         if not ret:
#             break
#
#         # Extract frame every 2 seconds
#         if frame_counter % frame_interval == 0:
#             # Save the resized frame as an image
#             output_path = os.path.join(output_folder, f"frame_{frame_counter // frame_interval}.jpg")
#             cv2.imwrite(output_path, frame)
#
#         # Increment frame counter
#         frame_counter += 1
#
#     # Release resources
#     video.release()
#     cv2.destroyAllWindows()
#
# # Define route to process video
# # @app.route('/process_video', methods=['POST'])
# def process_video_route():
#     # Check if a video file is present in the request
#     if 'file' not in request.files:
#         return jsonify({"message": "No video file found"}), 400
#
#     file = request.files['file']
#
#     # Save the uploaded video to a temporary location
#     video_path = os.path.join(output_folder, "temp_video.mp4")
#     file.save(video_path)
#
#     # Process the video
#     process_video(video_path, output_folder)
#
#     # Return success message
#     return jsonify({"message": "Video processed successfully"})
#
# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
