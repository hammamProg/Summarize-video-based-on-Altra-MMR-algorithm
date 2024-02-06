import cv2
import numpy as np

# Define a function to extract histogram features for a given frame
def extract_histogram_features(frame):
    # Assuming frame is in BGR format, you can convert it to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram features for each channel (Hue, Saturation, Value)
    hist_hue = cv2.calcHist([hsv_frame], [0], None, [256], [0, 256]).flatten()
    hist_saturation = cv2.calcHist([hsv_frame], [1], None, [256], [0, 256]).flatten()
    hist_value = cv2.calcHist([hsv_frame], [2], None, [256], [0, 256]).flatten()
    
    # Concatenate the histograms into a single feature vector
    histogram_features = np.concatenate((hist_hue, hist_saturation, hist_value))
    
    return histogram_features.flatten()

def frames_to_video(frames, output_video_path, fps=30):
    if not frames:
        raise ValueError("The 'frames' list is empty. Please provide frames to create the video.")

    frame_height, frame_width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for video compression
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)
    out.release()

def video_to_frames(video_path, output_size=(224, 224)):
    vidcap = cv2.VideoCapture(str(video_path))
    frames = []  # List to store frames
 
    success, image = vidcap.read()
    count = 0

    while success:
        # Resize the frame to the specified output size
        resized_frame = cv2.resize(image, output_size)

        frames.append(resized_frame)  # Append the current frame to the list
        success, image = vidcap.read()
        count += 1

    return np.array(frames) # return vector of frames 