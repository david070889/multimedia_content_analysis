"""
    ---- Overview ----
    Given that the script download.py downloaded videos from the FF++
    dataset, converts those videos into images (frame by frame) for
    machine learning purposes. Images are cropped using dlib around the 
    face area.
    ------------------
"""

# Library Imports

import argparse
import cv2
import dlib
import os
import random

from PIL import Image as pil_image

# Function declarations

def get_bounding_box(face, width, height, scale=1.3, minsize=None):
    """
    Generates a quadratic bounding box using the dlib library.
    --- 
    parameters:
        face    : dlib face class
        height  : frame height
        minsize : minimum b-box size
        scale   : b-box size multiplier to increase the face region
    outputs:
        x       : top left corner x coordinate 
        y       : top left corner y coordinate
        size_bb : length of sides of the square b-box
    """
    # Computes the size of the bounding box
    size_bb = max(face.right()-face.left(),face.bottom()-face.top())
    size_bb = int(size_bb*scale)
    if minsize and size_bb < minsize:
        size_bb = minsize
    # Computes the bounding box's top-left corner
    # with an out of bound check
    x = face.left() + face.right()
    y = face.top() + face.bottom()
    x = max(int((x - size_bb) // 2), 0)
    y = max(int((y - size_bb) // 2), 0)
    # Computes the final bounding box size
    size_bb = min(size_bb, width - x, height - y)
    return x, y, size_bb

def convert_video2image(
        video_path, label, image_path_real, image_path_fake,
        start_frame=0, end_frame=None
        ):
    """
    Converts a video path into a list of images.
    30 images are produced per video, randomly selected 
    without replacement.
    ---
    parameters:
        video_path     : Path of the video to convert
        image_path_real: Path where the real images will be saved
        image_path_fake: Path where the fake images will be saved
        start_frame    : Indicates where to start in the video
        end_frame      : Indicates where to stop in the video
    """
    print(f"Launching process on video: {video_path}")
    # Initiates a frame by frame reader object
    reader = cv2.VideoCapture(video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initiates the dlib face detector
    face_detector = dlib.get_frontal_face_detector()
    # Initiates frame numbers to process the tested video
    frame_number = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    # Computes which frames to extract (maximum of 30)
    nb = 30
    frames_to_extract = sorted(random.sample(range(end_frame), nb))
    last_frame = frames_to_extract[-1]

    # Get the label 
    image_path = image_path_fake if label == 0 else image_path_real

    ############################
    # PERFORMS THE READER LOOP #
    # OVER THE VIDEO FRAMES    #
    ############################
    while reader.isOpened():
        # Retrieves the next frame of the video
        frame_number += 1
        _, image = reader.read()
        if frame_number not in frames_to_extract: 
            if frame_number > last_frame: break
            else: continue
        if image is None: break
        # Retrieves the image size
        height, width = image.shape[:2]
        # Performs the face detection with dlib
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(grayscale_image, 1)
        if len(faces):
            # Retrieves the first/largest face
            face = faces[0]
            # Crops the face out of the frame picture
            x, y, size = get_bounding_box(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            # splits the path to retrieve a list with the 
            # video name and its extension. Splices in the frame
            # number
            name = video_path.split("/")[-1].split(".")
            name = name[:-1] + [str(frame_number)]
            name = "_".join(name)+".jpg"
            # Saves the picture
            im = pil_image.fromarray(cv2.cvtColor(
                cropped_face, cv2.COLOR_BGR2RGB
                ))            
            im.save(image_path+f"{name}")
        # Checks if at the end of the video
        if frame_number >= end_frame:
            break

def parse_video_list(file_path):
    video_entries = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                label = int(parts[0])
                video_path = parts[1]
                video_entries.append((label, video_path))
            else:
                print(f"Invalid line format: {line}")
    return video_entries

if __name__ == "__main__":
    txt = 'List_of_testing_videos.txt'
    video_entries = parse_video_list(txt)

    output_dir_real = 'extracted_frames/real/'
    output_dir_fake = 'extracted_frames/fake/'

    for label, video_path in video_entries:
        if os.path.isfile(video_path):
            convert_video2image(video_path, label, output_dir_real, output_dir_fake)
        else:
            print(f"Video file not found: {video_path}")
