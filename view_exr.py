import os
import argparse
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" 

import cv2
import matplotlib.pyplot as plt
import numpy as np

def display_EXR(filename):
    # use IMREAD_ANYCOLOR or IMREAD_ANYDEPTH to ensure all color channels and the full depth are loaded.
    image_bgr = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    if image_bgr is None:
        print(f"Error: Could not read image from {filename}")
        return

    # convert the color order.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title(f"Displaying {filename}")
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Path to the EXR file to display")
    args = parser.parse_args()

    filename = args.filename
    print(f"Processing file: {filename}")
    
    try:
        display_EXR(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()