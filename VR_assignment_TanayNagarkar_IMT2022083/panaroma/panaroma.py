#Make sure dependencies are installed can use this single command - pip install opencv-python opencv-contrib-python matplotlib numpy

#Importing necessary libraries 
import cv2
import matplotlib.pyplot as plt
import os
import glob

# Getting the directory where the script is currently located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Defining the relative path to the 'images' folder inside the 'panorama' directory
image_folder = os.path.join(script_dir, "images")

# Automatically fetching all JPEG images from the 'images' folder and sorting them
image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpeg")))

# Checking if there are any valid image files
if len(image_files) == 0:
    print("No images found in the 'images' folder.")
    exit()

# Loading images from the fetched file paths while ensuring they are not None
images = [cv2.imread(img) for img in image_files if cv2.imread(img) is not None]

# Checking if there are enough images for stitching (at least two images are required)
if len(images) < 2:
    print("Error: Not enough valid images for stitching.")
    exit()

# Initializing the ORB (Oriented FAST and Rotated BRIEF) detector for key point extraction
orb = cv2.ORB_create()

# Setting the maximum number of keypoints to 5000 for more detailed feature extraction
orb.setMaxFeatures(5000)

# Defining a function that is extracting and drawing keypoints on images
def extract_and_draw_keypoints(image):
    # Converting the image to grayscale to simplify keypoint detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecting keypoints and computing descriptors using ORB
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Drawing the detected keypoints on the original image in red color
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255), 
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Returning the image with keypoints, detected keypoints, and descriptors
    return image_with_keypoints, keypoints, descriptors

# Initializing lists to store images with keypoints, keypoints, and descriptors
images_with_keypoints = []
keypoints_list = []
descriptors_list = []

# Iterating through all images and extracting keypoints
for img in images:
    img_with_kp, kp, des = extract_and_draw_keypoints(img)  # Extracting keypoints and descriptors
    images_with_keypoints.append(img_with_kp)  # Storing the image with keypoints
    keypoints_list.append(kp)  # Storing keypoints for each image
    descriptors_list.append(des)  # Storing descriptors for each image

# Visualizing the detected keypoints for all images
plt.figure(figsize=(15, 15))  # Creating a figure of size 15x15 inches

# Iterating through the images with keypoints and displaying them in a grid layout
for i, img_with_kp in enumerate(images_with_keypoints):
    plt.subplot(3, 3, i+1)  # Arranging images in a 3x3 grid
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))  # Converting BGR to RGB for proper display
    plt.title(f"Image {i+1} with Key Points")  # Assigning titles to subplots
    plt.axis("off")  # Hiding axis for better visualization

plt.tight_layout()  # Adjusting layout to prevent overlap
plt.show()  # Displaying the images with detected keypoints

# Creating an instance of the OpenCV Stitcher class to stitch images together
stitcher = cv2.Stitcher_create()

# Stitching the images and storing the result along with the status code
status, panorama = stitcher.stitch(images)

# Checking if the stitching process is successful
if status == cv2.Stitcher_OK:
    # Saving the full-resolution stitched panorama as 'panorama.jpg'
    cv2.imwrite("panorama.jpg", panorama)

    # Resizing the stitched image for display without modifying the saved file
    display_width = 1200  # Setting the maximum width for the displayed image
    scale = display_width / panorama.shape[1]  # Maintaining the aspect ratio
    display_height = int(panorama.shape[0] * scale)  # Calculating the proportional height

    # Resizing the panorama image using INTER_AREA interpolation for better quality
    panorama_resized = cv2.resize(panorama, (display_width, display_height), interpolation=cv2.INTER_AREA)

    # Displaying the resized panorama using OpenCV's imshow function
    cv2.imshow("Panorama", panorama_resized)
    cv2.waitKey(0)  # Waiting indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Closing all OpenCV windows after key press
else:
    # Printing an error message if stitching fails along with the status code
    print(f"Stitching failed with error code: {status}")
