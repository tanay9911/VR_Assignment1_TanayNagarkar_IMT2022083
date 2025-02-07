#Importing necessary libraries
#Make sure your system has OpenCV installed can use command - pip install opencv-python

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

# Defining the path to the 'images' folder inside the 'coin_detection' directory
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "images")

# Function to dynamically load images from the images folder
def load_and_preprocess_image(image_filename):
    """
    Loading the image, enhancing it using CLAHE, and then applying bilateral filtering for noise reduction.
    """
    # Constructing the full image path
    image_path = os.path.join(IMAGE_FOLDER, image_filename)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Converting to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Applying bilateral filtering for edge-preserving noise reduction
    blurred = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
    
    return image, enhanced_gray, blurred

# Edge Detection with Morphological Fixes
def detect_edges(image):
    """
    Performing edge detection using the Canny algorithm and fixing broken edges with morphological operations.
    """
    # Canny Edge Detection
    edges = cv2.Canny(image, threshold1=19, threshold2=80)

    # Morphological closing to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closed_edges

# Checking circularity of a contour
def is_contour_circular(contour):
    """
    Checks if a contour is circular based on its circularity measure.
    """
    # Calculating contour area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Avoiding division by zero
    if perimeter == 0:
        return False
    
    # Calculating circularity: (4 * pi * area) / (perimeter^2)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # Circularity should be close to 1 for a perfect circle
    return 0.3 < circularity < 1.2  # 0.3 to 1.2 is a reasonable range

# Detecting coins and segmenting them
def detect_and_segment_coins(original_image, edges):
    """
    Using contours to detect and segment coins from the image.
    """
    # Finding contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtering contours based on area and circularity
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Checking for valid area and circularity
        if 80 < area and is_contour_circular(contour):
            valid_contours.append(contour)
    
    # Drawing contours and segmenting coins
    outlined_image = original_image.copy()
    segmented_coins = []
    
    for contour in valid_contours:
        # Drawing the contours
        cv2.drawContours(outlined_image, [contour], -1, (0, 255, 0), 2)
        
        # Creating a bounding box for the coin
        x, y, w, h = cv2.boundingRect(contour)
        coin = original_image[y:y+h, x:x+w]
        segmented_coins.append(coin)
    
    return outlined_image, segmented_coins

# Counting the total number of coins
def count_coins(contours):
    """
    Count the total number of valid contours (coins).
    """
    return len(contours)

# Visualization with zoom functionality
def visualize_results(original_image, edges, outlined_image, segmented_coins):
    """
    Displaying the results including edge detection, outlined coins, and segmented coins,
    with a dark blue background and zoom functionality.
    """
    # Creating a figure with a black background
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor('darkblue')  # Setting full figure background to dark blue
    ax.set_facecolor('darkblue')  # Setting the subplot background to dark blue
    ax.set_title("Outlined Coins(zoomable)", color='yellow', fontsize =20)  # Yellow title text for visibility

    # Displaying the outlined image
    img = ax.imshow(cv2.cvtColor(outlined_image, cv2.COLOR_BGR2RGB))
    ax.axis("off")  # Hiding axes

    # Adding Zoom functionality
    def zoom(event):
        nonlocal img
        zoom_factor = 1.2
        if event.button == 'down':  # Zoom In
            scale = zoom_factor
        elif event.button == 'up':  # Zoom Out
            scale = 1 / zoom_factor
        else:
            return

        # Getiting current limits
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        x_center, y_center = np.mean(xlim), np.mean(ylim)
        
        width = (xlim[1] - xlim[0]) * scale
        height = (ylim[1] - ylim[0]) * scale

        # Setting new limits based on zooming
        ax.set_xlim(x_center - width / 2, x_center + width / 2)
        ax.set_ylim(y_center - height / 2, y_center + height / 2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('scroll_event', zoom)
    
    # Display edge detection and segmented images
    fig2 = plt.figure(figsize=(15, 10))
    fig2.patch.set_facecolor('darkblue')  # Set figure background to dark blue

    # Showing Original Image
    plt.subplot(2, 3, 1)
    plt.title("Original Image", color='yellow',fontsize = 20)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Edge Detection with Yellow Border
    ax2 = plt.subplot(2, 3, 2)
    plt.title("Enhanced Edges", color='yellow',fontsize =20)
    plt.imshow(edges, cmap="gray")
    plt.axis("off")
    plt.figtext(0.5, 0.5, "Segmented Coins", ha="center", va="center", fontsize=20, color='yellow')
    # Showing outlined Coins
    plt.subplot(2, 3, 3)
    plt.title("Detected Coins", color='yellow', fontsize =20)
    plt.imshow(cv2.cvtColor(outlined_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    warnings.filterwarnings("ignore", message="tight_layout not applied")
    
    # Displaying the segmented coins
    for i, coin in enumerate(segmented_coins):
        plt.subplot(2, len(segmented_coins), len(segmented_coins) + i + 1)
        plt.title(f"Coin {i+1}", color='yellow')
        plt.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# defining main function to execute the program
def main():
    # Listing all images in the 'images' folder
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpeg")]
    
    # Checking if there are any images in the folder
    if not image_files:
        print("No .jpeg images found in the 'images' folder.")
        return
    
    # Displaying image options to the user
    print("Available images:")
    for idx, image_filename in enumerate(image_files, start=1):
        print(f"{idx}. {image_filename}")
    
    # Prompting the user to select an image by index
    try:
        user_choice = int(input("Enter the index of the image of which you want to detect the coins: "))
        
        if user_choice < 1 or user_choice > len(image_files):
            print("Invalid choice. Please enter a valid index.")
            return
        
        # Getting the selected image filename
        selected_image_filename = image_files[user_choice - 1]
        
        # Loading and preprocessing the selected image
        original_image, enhanced_gray, blurred_image = load_and_preprocess_image(selected_image_filename)

        # Detecting edges with morphological fixes
        edges = detect_edges(blurred_image)

        # Detecting and segmenting coins
        outlined_image, segmented_coins = detect_and_segment_coins(original_image, edges)

        # Counting the coins
        total_coins = len(segmented_coins)
        print(f"Total Coins Detected in {selected_image_filename}: {total_coins}")

        # Visualizing the results with zoom functionality
        visualize_results(original_image, edges, outlined_image, segmented_coins)

    except ValueError:
        print("Invalid input. Please enter a numeric index.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
