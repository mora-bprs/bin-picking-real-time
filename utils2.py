import cv2
import numpy as np
import torch
from fastsam import FastSAMPrompt
from matplotlib import pyplot as plt

import config


# updated code
# updated code
def get_bounding_box_coordinates(mask, DEBUG = False):
    """
    Calculate the coordinates of the bounding box surrounding the True region in the mask.

    Parameters
    ----------
    mask : np.array
        Binary mask where True indicates the region of interest.

    Returns
    -------
    dict
        Dictionary containing the coordinates of the corners of the bounding box.
        Dictionary keys: "top_left", "top_right", "bottom_right", "bottom_left".
        Dictionary elements: tuple : (y, x) coordinates of the corners.
    """
    
    DEBUG = DEBUG
    
    # Convert boolean mask to uint8 type
    mask_uint8 = mask.astype(np.uint8) * 255
    
        # Threshold the image
    _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a copy of the original image for drawing
    result = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)

    # Draw the largest contour
    cv2.drawContours(result, [largest_contour], 0, (0, 0, 255), 5)

    # Display the mask
    if DEBUG : cv2.imshow('Binary Mask with contours', result)
    
    # Reshape the contour to remove the extra dimension
    contour_points = largest_contour.reshape(-1, 2)

    # Get the top-most point (minimum y)
    top_most = tuple(contour_points[contour_points[:, 1].argmin()])
    bottom_most = tuple(contour_points[contour_points[:, 1].argmax()])

    # Get the left-most point (minimum x)
    left_most = tuple(contour_points[contour_points[:, 0].argmin()])
    right_most = tuple(contour_points[contour_points[:, 0].argmax()])

    top_most = top_most[1], top_most[0]
    bottom_most = bottom_most[1], bottom_most[0]
    left_most = left_most[1], left_most[0]
    # right_most = right_most[1], right_most[0]
    right_most =  right_most[1], right_most[0]   
    
    
    # Print the coordinates
    # print(f"Leftmost coordinate: {left_most}")
    # print(f"Rightmost coordinate: {right_most}")
    # print(f"Topmost coordinate: {top_most}")
    # print(f"Bottommost coordinate: {bottom_most}")
    
    top_left = left_most # red
    top_right = top_most # blue
    bottom_right = right_most # green (this is where the error is)
    bottom_left = bottom_most # yellow/ cyan in opencv
    
    # Blue and Cyan points line is the orientation of the box
    orientation_angle = np.arctan2((abs(top_right[1] - bottom_left[1])), (top_right[0] - bottom_left[0])) * 180 / np.pi
    

    # orientation_angle is tan inverse of top
    
    if abs(top_left[1] - top_right[1]) < 10 and abs(top_left[0] - top_right[0]) < 10:
    # Find the indices where the mask is True
        true_indices = np.argwhere(mask)

        # Get the bounding box of the True region
        top_left = tuple(np.min(true_indices, axis=0))
        bottom_right = tuple(np.max(true_indices, axis=0))

        # Remove the channels layer coordinate
        top_left = top_left[:-1]
        bottom_right = bottom_right[:-1]

        # Calculate the width and height of the bounding box
        width = bottom_right[1] - top_left[1]
        height = bottom_right[0] - top_left[0]

        # make a numpy array for top_right and bottom_left
        top_right = (top_left[0], top_left[1] + width)
        bottom_left = (top_left[0] + height, top_left[1])
        
        orientation_angle = np.arctan2((abs(top_left[1] - bottom_left[1])), (top_left[0] - bottom_left[0])) * 180 / np.pi
     
    
    
    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_right": bottom_right,
        "bottom_left": bottom_left,
        "orientation": orientation_angle
    }


def process_coordinates(mask, frame_h, frame_w):
    # find the indices where the mask is true
    true_indices = np.argwhere(mask)

    y1, x1 = tuple(np.min(true_indices, axis=0))[:-1]
    y4, x4 = tuple(np.max(true_indices, axis=0))[:-1]
    print(y1, y4, x1, x4)

    # calculate the width and height of the bounding box
    width = x4 - x1
    height = y4 - y1
    # print("box height: ", height, "box width: ", width)

    # make a numpy array for top_right and bottom_left
    # y2, x2 = y1, x1 + width
    # y3, x3 = y1 + height, x1

    # box center
    bx, by = x1 + width / 2, y1 + height / 2

    ty, tx = abs((1088 / 2) - by), abs((1920 / 2) - bx)
    theta = np.arctan(ty / tx)
    print("translation coords: ", ty, tx, "theta: ", theta)

    return {"ty": ty, "tx": tx, "theta": theta}


def get_device():
    """
    Get input from the terminal

    returns :
    device : str : "cuda" or "cpu"
    """
    device = input("Choose the device you want to use: 'cuda' or 'cpu' >>> ")

    # if cuda is available use "cuda" else use "cpu"
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    else:
        device = "cpu"

    return device


def get_model(model_name: str):
    if model_name == "fastSAM":
        from fastsam import FastSAM

        model_fast_sam = FastSAM(config.fast_sam_checkpoint)
        return model_fast_sam

    elif model_name == "fastSAM-s":
        from fastsam import FastSAM

        model_fast_sam_s = FastSAM(config.fast_sam_s_checkpoint)
        return model_fast_sam_s

    elif model_name == "SAM":
        pass

    else:
        pass


def plot_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("on")
    plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


import matplotlib.patches as patches


def plot_square(mask):
    # Find the indices where the mask is True
    true_indices = np.argwhere(mask)

    # Get the bounding box of the True region
    top_left = np.min(true_indices, axis=0)
    bottom_right = np.max(true_indices, axis=0)

    # Calculate the width and height of the bounding box
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]

    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Plot the mask
    ax.imshow(mask, cmap="gray")

    # Create a rectangle patch
    rect = patches.Rectangle(
        (top_left[1], top_left[0]),
        width,
        height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )

    # Add the rectangle patch to the axis
    ax.add_patch(rect)

    # Show the plot
    plt.show()


def annotate_square_corners(
    image, top_left, top_right, bottom_left, bottom_right, save_path
):
    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Plot the original image
    ax.imshow(image)

    # Annotate the corners
    ax.plot(top_left[0], top_left[1], "ro")  # Top Left corner
    ax.plot(top_right[0], top_right[1], "go")  # Top Right corner
    ax.plot(bottom_left[0], bottom_left[1], "bo")  # Bottom Left corner
    ax.plot(bottom_right[0], bottom_right[1], "yo")  # Bottom Right corner

    # Save the plot to the specified path
    # plt.savefig(save_path)

    plt.show()


def get_box_coordinates(
    img,
    model,
    device,
    showOriginalImage=False,
    showPoints=False,
    showPlotMaskWithHighestScore=True,
    DEBUG=False,
   
):
    """
    Parameters
    ----------
    img : np.array
        Image frame.
    model : object
        Model object.
    device : str
        Device identifier.
    showOriginalImage : bool, optional
        Whether to show the original image. Default is False.
    showPoints : bool, optional
        Whether to show points. Default is False.
    showPlotMaskWithHighestScore : bool, optional
        Whether to plot the mask with the highest score. Default is True.

    Returns
    -------
    bounding_box_coords_dict : dict
        Dictionary containing the coordinates of the corners of the bounding box.
        Dictionary keys: "top_left", "top_right", "bottom_right", "bottom_left".
        Dictionary elements: tuple : (y, x) coordinates of the corners.
    """
    
    
    # plot original image
    if showOriginalImage:
        plot_image(img)

    # get image dimensions
    img_height, img_width, _ = img.shape

    # get centre point coordinates
    center_point_coords = [int(img_width / 2), int(img_height / 2)]
    input_point = np.array([center_point_coords])
    input_label = np.array([1])

    if showPoints:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_points(input_point, input_label, plt.gca())
        plt.axis("on")
        plt.show()

    # generate the mask in the relevant area
    fast_sam_predictor = model(
        img, device=device, retina_masks=True, imgsz=img_width, conf=0.4, iou=0.9
    )
    fast_sam_prompt_process = FastSAMPrompt(img, fast_sam_predictor, device=device)

    # point prompt
    # points default [[0,0]] [[x1,y1],[x2,y2]]
    # point_label default [0] [1,0] 0:background, 1:foreground
    img_mask = fast_sam_prompt_process.point_prompt(
        points=input_point, pointlabel=input_label
    )
    

    # plot_mask_with_score(img, "FastSAM output", img_mask, input_point, input_label )

    # reshape image mask
    # print(img_mask.shape)
    img_mask = np.transpose(img_mask, (1, 2, 0))
    
    
    

    if showPlotMaskWithHighestScore:
        plot_square(img_mask)

    # get the rectangular boxes
    # if DEBUG: cv2.imshow("mask", img_mask.astype(np.int8))
    bounding_box_coords_dict = get_bounding_box_coordinates(img_mask, DEBUG)
    armctrl_dict = process_coordinates(img_mask, img_height, img_width)

    armctrl_dict["orientation"] = 180 - bounding_box_coords_dict["orientation"]
    del bounding_box_coords_dict["orientation"]
    
    # get the coordinates of the rectangular bounding box
    return bounding_box_coords_dict, armctrl_dict


def get_image_with_box_corners(frame, points_dict):
    """
    parameters
    ----------
    frame : np.array : image frame
    points_dict : dict : dictionary containing the coordinates of the corners of the bounding box
                         dictionary keys : "top_left", "top_right", "bottom_right", "bottom_left"
                         dictionary elements : tuple : (y, x) coordinates of the corners

    returns
    -------
    frame : np.array : image frame with the corners of the bounding box annotated
    """
    circle_radius = 5
    # Define colors for each point in RGB format (not BGR format)
    colors_dict = {
        "blue": (0, 0, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
    }

    # Draw points on the original image

    #### Point should be in (x, y) format
    # top_left : red
    cv2.circle(
        frame,
        (points_dict["top_left"][1], points_dict["top_left"][0]),
        circle_radius,
        colors_dict["red"],
        -1,
    )

    # top_right : blue
    cv2.circle(
        frame,
        (points_dict["top_right"][1], points_dict["top_right"][0]),
        circle_radius,
        colors_dict["blue"],
        -1,
    )

    # bottom_right : green
    cv2.circle(
        frame,
        (points_dict["bottom_right"][1], points_dict["bottom_left"][0]),
        circle_radius,
        colors_dict["green"],
        -1,
    )

    # bottom_left : yellow
    cv2.circle(
        frame,
        (points_dict["bottom_left"][1], points_dict["bottom_left"][0]),
        circle_radius,
        colors_dict["yellow"],
        -1,
    )

    return frame
