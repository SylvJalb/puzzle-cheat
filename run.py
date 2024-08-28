import argparse
from backgroundremover.bg import remove
import cv2
from math import *
import numpy as np

def imread_without_background(image_path):
    """
    Remove the background of the image.

    Args:
        image_path (str): Path to the image to remove the background.

    Returns:
        numpy.ndarray: Image without the background.
    """
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    model = model_choices[0]

    f = open(image_path, "rb")
    data = f.read()
    img = remove(data, model_name=model,
                 alpha_matting=True,
                 alpha_matting_foreground_threshold=250,
                 alpha_matting_background_threshold=250,
                 alpha_matting_erode_structure_size=10,
                 alpha_matting_base_size=1000)
    f.close()
    
    # Convert the result to a numpy array
    img_array = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Smooth the image
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Crop the image to remove the full black background
    w_min = 0
    w_max = image.shape[1]
    h_min = 0
    h_max = image.shape[0]
    for i in range(image.shape[0]):
        if np.sum(image[i, :, :]) != 0:
            h_min = i
            break
    for i in range(image.shape[0] - 1, 0, -1):
        if np.sum(image[i, :, :]) != 0:
            h_max = i
            break
    for i in range(image.shape[1]):
        if np.sum(image[:, i, :]) != 0:
            w_min = i
            break
    for i in range(image.shape[1] - 1, 0, -1):
        if np.sum(image[:, i, :]) != 0:
            w_max = i
            break
    image = image[h_min:h_max, w_min:w_max, :]

    return image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Person focus")

    # Puzzle image path
    parser.add_argument("--puzzle-image-path",
                        type=str,
                        help="Path to the puzzle main image.",
                        default=0)
    
    # Piece image path
    parser.add_argument("--piece-image-path",
                        type=str,
                        help="Path to the image of the piece to found.",
                        default=0)
    
    # Number of pieces in the puzzle
    parser.add_argument("--number-pieces",
                        type=int,
                        help="Number of pieces in the puzzle.",
                        default=1000)
    
    # Number of assembly pieces
    parser.add_argument("--number-assembly-pieces",
                        type=int,
                        help="Number of pieces already assembled.",
                        default=1)

    args = parser.parse_args()

    # Load images
    puzzle_image = cv2.imread(args.puzzle_image_path)
    piece_image = imread_without_background(args.piece_image_path)

    # Resize puzzle image to the good size
    puzzle_image = cv2.resize(puzzle_image, (1600, piece_image.shape[0] * 1600 // piece_image.shape[1]))
    ratioWH = puzzle_image.shape[1] / puzzle_image.shape[0]

    # set scale of the piece image
    scale_mul = args.number_assembly_pieces / (sqrt(args.number_pieces) * ratioWH)
    # Resize piece image to the good size
    piece_image = cv2.resize(piece_image, (int(piece_image.shape[1] * scale_mul), int(piece_image.shape[0] * scale_mul)))
    # set a new image
    new_piece_image = np.zeros((puzzle_image.shape[0], puzzle_image.shape[1], 3), np.uint8)
    # add piece image in center of new_piece_image
    x_offset = int((new_piece_image.shape[1] - piece_image.shape[1]) / 2)
    y_offset = int((new_piece_image.shape[0] - piece_image.shape[0]) / 2)
    new_piece_image[y_offset:y_offset + piece_image.shape[0], x_offset:x_offset + piece_image.shape[1]] = piece_image
    piece_image = new_piece_image

    # Convert images to RGB
    puzzle_image = cv2.cvtColor(puzzle_image, cv2.COLOR_BGR2RGB)
    piece_image = cv2.cvtColor(piece_image, cv2.COLOR_BGR2RGB)

    # Extract features from images
    orb = cv2.ORB_create(nfeatures=100)
    kp1, des1 = orb.detectAndCompute(puzzle_image, None)
    kp2, des2 = orb.detectAndCompute(piece_image, None)


    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_img = cv2.drawMatches(puzzle_image, kp1, piece_image, kp2, matches, None)


    # SHOW OUTPUTS

    match_img = cv2.cvtColor(match_img, cv2.COLOR_RGB2BGR)
    match_img = cv2.resize(match_img, (1000, match_img.shape[0] * 1000 // match_img.shape[1]))

    # Create a named window
    cv2.namedWindow("Matches", cv2.WND_PROP_FULLSCREEN)
    # Set the window to fullscreen
    cv2.setWindowProperty("Matches", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Matches", match_img)
    cv2.waitKey()