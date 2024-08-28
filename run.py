import argparse
from backgroundremover.bg import remove
import cv2
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

    args = parser.parse_args()

    # Load images
    puzzle_image = cv2.imread(args.puzzle_image_path)
    piece_image = imread_without_background(args.piece_image_path)

    # Convert images to RGB
    puzzle_image = cv2.cvtColor(puzzle_image, cv2.COLOR_BGR2RGB)
    piece_image = cv2.cvtColor(piece_image, cv2.COLOR_BGR2RGB)

    # Resize images with same ratio w / h
    puzzle_image = cv2.resize(puzzle_image, (1000, puzzle_image.shape[0] * 1000 // puzzle_image.shape[1]))
    piece_image = cv2.resize(piece_image, (400, piece_image.shape[0] * 400 // piece_image.shape[1]))


    # Reconvert images to BGR
    puzzle_image = cv2.cvtColor(puzzle_image, cv2.COLOR_RGB2BGR)
    piece_image = cv2.cvtColor(piece_image, cv2.COLOR_RGB2BGR)

    # Show images
    cv2.imshow("Puzzle Image", puzzle_image)
    cv2.imshow("Piece Image", piece_image)

    # Wait for a key press
    cv2.waitKey(0)