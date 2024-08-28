import argparse
from backgroundremover.bg import remove
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable

feature_model = models.resnet18(pretrained=True)

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

def extract_features(image):
    """
    Extract features from the image.

    Args:
        image (numpy.ndarray): Image to extract features.

    Returns:
        numpy.ndarray: Features extracted from the image.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)
    ])
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    return processed
    
    

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

    # Extract features from images
    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(feature_model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_model = feature_model.to(device)

    puzzle_features = extract_features(puzzle_image)
    piece_features = extract_features(piece_image)


    # SHOW OUTPUTS

    def visualize_features(features):
        """
        Visualiser les caractéristiques sur l'image.

        Args:
            features (numpy.ndarray): Caractéristiques extraites.
        """
        for i, feature in enumerate(features):
            feature = cv2.resize(feature, (400, feature.shape[0] * 400 // feature.shape[1]))
            cv2.imshow(f"Feature {i}", cv2.cvtColor(feature, cv2.COLOR_GRAY2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Visualiser les caractéristiques sur les images
    visualize_features(puzzle_features)
    visualize_features(piece_features)

    # Reconvert images to BGR
    puzzle_image = cv2.cvtColor(puzzle_image, cv2.COLOR_RGB2BGR)
    piece_image = cv2.cvtColor(piece_image, cv2.COLOR_RGB2BGR)

    # Show images
    # cv2.imshow("Puzzle Image", puzzle_image)
    # cv2.imshow("Piece Image", piece_image)

    # Wait for a key press
    cv2.waitKey(0)