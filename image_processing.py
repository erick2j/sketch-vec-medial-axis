import cv2 
import numpy as np
from typing import Tuple
from skimage.measure import find_contours

def normalize_to_measure(grid: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D array into a probability measure (sum equals 1).

    Parameters:
        grid (np.ndarray): A 2D array of non-negative values.
    Returns:
        np.ndarray: The normalized measure, same shape as input.
    """
    total_mass = np.sum(grid.astype(np.float32))

    if total_mass <= 0:
        raise ValueError("Input grid must have positive total mass to define a measure.")

    return grid / total_mass

def complement_measure(measure: np.ndarray) -> np.ndarray:
    """
    Compute the "complement" of a measure on a grid.
    
    The complement redistributes mass inversely across the domain,
    such that the total mass remains 1.

    Does this make any sense?

    Parameters:
        measure (np.ndarray): 2D array where np.sum(measure) == 1

    Returns:
        np.ndarray: Complemented measure with the same shape and total mass 1
    """
    if not np.isclose(np.sum(measure), 1.0, atol=1e-6):
        raise ValueError("Input must be a valid probability measure (sum = 1).")
    
    complement = 1.0000 - measure
    total = np.sum(complement)
    
    if total <= 0:
        raise ValueError("Complement has zero or negative total mass.")
    
    return complement / total


def process_image(file_path, padding=0):
    """
    docstring
    """
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image from: {file_path}")

    # Handle transparent images
    if img.shape[-1] == 4:
        # Separate alpha and composite over white background
        bgr, alpha = img[..., :3], img[..., 3:] / 255.0
        white = np.ones_like(bgr, dtype=np.float32) * 255
        img = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add optional white padding
    if padding > 0:
        img = cv2.copyMakeBorder(
            img,
            top=padding, bottom=padding,
            left=padding, right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=255  # white border
        )
    return img

def compute_dual_boundary(measure: np.ndarray) -> np.ndarray:
    '''
    Computes a boundary measure of an image by viewing the image as a dual 0-form
    and applying the dual d_0 operator on it and averaging the values on dual edges onto
    adjacent faces.
    '''
    left_kernel   = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    right_kernel  = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    top_kernel    = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    bottom_kernel = np.array([[0, 0, 0], [0,  1, 0], [0,-1, 0]])

    left   = cv2.filter2D(src=measure, ddepth=-1, kernel=left_kernel)
    right  = cv2.filter2D(src=measure, ddepth=-1, kernel=right_kernel)
    top    = cv2.filter2D(src=measure, ddepth=-1, kernel=top_kernel)
    bottom = cv2.filter2D(src=measure, ddepth=-1, kernel=bottom_kernel)

    boundary = 0.5 * (np.abs(left) + np.abs(right) + np.abs(top) + np.abs(bottom))
    #boundary = 0.5 * (left + right + top + bottom)

    return boundary

def compute_primal_boundary(measure: np.ndarray) -> np.ndarray:
    '''
    Compute the "boundary" on the dual grid .
    '''
    right_kernel = np.array([[0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0],
                             [0, 0, -1, 0, 1], 
                             [0, 0, -1, 0, 1], 
                             [0, 0, 0, 0, 0]])

    top_kernel = np.array(  [[0, 0, 1, 1, 0], 
                             [0, 0, 0, 0, 0],
                             [0, 0, -1, -1, 0], 
                             [0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0]])

    left_kernel = np.array( [[0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0],
                             [-1, 0, 1, 0, 0], 
                             [-1, 0, 1, 0, 0], 
                             [0, 0, 0, 0, 0]])

    bottom_kernel =np.array([[0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0], 
                             [0, 0, 0, 0, 0], 
                             [0, 0, -1, -1, 0]])

    left   = 0.25 * cv2.filter2D(src=measure, ddepth=-1, kernel=left_kernel)
    right  = 0.25 * cv2.filter2D(src=measure, ddepth=-1, kernel=right_kernel)
    top    = 0.25 * cv2.filter2D(src=measure, ddepth=-1, kernel=top_kernel)
    bottom = 0.25 * cv2.filter2D(src=measure, ddepth=-1, kernel=bottom_kernel)

    boundary = np.abs(left) + np.abs(right) + np.abs(top) + np.abs(bottom)

    return boundary


def compute_sobel_boundary(image: np.ndarray) -> np.ndarray:
    """
    Compute the Sobel gradient magnitude of a grayscale image using OpenCV.

    Parameters:
        image (np.ndarray): 2D grayscale image (uint8 or float32)

    Returns:
        np.ndarray: 2D array of gradient magnitudes (float32)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")

    # Convert to float32 for precision if not already
    image = image.astype(np.float32)

    # Compute Sobel gradients in x and y directions
    grad_x = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return magnitude


def upsample_bicubic(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Upsample a grayscale image using bicubic interpolation.

    Parameters:
        image (np.ndarray): Input 2D grayscale image.
        scale (float): Upsampling factor (e.g., 2.0 doubles the size).

    Returns:
        np.ndarray: Upsampled grayscale image.
    """
    if image.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image.")
    if scale <= 0:
        raise ValueError("Scale factor must be positive.")

    height, width = image.shape
    new_size = (int(width * scale), int(height * scale))
    upsampled = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return upsampled

