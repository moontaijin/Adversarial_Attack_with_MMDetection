import torch

def resize_bbox(bbox, original_size, new_size):
    """
    Resize the bounding box according to the new image size.

    Args:
    bbox (tuple): The original bounding box (x1, y1, x2, y2).
    original_size (tuple): The size (width, height) of the original image.
    new_size (tuple): The size (width, height) of the new image.

    Returns:
    tuple: The resized bounding box.
    """
    x1, y1, x2, y2 = bbox
    orig_width, orig_height = original_size
    new_width, new_height = new_size

    # Resize the bbox
    x1_new = x1 * new_width / orig_width
    y1_new = y1 * new_height / orig_height
    x2_new = x2 * new_width / orig_width
    y2_new = y2 * new_height / orig_height

    return torch.tensor([x1_new, y1_new, x2_new, y2_new])