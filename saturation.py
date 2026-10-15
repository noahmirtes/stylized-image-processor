import numpy as np

def adjust_saturation(img_u8, sat=1.0):
    """
    sat=1.0 no change
    sat<1.0 less saturated
    sat>1.0 more saturated
    """
    x = img_u8.astype(np.float32) / 255.0
    luma = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    y = luma + sat * (x - luma)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)

