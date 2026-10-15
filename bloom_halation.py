import numpy as np

from utils import srgb_to_linear, linear_to_srgb, gaussian_blur




def add_bloom_halation(img_u8, *,
                      bloom_amount=0.12,
                      bloom_sigma=6.0,
                      halation_amount=0.08,
                      halation_sigma=10.0,
                      highlight_threshold=0.75,
                      highlight_softness=0.15,
                      halation_tint=(1.0, 0.55, 0.25),
                      use_linear=True):
    """
    Bloom: neutral glow from highlights.
    Halation: warmer glow from highlights (often reddish/orange).
    """
    x = img_u8.astype(np.float32) / 255.0

    if use_linear:
        x = srgb_to_linear(x)

    luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

    t0 = highlight_threshold
    t1 = min(1.0, t0 + max(1e-6, highlight_softness))
    mask = (luma - t0) / (t1 - t0)
    mask = np.clip(mask, 0.0, 1.0)[..., None]

    bright = x * mask

    bloom_glow = gaussian_blur(bright, bloom_sigma) if bloom_amount > 0 and bloom_sigma > 0 else 0.0
    hal_glow = gaussian_blur(bright, halation_sigma) if halation_amount > 0 and halation_sigma > 0 else 0.0

    tint = np.array(halation_tint, dtype=np.float32)[None, None, :]
    hal_glow = hal_glow * tint

    y = x + bloom_amount * bloom_glow + halation_amount * hal_glow
    y = np.clip(y, 0.0, 1.0)

    if use_linear:
        y = linear_to_srgb(y)

    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)
