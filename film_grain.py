
import numpy as np
from utils import srgb_to_linear, linear_to_srgb, gaussian_blur


def add_film_grain_basic(img_u8, *,
                   amount=0.06,
                   shadow_boost=2.0,
                   grain_size=1.0,
                   chroma_amount=0.10,
                   seed=None):
    rng = np.random.default_rng(seed)

    x = img_u8.astype(np.float32) / 255.0
    h, w, _ = x.shape

    luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    weight = 1.0 + shadow_boost * (1.0 - luma) ** 1.5
    weight = weight[..., None]

    noise_h, noise_w = h, w
    if grain_size > 1.0:
        noise_h = max(1, int(h / grain_size))
        noise_w = max(1, int(w / grain_size))

    n_luma = rng.standard_normal((noise_h, noise_w, 1)).astype(np.float32)
    n_color = rng.standard_normal((noise_h, noise_w, 3)).astype(np.float32)

    if (noise_h, noise_w) != (h, w):
        rep_h = int(np.ceil(h / noise_h))
        rep_w = int(np.ceil(w / noise_w))
        n_luma = np.repeat(np.repeat(n_luma, rep_h, axis=0), rep_w, axis=1)[:h, :w, :]
        n_color = np.repeat(np.repeat(n_color, rep_h, axis=0), rep_w, axis=1)[:h, :w, :]

    noise = n_luma + chroma_amount * n_color
    y = x + (amount * weight) * noise
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)



def add_film_grain(
    img_u8,
    *,
    amount=0.045,          # overall strength; start 0.03–0.06
    shadow_boost=1.8,      # boosts shadows/mids; start 1.2–2.5
    highlight_suppress=0.7,# extra suppression in highlights; 0..1
    fine_sigma=0.6,        # fine grain correlation
    coarse_sigma=2.0,      # clumpy grain correlation
    coarse_mix=0.35,       # 0..1 how much coarse layer contributes
    chroma_amount=0.08,    # subtle color grain; start 0.03–0.12
    seed=None,
):
    """
    More film-faithful grain:
      - shadow/midtone emphasis (not infinite grain in highlights or deep blacks)
      - band-limited correlated noise (fine + coarse clumps)
      - mostly luminance grain; subtle correlated chroma grain
      - luma weighting computed in linear light

    img_u8: uint8 RGB (H,W,3)
    returns uint8 RGB
    """
    rng = np.random.default_rng(seed)

    x_srgb = img_u8.astype(np.float32) / 255.0
    x_lin = srgb_to_linear(x_srgb)

    # Linear-light luma for weighting
    luma = 0.2126 * x_lin[..., 0] + 0.7152 * x_lin[..., 1] + 0.0722 * x_lin[..., 2]  # 0..1

    # Weighting curve: emphasize shadows/mids, suppress highlights.
    # - (1-luma)^p gives more grain in shadows
    # - multiply by (0.2 + 0.8*luma) to avoid "infinite" grain in pitch black
    shadow_term = (1.0 - luma) ** 1.6
    floor_term = 0.20 + 0.80 * luma
    w = 1.0 + shadow_boost * shadow_term * floor_term

    # Extra highlight suppression
    if highlight_suppress > 0:
        # as luma approaches 1, reduce grain further
        sup = 1.0 - float(np.clip(highlight_suppress, 0.0, 1.0)) * (luma ** 2.2)
        w *= np.clip(sup, 0.15, 1.0)

    w = w[..., None]  # (H,W,1)

    h, w_img, _ = x_lin.shape

    # Build correlated luminance noise (fine + coarse)
    n_fine = rng.standard_normal((h, w_img, 1)).astype(np.float32)
    n_coarse = rng.standard_normal((h, w_img, 1)).astype(np.float32)

    n_fine = gaussian_blur(n_fine, fine_sigma)
    n_coarse = gaussian_blur(n_coarse, coarse_sigma)

    # Normalize each layer to roughly unit stddev (so params behave consistently)
    n_fine /= (np.std(n_fine) + 1e-6)
    n_coarse /= (np.std(n_coarse) + 1e-6)

    n_luma = (1.0 - coarse_mix) * n_fine + coarse_mix * n_coarse
    n_luma /= (np.std(n_luma) + 1e-6)

    # Subtle correlated chroma noise (3ch), mostly “shared” not independent RGB static
    n_chr = rng.standard_normal((h, w_img, 3)).astype(np.float32)
    n_chr = gaussian_blur(n_chr, max(fine_sigma, 0.8))
    n_chr /= (np.std(n_chr) + 1e-6)

    # Apply in linear space (more film-like energy behavior)
    y_lin = x_lin + (amount * w) * (n_luma + chroma_amount * n_chr)
    y_lin = np.clip(y_lin, 0.0, 1.0)

    y_srgb = linear_to_srgb(y_lin)
    return (np.clip(y_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

