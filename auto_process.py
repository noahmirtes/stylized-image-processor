import numpy as np

def _luma01(img_u8):
    x = img_u8.astype(np.float32) / 255.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]  # 0..1

def measure_image(img_u8):
    """
    Returns a dict of simple metrics:
      - median_luma: how bright overall
      - luma_std: global contrast proxy (higher = punchier)
      - sat_mean: average "distance from gray" (simple saturation proxy)
      - sat_p95: 95th percentile saturation (helps detect already-saturated images)
      - highlight_frac: fraction of pixels in highlights
    """
    x = img_u8.astype(np.float32) / 255.0
    l = _luma01(img_u8)

    # "Saturation" proxy: distance from per-pixel gray
    gray = l[..., None]
    sat = np.sqrt(np.mean((x - gray) ** 2, axis=2))  # 0..~0.6 typical

    metrics = {
        "median_luma": float(np.median(l)),
        "luma_std": float(np.std(l)),
        "sat_mean": float(np.mean(sat)),
        "sat_p95": float(np.percentile(sat, 95)),
        "highlight_frac": float(np.mean(l > 0.90)),
    }
    return metrics


# ----------------------------------------------------- PROCESS -------

def exposure_gain(img_u8, gain=1.0):
    """
    Multiply brightness. gain>1 brighter, <1 darker.
    (Simple exposure control.)
    """
    x = img_u8.astype(np.float32) / 255.0
    y = np.clip(x * gain, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)

def soft_contrast_curve(img_u8, amount=0.0):
    """
    amount: 0..1, where 0 = no change, higher = more "S-curve" contrast.
    This increases midtone separation while keeping extremes softer than a hard curve.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    x = img_u8.astype(np.float32) / 255.0

    # A gentle S-curve using smoothstep-ish shaping
    # y = x + a*(x - x^2) centered-ish; keeps it mild.
    a = 0.35 * amount
    y = x + a * (x - x * x) - a * (0.5 - 0.25)  # small centering tweak
    y = np.clip(y, 0.0, 1.0)

    return (y * 255.0 + 0.5).astype(np.uint8)

def warm_shift(img_u8, warm=0.0):
    """
    warm: -1..+1. Positive warms (more R, slightly less B).
    Keep warm small like 0.05 to 0.20.
    """
    warm = float(np.clip(warm, -1.0, 1.0))
    x = img_u8.astype(np.float32) / 255.0

    # Very small per-channel multipliers
    r = 1.0 + 0.12 * warm
    b = 1.0 - 0.10 * warm

    y = x.copy()
    y[..., 0] *= r
    y[..., 2] *= b
    y = np.clip(y, 0.0, 1.0)

    return (y * 255.0 + 0.5).astype(np.uint8)

def adjust_saturation(img_u8, sat=1.0):
    """
    Same saturation function you used earlier.
    sat=1 no change. sat<1 less saturated. sat>1 more saturated.
    """
    x = img_u8.astype(np.float32) / 255.0
    l = _luma01(img_u8)[..., None]
    y = l + sat * (x - l)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)












def choose_params(m):
    """
    Deterministic rules based on your stated taste:
      - darker overall (don't brighten too much)
      - soft contrast
      - slightly warm
      - vivid but not obviously saturated
      - glowing highlights only if highlights exist
    """
    # Targets (tune later)
    target_median = 0.40        # slightly dark
    target_contrast = 0.22      # softer contrast range
    target_sat_mean = 0.085     # "vivid but realistic" proxy

    # Exposure: gently steer median toward target (small moves)
    # If too bright, darken a bit; if too dark, only lift slightly.
    err = target_median - m["median_luma"]
    gain = 1.0 + 0.60 * err
    gain = float(np.clip(gain, 0.92, 1.08))  # keep subtle

    # Contrast curve amount: if very flat, add a little; if already punchy, add none.
    c_err = target_contrast - m["luma_std"]
    contrast_amt = float(np.clip(0.0 + 2.0 * c_err, 0.0, 0.35))  # soft

    # Warmth: keep slight warmth always, reduce if already warm-looking (sat proxy isn’t warmth,
    # so we keep this simple for V1)
    warm = 0.09  # default warm lean

    # Saturation: avoid "looks saturated"
    # If sat is high already, reduce a hair. If low, increase a hair.
    s_err = target_sat_mean - m["sat_mean"]
    sat = 1.0 + 1.2 * s_err
    sat = float(np.clip(sat, 0.94, 1.06))

    # Bloom/halation gating based on highlight presence
    has_highlights = m["highlight_frac"] > 0.002  # 0.2% pixels above 0.9
    bloom_amount = 0.10 if has_highlights else 0.0
    halation_amount = 0.06 if has_highlights else 0.0

    # LUT strength: default moderate; reduce if image already dark + contrasty (to avoid mud)
    lut_strength = 0.65
    if m["median_luma"] < 0.33 and m["luma_std"] > 0.24:
        lut_strength = 0.50
    lut_strength = float(np.clip(lut_strength, 0.35, 0.80))

    return {
        "gain": gain,
        "contrast_amt": contrast_amt,
        "warm": warm,
        "sat": sat,
        "lut_strength": lut_strength,
        "bloom_amount": bloom_amount,
        "halation_amount": halation_amount,
    }