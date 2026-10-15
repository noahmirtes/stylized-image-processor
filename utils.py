import os
import subprocess
import tempfile
from PIL import Image, UnidentifiedImageError, ExifTags
from datetime import datetime
import numpy as np


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1.0 + a) * (x ** (1.0 / 2.4)) - a)


def _gaussian_kernel_1d(sigma):
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = int(max(1, round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def _convolve1d_reflect(img, k, axis):
    pad = len(k) // 2
    img = img.astype(np.float32)

    if axis == 0:
        padded = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
        out = np.empty_like(img, dtype=np.float32)
        for i in range(img.shape[0]):
            window = padded[i:i + 2 * pad + 1, :, :]
            out[i, :, :] = np.tensordot(window, k, axes=([0], [0]))
        return out

    padded = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(img, dtype=np.float32)
    for j in range(img.shape[1]):
        window = padded[:, j:j + 2 * pad + 1, :]
        out[:, j, :] = np.tensordot(window, k, axes=([1], [0]))
    return out


def gaussian_blur(img_f, sigma):
    k = _gaussian_kernel_1d(sigma)
    y = _convolve1d_reflect(img_f, k, axis=1)
    y = _convolve1d_reflect(y, k, axis=0)
    return y










def open_image_rgb_with_fallback(path):
    """
    Try opening with Pillow first.
    If Pillow can't identify it, use macOS `sips` to transcode to a clean JPEG,
    then open that.
    Returns: (PIL.Image in RGB, icc_profile_bytes_or_None)
    """
    try:
        im = Image.open(path)
        icc = im.info.get("icc_profile")
        return im.convert("RGB"), icc
    except UnidentifiedImageError:
        pass

    # Fallback: transcode via macOS to a temporary JPEG
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "transcoded.jpg")
        # sips returns nonzero on failure
        subprocess.run(
            ["sips", "-s", "format", "jpeg", path, "--out", out_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        im = Image.open(out_path)
        icc = im.info.get("icc_profile")  # may be None after transcode
        return im.convert("RGB"), icc
    






def build_pd_filename(path, pd_number):
    """
    path: path to image file
    pd_number: int (e.g. 297 -> PD00297)

    Returns: filename string, including extension
    Example:
    PD00297 Sunday, May 28, 2017 at 8.27.52 PM.jpg
    """
    im = Image.open(path)
    exif = im.getexif()

    if not exif:
        raise ValueError("No EXIF data found")

    exif_data = {
        ExifTags.TAGS.get(tag, tag): value
        for tag, value in exif.items()
    }

    dt_str = exif_data.get("DateTimeOriginal")
    if not dt_str:
        raise ValueError("No DateTimeOriginal in EXIF")

    # Parse EXIF datetime
    dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")

    # Build PD code
    pd_code = f"PD{pd_number:05d}"

    # Format date
    date_part = dt.strftime("%A, %B %d, %Y")

    # Custom time format: 8.27.52 PM
    hour = dt.hour
    am_pm = "AM" if hour < 12 else "PM"
    hour = hour % 12 or 12

    time_part = f"{hour}.{dt.minute:02d}.{dt.second:02d} {am_pm}"

    # Preserve extension
    ext = os.path.splitext(path)[1]

    return f"{pd_code} {date_part} at {time_part}{ext}"