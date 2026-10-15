import numpy as np
from PIL import Image
import os

from lut import load_cube_lut, apply_lut, auto_fix_lut
import utils
import auto_process as ap
from bloom_halation import add_bloom_halation
from film_grain import add_film_grain


from time import monotonic


def auto_edit_v1(img_u8, lut, lut_size):
    m = ap.measure_image(img_u8)
    p = ap.choose_params(m)

    out = img_u8
    out = ap.exposure_gain(out, gain=p["gain"])
    out = ap.soft_contrast_curve(out, amount=p["contrast_amt"])
    out = ap.warm_shift(out, warm=p["warm"])

    out = add_bloom_halation(
        out,
        bloom_amount=p["bloom_amount"],
        halation_amount=p["halation_amount"],
        highlight_threshold=0.85,
        highlight_softness=0.10,
        bloom_sigma=5.0,
        halation_sigma=9.0,
        use_linear=True,
    )

    out = apply_lut(out, lut, lut_size, strength=0.47) # p["lut_strength"]

    out = ap.adjust_saturation(out, sat=p["sat"])




    out = add_film_grain(
        out,
        amount=0.014,
        chroma_amount=0.10,
        shadow_boost=2.0,
    )

    return out






if __name__ == "__main__":

    in_dir = "/Users/noah/Desktop/Image Process/PHOTOS/INPUT"
    out_dir = "/Users/noah/Desktop/Image Process/PHOTOS/OUTPUT - Kodak_2393 - More Grain"
    lut_path = "/Users/noah/Desktop/Image Process/LUTs/Rec709_Kodak_2393_D65.cube"

    # Load + fix LUT once
    lut, lut_size = load_cube_lut(lut_path)
    lut = auto_fix_lut(lut, lut_size)

    for fname in os.listdir(in_dir):
        if not fname.endswith((".jpg",".jpeg")):
            continue

        in_path = os.path.join(in_dir, fname)

        name, ext = os.path.splitext(fname)
        out_fname = f"{name}_AUTO{ext}"
        out_path = os.path.join(out_dir, out_fname)

        try:
            start_time = monotonic()
            im, icc = utils.open_image_rgb_with_fallback(in_path)
            arr = np.array(im, dtype=np.uint8)

            arr = auto_edit_v1(
                arr,
                lut=lut,
                lut_size=lut_size
            )

            out_im = Image.fromarray(arr, mode="RGB")

            if icc is not None:
                out_im.save(out_path, icc_profile=icc)
            else:
                out_im.save(out_path)

            print(f"Processed: {fname} in {monotonic() - start_time}")

        except Exception as e:
            print(f"Failed on {fname}: {e}")