import os
import utils
import numpy as np
from PIL import Image
from bloom_halation import add_bloom_halation
from film_grain import add_film_grain_basic
from lut import load_cube_lut, apply_lut, auto_fix_lut
from saturation import adjust_saturation


def process_image_pillow(
    in_path,
    out_path,
    *,

    lut,
    lut_size,
    lut_strength=1.0,

    # Saturation
    saturation=0.95,

    # Bloom/Halation
    bloom_amount=0.12,
    bloom_sigma=6.0,
    halation_amount=0.08,
    halation_sigma=10.0,
    highlight_threshold=0.75,
    highlight_softness=0.15,

    # Grain
    grain_amount=0.06,
    shadow_boost=2.0,
    grain_size=1.0,
    chroma_amount=0.55,
    grain_seed=None,
):
    im, icc = utils.open_image_rgb_with_fallback(in_path)
    arr = np.array(im, dtype=np.uint8)

    # 1) Color remap via LUT
    arr = apply_lut(arr, lut, lut_size, lut_strength)

    """
    # 2) Saturation
    arr = adjust_saturation(arr, sat=saturation)

    # 3) Bloom + halation
    arr = add_bloom_halation(
        arr,
        bloom_amount=bloom_amount,
        bloom_sigma=bloom_sigma,
        halation_amount=halation_amount,
        halation_sigma=halation_sigma,
        highlight_threshold=highlight_threshold,
        highlight_softness=highlight_softness,
        use_linear=True,
    )
    """

    # 4) Grain last
    arr = add_film_grain_basic(
        arr,
        amount=grain_amount,
        shadow_boost=shadow_boost,
        grain_size=grain_size,
        chroma_amount=chroma_amount,
        seed=grain_seed,
    )
    """
    """
    out_im = Image.fromarray(arr, mode="RGB")

    if icc is not None:
        out_im.save(out_path, icc_profile=icc)
    else:
        out_im.save(out_path)



def main():
    input_folder = "/Users/noah/Library/Mobile Documents/com~apple~CloudDocs/•My Files•/POTD/14 - 2025 Picture of the Day"
    output_folder = "/Users/noah/Desktop/images/out"
    lut_path = "/Users/noah/Desktop/Print+Film+Emulation+LUTs/Rec709_Fujifilm_3510_D65.cube"

    image_paths = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]


    lut, lut_size = load_cube_lut(lut_path)
    lut = auto_fix_lut(lut, lut_size)  # prints the chosen mapping

    total = len(image_paths)
    count = 1
    for path in image_paths:

        # construct input paths
        input_path = os.path.join(input_folder, path)
        output_path = os.path.join(output_folder, path.replace('.jpg', '_FILM.jpg'))

        if os.path.exists(output_path):
            print(f"Skipping existing output for {path}")
            count += 1
            continue

        print(f"Processing image {path} : {count} of {total}")

        # run main process flow
        process_image_pillow(
            in_path=input_path,
            out_path=output_path,

            # Suggested starting values (subtle)
            lut=lut,
            lut_size=lut_size,
            lut_strength=0.5,

            saturation=1.3,

            bloom_amount=0.10,
            halation_amount=0.06,

            grain_amount=0.05,
            chroma_amount=0.10,
            grain_size=1.0,
        )

        # increment count
        count +=1



if __name__ == "__main__":
    main()