from auto_edit_v1 import auto_edit_v1
import numpy as np
from PIL import Image
import os

from lut import load_cube_lut, auto_fix_lut
import utils

import multiprocessing
from dataclasses import dataclass

from time import monotonic


@dataclass
class ProcessImage:
    input_path : str
    out_filename : str
    out_filepath : str


# WORKER GLOBALS
worker_lut = None
worker_lut_size = None


def init_worker(lut_path):
    global worker_lut, worker_lut_size

    # LOAD + FIX LUT ONCE PER WORKER
    worker_lut, worker_lut_size = load_cube_lut(lut_path)
    worker_lut = auto_fix_lut(worker_lut, worker_lut_size)


def image_process_worker(image_entry):

    try:
        print(f"running {image_entry.out_filename}")
        start_time = monotonic()

        # LOAD THE IMAGE
        im, icc = utils.open_image_rgb_with_fallback(image_entry.input_path)
        arr = np.array(im, dtype=np.uint8)

        # EDIT THE IMAGE
        print("starting auto edit")
        arr = auto_edit_v1(
            arr,
            lut=worker_lut,
            lut_size=worker_lut_size
        )
        print("finished auto edit")

        out_im = Image.fromarray(arr, mode="RGB")

        if icc is not None:
            print('save1')
            #out_im.save(image_entry.out_filepath, icc_profile=icc)
        else:
            print('save2')
            #out_im.save(image_entry.out_filepath)

        print(f"Processed: {image_entry.out_filename} in {monotonic() - start_time}")

    except Exception as e:
        print(f"Failed on {image_entry.out_filename}: {e}")


def image_multiprocess_handler(process_data, lut_path):
    with multiprocessing.Pool(
        processes=3,
        initializer=init_worker,
        initargs=(lut_path,)
    ) as p:
        p.map(image_process_worker, process_data)



if __name__ == "__main__":
    IN_DIR = "/Users/noah/Desktop/SUMMER FALL '26/input"
    OUT_DIR = "/Users/noah/Desktop/SUMMER FALL '26/output"
    LUT_PATH = "/Users/noah/Desktop/MISC/images/Print+Film+Emulation+LUTs/Rec709_Fujifilm_3510_D65.cube"


    process_image_items = []
    for fname in os.listdir(IN_DIR):
        if not fname.endswith((".jpg",".jpeg")):
            continue

        in_path = os.path.join(IN_DIR, fname)
        name, ext = os.path.splitext(fname)
        out_fname = f"{name}_AUTO{ext}"

        out_path = os.path.join(OUT_DIR, out_fname)

        process_image_items.append(
            ProcessImage(
                input_path = os.path.join(IN_DIR, fname),
                out_filename = out_fname,
                out_filepath = os.path.join(OUT_DIR, out_fname)
            )
        )



    # BEGIN MULTIPROCESS
    with multiprocessing.Pool(
        processes=2,
        initializer=init_worker,
        initargs=(LUT_PATH,)
    ) as p:
        p.map(image_process_worker, process_image_items)