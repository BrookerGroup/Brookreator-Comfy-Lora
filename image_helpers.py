import os
import logging
import json
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import numpy as np
from comfy.cli_args import args
import shutil


class ImageHelpers:
    def __init__(self):
        self.output_dir = os.path.abspath("test")
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    def save_images(self, images, image_path, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        self.clean_dir(image_path)
        self.output_dir = os.path.abspath(image_path)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = self.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            counter += 1

        return

    def clean_dir(self, image_path):
        # Ensure the train_data_dir exists
        if os.path.exists(image_path):
            # Clear the contents of the directory (optional)
            shutil.rmtree(image_path)
            logging.info(f"Directory cleared: {image_path}")
        # Create the directory (again, after clearing)
        os.makedirs(image_path, exist_ok=True)
        logging.info(f"Directory created: {image_path}")

    def get_save_image_path(self, filename_prefix: str, output_dir: str, image_width=0, image_height=0) -> tuple[str, str, int, str, str]:
        def map_filename(filename: str) -> tuple[int, str]:
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return digits, prefix
        def compute_vars(input: str, image_width: int, image_height: int) -> str:
            input = input.replace("%width%", str(image_width))
            input = input.replace("%height%", str(image_height))
            return input
        filename_prefix = compute_vars(filename_prefix, image_width, image_height)
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(output_dir, subfolder)
        if os.path.commonpath((output_dir, os.path.abspath(full_output_folder))) != output_dir:
            err = "**** ERROR: Saving image outside the output folder is not allowed." + \
                "\n full_output_folder: " + os.path.abspath(full_output_folder) + \
                "\n         output_dir: " + output_dir + \
                "\n         commonpath: " + os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))
            logging.error(err)
            raise Exception(err)
        try:
            counter = max(filter(lambda a: os.path.normcase(a[1][:-1]) == os.path.normcase(filename) and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1
        return full_output_folder, filename, counter, subfolder, filename_prefix