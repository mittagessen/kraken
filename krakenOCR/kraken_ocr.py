from kraken import blla
from PIL import Image, ImageOps
from kraken.binarization import nlbin
from kraken.lib import vgsl


class KrakenOCR:
    def __init__(self):
        self.blla = blla
        self.model = vgsl.TorchVGSLModel.load_model("blla.mlmodel")

    def ocr(self, input_image):
        preprocessed_image = nlbin(input_image)

        baseline_seg = self.blla.segment(im=preprocessed_image, model=self.model, device='cpu')  # Baseline segmenter
        return baseline_seg['lines']


if __name__ == "__main__":
    image_path = '/home/dell/Documents/handwritten_images/testingimages/d1.jpeg'
    image = Image.open(image_path)
    im = ImageOps.exif_transpose(image)
    k = KrakenOCR()
    segmented_info = k.ocr(im)
    print(segmented_info)

