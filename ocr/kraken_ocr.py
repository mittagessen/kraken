from kraken import blla
from PIL import Image, ImageOps
from kraken.binarization import nlbin
from kraken.lib import vgsl


class KrakenOCR:
    def __init__(self):
        self.blla = blla
        self.model = vgsl.TorchVGSLModel.load_model('kraken/blla.mlmodel')

    def ocr(self, im, preprocessing=False):
        baseline_seg = self.blla.segment(im=im, model=self.model, device='cpu')  # Baseline segmenter
        return baseline_seg['lines']

    def img_preprocessing(self, im, inverse=False):
        """
        im: input image
        inverse: Default is False. If set to True, it will inverse the binary image to foreground = white and
        background = black.
        """

        binarized_image = nlbin(im)
        if inverse:
            return ImageOps.invert(binarized_image)

        return binarized_image


if __name__ == "__main__":
    image_path = "/home/dell/Documents/handwritten_images/testingimages/d2.jpeg"
    image = Image.open(image_path)
    im = ImageOps.exif_transpose(image)
    k = KrakenOCR()
    segmented_info = k.ocr(im)
    print(segmented_info)

