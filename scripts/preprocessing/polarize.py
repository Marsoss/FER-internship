import polarTransform
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path


def polarizing(im_path:Path) -> Image:
    print("polarizing...")
    image = imageio.imread(im_path)
    polarImage, _ = polarTransform.convertToPolarImage(image, initialRadius=0, hasColor=False,
                                                            finalRadius=39, initialAngle=0,
                                                            finalAngle=2 * 3.141592)
    return Image.fromarray(polarImage)