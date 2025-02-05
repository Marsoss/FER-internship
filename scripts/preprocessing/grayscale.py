from PIL import Image
def grayscaling(im_in:Image, save:bool=False) -> Image:
    print("gray scaling...")
    im_out = im_in.convert('L')
    if save:
        im_out.save(f'gray_{im_in.filename}')

    return im_out