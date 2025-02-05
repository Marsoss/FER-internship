from PIL import Image
def resizing(im_in:Image, new_width:int=48, new_length:int=48,save:bool=False) -> Image:
    print("resizing...")
    im_out = im_in.resize((new_width,new_length))
    if save:
        im_out.save(f'resized_{im_in.filename}')

    return im_out

