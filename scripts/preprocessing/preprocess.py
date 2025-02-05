import itertools
from PIL import Image
from pathlib import Path
from typing import List
from resize import resizing
from grayscale import grayscaling
from polarize import polarizing

def preprocess(image_path:Path, new_width:int=48, new_height:int=48,save:bool=False) -> Image:
    im_out = Image.open(image_path)

    if not is_grey_scale(im_out):
        im_out = grayscaling(im_out)

    if not is_size_equal_to(im_out, new_width, new_height):
        im_out = resizing(im_out, new_width, new_height)

    if save:
        im_out.save(f'../preprocess/preprocess{image_path.name}')

    return im_out

def is_grey_scale(img:Image) -> bool:
    img = img.convert('RGB')
    w, h = img.size
    for i, j in itertools.product(range(w), range(h)):
        r, g, b = img.getpixel((i,j))
        if r != g != b:
            return False
    return True

def is_size_equal_to(img:Image,  new_width:int, new_length:int)->bool:
    return img.size == (new_width,new_length)

def preprocess_folder(images_folder:Path, preprocess_folder_name:str, new_width:int=48, new_height:int=48) -> None:
    extension = '.jpg'
    for image_path in images_folder.rglob(f'*{extension}'):
        parent_paths = image_path.parents
        subfolder_names = [parent_path.name for parent_path in parent_paths]
        #print(f'{subfolder_names=}' )
        preprocess_dataset_path = Path(get_path(subfolder_names, 3), preprocess_folder_name, subfolder_names[1], subfolder_names[0])
        #print(f'{preprocess_dataset_path=}')
        if not preprocess_dataset_path.exists():
            preprocess_dataset_path.mkdir(parents=True, exist_ok=True)

        print(f'image preprocessed: {image_path.name}')
        saving_path = Path(preprocess_dataset_path, image_path.name)
        im_out = preprocess(image_path, new_width=new_width, new_height=new_height)
        #print(f'{saving_path}')
        im_out.save(str(saving_path))

def polarize_folder(images_folder:Path, preprocess_folder_name:str) -> None:
    extension = '.jpg'
    for image_path in images_folder.rglob(f'*{extension}'):
        parent_paths = image_path.parents
        subfolder_names = [parent_path.name for parent_path in parent_paths]
        #print(f'{subfolder_names=}' )
        preprocess_dataset_path = Path(get_path(subfolder_names, 3), preprocess_folder_name, subfolder_names[1], subfolder_names[0])
        #print(f'{preprocess_dataset_path=}')
        if not preprocess_dataset_path.exists():
            preprocess_dataset_path.mkdir(parents=True, exist_ok=True)

        print(f'image preprocessed: {image_path.name}')
        saving_path = Path(preprocess_dataset_path, image_path.name)
        im_out = polarizing(image_path)
        #print(f'{saving_path}')
        im_out.save(str(saving_path))


def get_path(list_folder:List[str], n:int) -> str:
    path = '.'
    n = min(n, len(list_folder))
    for i in range(-2,-n,-1):
        path = f'{path}/{list_folder[i]}'
    return path


def img_equal(img1:Image, img2:Image) -> bool:
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')
    w1, h1 = img1.size
    w2, h2 = img2.size
    if h1!=h2 or w1!=w2:
        return False

    for i, j in itertools.product(range(w1), range(h1)):
        r1, g1, b1 = img1.getpixel((i,j))
        r2, g2, b2 = img2.getpixel((i,j))

        if r1 != r2  or g1 != g2 or b1!=b2:
            return False
    return True

def img_diff(img1:Image, img2:Image)->Image:
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')
    w1, h1 = img1.size
    w2, h2 = img2.size
    im_out = Image.new('RGB',(w1,h1))
    pixels = im_out.load()
    if h1==h2 and w1==w2:
    
        for i in range(w1):
            for j in range(h1):
                r1, g1, b1 = img1.getpixel((i,j))
                r2, g2, b2 = img2.getpixel((i,j))
                pixels[i,j] = (100*abs(r1-r2), 100*abs(g1-g2), 100*abs(b1-b2))
                    
    return im_out

image_folder = Path('../datasets/FER2013_polarized')
preprocess_folder_name = 'datasets/FER2013_polarized_resized'
if image_folder.exists():
    preprocess_folder(image_folder, preprocess_folder_name)

