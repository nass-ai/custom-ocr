from pdf2image import convert_from_path
from PIL import Image

import os
import sys
import tempfile

def MTarikconvert(file_path, output_path):

    """
    M Tarik Yurt's code for converting a multi-page pdf document to images
    and merging them into a single jpeg file.
    """
    # save temp image files in temp dir, delete them after we are finished
    with tempfile.TemporaryDirectory() as temp_dir:
        # convert pdf to multiple image
        images = convert_from_path(file_path, output_folder=temp_dir)
        # save images to temporary directory
        temp_images = []
        for i in range(len(images)):
            image_path = temp_dir+'/'+str(i)+'.png'
            images[i].save(image_path, 'PNG')
            temp_images.append(image_path)
        # read images into pillow.Image
        imgs = []
        for img in temp_images:
            temp = Image.open(img)
            imgs.append(temp.copy())

            temp.close()
        #imgs = list(map(Image.open, temp_images))
    # find maximum width of images
    min_img_width = min(i.width for i in imgs)
    # find total height of all images
    total_height = 0
    for i, img in enumerate(imgs):
        total_height += imgs[i].height
    # create new image object with width and total height
    merged_image = Image.new(imgs[0].mode, (min_img_width, total_height))
    # paste images together one by one
    y = 0
    for img in imgs:
        merged_image.paste(img, (0, y))
        y += img.height
    # save merged image
    merged_image.save(output_path)
    return output_path

def batchConvert(absPath2PDFs):
    paths = [(x, os.stat(x).st_size) for x in os.listdir(absPath2PDFs) if x.endswith(".pdf")]

    output_folder = absPath2PDFs+"/Images"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        print("images will be saved inside the directory: {}".format(output_folder))

    print(len(os.listdir(output_folder))/(len(paths)/100), "percent complete")

    for idx, f in enumerate(sorted(paths,key=lambda x: x[1])):
      f, size = f
      if str(f[:-4])+".png" not in os.listdir(output_folder):

        imfile_name = os.path.join(output_folder, f[:-4]+".png")
        print("on the", idx,"th bill")


        try:
          if os.stat("./Bills/"+f).st_size >20000000:
            f= open("large files.txt", "a")
            f.write(f+"\n")
            f.close()
            continue
          print("trying to convert", f)
          print("saving to {}".format(imfile_name))
          out_ = MTarikconvert(os.path.join(absPath2PDFs, f), imfile_name)

          fpt = open("converted bills.txt", "a")
          fpt.write(imfile_name+"\n")
          fpt.close()
        except:
          print(imfile_name,"has issues")
          fpt= open("Not Converted.txt", "a")
          fpt.write(imfile_name+"\n")
          fpt.close()

      else:

          print(str(f[:-4])+".png", "has been converted already")

    return None

if __name__=="__main__":
    file_path = ".../Dataset/SA_Bills"
    batchConvert(file_path)
