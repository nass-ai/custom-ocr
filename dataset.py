from ocr import detectLines
from pathlib import Path
from PIL import Image
from pytesseract import image_to_string
from pdfs2Images import batchConvert

import os
# change this line to import csv later on
import pandas as pd


def chopImages(path):
    """
    Use Line segmentation algorithm defined in "detectLines" script
    to slice a full image into text lines and save the text lines in png format
    """
    files = [f for f in os.listdir(path) \
             if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".JPG")]

    for f in files:
        try:
            detectLines(os.path.join(path, f))
        except:
            print(f)
    return 

def transcribeImage(path2imfile):
    """
    Use pytesseract to convert text line image to string
    """
    fname= os.path.basename(path2imfile)
    lines_dirname= os.path.dirname(path2imfile)
    upper_dirname = os.path.dirname(lines_dirname)
    
    if not os.path.exists(upper_dirname+"/"+"gtTranscripts"):
        os.makedirs(upper_dirname+"/"+"gtTranscripts")
    
    if not os.path.exists(lines_dirname+"/"+fname+".gt.txt"):
        im_file = Image.open(path2imfile)
        gtTranscript = image_to_string(im_file)
        f = open(upper_dirname+"/"+"gtTranscripts/"+fname+".gt.txt", "a")
        f.write(gtTranscript)

        f.close()
    else:
        return

    
    return gtTranscript

def batchTranscribe(path2lines):
    """
    Transcribe all the text line image files in "path2lines" 
    """
    total_no_of_bills= len(os.listdir(path2lines))
    for idx, bill_folder in enumerate(os.listdir(path2lines)):    
        full_bill_folder=os.path.join(path2lines, bill_folder)
        
        # check out how to remove "./DS STORE" from the list of directories displayed by os.listdir()
        
        if "bill" in bill_folder:
            
            lines_folder = [x for x in os.listdir(os.path.join(path2lines, bill_folder)) if x=="lines"][0]
            full_lines_folder = os.path.join(full_bill_folder, lines_folder)

  
            path=Path(full_lines_folder)

            line_images = list(path.rglob("*.png"))

            for l in line_images:
                transcribeImage(l)
        else:
            
            continue
        print(idx, "of", total_no_of_bills)
        
def formatData(path2Images=None):
    """
    organize image files and their corresponding transcripts into 
    a csv.
    """
    df = pd.DataFrame(columns = ["FILE NAMES", "TRANSCRIPTS"])
    # change this line to include all the dataset into training 
    path="/Users/aobaruwa/Desktop/nassai/train/dataset/Untitled.txt"
    f = open(path, "r")
    
    dirs = [x[:-1] for x in f.readlines()[2:-7]]
    #print(dirs)
    pairs = []
    for dirname in dirs:
       # print(dirname)
        lines_folder = [x for x in os.listdir(dirname) if x=="lines"][0]
        transcripts_folder = [x for x in os.listdir(dirname) if x=="gtTranscripts"][0]
        full_lines_folder = os.path.join(dirname, lines_folder)
        full_transcripts_folder =os.path.join(dirname, transcripts_folder)
        
        for fname in os.listdir(full_lines_folder):
            transcript_path = os.path.join(full_transcripts_folder, fname)+".gt.txt"
            f = open(transcript_path, "r")
            txt=f.read()
            f.close()
           # print(transcript)
            pairs.append((os.path.join(full_lines_folder, fname), txt))
    df["FILE NAMES"] = [f for f,tr in pairs]
    df["TRANSCRIPTS"]=[tr for f, tr in pairs]
    csv_file = df.to_csv(os.path.dirname(path)+"/data.csv")
    

    return csv_file

def run(path2Pdfs, path2Images, path2lines):
    batchConvert(path2Pdfs)
    chopImages(path2Images)
    batchTranscribe(path2lines)
    formatData()

    return 

    
if __name__=="__main__":
    path2Images = "../Images"
    path2lines = "../dataset"
    path2Pdfs = ".../Dataset/SA_Bills"
        
    
    run(path2Pdfs, path2Images, path2lines)
    
