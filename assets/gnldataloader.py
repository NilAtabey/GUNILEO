import os
import pandas as pd 
import dlib
import cv2
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchnlp.encoders import LabelEncoder

class GNLDataLoader(Dataset):
    def __init__(self, labels_path: str, data_path: str, transform = None) -> None:
        """
        Creates a dataset given the path to the labels and the image directory

        Parameters:
            - `labels_path`: the path to the `csv` file containing the labels;
            - `images_dir`: the path to the directory with the images;
            - `transform`: states whether a transformation should be applied to the images or not.
        """
        super().__init__()
        self.data_dir, self.labels_dir = os.listdir(data_path).sort(), os.listdir(labels_path).sort()
        print(self.data_dir, self.labels_dir, sep="\n\n")
        self.transform = transform

        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")

        self.alphabet = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789 "]
        self.encoder = LabelEncoder(self.alphabet, reserved_labels=['unknown'], unknown_index=0)
        self.CROPMARGIN = 20

    def __len__(self) -> int:
        """Returns the length of the data folder
        
        Returns:
            - `length` (`int`): the length of the data folder"""
        return len(self.data_dir)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the ith item(s) in the dataset
        
        Parameters:
            - `index`: the index of the image that must be retrieven.
            
        Returns:
            - `item` (`img`): the image in the ith position in the dataset."""
        
        # Get the label + data
        label_path, data_path = self.labels_dir[index], self.data_dir[index]

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        
        print(self.data_dir, self.labels_dir, sep="\n\n")

        print(data_path, label_path, sep="\n\n")

        vid = self.videoload(data_path)
        label = self.alignload(label_path)
        return (vid, label)


                        # This is data path
    def videoload(self, path, debug=False):
        """Docs"""
        cap = cv2.VideoCapture(path)
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        to_return = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret,frame = cap.read()
            gframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            facedetect = self.face_detector(gframe)
            
            #HAVE A CHECK IF THE FACE IS FOUND OR NOT



            face_landmarks = self.landmark(gframe, facedetect[0])
            xleft = face_landmarks.part(48).x -self.CROPMARGIN
            xright = face_landmarks.part(54).x +self.CROPMARGIN
            ybottom = face_landmarks.part(57).y +self.CROPMARGIN
            ytop = face_landmarks.part(50).y -self.CROPMARGIN

            mouth = gframe[ytop:ybottom,xleft:xright]
            mouth = cv2.resize(mouth,(150,100))
            
            mean = np.mean(mouth)
            std_dev = np.std(mouth)
            mouth = (mouth - mean) / std_dev
                
            to_return.append(torch.tensor(mouth))
        cap.release()
        return np.array(to_return)
    

    def alignload(self, path, debug=False):
        encoding =[ {"b":"bin","l":"lay","p":"place","s":"set"},
                    {"b":"blue","g":"green","r":"red","w":"white"},
                    {"a":"at","b":"by","i":"in","w":"with"},
                    "letter",
                    {"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine"},
                    {"a":"again","n":"now","p":"please","s":"soon"}]
        code = path.split(".")[0].split("_")[-1]
        sentence = []
        for i, letter in enumerate(code):
            corresponding_dict = encoding[i]
            next =""
            if corresponding_dict == "letter":
                next = letter
            else:next = corresponding_dict[letter]
            sentence = sentence + [" "] + [x for x in next]
    

        enl = self.encoder.batch_encode(sentence)
        return enl