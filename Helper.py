from __future__ import print_function, division
import sys
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.patches as patches
from io import StringIO

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode
class ActionLocationDataset(Dataset):
    """Action Location dataset."""

    def __init__(self,root_dir,verbose=False,transform=None):

        """

        :param root_dir: Directory containing training data. Must be in form:

        ucfAction/
        ├── <actions> ex: Diving-Side
        │   ├── <instances> ex: 01
        │   │   ├── <action images>.jpg
        │   │   ├── gt
        │   │   │   ├── <labels>.tif.txt
        ...


        :param verbose(boolean): Set to True to print information
        :param transform: (callable, optional): Optional transform to be applied on a sample.
        """

        # if directory exists
        if not os.path.isdir(root_dir):
            print("Directory "+root_dir+" does not exist")
            sys.exit(1)

        self.rootDir = root_dir
        self.transform = transform

        self.labelsCSV = ""
        colLabels = "imageName,x1,y1,x2,y2,label\n"
        self.labelsCSV += colLabels

        self.labelCodeDict = {}
        labelCodeCounter = 0

        # for each action folder in data directory (Action)
        for action in get_immediate_subdirectories(root_dir):
            if verbose:
                print(action)
            # for video folder in action folder
            for instance in get_immediate_subdirectories(root_dir+"/"+action):
                if verbose:
                    print("\t"+instance)
                    imageLoc = root_dir+"/"+action+"/"+instance
                for labels in get_immediate_subdirectories(root_dir+"/"+action+"/"+instance):
                    if labels != "gt":
                        continue
                    path = root_dir+"/"+action+"/"+instance+"/"+labels
                    for file in os.listdir(path):
                        filenameEx2 = os.fsdecode(file)
                        # remove first extension
                        filenameEx1 = os.path.splitext(filenameEx2)[0]
                        filename = os.path.splitext(filenameEx1)[0]
                        # if verbose:
                        #    print("\t\t"+filename)

                        contents = open(path+"/"+filenameEx2).read()

                        commaSep = contents.replace("\t", ",")

                        stage = root_dir+"/"+action+"/"+instance+"/"+filename + ".jpg" + "," + commaSep

                        if verbose:
                            print(stage)

                        '''
                        sIO = StringIO(stage)
                        l = pd.read_csv(sIO,sep=",")
                        label = l.iloc[0, 4]
                        '''
                        l = stage.split(",")
                        label = l[len(l)-1]

                        if not self.labelCodeDict.__contains__(label):
                            self.labelCodeDict[label] = labelCodeCounter
                            labelCodeCounter += 1

                        self.labelsCSV += (stage+"\n")



        labelsCSVFile = open('labels.csv', 'w')
        labelsCSVFile.write(self.labelsCSV)

        # set up label int code

        self.labelsFrame = pd.read_csv("labels.csv")
        # display training data with labels if verbose
        if verbose:
            # len = self.labelsFrame.__len__()
            n=7 # 2 points needed for rectangle? or 2nd image
            imageName = self.labelsFrame.iloc[n, 0]
            box = self.labelsFrame.iloc[n, 1:5].as_matrix()
            box = box.reshape(-1, 2)
            label = self.labelsFrame.iloc[n, 5]

            print('Image name: {}'.format(imageName))
            print('Landmarks shape: {}'.format(box.shape))
            print('Box location: \n {}'.format(box))
            print('Label: {}'.format(label))

            img = io.imread(imageName)

            show_lable(img, box, label)

    def __len__(self):
        return len(self.labelsFrame)

    def __getitem__(self, idx):
        imgName = self.labelsFrame.iloc[idx, 0]

        image = io.imread(imgName)
        # image =   torch.tensor(image)
        image = image.reshape(-1,10)
        label = self.labelsFrame.iloc[idx, 5]
        labelCode = self.labelCodeDict[label]

        sample = {'image': image, 'label': label, 'labelCode': labelCode}
        rescale= Rescale(200)
        sample = rescale(sample)
        toTens = ToTensor()
        sample = toTens(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample



class Rescale(object):
    """Rescale the image in a sample to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, labelCode = sample['image'], sample['label'], sample['labelCode']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'label': label, 'labelCode': labelCode}



class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label, labelCode = sample['image'], sample['label'], sample['labelCode']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        #landmarks = landmarks - [left, top]

        return {'image': image, 'label': label, 'labelCode': labelCode}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, labelCode = sample['image'], sample['label'], sample['labelCode']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),'label': label, 'labelCode': labelCode}

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def help():
    print("Usage: python3 *.py <training directory>")


#def run():
#    ds = ActionLocationDataset(sys.argv[1], True)

def show_lable(image, box, label):
    """Show image with label"""
    fig, ax = plt.subplots(1)

    # plt.imshow(image)
    ax.imshow(image)
    # get points from box
    x1 = box[0][0]
    y1 = box[0][1]

    x2 = box[1][0]
    y2 = box[1][1]
    # draw box


    width = np.math.fabs(float(x1-x2))
    height = np.math.fabs(float(y1 - y2))

    rect = patches.Rectangle((float(x2), float(y1)), width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    # display label
    # plt.lin   scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(2.010)  # pause a bit so that plots are updated


# run()
