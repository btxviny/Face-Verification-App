import os 
import argparse
import random
import numpy as np
import cv2
import torch

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from models.inception_resnet_v1 import  InceptionResnetV1


def parse_args():
    parser = argparse.ArgumentParser(description='Create embeddings for face verification')
    parser.add_argument('--images_path',type=str, help = 'Folder with images to extact the embeddings from')
    parser.add_argument('--save_path',type=str, help = 'where to save the resulting embeddings')
    return parser.parse_args()

def augment_image(image, num_augmentations=15):
    # Read the original image using cv2
    # Define an augmentation function for brightness adjustment using torchvision.transforms
    transform = transforms.ColorJitter(brightness=0.1,contrast=0.1,hue = 0.01)

    augmented_images = [image]

    for _ in range(num_augmentations):
        # Flip the image horizontally
        flip = random.randint(0,1) 
        if flip:
            image = cv2.flip(image, 1)
        # Rotate the image by a random angle between -30 and 30 degrees
        angle = np.random.uniform(-10, 10)
        rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Adjust brightness
        image = transform(Image.fromarray(image))
        image = np.array(image)
        augmented_images.append(image)
         

    return augmented_images


if __name__ == '__main__':
    '''class args():
        images_path = 'C:/Users/VINY/Desktop/Face_ID/Steve_Buscemi'
        save_path = 'steve_buscemi.npy'''
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    augmented_path = args.images_path + '_augmented'
    os.makedirs(augmented_path, exist_ok=True)
    paths = [os.path.join(args.images_path,x) for x in os.listdir(args.images_path)]
    #augment images and save
    print('Augmenting Images\n')
    for img_path in tqdm(paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img,(640,360))
        idx = len(os.listdir(augmented_path))
        augmented_images = augment_image(img)
        for i in range(len(augmented_images)):
            cv2.imwrite(os.path.join(augmented_path, f'{idx+i}.jpg'),augmented_images[i])
    #extract embeddings
    print('Extracting Embeddings\n')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    face_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_default.xml')
    photos = os.listdir(augmented_path)
    embeddings = torch.zeros((len(photos),512))
    for idx,f in tqdm(enumerate(photos)):
        img = cv2.cvtColor(cv2.imread(os.path.join(augmented_path,f)),cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        box = face_cascade.detectMultiScale(gray)
        if len(box) != 0: 
            for (x, y, w, h) in box:
                face = img[y: y+h, x: x+w, :]
                face = cv2.resize(face,(128,128))
                face = torch.from_numpy(face).permute(2,0,1).float()
                face = (face - 127.5)/128.0
                with torch.no_grad():
                    face_embedding = resnet(face.unsqueeze(0).to(device))
                    face_embedding = face_embedding.cpu().detach().squeeze(0)
                embeddings[idx,...] = face_embedding
    mean_embeddings = torch.mean(embeddings,dim = 0)    
    np.save(args.save_path,mean_embeddings.numpy())