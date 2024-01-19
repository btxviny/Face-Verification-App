## Deep Learning Face Verification with Transfer Learning
![example](https://github.com/btxviny/Face-Verification-Application/blob/main/demo.gif)

In this project, I have developed a robust face verification system that leverages the power of deep learning and transfer learning. I utilize a pretrained neural network for face classification and extract compact embeddings to represent faces in a high-dimensional space. The pretrained network can be found [here](https://github.com/timesler/facenet-pytorch). It is an Inception ResNet V1 trained on the VGGFace2 dataset, for face classification. The embeddings we extract are essentially the internal representation of the face provided by the pretrained model. In our case, each face is represented as a vector in a high-dimensional space (512 dimensions). This vector encodes the features of the face in a way that is useful for distinguishing between different faces.

To train the model, a dataset comprising 10 base images was augmented to 160 images using various techniques. This augmented dataset ensures the model's robustness in handling diverse facial expressions, poses, and lighting conditions. With the pretrained model I extract the embeddings of all images resulting in a tensor of shape [160,512]. I then create the reference embedding by averaging all embeddings resulting in a final embedding of shape [1,512]. 

During the inference phase, the MTCNN (Multi-Task Cascaded Convolutional Networks) is employed for face detection. The detected face region is cropped, preprocessed, and fed into the pretrained model to obtain the face embedding. The verification process relies on the L1 distance metric, comparing the calculated distance between the anchor embedding and the detected face's embedding to an empirically determined threshold of 0.03.

## 1. Create the reference embeddings.
This project can be used to verify a number of faces, by simply providing the reference embeddings. I provide the following script to augment images and get the reference embeddings.
- Run the following command:
     ```bash
     python get_embeddings.py --images_path ./path_to images_folder --save_path ./path_to_resulting_embedding_file
     ```
- Use the format 'FirstName_LastName.npy' 
## 2. Place the embedding inside the 'embeddings' folder.

## 3. I provide the inference_mtcnn.ipynb notebook for inference.
