# Retinal Segmentation for Glaucoma Diagnosis Using Deep Learning
Vasu Gupta, Laroy Milton, Erwin Pascual

## Introduction  
Glaucoma is an eye disease that results from optic nerve head damage due to high Intraocular Pressure (IOP). It is one of the leading causes of irreversible blindness around the world [1]. In the early stages, patients are asymptomatic, but as the disease progresses, peripheral vision loss occurs. In the advanced stages, the disease leads to total blindness. One of the major indicators of Glaucoma is the enlargement of the Optic Cup (OC) with respect to the Optic Disc (OD) (Figure 1). This makes it essential to identify the OD and OC and calculate the ratio of their areas. Unfamiliarity with Glaucoma is likely a risk factor for missed diagnosis, and accurate detection requires experts specializing in Glaucoma identification. The current methods of diagnosis include the use of Optic Coherence Tomography (OCT) — a technique used to obtain a topographical map of the Optic nerve. The equipment used for OCT is expensive and not readily available at every eye clinic, making it a nonviable technique. On the other hand, Color Fundus Photography (CFP) is a cost-effective way to take retinal images and is available at most eye clinics. However, the process of manually diagnosing Glaucoma using CFP is time-consuming, arduous, and may be compromised due to lack of experience from the specialists.  

![1](/Images/IMG-10.png)  
Fig. 1: Representing the OD and OC variation in Glaucoma and Non-Glaucoma cases

Since CFP analysis requires high specialization and expertise to examine retinal images, there exists a prime need for automating Glaucoma detection. Therefore, the previous studies [2] - [6] primarily make use of machine learning to perform optic segmentation for the calculation of the cup-to-disc ratio (CDR). Salam et al. [2] uses a Support Vector Machine (SVM) — a supervised learning model with associated learning algorithms to analyze data used for classification and regression analysis. The color and texture features, along with some other properties of CDR, were taken into account by using the OD and OC ratio to determine the Glaucoma stage. This approach was tested on 100 patients in which the authors obtained the specificity values and average sensitivity as 87% and 100%, respectively. On the other hand, Claro et al. [3] developed an automated Glaucoma detection technique using ensemble machine learning classifiers. The system to segment OD was created by extracting texture features in different color models, which were classified by the Multilayer Perceptron (MLP) model. Recent methods like [4-6] proposed solutions using deep learning for optic segmentation. Chen et al. [4] created Convolutional Neural Networks (CNN) which extracted and learned features with linear/non-linear activation functions. In the training phase of the CNN, Glaucoma and non-Glaucoma patterns were distinguished. The reported AUC values for their implementation were 0.838 and 0.898 on the ORIGA and SCES datasets. Lu, Ke, and Yang [5] proposed a deep learning-based framework semantic segmentation of optic disc and cup. They used the original U-Net architecture with some modifications including fewer filters in the convolutional layers. Also, a consistent number of filters were used for decreasing resolution, parameters, and training time.  Moreover, Diaz-Pinto and Naranjo [6] took a two-step approach for OD/OC segmentation. They passed a manually cropped image to the U-Net for OD segmentation and the result of it was passed to the same U-Net for OC segmentation.  

![2](/Images/IMG-09.png)  
Fig. 2: Overview of the proposed method of the Optic disc and cup segmentation  

The proposed method involves a two-step approach for OD/OC segmentation. The first step includes the identification of the ONH using a thresholding algorithm to eliminate the excess background. After the ONH area is detected, the original images and their corresponding masks are passed to the next step of the framework. The second step includes the application of a few preprocessing techniques and segmentation of OD and OC using the U-Net deep learning architecture. Fig. 2 shows a detailed representation of the proposed algorithm. 

The report includes information about the data pre-processing techniques used, details about the network architecture, implementation details, followed by the results for each scenario.

![3](/Images/IMG-08.png)  
Fig. 3: Retinal Fundus Image (left) and its corresponding Optic Disc and Cup Mask (right)


## Method
### Data
The dataset consists of 1,200 colour fundus images provided by the REFUGE challenge (https://grand-challenge.org). The training set contains 400 images of size 2124×2056 pixels (captured by Zeiss Visucam 500) whereas the testing and validation set contains 400 images, each with the size 1634×1634 pixels (captured by Canon CR-2). Each set features 40 glaucoma images and 360 non-glaucoma images with their ground truth masks to distinguish between Optic Disc, Optic Cup and the background. Fig. 3 represents a retinal fundus image and its corresponding mask.

### Optic Nerve Head (ONH) Detection
In order to increase the accuracy of the model, the excess background in the fundus images is eliminated by passing them through an algorithm that locates the ONH. This detection of the ONH is done by first converting the image to gray-scale and applying a bilateral filter and median blur for image smoothing. The smoothed grayscale images are thresholded to find the brightest spot, which is further passed to an erosion and dilation function to remove any small blobs and noise in the image. The largest blob found after performing these operations is labeled as the ONH area with a bounding box of size 640x640 pixels around it. The original image and its corresponding mask are cropped with ONH bounding box dimensions.

### Pre-processing
After the ONH detection, the images are then resized to 128x128 pixels to reduce the required system resources and processing time. Based on our observations, the model predictions with smaller image sizes were similar to the cases with bigger image sizes. After resizing, the images are normalized to ensure that each pixel in each color channel has similar data distribution. Normalization of images also has the advantage of reaching convergence faster while training the network.

### Network Architecture
The architecture in Fig. 4 is similar to the original implementation of U-Net such that it comprises a contracting path (down-sampling layers) and an expansive path (up-sampling layers). Each group of contraction paths is composed of two padded convolutions which double the number of feature channels at each step, one 2x2 average pooling layer, and a skip connection and dropout rate of 0.3. Each group of the expansive path also comprises an up-sampling of feature maps, two convolution layers that halves the number of feature channels at each step, and a skip connection. All convolution layers in the contractive and expansive path have a kernel size of 3x3 and a stride of 1x1 followed by a Rectified Linear Unit (ReLU) as the activation function. The last layer (output layer) of the network is an unpadded convolution layer of kernel size 1x1. In the training process, the input and output size is set to 3x128x128 (CxHxW). The loss functions used in this architecture are Cross-Entropy Loss and Hybrid Logistic Dice Loss (a loss function that incorporates both Logistic and Dice Loss).  In order to update the weight parameters to minimize the loss function, Adam, and SGD optimizers with an L2 regularization value of 1e-2 are used. The initial learning rate of 3.9e-4 which decreases 10% every 2 epochs.

![4](/Images/IMG-07.png)  
Fig. 4: Modified U-Net architecture

### Implementation Details
The ONH detection algorithm based on thresholding is developed using OpenCV. The U-Net network is implemented using the PyTorch library. To avoid over-fitting and to provide a generalized image set to the deep learning model, data augmentation techniques are applied. Some of them include random brightness, contrast, gamma, rotation, RGB shifts, transposition, translation, and scaling, each with their respective probability of 0.5. All data augmentation operations are done randomly. The segmentation network is trained for 40 epochs with a batch size of 4 on a 12GB Tesla K80 GPU on Google Colab.

## Experiments and Results
There were two main approaches taken to get the results for the model. These approaches were tested using different loss functions like Cross-Entropy (CE) and a Hybrid Logistic Dice Loss (HLDL), along with different optimizers like Adam and Stochastic Gradient Descent (SGD). The first approach includes the multi-class segmentation using Optic Disc and Cup as separate channels. The second approach involves the segmentation of Optic Disc and Cup separately using the same U-Net architecture.

The best results on the validation dataset occurred when training the OD and OC with separate models. The model trained only on OD performed best with the HLDL function paired with the Adam optimizer (Table 2). On the contrary, the model that was trained only on OC performed best with the HLDL function paired with the SGD optimizer (Table 3). The average dice score on the best model was 89.30% for OD and 73.60% for OC. The best results for the model trained on OD and OC segmentation utilized CE and Adam, with the dice score for OD and OC as 72.68% and 71.41% respectively. As seen in Fig. 5 to 7, the learning curves clearly show overfitting; however, OD/OC with CE and Adam may converge if trained for more epochs.

![5](/Images/IMG-06.png)  
Table 1: OD and OC Dice Coefficient evaluation of REFUGE validation dataset

![6](/Images/IMG-05.png)  
Table 2: OD Dice Coefficient evaluation of REFUGE validation dataset 

![7](/Images/IMG-04.png)  
Table 3: OC Dice Coefficient evaluation of REFUGE validation dataset 

![8](/Images/IMG-03.png)  
Fig. 5:  OD/OC Learning Curves: Cross-Entropy, Adam

![9](/Images/IMG-02.png)  
Fig. 6: OD Learning Curves: HLDL, SGD

![10](/Images/IMG-01.png)  
Fig. 7: OC Learning Curves: HLDL, Adam

![11](/Images/IMG-00.png)
Fig. 8:  (a) Result from OD/OC framework, (b) Result from OD framework, (c) Result from OC framework


### Discussion and Conclusion
In the early stages of this project, the original U-net implementation [7] was used, which posed a major challenge. The model did not accept variable-sized input images which required resizing them. Moreover, the predicted output images required upsampling which resulted in image information loss. As the project progressed, a modified version of U-Net was used, which resolves the flexibility issues from the original implementation. Other challenges include the difference in images between each dataset. Since each set is from a different fundus camera, this affected the brightness and contrast of each image. These variances were overcome by utilizing random image augmentations. 

The future work of this project will include enhancing the model by making some additional architectural changes, elliptical fitting, and parameter tuning to get better results. In addition, different models such as DeepLabV3 and U-Net++ can be leveraged in future work. After improving the results, the future aim will also include the classification of Glaucoma based on the ratio of OD and OC.


### References
[1] Y. Chung, X. Li, T. Wong, H. Quigley, T. Aung, C. Cheng, “Global Prevalence of Glaucoma and Projections of Glaucoma Burden through 2040: A Systematic Review and Meta-Analysis,” Ophthalmology, vol. 121, no. 11, pp. 2081–2090, November 2014.
[2] A. A. Salam, T. Khalil, M.U. Akram, A. Jameel and I. Basit, “Automated detection of glaucoma using structural and non structural features,” Springerplus, vol. 5, pp. 1–22, 2016.
[3] M. Claro, L. Santos, W. Silva, F. Araújo, N. Moura, “Automatic Glaucoma Detection Based on Optic Disc Segmentation and Texture Feature Extraction,” Clei Electronic Journal, vol. 19, pp. 1–10, August 2016.
[4] X. Chen, Y. Xu, S. Yan, D. Wong, T. Wong, and J. Liu, “Automatic Feature Learning for Glaucoma Detection Based on Deep Learning,” MICCAI (3), vol. 9351 of Lecture Notes in Computer Science. pp. 669–677, 2015.
[5] S. Lu, S. Ke, Y. Yang, “Feature Extraction and Classification of Glaucoma,” 2018
[6] A. Diaz-Pinto, V. Naranjo, “Glaucoma Assessment, Optic Disc and Optic Cup Segmentation on Retinal Images using Deep Learning,” 2018
[7] O. Ronneberger, P. Fischer, T. Brox,  2015. “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, vol. 9351 of Lecture Notes in Computer Science. pp. 234–241, 2015


