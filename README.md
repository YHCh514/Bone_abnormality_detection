# 110-1-NTU-DBME5028

#FinishByDL

The images is simply resiszed to 512 * 512 and equalized. Each input data, training and testing, cosists of 4 images of the same patient.
If the image of a patient is insufficient, we fill the data by replicating the first image of the patient.

The model is pretrained resnet50. However, we modified the input channels of the first convolutional layer to 4,
and generate the weight of the forth channel from that of the other three.
Additionally, the fc layer is also modified so as to fit the input format of negative logarithmic loss function for classification,
which is log of the prior of each class.
