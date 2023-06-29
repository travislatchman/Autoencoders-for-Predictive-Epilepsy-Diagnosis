# Autoencoders-for-Predictive-Epilepsy-Diagnosis

## Representation Learning with Autoencoders for Predictive Epilepsy Diagnosis Models

**Brief Description:** This project focused on utilizing representation learning with autoencoders to build predictive models for epilepsy diagnosis. We will solve a 2-class classification problem using Epilespy EEG data.

The dataset is available from https://zenodo.org/record/3684992

This dataset was generated with a motive to build predictive epilepsy diagnosis models. It was generated on a similar acquisition and settings i.e., sampling frequency, bandpass filtering and number of signals and time duration as its much more famous counterpart - the University of Bonn dataset. It has overcome the limitations faced by the University of Bonn dataset, such as different EEG recording place (inter-cranial and scalp) for healthy and epileptic patients. All the data were taken exclusively using surface EEG electrodes.

Dataset for Training/Testing of Existing Models and Algorithms:

"This dataset was collected exclusively from scalp EEG taken from the same EEG machine and contains data of 30 subjects- 15 healthy and 15 epileptic subjects. Like the previous studies, the data of 5 healthy and 5 epileptic subjects was used for training and validation. The rest, i.e., the data of 10 healthy and 10 epilepsy subjects, was used for testing such that there was no overlap of subjects between the training/validation dataset and the testing dataset. In the training/validation dataset, forty EEG segments, of 23.6 seconds duration each, were utilized. In the testing dataset, twenty EEG segments of 23.6 seconds duration each per subject were utilized. Similar to the benchmark dataset, each class in the training/validation and test sets had 200 artifact-free EEG segments. Along with the duration, all the EEG segments in these datasets also met the bandwidth (0.5-40 Hz), sampling rate (173.6 Hz) and stationarity criteria of the benchmark dataset. This dataset is used as an alternate dataset to the benchmark dataset because benchmark dataset is data collected differently for the two classes (from scalp and from cortex). Thus, this dataset allows us to evaluate the performance of existing features and algorithms when trained and tested on a dataset collected consistently for the two classes, i.e., from the scalp."

S. Panwar, S. D. Joshi, A. Gupta and P. Agarwal, "Automated Epilepsy Diagnosis Using EEG With Test Set Evaluation," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 27, no. 6, pp. 1106-1116, June 2019, doi: 10.1109/TNSRE.2019.2914603.
https://ieeexplore.ieee.org/abstract/document/8705361

### Implementation (see notebook for each task):
* Replicated an auto-encoder from a research paper.
* Trained auto-encoders for unsupervised representation learning.
* Leveraged the encoder for supervised learning by attaching a classifier.
* Visualized the reconstructed signals.
* Explored designing bottlenecks in an auto-encoder.


### **`TASK 1: (Data Preparation) `** 

Please read the description of the dataset carefully (provided in the download link).

For this lab, we will not use the standard dataset partition suggested in the paper describing the corpus. We will employ all the files starting with 'Train' along with the files starting with E1 to E8 and H1 to H8 as our training data (that is a total of 26 participants.)

Files of subjects E9, E10, H9 and H10 (4 participants) will be our test data.

Load all the EEG files into two data matrices for train and test. The matrices should have dimensions N x S. Where is N is number of data points (~2880 for train and ~320 for test).
A single data point corresponds to a contiguous S=868 sample length (about 5s given the sampling rate of 173.6 Hz) from an EEG file. These 5 second segments are non-overlapping portions from the datasets.

The labels should be 0 or 1 (epilepsy or healthy). If you choose to store labels in other format like "one-hot encoding", y.shape would be Nx2.

Next, normalize all data to range of [-1,1]

Note: If you are having issues downloading the dataset from the link provided above, you can download it from this Google Drive folder

https://drive.google.com/file/d/1To0qBit_OVfC8ocV8xXSpHRh52Heq2lH/view?usp=share_link

### **`TASK 2: (Auto-encoder training) `** 
We will create an autoencoder based on the analysis methodology described in this paper.
https://www.sciencedirect.com/science/article/pii/S0957417420306114

Using Pythorch, create and train the auto-encoder specified in the paper.
An autoencoder is a specific type of a neural network, which is mainly designed to encode the input into a compressed and meaningful representation, and then decode it back such that the reconstructed input is similar as possible to the original one.
We will refer to this model as *original auto-encoder* in this lab.
We provide the Table 9 from the paper which specifies all details of the network below. Choose MSE (mean squared error) as loss function.
Your loss must decrease with epochs.

Hint 1: It would be useful to use `summary(autoencoder, input_size=())` to verify if your architecture is correct.

Hint 2: Convolutional layers need to be modified such that they do not modify the length of input signal. Also, ensure the output of network is same shape as input.

![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/43918f0b-8188-46f4-88b3-5946434cea37)

![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/60dd0f8b-eecc-45d7-ae36-b7a557f58626)


### **`TASK 3: (Visualization of reconstruction) `** 

Let x be a 868-sample long data point from your test set, e() be encoder, d() be decoder. The reconstuction/auto-encoding of x is given by d(e(x)) i.e. simply forward pass of x through the auto-encoder. On separate figures, (1) plot one signal from test set and its reconstructed version.

![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/796876ff-8fd5-46be-997f-018a738cddbe)
![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/61dd2c35-b1cc-4c98-b4b4-48ea1623967c)

Encoding dimension: 1728  
Relation between dimension of output_vector and learned representation vector: 2

### **`TASK 4 (Training classifier) `** 
Discarding the decoder, append a simple classifier at the end of your trained encoder. The classifier (excluding your encoder) is a M=100 unit dense layer followed by your favorite activation function followed by 2-class softmax classification. Encoder must be frozen in this training. Report accuracy on test set.

![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/82890a02-77c0-4a97-8028-35f99df48174)


Accuracy on test set: 54.37%


### **`TASK 5 (Modifying bottleneck of auto-encoder) `** 

Let us treat K (encoded dimension) a as hyper-parameter. Your goal is to ensure K is approximately half of signal length i.e. 868/2 = 434. It is OK if your reduced dimension is in +/-10% range of 434. You will accomplish this by modifying/adding/removing ONLY the following three types of layers: Upsampling, ZeroPadding, and MaxPooling layer. Like previous tasks, first train the autoencoder, then train the subsequent classifier (keeping encoder frozen), and finally report test accuracy. The classifier architecture remains same as before. Also, display a test image and its reconstruction using this auto-encoder. 

![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/47559559-b281-485f-846c-03b63bad2a35)

![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/cfe27016-98d3-4659-be42-dcaa310a670a)

Test Accuracy: 50.31%

![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/2c31860d-e907-4511-a1c0-99538caa1153)

![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/09cfcd32-9f78-422d-b9f6-a6f9e605cb74)

Encoding dimension: 13824  
Relation between dimension of output_vector and learned representation vector: 16

After the modification of the autoencoder, the encoded dimension of the output vector was increased to 16. This means that the encoder is now compressing the input signal into a 16-dimensional latent space. This higher dimensional representation should contain more information about the input signal than the 2-dimensional representation, making it easier for the classifier to distinguish between different input signals.

### **`TASK 6 `** 
Plot the t-SNE visualization (https://opentsne.readthedocs.io/) of the representations of training data learned by the auto-encoders of PART 2, Tasks 5 and 6. Use different color for data from different classes. Repeat this visualization but now use a different color per participant (each participant has several files and frames and will have several points in the TSNE plot).

![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/5f587825-e76e-4cf8-9130-835b2ca8a44b)  
![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/e1c3aa90-f011-4f68-81e3-b06eeec4f40b)  
![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/4c59f8cb-234e-4c93-985d-8b5c1dc2ea2b)
![image](https://github.com/travislatchman/Autoencoders-for-Predictive-Epilepsy-Diagnosis/assets/32372013/58e24ec0-b746-4bd4-b370-4281a7dcb212)

A t-SNE visualization of trained data from an autoencoder can provide useful insights into the quality of the learned representations and areas that could be improved. From the plots, it is evident that the healthy-labeled data points are clustered together in both cases. However, in the second autoencoder's graph, the red points are more tightly clustered than in the first, suggesting that it has learned more meaningful representations of the data. Additionally, we can infer from the second autoencoder's graph that most of the outliers correspond to epilepsy-labeled data points. There is no clear additional conclusion regarding the model's ability to learn meaningful representations of patient-labeled data from the graphs.


