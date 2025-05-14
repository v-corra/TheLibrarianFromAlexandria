# Font Classification with CNN and Transfer Learning

## Title:
**Font Classification Using Convolutional Neural Networks (CNN)**

## Team Members:
- Corrado Valeri
- Michele Baldo


---

## [Section 1] Introduction:
This project focuses on using **Convolutional Neural Networks (CNNs)**, specifically a **ResNet50 pre-trained model**, for the task of classifying fonts in images. The dataset consists of 1250 images of different fonts, and our goal is to train a CNN model to classify the fonts based on these images. We applied various **preprocessing techniques**, including **data augmentation** and **denoising**, to improve model performance.

---

## [Section 2] Methods:

### 1. **Data Preprocessing**:
   - **Grayscale Conversion**: The images are converted to grayscale using `PIL` to reduce complexity.
   - **Binarization**: Each image is binarized by applying a threshold to create a high contrast image.
   - **Denoising**: A median filter is applied to remove noise from the binary image.
   - **Data Augmentation**: Random rotations, flips, and color jitter are applied to enhance the dataset, making the model more robust.

### 2. **Data Augmentation**:
   - **RandomAffine**: Applied random translation of up to 10%.
   - **RandomFlip**: Random horizontal and vertical flips.
   - **ColorJitter**: Random changes in brightness and contrast.
   
   These augmentations help simulate variations of font images to ensure the model generalizes well to new images.

### 3. **CNN Architecture (ResNet50)**:
   - **Transfer Learning**: We used **ResNet50**, a model pre-trained on **ImageNet**. The model is modified by replacing its final classification layer to match the number of classes in our dataset (the number of unique fonts).
   - **Training**: We fine-tuned the last fully connected layer while freezing the weights of the pre-trained layers to retain learned features.
   - **Loss Function**: **CrossEntropyLoss** was used for multi-class classification.
   - **Optimizer**: **SGD** with a learning rate of **0.002** and momentum **0.9**.

---

## [Section 3] Experimental Design:

### Purpose:
- **Main Goal**: To classify fonts in images using CNNs and Transfer Learning with ResNet50.
- **Baseline**: We compare our results against a **baseline** using a simple CNN model and no data augmentation.

### Metrics:
- **Accuracy**: The primary metric for evaluating the performance of the model on the training and test sets.
- **Loss**: Monitored using **CrossEntropyLoss** to ensure the model is learning effectively.

---

## [Section 4] Results:

### 1. **Training Loss and Accuracy**:
   - The training loss decreased steadily with each epoch, indicating that the model was learning the features associated with the fonts.
   - The accuracy for the training set increased, showing that the model was able to correctly classify the fonts in the training data.
   
   ![Training and Accuracy Graph](images/training_and_accuracy_graph.png)

### 2. **Test Accuracy**:
   - After training, the model was evaluated on a held-out test set to assess its ability to generalize. The test accuracy was **X%**, indicating good performance.

   ![Test Accuracy](images/test_accuracy.png)

### 3. **Results File**:
   - The results are saved in an Excel file that includes the training and test accuracies for each epoch. It also includes a graphical representation of the training process.

   ![Training Results](images/training_results.png)

---

## [Section 5] Conclusions:

### Summary:
This project successfully applied CNNs with Transfer Learning to classify font types in images. The model performed well with a final test accuracy of **X%**, and by using data augmentation and fine-tuning a pre-trained ResNet50 model, we were able to achieve significant performance improvements with relatively small data.

### Future Work:
- **Increasing Dataset Size**: Expanding the dataset could help improve the generalization ability of the model.
- **Improved Data Augmentation**: Experimenting with additional data augmentation techniques, such as more advanced image transformations, could further boost model performance.
- **Exploring Other CNN Architectures**: Using deeper models or lightweight models like **MobileNet** could be considered for better efficiency.

---

## Folder Structure:

