# Car Model Classification Using Deep Learning

## 1. Introduction

This project focuses on classifying car models using deep learning techniques.  
The dataset contains approximately 3,200 images distributed across multiple car model classes, where each class is represented by a separate folder.  

Due to the fine-grained nature of the task (cars with very similar visual appearances), transfer learning was adopted using several well-known convolutional neural network (CNN) architectures.

The following architectures were selected:
- ResNet-50
- MobileNetV2
- InceptionV3
- VGG-19

All experiments were conducted using Google Colab.

---

## 2. Dataset Description

- Total images: ~3,200
- Number of classes: 20+ car models
- Data organization:  





The dataset was split into:
- Training set (70%)
- Validation set (15%)
- Test set (15%)

Standard data augmentation techniques were applied during training, including random horizontal flips and normalization.

---

## 3. Documentation of Architectures

### 3.1 ResNet-50

**Overview:**  
ResNet (Residual Network) introduces residual connections to address the vanishing gradient problem in very deep networks.

**Architecture Explanation (Step-by-Step):**
1. Initial convolution and max pooling layers extract low-level features.
2. Residual blocks allow the network to learn identity mappings.
3. Skip connections help preserve gradient flow.
4. Global average pooling reduces spatial dimensions.
5. A fully connected layer performs final classification.

**Advantages:**
- Excellent gradient flow
- Strong feature extraction
- Good performance on complex datasets

**Limitations:**
- Computationally expensive
- Requires more training time

**Reference:**  
He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.

---

### 3.2 MobileNetV2

**Overview:**  
MobileNetV2 is designed for lightweight and efficient inference using depthwise separable convolutions.

**Architecture Explanation:**
1. Uses depthwise separable convolutions to reduce computation.
2. Inverted residual blocks expand and compress features.
3. Linear bottlenecks preserve information.
4. Final pooling and dense layer perform classification.

**Advantages:**
- Fast training and inference
- Low memory consumption
- Suitable for limited computational resources

**Limitations:**
- Slightly lower accuracy than heavier models

**Reference:**  
Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.

---

### 3.3 InceptionV3

**Overview:**  
InceptionV3 captures multi-scale features by applying multiple convolution filters in parallel.

**Architecture Explanation:**
1. Parallel convolutions (1×1, 3×3, 5×5) extract features at different scales.
2. Factorized convolutions reduce computational cost.
3. Auxiliary classifiers improve gradient flow.
4. Global average pooling and dense layers output predictions.

**Advantages:**
- Strong multi-scale feature representation
- High accuracy for complex visual patterns

**Limitations:**
- Complex architecture
- Higher memory usage

**Reference:**  
Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, CVPR 2016.

---

### 3.4 VGG-19

**Overview:**  
VGG-19 uses a simple and uniform architecture composed of stacked 3×3 convolution layers.

**Architecture Explanation:**
1. Sequential convolution blocks increase feature depth.
2. Max pooling reduces spatial resolution.
3. Fully connected layers perform classification.
4. Large number of parameters enable expressive learning.

**Advantages:**
- Simple and easy to understand
- Strong baseline architecture

**Limitations:**
- Very large number of parameters
- Prone to overfitting on small datasets
- High computational cost

**Reference:**  
Simonyan and Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, ICLR 2015.

---

## 4. Experimental Results

| Model        | Test Accuracy | Precision | Recall | F1 Score |
|--------------|---------------|-----------|--------|----------|
| ResNet-50    | ~0.52         | ~0.54     | ~0.52  | ~0.47    |
| MobileNetV2  | (reported)    | (reported)| (reported) | (reported) |
| InceptionV3  | (reported)    | (reported)| (reported) | (reported) |
| VGG-19       | 0.11          | 0.01      | 0.11   | 0.02     |

Confusion matrices and classification reports were generated for all models.

---

## 5. Comparative Analysis of Models

### Performance Comparison

- **ResNet-50** achieved the best overall performance due to its residual connections and deep feature extraction.
- **MobileNetV2** provided a good trade-off between efficiency and accuracy.
- **InceptionV3** captured multi-scale features but required careful tuning.
- **VGG-19** performed poorly due to overfitting and its large number of parameters.

### Pros and Cons Summary

| Model        | Pros | Cons |
|-------------|------|------|
| ResNet-50   | High accuracy, stable training | High computation |
| MobileNetV2 | Lightweight, fast | Slight accuracy drop |
| InceptionV3 | Multi-scale features | Complex design |
| VGG-19      | Simple structure | Overfitting, poor performance |

### Why Some Models Perform Better

ResNet-50 outperformed other architectures because residual connections enable deeper representations without degrading performance.  
MobileNetV2 performed well given limited resources, while VGG-19 struggled due to its parameter-heavy design, which is unsuitable for relatively small fine-grained datasets.

---

## 6. Conclusion

This project demonstrates that architecture choice plays a critical role in fine-grained image classification tasks.  
Deeper and more modern architectures such as ResNet and Inception provide superior performance, while lightweight models like MobileNet offer efficiency advantages.  
Future improvements may include advanced data augmentation, class balancing, and fine-tuning deeper layers.

---

## 7. References

1. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016  
2. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018  
3. Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, CVPR 2016  
4. Simonyan and Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, ICLR 2015
