# Where's Whaledo?
Beluga whale classification through Metric Learning.
**Group 18**

This blog post describes the development and implementation of a metric learning network for the [Where's Whale-do?](https://www.drivendata.org/competitions/96/beluga-whales/) competition Hosted by Driven-Data.

# Introduction
## Problem Statement
The Cook Inlet Beluga whale is an endangered species. To monitor populations of these whales, the Marine Mammal Laboratory at the NOAA Alaska Fishery Science Center began making overhead and lateral photographs of these belugas. However, analyzing these new images takes a lot of time for an individual to do.  **The goal of this challenge is to help wildlife researchers accurately identify endangered Cook Inlet beluga whale individuals from photographic images.** [3] 

The main task is to match a query image of an unidentified whale against a database of labeled whale images. The algorithm should return a ranking of the images most likely to be the same whale. The final performance metric of a model is scored with the mean average precision (mAP).

# Data
Let's look at the data to get a clearer view of the problem. 
### Training Data
The dataset consists of 5902 images labeled with 788 unique whale IDs. The images are taken from three viewpoints: top view and side views from the left and the right. The majority of images are top views.
We observed two main difficulties with the data. 

The first difficulty is the stark difference between top and side views. 
Here are 4 samples of the same whale:
| Top view | Top view  | Side view | Side view |
| :---: | :---: | :---: | :---: |
| <img height="400" src="https://i.imgur.com/nhqxRNZ.jpg"> | <img height="400"  src="https://i.imgur.com/IuLTPXH.jpg"> | <img height="400" src="https://i.imgur.com/wXmKtpx.jpg"> | <img height="400" src="https://i.imgur.com/pgF3zZd.jpg"> 

The second difficulty is the data distribution. For the majority of whales, there are only one or two images. This makes the problem fall into the category of one-shot learning.
![](https://i.imgur.com/AzslXYB.png)

### Test data
To evaluate how the model performs under various conditions the challenge uses a scenario-based evaluation approach. A scenario contains a distinct set of query images and a set of database images.

### Metadata
We also got access to a `metadata.csv` file which contains more information on each image. 

The following information was provided for both train and test data.
`image_id`: a unique identifier for each image.
`path`: path to the corresponding image file.
`height`: the height of the image in pixels.
`width`: the width of the image in pixels.
`viewpoint`: the direction from which the whale is being photographed; may be "left", "right" or "top". 
`date`: date that the photograph was captured.

The following information was only available during training:
`timestamp`: the full timestamp that the photograph was captured with second-level resolution.
`whale_id`: an identifier for the individual whale in the photograph. However, note that an individual whale can have multiple whale IDs. 
`encounter_id`: an identifier for the continuous encounter of the individual whale during which the photo was taken. 




# Our Approach
We decided to tackle this problem with metric learning. Since it is a ranking task rather than a classification task because the goal is to return the database images that are most similar, or closest, to the query image.

In that vein, our idea was to learn the most important features to discriminate between whales across images. We took inspiration from seminal works in this field [5] whose contributions were mainly in the form of new architectures and loss metrics.

Schneider et. al [2], in their paper, come up with an approach that uses triplet learning to do animal reidentification in a database of images. This is exactly what we are looking for!

To sum up, our approach for this task is to train a neural network to learn rich representations from images of whales and then use these representations in finding similar images of a query whale in a database of whale images.

To accomplish this, we made use of the PyTorch package Metric Learning which is an open-source library that eases the tedious and time-consuming task of implementing various deep metric learning algorithms [10].

In the next couple of sections, we will go into detail about our approach starting from image preprocessing to our final model evaluations. 


## Image preprocessing

To ensure the robustness of our model, we apply the following random augmentations to the training dataset. It is important to note that we keep the aspect ratio of the original images even through all the other augmentations. 

1. **Rotate landscape images & resize with padding** 
A fraction of the images is also taken from the left or the right of the whale. These "landscape" images are rotated into portrait images to make them consistent with the other top view images. 
2. **Resize with padding**
Images were resized to fit the mean height and width of all the images. Padding was added to maintain the ratios.
3. **Add random contrast** 
 To help features like the scratches on the whales to be more easily detected.
4. **Add random brightness** 
5. **Random crop** 
6. **Normalize images**
7. **Add viewpoint as 4th channel** 
To help the model distinguish between these left/right and top view images we add a mask, which is a matrix  (same height and width as the augmented image) of either -1 (left), 0 (top), or 1 (right) as the 4th channel. 

## Whaledo Model
We developed the WhaleDo model, which consists of two submodels. The first one is a ResNet50 backbone model which is the main model that learns the representations from the images. The second submodel is a projector network, which is a simple MLP network that projects the learned features into a lower-dimensional subspace.

An important note here is that the ResNet backbone works with only 3-channels but we add an additional channel to our data in the augmentation step. To accommodate this change, we modify the first layer of the ResNet backbone to accept 4 channels as input. 


## Triplet Margin Loss
We used the `TripletMarginLoss`, first proposed by Schroff et al. with FaceNet [1]. The `TripletMarginLoss` tries to separate dissimilar pairs from any similar pairs by at least a given margin. If the model underfits, the margin should be increased. When the model overfits, the margin should be decreased. The following loss formula describes the `TripletMarginLoss`:

$$
L(a, p, n)=\max \left\{d\left(a_{i}, p_{i}\right)-d\left(a_{i}, n_{i}\right)+\operatorname{margin}, 0\right\}
$$
where 
$$
d\left(x_{i}, y_{i}\right)=\left\|\mathbf{x}_{i}-\mathbf{y}_{i}\right\|_{p}
$$

### Triplet Miner
Training on all possible triplets leads to poor performance [5], the model should instead only be trained on pairs that violate the constraint of the `TripletMarginLoss`. Therefore we make use of a `TripletMarginMiner`. 

This miner takes a batch of embeddings and returns triplets (anchor ($a$), positive ($p$), negative ($n$)) which can be used to calculate the loss. The margin is the difference between the anchor-positive distance and the anchor-negative distance. 

There are three triplet types: *hard*, *semi-hard* and *easy*. For the *hard* triplets the negative is closer to the anchor than the positive. While for *semihard* triplets the negative is further from the anchor than the positive, but still violates the margin constraint. For the *easy* triplets the margin constraint is met [4]. We only used the *hard* and *semihard* options.

It can be visualized as follows:

<img src="https://i.imgur.com/tEjCX25.png " width="400">


## Training Pipeline
Together these components form our training pipeline. Images are transformed into embeddings using the Whaledo model. Then the miner uses the embeddings and labels to produce a set of hard or semi-hard triplets. These triplets are then used to compute the triplet margin loss, and with this loss, the weights in our Whaledo model get updated. 

The final training procedure can be visualized as follows:
![](https://i.imgur.com/uIX9JXm.png)

## Testing Pipeline
Because of the specific nature of the problem, testing the performance of the algorithm during the training procedure was difficult. Testing the performance required a specific setup well described on the challenge website [6]. We used a scoring script that simulated the task on the training dataset.

During testing the projector layer was omitted, a measure proposed in the work of Grill et al. [5]. This consistently improved performance on the test set. 

The final testing procedure can be visualized as follows:

![](https://i.imgur.com/tORXfM7.png)

# Experiments
This chapter describes the loss results, hyperparameter tuning and performance results.

## Loss 
While training the model we tracked the average training loss for each epoch as well as the validation loss, both being triplet margin losses. We observed a very rapid stagnation of the loss value for all runs. 

Our first model suffered from noisy losses and rapid stagnation around the margin, as can be seen in the following example.
| | Hard Pair Example | 
| :---: | :---: |
Margin | 0.1|
Plot | ![](https://i.imgur.com/zx38gXW.png) 

For every margin, the losses stagnated around the margin. This indicates that the distances between the negative and positive pairs become almost zero. From a geometric perspective, this means that the algorithm learns to map all the images to the same area in the vector space.

This behavior was described by Wu et al. [8] which led us to the main cause: the triplet miner being configured to mine hard pairs.

When the miner was configured to mine semi-hard pairs the loss dropped below the margin but still stagnated almost immediately. Next to that the loss also seemed to become less noisy.
In addition to the margin we also evaluated the other hyperparameters but they did not seem to affect the loss functions in any meaningful way. 

| | Semi-Hard Pair Example A | Semi-Hard Pair Example B  | 
| :--: | :---: | :---: | 
Margin  | 0.1 |  0.3
|Plot | ![](https://i.imgur.com/WkBK1dl.png) | ![](https://i.imgur.com/EMeuzRL.png)

Despite being lower the losses still stagnated. This indicated that our model still collapsed 
We investigated this hypothesis by constructing a TSNE scatter plot, which confirmed this.
| 2D TSNE Plot | 3D TSNE Plot | 
| :--: | :---: |
![](https://i.imgur.com/jR58QEl.jpg) | ![](https://i.imgur.com/adB1Zc8.jpg)

## Performance
During the testing procedure highest the mean average precision attained was 26.04 percent. When submitted on the official competition dataset the same model reached a performance of 1.30 percent map. At the time of writing the highest achieved score in the competition is 47.70 percent map. 

In the next section, we discuss potential causes for this extremely low performance.


## Hyperparameters
We explored the influence of a few hyperparameters, which we suspected to have a significant impact on performance. In particular, we studied margin, training batch size, learning rate, output dimension of the projector and distance norm.

The **margin** is the minimal distance between two different classes. Generally, increasing a margin leads to less underfitting and decreasing a margin leads to less overfitting. Because we normalize our embeddings, all points get projected on a high-dimensional unit-sphere, and therefore the theoretical maximum margin that could be achieved between two classes is 2. 

The **training batch size** should be as big as possible. A bigger batch size allows the miner to find more pairs because a miner can only mine for pairs inside one batch.
Therefore the training batch size should take up the maximum size that the GPU can handle, for our machine which was 64. 

The **learning rate** is decreased systematically with a`ReduceLROnPlateau` scheduler. It did not seem to have a significant effect on the overall value of the loss nor the amount of noise.

The **output dimension of the projector** was decreased to combat the effects of high dimensions. We found no significant effect.

The **distance norm** was also decreased to combat the effects of high dimensions, as proposed by Aggarwal et al. [11]. We mainly used the standard Euclidean distance metric, which has a distance norm of 2. The examples below show that a lower distance norm did not have a significant effect on performance.

| | Example A | Example B  | 
| :--: | :---: | :---: | 
Margin | 0.1 |  0.1
Distance Norm  | 0.1 |  0.3
|Plot | ![](https://i.imgur.com/F2dO7AW.png) | ![](https://i.imgur.com/EPCiacI.png)


# Discussion
Because of the stagnating losses around the margin we hypothesize that our model suffers from a "model collapse". We were able to decrease the loss under the margin value, meaning the negative pairs were more distant than the positive pairs, but this nevertheless resulted in similar performance.

We hypothesize that this model collapse is due to the "curse of dimensionality" where for high dimensions all points come to lie close to each other [11] and the ratio between the furthest points and the closest point becomes close to 1. We made some attempts to lower the embedding dimensionality and change our distance metrics but did not find any improvements.


# Future work
We did not have the time to incorporate changes, therefore we made some suggestions that can be tried out:

1. Train the model only on one image perspective instead of all three.
2. Since the images are very similiar, try different data augmentations.
3. Some images do have a white spot due to cutting off the image, drop those images or augment them.
4. Implement techniques to avoid model collapse such as "Learning Spread-out Local Feature Descriptors" and "Simple Positive Sampling" [9, 12].






# Conclusion 
Our attempt at classifying whales using metric learning was not successful. Despite this, we gained significant insights into the inner workings of metric learning. We also revealed the probable cause of the poor performance: a model collapse, where the model learns to map all the embeddings to the same area in the vector space, which we hypothesise to be an effect of the effects of high dimensionality.

All our code is available on [GitHub](https://github.com/gauthsvenkat/whaledo).


# References

[1]: Schroff, F., Kalenichenko, D., &amp; Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2015.7298682 

[2]: Schneider, S., Taylor, G. W., &amp; Kremer, S. C. (2020). Similarity learning networks for animal individual re-identification - beyond the capabilities of a human observer. 2020 IEEE Winter Applications of Computer Vision Workshops (WACVW). https://doi.org/10.1109/wacvw50321.2020.9096925 

[3]: DrivenData, *Where's Whale-do?*, https://www.drivendata.org/competitions/96/beluga-whales/page/478/

[4] Musgrave, K., (2019). *The New PyTorch Package that makes Metric Learning Simple*, https://medium.com/@tkm45/the-new-pytorch-package-that-makes-metric-learning-simple-5e844d2a1142

[5]: Grill, J.B., Strub, F., Altché F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Pires, B.A., Guo, Z.D., Azar, M.G., Piot, B.,  Kavukcuoglu K., Munos, R., Valko, M. (2020). Bootstrap your own latent: A new approach to self-supervised Learning. https://doi.org/10.48550/arXiv.2006.07733

[6]: DrivenData, *Code Submission Format*, https://www.drivendata.org/competitions/96/beluga-whales/page/478/

[7]: Hermans, A., Beyer, L., Leibe, B.: In defense of the triplet loss for person re- identification. arXiv preprint arXiv:1703.07737 (2017)

[8]: Wu, C.Y., Manmatha, R., Smola, A.J., Krahenbuhl, P.: Sampling matters in deep embedding learning. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 2840–2848 (2017)

[9] Zhang, X., Yu, F., Kumar, S., (2017). Learning Spread-out Local Feature Descriptors, 
https://doi.org/10.48550/arXiv.1708.06320
Focus to learn more

[10] Musgrave, K., (2019), *PyTorch Metric Learning*, https://kevinmusgrave.github.io/pytorch-metric-learning/

[11] Aggarwal, C. C., Hinneburg, A., &amp; Keim, D. A. (2001). On the surprising behavior of distance metrics in high dimensional space. Database Theory — ICDT 2001, 420–434. https://doi.org/10.1007/3-540-44503-x_27 

[12] Levi, E., Xiao, T., Wang, X., &amp; Darrell, T. (2021). Rethinking preventing class-collapsing in metric learning with margin-based losses. 2021 IEEE/CVF International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv48922.2021.01015 