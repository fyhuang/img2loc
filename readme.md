# Overview

Identify a location in the world without GPS, using only camera images.

Project goals:

- [x] Train a PyTorch model that predicts location from an image.
- [x] Convert model to ONNX. Write demo page to retrieve images from camera and run inference using model.
- [ ] Try using the model to play GeoGuessr.

Constraints:

* All computation, API access, etc. is self-funded
* Model size (# parameters) should be small enough to run in a mobile (iPhone) browser

Source code:

* [Main training script](s2cell_ml.py)
* [Dataset creation](dataset/geodataset)
* [Utility classes](mlutil)
* [JavaScript inference demo](webdemo)

# Approach & Modeling

I went with a modified version of the approach from PlaNet [^1].
The model is a multi-**label** classification model over regions of the Earth (S2 cells).

Like PlaNet, I used an adaptive S2-cell splitting approach.
Unlike PlaNet, I formulated this problem as multi-label, rather than multi-class classification.

Multi-label classification allows the model to express low confidence in a precise prediction, but high confidence in an imprecise one.
For example, the model might confidently predict France for an image, but not be able to determine which city/region of France.

Unlike PlaNet, the adaptive S2 cell splitter does not delete parent cells when spltiting.
This aggregates training examples at larger scales, allowing for more efficient use of the limited training data.

With a smaller training set, and different parameters for adaptive partitioning, I ended up with a set of 930 S2 cells as labels.

This project presented several challenges:

* **Heavy class imbalance.**

  Each example has <6 labels that are 1.

  I observed some instances in which the standard BCE loss was getting stuck in the all-zeros local minimum.
  Switching to positive-weighted BCE loss improved this.

* **Relatively little training data compared to "Big" models.**

  PlaNet used a training set of 126M images.
  Our training set has just over 1M images.

  Adjusting the S2 cell splitting algorithm to preserve parent cells allow some (lower-precision) labels to have a much larger number of training examples.

  <!-- Unsupervised clustering of dataset image embeddings improved data quality.
  Clusters containing images that contain little learnable information (indoor scenes, portraits with background blur, etc.) are removed from the dataset. -->

* **Limited to smaller models in order to run on mobile devices.**

  I used SotA small base models and multi-resolution training to get reasonable performance.

* **Limited resources for training.**

  The [tech stack](tech_stack.md) was chosen for simplicity of dev setup, and to make it easy to switch between my local workstation and cloud instances.

  I created smaller dataset subsets to debug model issues locally before launching full training runs.

  The environment was created in miniconda, and I wrote Ansible scripts to easily recreate the setup on Lambda Cloud.

[^1]: [Weyand, Kostrikov, and Philbin, “PlaNet - Photo Geolocation with Convolutional Neural Networks.”](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/45488.pdf)

# Dataset

The dataset is composed of images from two sources:

1. **Geotagged images from Flickr.**
   This is an extended/modified version of the Im2GPS dataset [^2]. Total: ~1M images.
2. **Images from Google Street View.**
   This is a custom dataset created for this project. Total: ~40k images.

The Flickr dataset is high volume, but has low quality.
These simply photos with geolocation metadata attached.

Street View data is high quality, and representative of the distribution at inference time.
(The motivating use case -- Geoguessr -- uses Street View images.)
However, this dataset has low volume (Street View images cost money), and is also very homogeneous.

To try to correct for these dataset issues, I tried:

* Overweighting the Street View data during training.
* Applying heavy data augmentation during training.
  In particular

Inspired by PyTorch's implementation of Vision Transformers [^3], I use a slightly cropped image at inference time (scale to 256px, followed by center crop at 224px).
This aims to correct for the size discrepancy introduced by train-time data augmentation.
<!-- This dataset is heavily filtered using unsupervised techniques to improve its usability. -->

[^2]: [James Hays, Alexei A. Efros. IM2GPS: estimating geographic information from a single image. Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2008.](http://graphics.cs.cmu.edu/projects/im2gps/)

[^3]: https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16

# Tech Stack

See [tech stack writeup](tech_stack.md) for an analysis of the tech stack that I used for this project.