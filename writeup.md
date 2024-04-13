# Overview

Identify a location in the world without GPS, using only camera images.

Constraints:

* All computation, API access, etc. is self-funded

# Tech Stack

See [tech stack writeup](tech_stack.md) for an analysis of the tech stack that I used for this project.

# Work Log

## Simple baseline (binary classification)

I started with a very simplified model of the problem to get a baseline for performance.
This first version trains a binary classifier on Northern Califoria vs Southern California.

For this, I generated a tiny dataset (Street View images from major California cities).

### Tiny CA dataset

This overall approach was used to generate the dataset:

* Sample a set of raw lat/lng locations from the target area (California).
* For each raw location, get the closest Street View panorama location from the API.
* For each panorama, download several Street View images at different headings.

For sampling the raw locations, I decided to sample addresses from the OpenAddresses database, rather than uniformly sampling points in a polygon.

My reasoning was that sampled addresses were much more likely to be near a Street View location.
Addresses are also more tightly packed in "interesting" areas (towns, city centers), so there would be fewer uninteresting samples in the dataset (long flat roads in the middle of nowhere).

### Analysis

I did error analysis on the examples that the model got "the most wrong" (10 examples with largest loss).
Most of them were genuinely difficult even for a human.
Because these images came straight from Street View, there were examples that were simply images of a parking lot, or facing a blank wall of a building.

I suspected the model might be capable of predicting much more precise locations than simply northern vs southern CA.
Because this problem formulation only has 2 classes, it severely limits the ability of the model to express higher accuracy.


## S2 cell multi-class classification (1776 cells)

I wanted to try modeling the problem as classification over small patches of the earth's surface (S2 cells), similar to PlaNet.
Each class is one S2 cell.
S2 cells are picked from the training dataset using an adaptive algorithm which limits the min/max number of examples in each cell.

I needed a much larger and more diverse dataset in order to have enough cells.
It didn't seem realistic to download such a large dataset from Street View.
So I tried to recreate the im2gps dataset from the PlaNet paper.

Using the im2gps dataset, adaptive cell selection created a set of 1776 S2 cells.

### im2gps: recreating the dataset in 2024

The [original code](http://graphics.cs.cmu.edu/projects/im2gps/flickr_code.html) is quite old (2009), uses Python 2, and no longer runs properly with the current Flickr API.
I created an updated version in [im2gps.ipynb](dataset/im2gps/im2gps.ipynb).

The query portion uses the updated flickrapi package and uses a simpler algorithm to find the best time-block sizes.

The downloader portion is written in Python rather than Matlab.
It just focuses on one format (1024px JPG).
Downloading the originals potentially adds a lot more complexity: for example, the original may not be a JPG.
The downloader also skips resizing the image, as this transform can be done at training time.

It's now 2024, and the im2gps dataset contains images from 2007.
It's interesting to see how this dataset has "evolved" in the 17 years since.
As of 2024, this dataset contains ~630k images for a total of 109 GB.
However, according to the PlaNet paper, the original dataset contained 6.5M images.
In 17 years, only ~10% of the original dataset is left.

### Attempt #1 (MobileNet v3)

This model is MobileNet v3 pre-trained on ImageNet.
I replaced the classifier "head" with one that predicts 1776 classes, and trained this on the im2gps (2007) dataset.
Training until validation accuracy plateaued took about $20 of GPU time on Lambda Cloud (A10).

#### Attempt #1 analysis

Initial performance was not very impressive:

<table>
<tr>
    <th>Model (# params)</th>
    <th>Street 1km</th>
    <th>City 25km</th>
    <th>Region 200km</th>
    <th>Country 750km</th>
    <th>Continent 2500km</th>
</tr>
<tr>
    <td>PlaNet (6.2M)</td>
    <td>6.3%</td>
    <td>18.1%</td>
    <td>30.0%</td>
    <td>45.6%</td>
    <td>65.8%</td>
</tr>
<tr>
    <td>Ours v1 (8.6M)</td>
    <td>0.03%</td>
    <td>1.6%</td>
    <td>3.0%</td>
    <td>4.3%</td>
    <td>7.4%</td>
</tr>
</table>

One major difference in the dataset is that many of the original im2gps dataset images are no longer available.
Consequently, this model is trained on a dataset that is ~10x smaller.

The choice of S2 cells also impacts the model's performance, and I wondered whether the 10x smaller dataset was creating a sub-optimal choice of S2 cells.
As a quick sanity check, I wrote a function to predict the closest S2 cell for each test set example.
Results were as follows:

<table>
<tr>
    <th>Model</th>
    <th>Street 1km</th>
    <th>City 25km</th>
    <th>Region 200km</th>
    <th>Country 750km</th>
    <th>Continent 2500km</th>
</tr>
<tr>
    <td>PlaNet (6.2M)</td>
    <td>6.3%</td>
    <td>18.1%</td>
    <td>30.0%</td>
    <td>45.6%</td>
    <td>65.8%</td>
</tr>
<tr>
    <td>Best S2Cell</td>
    <td>0.9%</td>
    <td>24.5%</td>
    <td>49.2%</td>
    <td>93.0%</td>
    <td>97.8%</td>
</tr>
</table>

It appears our S2 cell set isn't fine-grained enough to match PlaNet on street-level accuracy.
However, the performance of region, country, and continent-level predictions isn't limited by the S2 cell set.
So our model should have headroom to beat PlaNet on those dimensions.

What I intend to try next:

* Manual error analysis on validation set.
* Fine-tune the full model, not just final layers.
* Pick the learning rate with hyperparameter optimizer.
* Increase size of dataset in worst-performing areas.
* Try multi-label classification, to see if learning the parent-child relationship of S2 cells helps.

# Datasets

# Misc
