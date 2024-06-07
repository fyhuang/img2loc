# Overview

Identify a location in the world without GPS, using only camera images.

Project goals:

- [ ] Train a PyTorch model that predicts location from an image.
- [ ] Convert model to ONNX. Write demo page to retrieve images from camera and run inference using model.
- [ ] Try using the model to play GeoGuessr.

Constraints:

* All computation, API access, etc. is self-funded
* Model size (# parameters) should be small enough to run in a mobile (iPhone) browser

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

* [x] Manual error analysis on validation set.
* [x] Pick the learning rate with LR finder. (LR finder suggested same LR I was already using, 1e-3)
* [x] Rewrite training script as plain old Python file (vscode Jupyter isn't too stable over SSH tunneling).
* [x] Shuffle dataset before compilation.
* [x] Fine-tune the full model, not just final layers.
* [x] Increase size of dataset in worst-performing areas.
* [x] Reduce number of labels, try overfitting again.
* [x] Try multi-label classification, to see if learning the parent-child relationship of S2 cells helps.

#### Error analysis

I did a quick error analysis by finding the top 10 examples where the model made the worst prediction (largest distance from true location).
Two things stood out to me:

* Many wrong guesses were made in 94cf (Sao Paolo), mainly images from other major cities.
* Top 10 wrong guesses were all four-letter tokens, i.e. larger cells.

The wrong predictions were all large cells, which cover more area, and might contain a larger number of examples.
This made me think the class distribution might be unbalanced, and the model was overfitting by simply selecting cells with more images.

* [x] Double check the distribution of classes in training set.

This only turned out to be mildly informative:

* The class guessed most-incorrectly (94cf) was not in the top 100 most common classes.
* The most common class (47b7) has 2123 training examples.
  This seems rather odd since the S2 cell splitting algorithm should have split this cell (with a limit of 500 examples).

* [ ] Check confusion matrix for least-correct cells.

#### Re-reading the paper

I re-read the PlaNet paper and realized that I misinterpreted the results tables.
The parenthesized numbers are apparently not the number of parameters in the model, as there is a separate table showing the # of parameters by number of classes.
The actual number of parameters is ~4x higher than I thought.
Next, I will try training on a larger base model.

### Attempt #2 (EfficientNet-B5)

EfficientNet-B5 (30M) more closely matches the # of parameters (29.1M for 2056 classes) from the original PlaNet paper.
I also tweaked the fine-tuning configuration by adding [FinetuningScheduler](https://github.com/speediedan/finetuning-scheduler).
This automates the process of freezing/unfreezing layers during finetuning.
Some things are a bit rough around the edges (e.g. the fast_dev_run argument to Trainer no longer works), but overall it seems like an improvement over my original hacky finetuning setup.

I first tried full fine-tuning on Lambda Cloud.
Training took ~10 hours.
Unfortunately, the results looked very poor, even worse than the MobileNet-based version.
Accuracy on the validation set was very low, almost zero.

#### Debugging with overfitting

To try to find out why, I tried debugging the model by creating a [smaller overfitting dataset](dataset/im2gps/im2gps_overfit.ipynb).
There are two overfitting sets: one with exactly one image from each class (1776 images), and one with five images.

On the first run, it became clear very quickly that the model was doing very poorly on the overfitting set.
I used the exact same images in training and validation, and accuracy was about 3%.

I tried to fix this by adding an extra hidden layer to the classifier head.
This version got to around 80% accuracy on the overfitting (1) set.
This was much better, but honestly still not as good as I expected for a model with 36M parameters.

I then tried the overfitting (5) set.
Encouragingly?

Weird behavior when unfreezing all layers (test loss shoots up).
Maybe need to adjust learning rate when unfreezing (lower learning rates seem to have made the jump smaller).

### Attempt #3 (EfficientNet with multilabel)

Changes:

* Multilabel classification
* Larger dataset
* Fewer classes -- more examples per class
* Includes some high quality data from Street View

To do:

- [ ] Debug the GeoGuessr score implementation