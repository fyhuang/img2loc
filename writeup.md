# Modeling the problem

## Simple baseline (binary classification)

I started with a very simplified model of the problem to get a baseline for performance.
This first version trains a binary classifier on Northern Califoria vs Southern California.

For this, I generated a tiny dataset (Street View images from major California cities).

After some experimentation, I found that this "baseline" was actually not representative.
The model might be capable of predicting much more precise locations than simply northern vs southern CA, but having only 2 classes severely limits the ability of the model to express higher accuracy.

I quickly moved on to the approach used by several other papers.

## S2 cell multi-class classification

Similar to PlaNet.

The problem is modeled as a multi-class classification, where each class is one S2 cell.
S2 cells are picked from the training dataset using an adaptive algorithm which limits the min/max number of examples in each cell.

Using im2gps dataset, created a set of 1776 S2 cells.

### First results

This model is MobileNet v3 pre-trained on ImageNet.
I replaced the classifier "head" with one that predicts 1776 classes, and trained this on the im2gps (2007) dataset.
Training until validation accuracy plateaued took about $20 of GPU time on Lambda Cloud (A10).
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
See [dataset section](#im2gps-recreating-the-dataset-in-2024) for more details.

What I intend to try next:

* Manual error analysis on validation set.
* Fine-tune the full model, not just final layers.
* Pick the learning rate with hyperparameter optimizer.
* Increase size of dataset in worst-performing areas.

# Datasets

## Tiny CA dataset

## im2gps: recreating the dataset in 2024

The [original code](http://graphics.cs.cmu.edu/projects/im2gps/flickr_code.html) is quite old (2009), uses Python 2, and no longer runs properly with the current Flickr API. I created an updated version.

The query portion uses the updated flickrapi package and uses a simpler algorithm to find the best time-block sizes.

The downloader portion is written in Python rather than Matlab. It just focuses on one format (1024px JPG). Downloading the originals adds a lot more complexity: for example, the original may not be a JPG. The downloader also skips resizing the image, as this transform can be done at training time.

The query dates for the original im2gps dataset are quite old at this point (~2007). As of 2024, this dataset contains ~630k images for a total of 109 GB.
According to the PlaNet paper, the original dataset contained 6.5M images.
In 17 years, only ~10% of the original dataset is left.


# Misc

## Creating datasets: store metadata in DataFrames

Pandas DataFrame for intermediate results. Saving intermediate data using df.to_pickle. Similar to checkpointing for model training, but on the dataset retrieval side. Makes it much easier to resume if something goes wrong. Makes it easier to tweak the dataset later if needed, without recreating everything from scratch.

# Tech Stack

Also see [tech stack thoughts](tech_stack.md).