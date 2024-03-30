# Modeling the problem

## Simple baseline (binary classification)

I started with a very simplified model of the problem to get a baseline for performance.
This first version trains a binary classifier on Northern Califoria vs Southern California.
For this, I generated a tiny dataset (Street View images from major California cities).
However, after some experimentation, I found that this "baseline" was actually not representative.
The model is capable of predicting much more precise locations than simply northern vs southern CA, but having only 2 classes severely limits the ability of the model to express higher accuracy.

I quickly moved on to the approach used by several other papers.
Namely, the problem is modeled as a multi-class classification, where each class is one S2 cell.
S2 cells are picked from the training dataset using an adaptive algorithm which limits the min/max number of examples in each cell.

# Datasets

## Tiny CA dataset

## im2gps: recreating the dataset in 2024

The [original code](http://graphics.cs.cmu.edu/projects/im2gps/flickr_code.html) is quite old (2009), uses Python 2, and no longer runs properly with the current Flickr API. I created an updated version.

The query portion uses the updated flickrapi package and uses a simpler algorithm to find the best time-block sizes.

The downloader portion is written in Python rather than Matlab. It just focuses on one format (1024px JPG). Downloading the originals adds a lot more complexity: for example, the original may not be a JPG. The downloader also skips resizing the image, as this transform can be done at training time.

The query dates for the original im2gps dataset are quite old at this point (~2007). As of 2024, this dataset contains ~630k images for a total of 109 GB.


# Misc

## Creating datasets: store metadata in DataFrames

Pandas DataFrame for intermediate results. Saving intermediate data using df.to_pickle. Similar to checkpointing for model training, but on the dataset retrieval side. Makes it much easier to resume if something goes wrong. Makes it easier to tweak the dataset later if needed, without recreating everything from scratch.

# Tech Stack

Also see [tech stack thoughts](tech_stack.md).