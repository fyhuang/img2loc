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

Findings

* Without dropout, seemed to perform OK (f1 = 0.43)
* Training feels so slow with the larger dataset
* Dropout (p=0.4) at the hidden layer makes it perform way worse (although this is what EFN defaults to)

To do:

- [x] Debug the GeoGuessr score implementation
- [x] Set up "infinite training" (do validation, LR scheduler, etc. in the middle of long epoch)
- [x] Try out tinyvit
- [x] Do ONNX js with best results from EFN
- [x] Cluster dataset and remove bad examples

### Attempt #4 (EFN small v2, with different dataset sizes)

Larger dataset didn't seem to improve performance.
Even after removing weird clusters, performance still seems lower than with the smaller dataset.
Experiment with: same val set, but different subsets of the training set.

Common hparams:
* Max steps: 200k.
  Calculation: 58k steps per epoch, 200k steps is ~4 epochs. (12h per run)
* LR scheduler: 1k steps x 50 patience, using train_loss_step
* Data: 224px, random center crop, color jitter (0.1x4)
* Val values unsmoothed.
* Train loss from last step (smoothed).
* Val metrics are best from run.

| Training set fraction | Log version | Train loss | Val loss | Val F1 | Val acc |
| --------------------- | ----------- | - | - | - | - |
| world              | 1 | 0.1981 | 1.495 | 0.0403 | 0 |
| world + im2gps 10% | 2 | 0.2855 | 0.512 | 0.0259 | 0 |
| world + im2gps 50% | 3 | 0.4299 | 0.502 | 0.0241 | 0 |

For following experiments:

* Dataset: world + im2gps 10%
* Batch size 16, max steps 200k

| Experiment | Log version | Train loss | Val loss | Val F1 | Val acc |
| ---------- | ----------- | - | - | - | - |
| hidden 2048          | 4 | 0.2704 | 0.504 | 0.0300 | 0 |
| hidden 2048 aug 0.01 | 5 | 0.3834 | 0.502 | 0.0263 | 0 |
| hidden 2048 ndo      | 6 | 0.3441 | 0.523 | 0.0287 | 0 |
| hidden 2048 aug2     | 7 | 0.3177 | 0.498 | 0.0256 | 0 |
| hidden 2048 aug2 fval| 8 | - | - | - |

Notes:

* Hidden 2048. Helped somewhat
* aug 0.01: reduce hue/sat jitter to 0.01. Didn't help (val worse)
* ndo: no dropout. Didn't help (val worse)
* aug2: brightness/contrast jitter to 0.01. hue/sat jitter to 0. Seemed to help, but scheduler skipped lowering the LR.
* fval: full val crop (no zoom). Helped val performance in early epochs. Stopped early b/c I realized it doesn't affect train-time at all.

For following experiments:

* Kept hidden size at 2048.
* Kept augmentation at "aug2" (no hue/sat jitter; b/c jitter at 0.01).
* Kept validation at no zoom.
* Dropout remains at default (p=0.2).

| Experiment | Log version | Train loss | Val loss | Val F1 | Val acc |
| ---------- | ----------- | - | - | - | - |
| 384                 | 10-11 | 0.3937 | 0.540 | 0.0235 | 0 |
| hidden-do           | 12    | 0.2936 | 0.490 | 0.0304 | 0 |
| mnet3               | 13    | 0.3424 | 0.554 | 0.0316 | 0 |
| vitb16              | 14    | 0.3669 | 0.613 | 0.0286 | 0 |
| tvit-v2             | 15    | 0.4917 | 0.680 | 0.0026 | 0 |

Notes:

* 384: training + val at 384px. Batch size 8, increased max_steps to 400k.
  Performed worse; however, I forget to change the LR scheduler interval.
  Tests with smaller batch size are also taking way too long.
* 512: training + val at 512px.
* hidden-do: move dropoff after hidden layer. p=0.2. better
* mnet3: mobilenet v3. starts off with much better performance. gap decreases over time. at 200k, almost equal with other models.
* vitb16: larger model (vit-base-patch16-224). image size 224px. normalized to 0.5x6 (not the same as others). pre-logits size 2048. Slower at first by almost caught up by 200k steps. Probably a good change, but increases training time a lot.
* tvit-v2: dataset with std imagenet normalization. classifier head hidden 2048, DO after hidden p=0.2. AdamW optimizer, settings from paper (ImageNet-1k training from scratch). Didn't implement cosine annealing LR (too few epochs).
  
  Didn't make any progress at all. Training completely stalled.

#### Manually inspect val set

* World1 looks good, as expected.
* Im2gps_v2 is >90% good.
* Some shots are still indoors, or too narrow FOV.
* Can potentially do some more cleaning.

### Attempt #5 (optimizing TinyViT)

The first attempt with TinyViT was very poor, much poorer than expected.
Train loss was basically flat.
Here we do some experiments to try and determine why.

Baseline settings:

* Standard ImageNet normalization.
* Data augmentation: same jitter as "aug2".
* Classifier head: one hidden layer, size 2048.
* Dropout p=0.2 after hidden layer.
* Optimizer: AdamW with weight_decay=0.05.
* LR: start at 1e-3, use ReduceLROnPlateau scheduler
* Dataset: world + 10% im2gps_v2

| Experiment | Log version | Train loss | Val loss | Val F1 |
| ---------- | ----------- | - | - | - |
| tvit-v2             | 15    | 0.4917 | 0.680 | 0.0022 |
| adam                | 16    | 0.5195 | 0.685 | 0.0022 |
| no-hidden           | 17    | x | x | x |

Experiments:

* adam: use Adam optimizer with no weight decay.
  
  No difference from baseline.
* no-hidden: classifier head is direct. No dropout. AdamW. Forgot to initialize weights of layer...

  No difference from baseline.

* debug with overfit set
* double check if layers frozen
* gradient clipping
* other stuff from the paper (depth something)

### Debugging TinyViT

* Overfit-1. log_version=0. With no hidden layer or dropout. No problem, gets to val_f1=0.3 within 3000 steps.
* Overfit-5. log_version=1. Could not learn, loss is flat.
* Overfit-5. log_version=4. Fix LR scale on classification head layer. Also initialize head layer weights/bias. Might not do much, seems like decay_scale is 1.0 anyway.

  Didn't do anything. Could not learn, loss is flat.
* Overfit-5. log_version=5. Make sure all layers requires_grad=True. Didn't do anything.
* Overfit-5. log_version=8. Use model directly from timm (tiny_vit_21m_224.dist_in22k) with num_classes=X. Didn't do much.
* log_version=10. Use another model (resnet50.a1_in1k) from timm for sanity check. Worked.
* log_version=11. TinyViT. Much higher LR=1e-1. Bad.
* log_version=12. TinyViT. Much lower LR=1e-5. Slightly better (loss decreases for 2-3k steps), but performance is still very bad.
* log_version=13. TinyViT. LR=1e-6. Not as good as 1e-5.
* log_version=14. TinyViT. LR=1e-4. Actually working!

### Attempt #6 (optimizing TinyViT again)

First try to find the best settings on overfit set.

* Standard ImageNet normalization.
* Data augmentation: same jitter as "aug2".
* Optimizer: AdamW with weight_decay=0.05.
* Dataset: world overfit-5.
* Max steps: 20k (~30 epochs).

| Experiment | Log version | Train loss | Val loss | Val F1 |
| ---------- | ----------- | - | - | - |
| lr=1e-4              | 0 | 0.0411 | 0.0155 | 0.4488 |
| cosine lr            | 4 | 0.0398 | 0.0102 | 0.4321 |
| cosine-lr-30         | 5 | 0.0247 | 0.0047 | 0.5964 |

* lr=1e-4 with ReduceLROnPlateau scheduler.
* Cosine LR with warmup. (Use values from paper for 22k-to-1k fine-tuning, 10 warm-ups epochs instead of 5, ~60 total epochs instead of 30. Unclear if optimizer/weight decay is Adam or AdamW, here we use AdamW.) Oops, messed up the epoch count.
* Cosine LR, 5 epochs warmup + 25 epochs cosine. Very good.

### Attempt #7 (TinyViT after finding a good LR)

Baseline settings:

* TinyViT from timm.
* Standard ImageNet normalization.
* Data augmentation: same jitter as "aug2".
* Optimizer: AdamW with weight_decay=0.05.
* LR: start at 1e-3, use ReduceLROnPlateau scheduler
* Dataset: world + 10% im2gps_v2
* Max steps: 50k @ batch size 64 (~25 epochs)

| Experiment | Log version | Train loss | Val loss | Val F1 |
| ---------- | ----------- | - | - | - |
| tvit                 | 4 | 0.1139 | 0.5600 | 0.1338 |
| hidden-2048          | 5 | 0.1184 | 0.6547 | 0.1306 |
| hidden-2048-do       | 6 | 0.1409 | 0.6095 | 0.1228 |
| tvit-aug             | 7 | 0.1301 | 0.5483 | 0.1341 |
| w+20%                | 8 | 0.1634 | 0.3774 | 0.1087 |
| w+100%               | 9 | 0.1105 | 0.2827 | 0.1379 |
| hier-w+20           | 11 | 0.1538 | 0.3785 | 0.1050 |

* tvit. Cosine LR, 5 epochs warmup + 25 epochs cosine, max LR=5.0e-4. No pre-logits hidden layer. Oops, forgot to adjust val_check_interval.

  Train loss looked good and went down steadily. However, both val loss and val F1 went up over the course of training.

* hidden-2048. Same as above, but with hidden layer 2048. No dropout. Also adjusted val_check_interval.

  Suspect overfitting. Val loss even higher.

* hidden-2048-do. Same as hidden-2048 but with dropout=0.2. OK.
* tvit-aug. No hidden. Increased jitter (0.1 for b/c, 0.01 for h/s). Better than tvit.
* w+20%. world1 + 20% of im2gps. Same model (tvit with no hidden). Adjusted LR schedule (total 15 epochs).

  While val_loss curve looked much better (only going up slightly at the end), the overall performance was not better.
  However, this could have been because of fewer training epochs.

* w+100%. LR schedule adjusted for 30 epochs. max_steps=400k

* hier-w+20. same dataset as w+20%. hierarchical classification head. No large difference.