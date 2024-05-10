## WebDataset

Why I picked webdataset?

- Easy to transfer datasets to lambda cloud
- Fixed iteration order removed one source of possible unreproducibility
- Streaming from object storage (S3)

Downsides that I discovered during this project

1. Finding specific samples is a bit complicated. For example, during error analysis, I wanted to see which samples from the training set the model was performing worst at. Visualizing the specific samples required another full pass through the training set, since it's not possible to index into a WebDataset.
1. For the same reason as above, shuffling must be done with a cache. This was one of the reasons I had to move training to cloud: the shuffling ended up using all the RAM on my workstation.
1. Next time, I should shuffle the data (e.g. in the DataFrame) before building the webdataset.
   This would make it easier to split the dataset between train/val/test at runtime.
   Currently splitting must be pre-determined, and any change to splits requires recompiling the dataset (and reuploading/downloading).
   This would probably also improve the quality of the shuffling.

## PyTorch Lightning

Pros

1. I thought that reducing the training boilerplate was very helpful. Restoring from checkpoint is a one-liner and prevents some sources of error. Built in metrics & logging to TensorBoard was very helpful.

Cons

1. Didn't find that much stuff for error analysis. This part I had to write myself.

## Error Analysis

Had to write myself.

## VSCode, GitHub Copilot

I used VSCode (through code-server) in order to be able to use Copilot in Jupyter Notebooks. Overall, I would say this saved a lot of time.

Typos. One time, Copilot typed `sample_idx` instead of `ps.sample_idx` (in a loop `for ps in ...`). It just so happened that `sample_idx` was defined, and this little typo cost me about 30 minutes of confused debugging.

## Lambda Cloud

### Buy vs build

For someone who doesn't have a ton of time to spend on side projects, Lambda Cloud is a great value compared to upgrading my workstation at home. For $0.75/hour, the A10 instance has 24 GB VRAM, 30 vCPUs, and 200 GB of (ordinary) RAM. These specs are far beyond what I could reasonably install in my mini-ITX desktop. And with my usage of a few hours a week max, it would never be cheaper to buy the parts myself.

### Setup

I opted not to use the persistent filesystem.
This means I need to recreate my working environment on the cloud instance every time I start it up.
I set up some helper scripts to make this process easier.

Of course, I ignored the [classic xkcd](https://xkcd.com/1205/) about whether the time saved was worth it.
But I think it worked out in the end: I spent about 2-3 hours setting up these scripts, and it probably saves 5-10 minutes every time I start up the instance.

#### Setup: Ansible

Ansible is used to:

1. Install Miniconda, and create a Conda environment with the needed packages installed.
1. Copy a quick-start script, which starts Jupyter and tensorboard in separate tmux windows.

I run VS Code on my workstation (using code-server).
My VS Code connects to the Jupyter notebook server running on Lambda Cloud.
For the most part, this means that I don't need to continually copy the code over as I change it.
Running notebook cells "just works", including changes.
The only exception to this is when I reference other notebook files (with `%run`): those *do* need to be copied to the cloud instance.

#### Setup: SSH config file for port forwarding

I wrote an SSH config section to automate the port forwarding needed (for Jupyter and tensorboard).
The config is split into a "raw" and normal section.
`ssh lambdacloud-raw` skips the port forwarding stuff, which is useful to avoid noisy errors when we're SSH'ing to run Ansible or to copy files.
`ssh lambdacloud` does the port forwarding, and it's expected that we only run this one time.
This setup is pretty simple and mostly works, although I do unfortunately have to change the IP address every time I start a new instance.

#### Setup: copying code and datasets

I wrote a quick script with `rsync` to copy over code and datasets.
Fortunately, I do have good upload speeds with my home internet (1 Gbps symmetric).
As I use webdataset for the data involved, there are a small number of large files to copy.
This makes the copy fast, compared to copying a large number of small files (e.g. individual images or JSON files).