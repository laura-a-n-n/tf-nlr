# tf-nlr
**An unofficial implementation of Neural Lumigraph Rendering in TensorFlow 2.** <br />
<sub>Undergraduate project by Laura Ann Perkins, [New College of Florida](https://ncf.edu/).</sub>

<p align='center'>
  <img src='img/M2.gif?raw=true' alt='GIF rendered with NLR.' />
</p>

This project is based on the 2021 paper [Neural Lumigraph Rendering](http://www.computationalimaging.org/publications/nlr/). I first started working on this project back in July 2021, before the code for [MetaNLR++](https://github.com/alexanderbergman7/metanlrpp) had been released. This is the first open-source implementation of their original paper.

## Requirements

 - This project requires Python 3.6+ with TensorFlow 2.5.0 (among other dependencies). 
 - For the [SIREN](https://www.vincentsitzmann.com/siren/) models, it builds upon [tf_siren](https://github.com/titu1994/tf_SIREN).
 - To run the Jupyter notebook, download the free [NLR dataset](https://drive.google.com/file/d/1BBpIfrqwZNYmG1TiFljlCnwsmL2OUxNT/view?usp=sharing) and place `nlr_dataset` in a folder named `data` in the root directory.

## Usage

To get started with `tf-nlr`, check out [the Jupyter notebook](tf-nlr.ipynb). There you can see how to:

 - load a pre-trained model.
 - train your own model.
 - render fitted views.
 - render novel views.

You can train a model from scratch with `train.py`.

```
python train.py --dataset_path path/to/data/folder --img_ratio scale_ratio --out_folder path/to/output/folder
```

To evaluate a fitted model, run `test.py`.

```
python test.py --model path/to/fitted/model --dataset_path path/to/data/folder --img_ratio scale_ratio
```

Note that `img_ratio` divides the image dimensions. So if you enter `img_ratio=5`, the images will be five times smaller.

For more options, try running either script with the help flag, e.g.  `python train.py -h`, or edit the [config file](conf/config.py).

## Pretrained models
The `h5` directory contains two trained models. The folder `pretrain` is a training initialization; the neural SDF is initialized to a radius of 0.5. The folder `L1` contains a trained model for the scene titled L1 in the NLR dataset.
