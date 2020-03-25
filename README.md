# Deep learning-based total electron content (TEC) map completion
[![DOI](https://zenodo.org/badge/249016877.svg)](https://zenodo.org/badge/latestdoi/249016877)

A deep convolutional generative adversarial network with Poisson blending  (DCGAN-PB) for TEC map completion.

## Prerequisites

This project is run in Python 3 and mainly uses Tensorflow under the guidance of Anaconda to manage the environment. You can use platform other than anaconda at your convenience. The detailed anaconda package list can be found in the file [pkgs.yml](https://github.com/pancookie/DCGAN-PB/blob/master/pkgs.yml). 

## Database

The training data is included [here](https://drive.google.com/file/d/1qlC0I2kzw_iKSnKnkW2xg1AngNGa-OrK/view?usp=sharing). Please download and unzip to the preferred location. 
For the mask files, you can get from [link1](https://drive.google.com/file/d/1jIHkb1ZNYCKBy4FsAFu3jy6u2D7GTW7C/view?usp=sharing) and [link2](https://drive.google.com/file/d/1e3PQHhwvVMUPDYjzCfu2G_9kvy-4gT7M/view?usp=sharing). They are some examples of the mask files, you are welcome to make your own mask files.

## To run the project

After configuring the environment, put your parameters such as learning rate, epochs, etc in **train-dcgan.py**.
```
python train-dcgan.py
```
It will run the project.

After getting the trained model, to inpaint a single mask file.
```
python3 ./complete.py [path/to/mask/file] --inmaskfile [path/to/mask/file] --nIter 4000 --outDir [path/to/save/output] --lam 0.05
```
You can tune the parameters either in the command line or in **complete.py**. After which, the inpainted results would be saved to the output path.

## Authors
Y. Pan (yang.pan@mavs.uta.edu)

M.W. Jin

Y. Deng

S.R. Zhang

## References
[Taehoon Kim](https://github.com/carpedm20/DCGAN-tensorflow)

[Brandon Amos](https://github.com/bamos/dcgan-completion.tensorflow)

[Yeh (semantic inpainting)](https://github.com/moodoki/semantic_image_inpainting)
