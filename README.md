# CRNet
This repository provides a simple implementation of our work entitled **CRNet: A Fast Continual Learning Framework with Random Theory** (TPAMI 2023)
https://ieeexplore.ieee.org/abstract/document/10086692

## Installation & Requirements
The current version of the codes has been tested with Python 3.7.13 on both Windows and Linux operating systems with the following versions of  requirements:
numpy==1.17.0
scipy==1.5.0
scikit-learn==0.23.1
torch==1.10.2
torchvision==0.11.3


## How To Use
Please create an environment to run the basic code: python CRNet_I_MR.py


##  Note
The default setting *performs Multiple Runs under five random task orderings*, as reported in our experiments.


##  Citation
```
@article{li2023CRNet,
  title={{CRNet}: A Fast Continual Learning Framework With Random Theory}, 
  author={Li, Depeng and Zeng, Zhigang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2023},
  note={Doi: 10.1109/TPAMI.2023.3262853},
  publisher={IEEE}
}
```


