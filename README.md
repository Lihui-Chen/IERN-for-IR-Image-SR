# IRSR

# 0. Introduction

Codes for paper: Chen, L., Tang, R., Anisetti, M., & Yang, X. (2020). A Lightweight Iterative Error Reconstruction Network for Infrared Image Super-Resolution in Smart Grid. *Sustainable Cities and Society*, 102520.

```latex
@article{chen2020lightweight,
  title={A Lightweight Iterative Error Reconstruction Network for Infrared Image Super-Resolution in Smart Grid},
  author={Chen, Lihui and Tang, Rui and Anisetti, Marco and Yang, Xiaomin},
  journal={Sustainable Cities and Society},
  pages={102520},
  year={2020},
  publisher={Elsevier}
}
```

# 1. Requirements

1. python3

2. tqdm
3. opencv-python
4. pytorch(>=1.6)
5. torchvision
6. yaml

# 2.  Test 

1. Clone this repository:

   ```bash
   git clone https://github.com/Huises/IERN-for-IR-Image-SR.git
   ```

2. Then, cd to **IERN-for-IR-Image-SR** and run **the commands** for evaluation on *GIR50 and Infrared20*   (or your own images) :

   ```bash
   python test.py -opt options/test/test.yml #test GIR50 and Infrared20
   python test.py -opt options/test/test.yml -lr_path your_img_path # test your own images
   ```

3. Finally, you can find the reconstruction images in `./results`.



# 3. Train



1. Prepare train set and validation set use **./scripts/Prepare_TrainData_HR_LR.m** or **./scripts/Prepare_TrainData_HR_LR.py**

2. Clone this repository:

   ```bash
   git clone https://github.com/Huises/IERN-for-IR-Image-SR.git
   ```

3. Open **IERN-for-IR-Image-SR/options/train/train.yml**. Then, modify image paths for  train and validation set 

4. Then，cd to **IERN-for-IR-Image-SR** and run **the commands** for evaluation 

   ```bash
   python train.py -opt options/train/train.yml  # train your own models
   ```



# Acknowledgements

​	Thank [Paper99](https://github.com/Paper99). Our code structure is derived from his repository [SRFBN](https://github.com/Paper99/SRFBN_CVPR19).

