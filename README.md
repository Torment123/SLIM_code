# SLIM_code

**Implementation of the paper:" SLIM: Skip-Layer Inception Modulation for Image Synthesis"**

# FastGAN-SLIM

**0.Data**

The few-shot learning datasets used in the paper can be found at <https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view>

**1.How to run**

For 256 x 256 resolution datasets, enter the folder 'FastGAN-SLIM-256', and call:
```bash
python train.py --path /path/to/RGB-image-folder  --im_size 256
```
For 1024 x 1024 resolution datasets, enter the folder 'FastGAN-SLIM-1024', and call:
```bash
python train.py --path /path/to/RGB-image-folder  --im_size 1024
```

Once finish training, for evaluation, you can generate a certain number of images by:
```bash
cd ./train_results/num_of_your_training/
python eval.py --n_sample 5000  --im_size your_im_size
```
and calculate the FID score using the fid.py file in the sub-folder benchmarking

# SLIM-SinGAN

**0.Install dependencies**
Enter the folder 'SLIM-SinGAN', and run
```bash
python -m pip install -r requirements.txt
```
**1.Data**
The images used in this paper can be found at <https://drive.google.com/drive/folders/17kxp715a875K3Qb5rSb9M1ShUcduAeZ-?usp=sharing>

**2.Train**
To train the proposed SLIM-SinGAN on a image, put the desired image file under Input/Images







