# SLIM_code

**Implementation of the paper:" SLIM: Skip-Layer Inception Modulation for Image Synthesis"**

# FastGAN-SLIM

The instructions is based on the repo of the fastgan paper [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch), please also refer to their instructions

The file with the SLIM module definition is at models.py, the function name is 'InceptionBlock'

**0.Data**

The few-shot learning datasets used in the paper can be found at <https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view>

**1.How to run**

For 256 x 256 resolution datasets, enter the folder 'FastGAN-SLIM-256', and call:
```bash
python train.py --path /path/to/RGB-image-folder  --im_size 256
```

![Obama](https://i.ibb.co/VS3ycB9/30000.jpg)

Figure1. Synthesized 256x256 images of the proposed FastGAN-SLIM

For 1024 x 1024 resolution datasets, enter the folder 'FastGAN-SLIM-1024', and call:
```bash
python train.py --path /path/to/RGB-image-folder  --im_size 1024
```
![Skulls](https://i.ibb.co/DpvyK1w/10000.jpg)

Figure2. Synthesized 1024x1024 images of the proposed FastGAN-SLIM


Once finish training, for evaluation, you can generate a certain number of images by:
```bash
cd ./train_results/num_of_your_training/
python eval.py --n_sample 5000  --im_size your_im_size
```
and calculate the FID score using the fid.py file in the sub-folder benchmarking

# SLIM-SinGAN

The instructions are based on the repo of SinGAN paper: [SinGAN](https://github.com/tamarott/SinGAN), pleae also refer to their instructions

The file with SLIM module definition is at SLIM-SinGAN/SinGAN/models.py, the function name is 'InceptionBlock' 

**0.Install dependencies**
Enter the folder 'SLIM-SinGAN', and run
```bash
python -m pip install -r requirements.txt
```
**1.Data**
The images used in this paper can be found at <https://drive.google.com/drive/folders/17kxp715a875K3Qb5rSb9M1ShUcduAeZ-?usp=sharing>

**2.Train**
To train the proposed SLIM-SinGAN on a image, put the desired image file under Input/Images,
and run
```bash
python main_train.py --input_name <input_file_name>
```
This will also use the resulting trained model to generate random samples starting from the coarsest scale (n=0)

**3.Random samples**
To generate random samples from any starting generation scale, please first train the proposed SLIM-SinGAN model on the desired image (as described above), then run
```bash
python random_samples.py --input_name <training_image_file_name> --mode random_samples --gen_start_scale <generation start scale number>
```
![SLIM-SinGAN-uncontional](https://i.ibb.co/82nMCrV/temp-eight.png=300x150)

Figure3. Example synthesized images of SLIM-SinGAN for unconditional image generation task

**4.Harmonization**
To harmonize a pasted object into an image, please first train SinGAN model on the desired background image (as described above), then save the naively pasted reference image and it's binary mask under "Input/Harmonization" (see the images in the data link for an example). Run the command
```bash
python harmonization.py --input_name <training_image_file_name> --ref_name <naively_pasted_reference_image_file_name> --harmonization_start_scale <scale to inject>
```
![SLIM-SinGAN-harmonization](https://i.ibb.co/xqQ03tp/SLIM-Sin-GAN-teaser-harmonization.png)

Figure4. Example harmonizaed images of SLIM-SinGAN


**5.Editing**
To edit an image, please first train SinGAN model on the desired non-edited image (as described above), then save the naive edit as a reference image under "Input/Editing" with a corresponding binary map (see the images in the data link for an example). Run the command
```bash
python editing.py --input_name <training_image_file_name> --ref_name <edited_image_file_name> --editing_start_scale <scale to inject>
```
![SLIM-SinGAN-editing](https://i.ibb.co/dtZND3k/SLIM-Sin-GAN-teaser-editing.png)

Figure5. Example Edited images of SLIM-SinGAN

# Additional Functions
**Single Image FID**
To calculate the SIFID between real images and their corresponding fake samples, please run:
```bash
python SIFID/sifid_score.py --path2real <real images path> --path2fake <fake images path> 
```
Make sure that each of the fake images file name is identical to its corresponding real image file name.







