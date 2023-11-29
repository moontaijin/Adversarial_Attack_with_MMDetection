# Adversarial_Attack_with_MMDetection
Generate Adversarial Attacked Image for Object Detection models using MMDetection

* You should put the files on */mmdetection/{here}*
  * This code use mmdetection 3.x version. DO NOT USE 2.x
* In this version, you can use only DAG method. I might update another methos later.
* **This method very sensitive to changes in pixel values**, so it stores the attacked image in [0,1] format rather than [0,255] format. For this purpose, the attacked images are stored in the form of a numpy array(.npy).

## Explain the pipeline
* This pipeline follows the following scheme
  1. Import a file with a video cropped to an image.
  2. Create a Pseudo Ground Truth(GT) of the file using Model Predcition.
  3. Create a Fake GT by randomly changing the label value of the model's Pseudo GT.
  4. Create an adversarial image using the fake GT.
  5. Save the image with Detection Visualization.
![image1](https://github.com/moontaijin/Adversarial_Attack_with_MMDetection/blob/main/upload_imgs/11.22.png)
![image2](https://github.com/moontaijin/Adversarial_Attack_with_MMDetection/blob/main/upload_imgs/11.22%20(2).png)
![image3](https://github.com/moontaijin/Adversarial_Attack_with_MMDetection/blob/main/upload_imgs/11.22%20(3).png)
![image4](https://github.com/moontaijin/Adversarial_Attack_with_MMDetection/blob/main/upload_imgs/11.22%20(4).png)
![image5](https://github.com/moontaijin/Adversarial_Attack_with_MMDetection/blob/main/upload_imgs/11.22%20(5).png)

# Sample Result
<p align="center">
<img src="https://github.com/moontaijin/Adversarial_Attack_with_MMDetection/blob/main/upload_imgs/sample_result.gif">
</p>
