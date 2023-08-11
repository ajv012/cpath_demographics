# Examining Demographic Bias in Misdiagnosis by AI-Driven Computational Pathology Models
[HTML](https://arxiv.org/abs/2304.06819)

**Summary:**   With an increasing number of regulatory approvals, computational pathology (CPATH) systems are poised to significantly impact diagnostic clinical pathology practice. Concurrently, there has been an increased focus on the role of demographic factors such as race in medicine, coupled with machine learning research that has repeatedly shown that deep learning models can exhibit unanticipated biases. Given the CPATH community's common paradigm of training models on large, publicly-available datasets such as the Cancer Genome Atlas (TCGA), there is an urgent need to investigate the possibility of bias for such models, especially since these public consortia severely under-represent certain demographic groups. We therefore conduct a systematic investigation into demographic disparities in performance using common multiple instance learning methods training on TCGA and testing on independent cohorts curated from Mass General Brigham Hospital. We find racial disparities in misdiagnosis rates are significant for breast and lung cancer subtyping tasks. For instance, differences in the area under the receiver operating characteristic curve (AUC) were identified between Black and White patients in breast (Black: 0.812, White: 0.902) and lung subtyping (Black: 0.743, White: 0.920). Even with the introduction of state-of-the-art modeling choices and commonly employed bias mitigation strategies, we observe strong performance disparities persist between different demographic subgroups. We also identify a trade-off between model performance and unfairness, where higher AUC is often accompanied by an increased recall gap between White and Black patients. Finally, we demonstrate that our findings extend to other demographic factors, such as zip code-inferred income. Given these findings, we encourage regulatory and policy agencies to integrate demographic-stratified evaluation into their testing guidelines.

<img width="973" alt="Screen Shot 2023-08-11 at 2 10 21 PM" src="https://github.com/ajv012/cpath_demographics/assets/55669017/e1f0c37e-7201-42c2-8b54-d94d7939fa1e">

## Installation Guide for Linux (using anaconda)
### Pre-requisities: 
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3090 Ti x 3) with CUDA 11.0 and cuDNN 7.5
- Python (3.8.13), h5py (3.8.0), matplotlib (3.7.1), numpy (1.24.2), opencv-python (4.5.3.56), openslide-python (1.2.0), pandas (2.0.0), pillow (9.3.0), Pytorch (version 2.0.0), CUDA (version 11.7), scikit-learn (1.2.2), scipy (1.9.1), torchvision (0.15.1), and timm (version 0.9.2). 

### Downloading TCGA Data
To download diagnostic WSIs (formatted as .svs files), molecular feature data and other clinical metadata, please refer  to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov)and the [cBioPortal](https://www.cbioportal.org/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/). 

## Processing Whole Slide Images 
To process Whole Slide Images (WSIs), first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. The 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. We use two different feature encoders: $\text{ResNet50}$ trained on ImageNet and $\text{Swin-T}$ transformer trained on histology slides. The feature encodings are stored as .pt files. We also extract Macenko stain normalized features for both of this feature extractors.  

## Measuring Fairness
We use three metrics to measure the fairness of AI models. First, we compare race-stratified area under the receiver operating characteristic curve (AUC). Second, we use the race-stratified macro-average F1 score, which is more robust to data imbalances. Finally, under the Equalized Opportunity framework, we define True Positive Rate (TPR) disparity, which measure how well does the recall for a demographic subgroup conditioned on the subtype compares to the overall population's recall. You can find the code to calculate these metrics in the directory `analysis`. 

## Training-Validation Splits on TCGA 
We experiment with two data-splitting strategies on TCGA. First, following common study designs in computational pathology, we use a 20 fold Monte Carlo splits (found in `splits_MonteCarlo` directory). Secondly, we use 10 fold site-preserving splits, where cases from one tissue contirbuting site in TCGA are not split over training and validation (found in `splits_siteStratified` directory).  

## Exploring modeling techniques
We experiment with various modeling choices for all components of the typical computational pathology pipeline:
1. `Patch feature extractor:` We use $\text{ResNet50}$ trained on ImageNet and $\text{Swin-T}$ transformer trained on histology slides
2. `Patch aggregators:` We use ABMIL, CLAM, and TransMIL, which make different assumptions about the relations between patches 
3. `Bias mitigation strategies:` We investigate Importance Weighting and Adversarial Regularization

You can find these implementations in the `models` directory. 

## Running Experiments 
Refer to `commands` folder for source files to create patches, extract features, train, and test models. Hyper-parameters used to train these models are directly taken from their respective papers. We cannot make the in-house private datasets public at the moment as they contain protected patient information. 

## Issues 
- Please open new threads or report issues directly (for urgent blockers) to `avaidya@mit.edu`.
- Immediate response to minor issues may not be available.

## License and Usage 
[Mahmood Lab](https://faisal.ai) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
