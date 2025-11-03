                                                CCAP Dataset Description

This dataset is systematically organized and standardized, with the top-level directory divided into two main parts: expert-data and data, which are used respectively for expert validation and model training/evaluation.

The data section contains 1,500 coronary angiography images and their corresponding segmentation masks. All original images are derived from the ARCADE dataset and were obtained through extraction and standardized preprocessing of the original DICOM files. The data are split into three subsets with a ratio of 7:1:2, including training (1,050 cases), validation (150 cases), and testing (300 cases).

Each subset consists of two subfolders: images and annotations.
The images folder stores coronary angiography images in .png format.
The annotations folder contains the corresponding segmentation masks, with filenames kept identical to ensure accurate indexing and traceability.

The expert-data section contains 100 cases randomly selected from the data subset for expert-level annotation and model performance validation. Its internal structure mirrors that of the data directory, comprising images and annotations subfolders. Both images and annotation masks are stored in .png format and follow the same one-to-one naming convention.

![image text](https://github.com/ruining-gi/image/blob/aa18be81063933081a647ccc9269e8c359520387/all.jpg)
