# FDD-48: Towards Real-World Food Defect Detection with Fine-grained Annotations across Diverse Scenarios

This repository contains the official dataset and code for the paper "FDD-48: Towards Real-World Food Defect Detection with Fine-grained Annotations across Diverse Scenarios", under review in Proceedings of ACM Multimedia 2025 (MM'25) Dataset Track.

**Authors:** Ruihao Xu, Yong Liu, Yansong Tang* (* Corresponding author)
**Affiliation:** Tsinghua Shenzhen International Graduate School, Tsinghua University, Shenzhen, Guangdong, China

## Abstract

Food defect detection is a critical task in quality control for the food industry. Recent research primarily employs deep learning methods to improve detection accuracy, yet existing studies lack a unified evaluation benchmark and are constrained by data scarcity due to high annotation costs. To address these challenges, we first introduce FDD-48, a comprehensive food defect detection dataset with fine-grained annotations across 13 food types and 48 distinct defect categories, captured under diverse real-world conditions. Subsequently, we evaluate the performance of several mainstream general object detection models on FDD-48. To enhance detection performance under limited annotated data, we propose a semi-supervised food defect detection framework named FDDet. FDDet first introduces a novel data augmentation technique specifically designed for food defect detection, named BBoxMixUp, which breaks the erroneous associations between defect-irrelevant features and defect types by mixing regions corresponding to instances of the same defect category in input images. Additionally, FDDet incorporates a pseudo-label filtering mechanism called CGPC (Consistency-Guided Pseudo-Label Calibration), a strategy based on the assumption of label consistency among similar regions within individual samples. Experiments demonstrate that, compared to mainstream object detection models, our method significantly improves detection performance on the FDD-48 dataset, validating the effectiveness of the proposed approach in enhancing the accuracy and robustness of complex food defect detection tasks under limited data conditions.

## FDD-48 Dataset

FDD-48 is a novel benchmark dataset for food defect detection designed to address the limitations of existing datasets by providing diverse real-world scenarios, fine-grained defect categories, and instance-level annotations.

### Dataset Overview
* **Food Types:** 13 (apple, apricot, banana, cantaloupe, cherry, lychee, mango, orange, peach, pear, plum, strawberry, and watermelon)
* **Defect Types:** 48 distinct defect categories.
* **Real-World Scenarios:** Covers diverse scenarios including Planting (52%), Transportation (34%), and Consumption (14%).
* **Annotation Level:** Instance-level annotations (bounding boxes).
![](./README_image/dataset.jpg)

### Food Defect Categories

![](./README_image/datasetinfo.jpg)

### Data Statistics

* **Total Images:** 4,000. [cite: 72, 85]
    * **Labeled Images:** 1,503. [cite: 75, 85]
    * **Unlabeled Images:** 2,497. [cite: 37, 85]
* **Labeled Instances:** 15,913 fruit instances in the labeled data. [cite: 86]
* **Average Instances per Image:** Approximately 10.58 objects per image in the labeled data. [cite: 86]
* **Defective Fruit Images:** 39% of the labeled images include defective fruits. [cite: 87]
* **Data Split:** The labeled data is randomly split into training and testing sets in a 7:3 ratio. [cite: 90]
* **Balanced Distributions:** The dataset ensures balanced distributions of image sizes and bounding box sizes. [cite: 3]

### Data Acquisition and Annotation Workflow

The data acquisition and annotation pipeline involved several steps (Figure 2)[cite: 48]:
1.  **Web Crawling:** Initial collection of 50,000 raw images using "food category + defect type" as keywords. [cite: 68]
2.  **MLLM Discrimination (Noisy Data Filtering):** A multimodal large model (MiniCPM-V-2.6) was used to automatically assess and remove images not containing the target food type. [cite: 70]
3.  **Duplicate Removal:** DINOv2 was used to extract image features, and images with high similarity were removed. [cite: 71]
4.  **Manual Screening:** Resulted in 4,000 images with diverse scenes and balanced food categories. [cite: 72]
5.  **Pre-annotation (OVD Pre-annotation):** YOLO-World, an open-source object detector, was used with textual prompts (13 fruit names) to generate initial bounding box annotations for normal and defective fruits. The confidence threshold was lowered to improve detection of defective fruits. [cite: 76, 77, 79]
6.  **Post-processing:** An algorithm was implemented to remove redundant larger bounding boxes that encapsulated multiple smaller ones based on an intersection over area threshold of 95%. [cite: 82]
7.  **Manual Annotation:** Manual verification was conducted to annotate missed fruit instances and assign specific defect labels to all bounding boxes, ensuring high-quality annotations. [cite: 84]

## FDDet Method

FDDet is a food defect detection model tailored for real-world scenarios, particularly under limited annotated data conditions. [cite: 91, 170] It employs RTMDet as its baseline. [cite: 92]

### Key Components

1.  **BBoxMixUp:**
    * A novel data augmentation technique specifically designed for food defect detection. [cite: 8, 49]
    * It performs localized mixing exclusively within bounding boxes of the same defect category. [cite: 35]
    * **Goal:** To break erroneous associations between defect-irrelevant features (e.g., shape, color of the fruit) and defect types, enhance diversity in non-defect features, and mitigate overfitting due to data scarcity. [cite: 8, 36, 49, 101, 103]
    * **Process (Figure 3):**
        * **Mixing Candidate Selection:** For a target bounding box, a candidate bounding box of the same class is randomly selected from a different image. [cite: 94, 108, 109]
        * **Input-level Mixing (Localized Mixing):** The candidate region is resized to match the target region's dimensions. A new mixed region is created by a weighted combination (using a mixing ratio sampled from a Beta distribution) of the original target region and the resized candidate region. This mixed region replaces the original target region in the image. [cite: 96, 105, 113, 114, 115, 116]

2.  **Semi-Supervised Learning (SSL) Adaptation:**
    * Leverages the 2,497 unlabeled samples in FDD-48 to improve model robustness. [cite: 37]
    * Addresses training collapse issues encountered with common semi-supervised object detection frameworks (e.g., [14]) on FDD-48. [cite: 38, 121]
    * **Modifications to stabilize training and improve performance:**
        * **Buffer Weight Updates:** The teacher model's buffer weights (e.g., Batch Normalization parameters) are updated via Exponential Moving Average (EMA). This allows the teacher model to adapt to domain shifts between labeled and unlabeled data. [cite: 41, 124, 126, 130, 131]
        * **Pseudo-label Filtering Threshold:** A lower fixed threshold (e.g., 0.35) is used for pseudo-label filtering instead of conventional high-confidence thresholds (e.g., 0.9). This retains more potentially valuable pseudo-labels, especially given the high similarity between certain food defect types and inherently lower model confidence scores on FDD-48. [cite: 41, 127, 128, 129, 130, 132, 133]

3.  **CGPC (Consistency-Guided Pseudo-Label Calibration):**
    * A novel pseudo-label filtering and refinement strategy to improve the quality of pseudo-labels generated by the teacher model in the SSL framework. [cite: 9, 42, 46, 50, 97]
    * **Core Idea:** Enforces multi-dimensional consistency constraints to calibrate initial pseudo-labels (Figure 4). [cite: 46, 50, 111, 137]
    * **Consistency Dimensions:**
        * **Context-Semantic Consistency:** Assumes food instances within the same image typically belong to the same high-level food category. All pseudo-labels in a single image are unified to the most frequent food type identified among them. [cite: 43, 139, 140, 141]
        * **Visual-Semantic Consistency:** Ensures visually similar regions receive consistent labels. Region features are extracted using an external pretrained backbone (e.g., RegNet). For each pseudo-label, its "peers" (regions with feature similarity exceeding a threshold) are identified, and the label is replaced with the most frequent label among these peers. [cite: 44, 143, 144, 149, 150, 151, 153] Using an external backbone helps mitigate the detection model's overfitting and biases. [cite: 153]
        * **Spatial Consistency:** Removes spatially redundant pseudo-labels. An IOU-based method similar to Non-Maximum Suppression (NMS) is applied after context-semantic and visual-semantic corrections to eliminate duplicate detections for the same object instance. [cite: 45, 155, 156, 157]

## Main Contributions

* Introduction of **FDD-48**, a novel benchmark dataset for food defect detection featuring diverse real-world scenarios, fine-grained defect categories, and instance-level annotations. [cite: 47]
* Proposal of **BBoxMixUp**, a data augmentation technique tailored for food defect detection that performs localized mixing to enhance non-defect feature diversity and mitigate overfitting under data scarcity. [cite: 49]
* Enabling effective **semi-supervised learning (SSL)** on FDD-48 by adapting existing frameworks for stable training. [cite: 50]
* Introduction of **CGPC (Consistency-Guided Pseudo-Label Calibration)**, a novel method that refines pseudo-labels through multi-dimensional consistency constraints to further improve SSL performance. [cite: 50]
* Extensive experiments demonstrating the limitations of existing models on FDD-48 and validating the significant performance gains achieved by the proposed methods. [cite: 51]

## Citation

If you use FDD-48 or FDDet in your research, please cite the following paper:

```bibtex
@inproceedings{Xu2025FDD48,
  author    = {Ruihao Xu and Yong Liu and Yansong Tang},
  title     = {{FDD-48: Towards Real-World Food Defect Detection with Fine-grained Annotations across Diverse Scenarios}},
  booktitle = {Proceedings of ACM Multimedia 2025 (MM'25)},
  year      = {2025},
  url       = {[https://doi.org/XXXXXXX.XXXXXXX](https://doi.org/XXXXXXX.XXXXXXX)}
}