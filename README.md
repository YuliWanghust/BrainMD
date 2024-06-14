# Enhancing vision-language models for medical imaging: bridging the 3D gap with innovative slice selection

Code for paper [Enhancing vision-language models for medical imaging: bridging the 3D gap with innovative slice selection]()

<img src="Pics/framework.png" align="middle" width="75%">

Recent approaches to vision-language tasks are built on the remarkable capabilities of large vision-language models (VLMs). These models excel in zero-shot and few-shot learning, enabling them to learn new tasks without parameter updates. However, their primary challenge lies in their design, which primarily accommodates 2D input, thus limiting their effectiveness for medical images, particularly radiological images like MRI and CT, which are typically 3D. To bridge the gap between state-of-the-art 2D VLMs and 3D medical image data, we developed an innovative, one-pass, unsupervised representative slice selection method called Vote-MI, which selects representative 2D slices from 3D medical imaging. To evaluate the effectiveness of Vote-MI when implemented with VLMs, we introduce BrainMD, a robust, multimodal dataset comprising 2,453 annotated 3D MRI brain scans with corresponding textual radiology reports and electronic health records. Based on BrainMD, we further develop two benchmarks, BrainMD-select (including the most representative 2D slice of a 3D image) and BrainBench (including various vision-language downstream tasks). Extensive experiments on the BrainMD dataset and its two corresponding benchmarks demonstrate that our representative selection method significantly improves performance in zero-shot and few-shot learning tasks. On average, Vote-MI achieves a 14.6\% and 16.6\% absolute gain for zero-shot and few-shot learning, respectively, compared to randomly selecting examples. Our studies represent a significant step toward integrating AI in medical imaging to enhance patient care and facilitate medical research. We hope this work will serve as a foundation for data selection as vision-language models are increasingly applied to new tasks.

## Data examples
For those unfamiliar with brain MRI data, please refer to the following example. It includes images of a brain MRI and the corresponding selections of representative 2D slices in axial, sagittal, and coronal views, as chosen by a radiologist.

![Visualizations of the BrainMD dataset.](Pics/figure_example.png)

## Dataset

The BrainMD dataset is available [here](). It consists of the following data splits:

### Data Structure

| Type                      | No. (cases) | Format     | Access Link |
| --------------------------| ------------| ---------- | ------------|
| BrainMD (3D MRI)          | 2453        | DICOM      | [link]()    |
| BrainMD-select (2D MRI)   | 2453        | NIFTI      | [link]()    |
| BrainBench (text)         | 2453        | CSV/JSON   | [link]()    |

For each data type, the dataset includes the following files:

**Raw_data and Synthesized_data**:

1. pulse_xxxxx_volume.nii.gz  % original volume
2. pulse_xxxxx_centerline.nii.gz  % extracted centerline
3. pulse_xxxxx_graph.graphml  % graph with edge information

## Dependencies
To establish the environment, run this code in the shell:
```
conda env create -f vote_MI.yml
conda activate vote_MI
pip install -e .
```
That will create the environment selective_annotation we used.

## Usage

### Environment setup

Activate the environment by running
```
conda activate vote_MI
```

### End-to-end pipeline: selection, inference, evaluation
GPT-J as the in-context learning model, DBpedia as the task, and vote-k as the selective annotation method (1 GPU, 40GB memory)
```
python main.py --task_name dbpedia_14 --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs
```

## Citation
If you find our work helpful, please cite us
```
