---
language:
- en
license: apache-2.0
task_categories:
- text-generation
- image-to-text
dataset_info:
  features:
  - name: file_name
    dtype: string
  - name: bbox
    sequence: float64
  - name: instruction
    dtype: string
  - name: data_type
    dtype: string
  - name: data_source
    dtype: string
  - name: image
    dtype: image
  splits:
  - name: test
    num_bytes: 1104449470.928
    num_examples: 1272
  download_size: 602316816
  dataset_size: 1104449470.928
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
# Dataset Card for ScreenSpot

GUI Grounding Benchmark: ScreenSpot. 

Created researchers at Nanjing University and Shanghai AI Laboratory for evaluating large multimodal models (LMMs) on GUI grounding tasks on screens given a text-based instruction.

## Dataset Details

### Dataset Description

ScreenSpot is an evaluation benchmark for GUI grounding, comprising over 1200 instructions from iOS, Android, macOS, Windows and Web environments, along with annotated element types (Text or Icon/Widget). 
See details and more examples in the paper.

- **Curated by:** NJU, Shanghai AI Lab
- **Language(s) (NLP):** EN
- **License:** Apache 2.0

### Dataset Sources

- **Repository:** [GitHub](https://github.com/njucckevin/SeeClick)
- **Paper:**  [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/abs/2401.10935)

## Uses

This dataset is a benchmarking dataset. It is not used for training. It is used to zero-shot evaluate a multimodal model's ability to locally ground on screens. 

## Dataset Structure

Each test sample contains:
- `image`: Raw pixels of the screenshot
- `file_name`: the interface screenshot filename
- `instruction`: human instruction to prompt localization
- `bbox`: the bounding box of the target element corresponding to instruction. While the original dataset had this in the form of a 4-tuple of (top-left x, top-left y, width, height), we first transform this to (top-left x, top-left y, bottom-right x, bottom-right y) for compatibility with other datasets.
- `data_type`: "icon"/"text", indicates the type of the target element
- `data_souce`: interface platform, including iOS, Android, macOS, Windows and Web (Gitlab, Shop, Forum and Tool)

## Dataset Creation

### Curation Rationale

This dataset was created to benchmark multimodal models on screens. 
Specifically, to assess a model's ability to translate text into a local reference within the image. 

### Source Data

Screenshot data spanning dekstop screens (Windows, macOS), mobile screens (iPhone, iPad, Android), and web screens. 

#### Data Collection and Processing

Sceenshots were selected by annotators based on their typical daily usage of their device.
After collecting a screen, annotators would provide annotations for important clickable regions. 
Finally, annotators then write an instruction to prompt a model to interact with a particular annotated element.

#### Who are the source data producers?

PhD and Master students in Comptuer Science at NJU. 
All are proficient in the usage of both mobile and desktop devices. 

## Citation

**BibTeX:**

```
@misc{cheng2024seeclick,
      title={SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents}, 
      author={Kanzhi Cheng and Qiushi Sun and Yougang Chu and Fangzhi Xu and Yantao Li and Jianbing Zhang and Zhiyong Wu},
      year={2024},
      eprint={2401.10935},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```