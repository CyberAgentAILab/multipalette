# Color recommendation for vector graphic documents 

This repository is the official implementation of Color Recommendation for Vector Graphic Documents based on Multi-Palette Representation

Paper | System video

![Overview_image](docs/overview.png)

### Prerequisites

- Python:3.8
- Poetry: 1.1.*

### Setup

Install requirements and run jupyter.

```
poetry install
poetry run jupyter lab
```

### Quick demo

`notebooks/recomm_colors.ipynb`: recommend colors for multiple palettes in a design
- Trained model of color prediction are in trained_models/.
- Json files for test are pre-created in data/model_test_input/crello_samples/.

You can train a color model on a notebook `notebooks/train_model.ipynb`. We recommended GPU resources to train this model (e.g. Tesla T4 * 1).
You can also create a json file for test from crello dataset on a notebook `notebooks/create_json_file.ipynb`.

### Data

`data/training_data/metadata_colors`: extracted color palettes for Image-SVG-Text elements from [Crello-dataset-v1](https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip) ([the lastest Crello-dataset](https://github.com/CyberAgentAILab/canvas-vae/blob/main/docs/crello-dataset.md))

`data/training_data/data_bert/data_color`: color corpus of train, validation, and test dataset, and color vocabulary from train dataset

`data/trained_models`: trained model for color recommendation

`model_test_input`: json sample files for testing the results of color recommendation