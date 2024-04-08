# Music Genre Classification in Pytorch
The identification and understanding of sound are an important activity in  several challenges in the context of multimedia processing, since music is present in  people’s daily lives, there are countless musical genres, and each person has their  preference. 

# Problem Description
The main objective is to automatically classify musical genres from audio files. For this task, a dataset called GTZAN, known as the MINIST of audio files, was made available.

For this project, a convolutional neural network (CNN) will be used to predict musical style from a piece of music through images of audio files that have been converted into Mel spectrograms. To evaluate the model, balanced accuracy, precision, recall, and F1 score metrics will be used, as well as visualization of predictions and incorrect classifications of the model, such as confusion matrix.

# Project Description

Techniques Involved
- Image processing.
- Multi classification.

## Data set
The dataset consists of 1,000 audios of 30 seconds each. In total, there are 10 different styles: blues, classical music, country, disco, hip hop, jazz, metal, pop, reggae, and rock, with 100 audios for each class.
The dataset has three types of audio representations:
- Original Audios: a collection of 10 genres with 100 audio files each, all with a length of 30 seconds.
- Images: a visual representation for each 30s audio file.
- Feature files: these files contain features from the audio files for 30s and 3s.

For this project, only the image dataset will be considered.

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

![image](https://github.com/Debsat/--Music-Genre-Classification-in-Pytorch/assets/114963575/68178cd7-19c4-4b0b-873c-af3a27e704cb)

It is possible to notice that the dataset is balanced.
Only one `jazz` class file is missing, but it does not compromise class balance.

![image](https://github.com/Debsat/--Music-Genre-Classification-in-Pytorch/assets/114963575/dcaad424-c66e-448a-9488-828b0376d4c3)

## References

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

https://discuss.pytorch.org/t/how-to-do-a-stratified-split/62290

https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7

https://rumn.medium.com/part-1-ultimate-guide-to-fine-tuning-in-pytorch-pre-trained-model-and-its-configuration-8990194b71e

https://github.com/huggingface/pytorch-image-models/tree/main/timm/models
