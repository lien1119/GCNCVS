# GCNCVS
We implemented our study by modifing the code provided by DSNet and PGL-SUM.
We thank DSNet and PGL-SUM for their contributions.
We will publish more information soon.

# Dataset and pretrained model download link
Dataset:
https://drive.google.com/drive/folders/1ViwXUaPcmuQxI97lHkySP-ozIk-6zegO?usp=drive_link

pretrained model:
https://drive.google.com/drive/folders/1yoTxKSvrdNaOGoCrJXYWQUvL0CTL6evJ?usp=sharing
# Inference

```sh
python evaluate.py --model-dir model/ --splits splits/summe.yml splits/tvsum.yml
```