CDRIB
===

The source code is for the term project in course 5331.

This branch is about approach of Meta Information Encoding, mainly contributed by CHAN Tsz Ho, WANG Zihao(in alphabet).

In this branch, we mainly add a embedding after the item index embedding in the model file. To acomplish the embedding, we also modified the data preprocessing files.


Requirements
---

Python=3.7.9

PyTorch=1.6.0

Scipy = 1.5.2

Numpy = 1.19.1

Spacy and python -m spacy download en_core_web_sm

Usage
---

To run this project, please make sure that you have the above packages being downloaded. Our experiments are conducted on a PC with an Intel Xeon E5 2.1GHz CPU, 256 RAM and a 3090 24GB GPU. 

Running example:

```shell
CUDA_VISIBLE_DEVICES=1 python -u train_rec.py --id gv --dataset game_video --model CDRIB --GNN 3 --beta 0.5 --source_item_text_file ../game_video_index2Title.csv --target_item_text_file ../video_game_index2Title.csv
```


