COMP5331 HKUST 2022 Fall
===

This is the course project of COMP5331 where we try to modify and improve the existing source codes of the paper “Cross-Domain Recommendation to Cold-Start Users
via Variational Information Bottleneck” accepted in ICDE 2022 by Jiangxia Cao, Jiawei Sheng, Xin Cong, Tingwen Liu and Bin Wang.

**In this branch (GCL)**, we add graph contrastive learning into the GNN of CDRIB. The trained models and training logs can be found at this [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ssongad_connect_ust_hk/En34Pt0xmwBGoP_4LNWDwUYBrgVW4G2OoVOb5IUBnGjhwg?e=5LH6ve) link.

```
@inproceedings{cao2022cdrib,
  title={Cross-Domain Recommendation to Cold-Start Users via Variational Information Bottleneck},
  author={Cao, Jiangxia and Sheng, Jiawei and Cong, Xin and Liu, Tingwen and Wang, Bin},
  booktitle={IEEE International Conference on Data Engineering (ICDE)},
  year={2022}
}
```

Requirements
---

Python=3.7.9

PyTorch=1.6.0

Scipy = 1.5.2

Numpy = 1.19.1

Usage
---

```shell
CUDA_VISIBLE_DEVICES=0 python -u train_rec.py --id gv --dataset game_video --remove_rate 0.1
```


