# DANet
  This is basic code of [CSGSA-Net: Canonical-Structured Graph Sparse Attention Network for Fetal ECG Estimation] (https://www.sciencedirect.com/science/article/pii/S1746809422010102)
  
  If you need any help for the code and data, do not hesitate to leave issues in this repository.
****
## Citation
 
```
 @article{CS2023Wang,
  title={CSGSA-Net: Canonical-Structured Graph Sparse Attention Network for Fetal ECG Estimation},
  author={Xu Wang, Yang Han and Yamei Deng},
  journal={Biomedical signal processing and control},
  pages={1--10},
  volume={82},
  year={2023. Art. no. 104556}
}

```
## Method
### Training
```

python train.py

```

### Testing

```

python test.py

```

## Notes

```

You can adjust the network by the following:
1. adjust the "nhid" of GCN in gpa.py;
2. add the layer of backbone in gpa.py;
3. adjust the learning rate.

```
