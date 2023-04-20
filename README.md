# CSGSA-Net
  This is basic code of [CSGSA-Net: Canonical-Structured Graph Sparse Attention Network for Fetal ECG Estimation](https://www.sciencedirect.com/science/article/pii/S1746809422010102)
  
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

## Notes

```
This project tends to share the basic code of our idea, and the test code can not share due to the copyright. But it can be implemented from the references cited in the available paper. In addition, the other dataset, named B2DB dataset, is used from "Fetal electrocardiograms, direct and abdominal with reference heartbeat annotations". Moreover, if you want to improve the performance of the network, the following methods are provided:

1. Adjust the A"nhid" of GCN in gpa.py;
2. Add the layer of backbone in gpa.py;
3. Update the learning rate.

```
