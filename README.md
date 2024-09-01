## MLCKD
Official pytorch implementation of []()

## Requirements
- Python3
- PyTorch (> 1.2.0)
- torchvision
- numpy


## Training
Run ```main.py``` with student network as WRN-16-2 and teacher as WRN-40-2 to reproduce experiment result on CIFAR100.
```
python main.py  --data CIFAR100 --trained_dir /trained/wrn40x2/model.pth --trail 1\
 --model wrn16x2 --model_t wrn40x2 --alpha 3 --beta 1 --gamma 3 --lr 0.05
```

## License

# CMKD
