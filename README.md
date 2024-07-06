# pytorch-lightning-template

## File Structure
```
pytorch-lightning-template/
├── src/
│   ├── common/ (Common files.)
│   │   └── enums.py
│   │   └── logger.py
│   │
│   ├── data_module/ (Data module for PyTorch Lightning.)
│   │   └── mnist_data_module.py
│   │
│   ├── model/ (Model module for PyTorch Lightning.)
│   │   ├── mnist_model.py
│   │   └── model_base.py
│   │
│   ├── trainer/ (Trainer module for PyTorch Lightning.)
│   │   └── mnist_trainer.py
│   │
│   ├── training/ (Main for training.)
│   │   └── mnist_training.py
│   │
│   └── utils/ (Utility functions.)
│       └── model_utils.py
```

## Start
```
pip install .
python src/training/mnist_training.py
```
