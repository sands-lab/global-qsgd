# DeepLight

This repo is modified from https://github.com/sands-lab/omnireduce-experiments/tree/master/models/DeepLight.

## Dataset

The folder dataset has a tiny dataset (several batches) from [Criteo's 1TB Click Prediction Dataset](https://docs.microsoft.com/en-us/archive/blogs/machinelearning/now-available-on-azure-ml-criteos-1tb-click-prediction-dataset) for evaluation.

## Launch
```shell
mkdir logs
bash launch.sh
```

## Citation

```bibtex
# DeepLight Model (WSDM'21)
@inproceedings{deeplight,
  title={DeepLight: Deep Lightweight Feature Interactions for Accelerating CTR Predictions in Ad Serving},
  author={Wei Deng and Junwei Pan and Tian Zhou and Deguang Kong and Aaron Flores and Guang Lin},
  booktitle={International Conference on Web Search and Data Mining (WSDM'21)},
  year={2021}
}
```