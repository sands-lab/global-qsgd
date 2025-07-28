# Transformer-XL For PyTorch

This repo is modified from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL

## Dataset
User should prepare the wikitext103 dataset by themselves, and put to the `pytorch/wikitext-103/` folder.

## Launch
```shell
cd pytorch
bash launch.sh
```

## Citation

```bibtex
# TransformerXL Model (ACL'19)
@misc{dai2019transformerxlattentivelanguagemodels,
      title={Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context}, 
      author={Zihang Dai and Zhilin Yang and Yiming Yang and Jaime Carbonell and Quoc V. Le and Ruslan Salakhutdinov},
      year={2019},
      eprint={1901.02860},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1901.02860}, 
}
```