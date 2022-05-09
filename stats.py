#
import fire

#
import torch
import torch.nn as nn
#
import numpy as np

#
from tqdm import tqdm

#
import models
from config import DefualtConfig

#
from fvcore.nn import FlopCountAnalysis, flop_count_table

###################################################################################

config = DefualtConfig()
device = torch.device(f'cuda:{config.use_gpu_index}' if torch.cuda.is_available(
) else'cpu') if config.use_gpu_index != -1 else torch.device('cpu')


def main():

    model = getattr(models, config.model_name)(config)
    model.to(device)

    flops = FlopCountAnalysis(model, torch.zeros((1, 3, 224, 224)).to(device))
    print(f'FLOPs : {flops.total()}')
    print(f'Operators : \n{flops.by_operator()}')
    print(flop_count_table(flops))


if __name__ == '__main__':
    main()


'''
Unsupported operator aten::max_pool2d encountered 4 time(s)
Unsupported operator aten::affine_grid_generator encountered 1 time(s)
Unsupported operator aten::add encountered 25 time(s)
Unsupported operator aten::div encountered 12 time(s)
Unsupported operator aten::softmax encountered 12 time(s)
Unsupported operator aten::gelu encountered 12 time(s)
The following submodules of the model were never called during the trace of the graph. They may be unused, or they were accessed by direct calls to .forward() or via other python methods. In the latter case they will have zeros for statistics, though their statistics will still contribute to their parent calling module.
vit.pooler, vit.pooler.activation, vit.pooler.dense
FLOPs : 20626653916
Operators :
Counter({'linear': 16732372572, 'conv': 3159439744, 'matmul': 715327488, 'layer_norm': 18912000, 'grid_sampler': 602112})
| module                                        | #parameters or shape   | #flops    |
|:----------------------------------------------|:-----------------------|:----------|
| model                                         | 86.791M                | 20.627G   |
|  localization                                 |  0.156M                |  3.044G   |
|   localization.0                              |   4.736K               |   0.224G  |
|    localization.0.weight                      |    (32, 3, 7, 7)       |           |
|    localization.0.bias                        |    (32,)               |           |
|   localization.1                              |   50.208K              |   2.255G  |
|    localization.1.weight                      |    (32, 32, 7, 7)      |           |
|    localization.1.bias                        |    (32,)               |           |
|   localization.4                              |   50.208K              |   0.502G  |
|    localization.4.weight                      |    (32, 32, 7, 7)      |           |
|    localization.4.bias                        |    (32,)               |           |
|   localization.7                              |   25.632K              |   54.17M  |
|    localization.7.weight                      |    (32, 32, 5, 5)      |           |
|    localization.7.bias                        |    (32,)               |           |
|   localization.10                             |   25.632K              |   9.242M  |
|    localization.10.weight                     |    (32, 32, 5, 5)      |           |
|    localization.10.bias                       |    (32,)               |           |
|  fc_loc                                       |  0.234M                |  0.234M   |
|   fc_loc.0                                    |   0.233M               |   0.233M  |
|    fc_loc.0.weight                            |    (90, 2592)          |           |
|    fc_loc.0.bias                              |    (90,)               |           |
|   fc_loc.2                                    |   0.546K               |   0.54K   |
|    fc_loc.2.weight                            |    (6, 90)             |           |
|    fc_loc.2.bias                              |    (6,)                |           |
|  vit                                          |  86.389M               |  17.582G  |
|   vit.embeddings                              |   0.743M               |   0.116G  |
|    vit.embeddings.cls_token                   |    (1, 1, 768)         |           |
|    vit.embeddings.position_embeddings         |    (1, 197, 768)       |           |
|    vit.embeddings.patch_embeddings.projection |    0.591M              |    0.116G |
|   vit.encoder.layer                           |   85.054M              |   17.466G |
|    vit.encoder.layer.0                        |    7.088M              |    1.455G |
|    vit.encoder.layer.1                        |    7.088M              |    1.455G |
|    vit.encoder.layer.2                        |    7.088M              |    1.455G |
|    vit.encoder.layer.3                        |    7.088M              |    1.455G |
|    vit.encoder.layer.4                        |    7.088M              |    1.455G |
|    vit.encoder.layer.5                        |    7.088M              |    1.455G |
|    vit.encoder.layer.6                        |    7.088M              |    1.455G |
|    vit.encoder.layer.7                        |    7.088M              |    1.455G |
|    vit.encoder.layer.8                        |    7.088M              |    1.455G |
|    vit.encoder.layer.9                        |    7.088M              |    1.455G |
|    vit.encoder.layer.10                       |    7.088M              |    1.455G |
|    vit.encoder.layer.11                       |    7.088M              |    1.455G |
|   vit.layernorm                               |   1.536K               |   0.756M  |
|    vit.layernorm.weight                       |    (768,)              |           |
|    vit.layernorm.bias                         |    (768,)              |           |
|   vit.pooler.dense                            |   0.591M               |           |
|    vit.pooler.dense.weight                    |    (768, 768)          |           |
|    vit.pooler.dense.bias                      |    (768,)              |           |
|  classifier                                   |  11.535K               |  11.52K   |
|   classifier.weight                           |   (15, 768)            |           |
|   classifier.bias                             |   (15,)                |           |
'''
