# Risk Bounds for Cross-Domain Mapping with IPMs

Pytorch implementation of "Risk Bounds for Cross-Domain Mapping with IPMs" 
Prerequisites
--------------
- Python 2.7
- Pytorch
- Numpy/Scipy/Pandas
- Progressbar
- OpenCV

## Download dataset
Download dataset [edges2shoes, edges2handbags, cityscapes, maps, facades]: 
```sh
bash datasets/download_pix2pix.sh $DATASET_NAME.
```

## General GAN Bound (Alg.1 and Alg.2)
DiscoGAN:
```sh
python ./discogan_arch/general_gan_bound_discogan.py --task_name=$DATASET_NAME
```

DistanceGAN:
```sh
python ./discogan_arch/general_gan_bound_distancegan.py --task_name=$DATASET_NAME
```

## Per Sample Bound (Alg.3)

### Train G_1 model:
DiscoGAN:
```sh
python ./discogan_arch/disco_gan_model.py --task_name=$DATASET_NAME --num_layers=3
```

DistanceGAN:
```sh
python ./discogan_arch/general_gan_bound_distancegan.py --task_name=$DATASET_NAME
```

### Then Train G_2:
DiscoGAN:

```sh
python ./discogan_arch/gan_bound_per_sample_discogan.py --task_name=$DATASET_NAME --pretrained_generator_A_path='./models/model_gen_A-10' --pretrained_generator_B_path='./models/model_gen_B-10' --pretrained_discriminator_A_path='./models/model_dis_A-10' --pretrained_discriminator_B_path='./models/model_dis_B-10' --one_sample_index=$SAMPLE_NUMBER
```

DistanceGAN:

```sh
python ./discogan_arch/gan_bound_per_sample_distancegan.py --task_name=$DATASET_NAME --pretrained_generator_A_path='./models/model_gen_A-10' --pretrained_generator_B_path='./models/model_gen_B-10' --pretrained_discriminator_A_path='./models/model_dis_A-10' --pretrained_discriminator_B_path='./models/model_dis_B-10' --one_sample_index=$SAMPLE_NUMBER
```

## Options
Additional options can be found in ./discogan_arch/discogan_based_options/options.py

For specific configuration see [DistanceGAN](https://github.com/sagiebenaim/DistanceGAN) and [DiscoGAN](https://github.com/SKTBrain/DiscoGAN)

## Reference
If you found this code useful, please cite the following paper:
```
@article{galanti2020risk,
   author={Tomer Galanti and Sagie Benaim and Lior Wolf},
  title={Risk Bounds for Unsupervised Cross-Domain Mapping with IPMs},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
}
```

## Acknowledgements
This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant ERC CoG 725974).

The code is based on the following github repositories:
1. CycleGAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
2. DiscoGAN (https://github.com/SKTBrain/DiscoGAN)
3. DistanceGAN (https://github.com/sagiebenaim/DistanceGAN)
4. Hyperband (https://github.com/zygmuntz/hyperband).


