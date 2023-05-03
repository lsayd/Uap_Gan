# GD_Uap_StarGan
## StarGan

```bash
cd stargan
```

### Train StarGan

To train StarGAN on CelebA, run the training script below. See [here](*https://github.com/yunjey/StarGAN/blob/master/jpg/CelebA.md*) for a list of selectable attributes in the CelebA dataset. If you change the `selected_attrs` argument, you should also change the `c_dim` argument accordingly.

```bash
 python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young

```

### 还没改完

所有都需要在main中修改想要执行的部分


###  为图片批量添加UniversalPert

没写在config中指定输出路径，需要自己去ProcessImage.py里改

### Get UniversalPert With Celeba

```bash
 python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

### Test UniversalPert with Celeba

```bash
python main.py --mode test --dataset CelebA --image_size 256 --c_dim 5 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

## AttGan

```bash
cd AttGan
```

### 测试UniversalPert的攻击效果

```bash
python test.py --experiment_name 256_shortcut1_inject0_none_hq --test_int 1.0  --custom_img 
--custom_data D:\lkq\UniversalImage --custom_data_1 D:\lkq\Image
```

### To test UniversalPert with Celeba

```bash
python test_Pert.py --experiment_name 256_shortcut1_inject0_none_hq --test_int 1.0  --custom_img --custom_data D:\lkq\UniversalPert_Gan\stargan\data\celeba\images
```

# UapGan

## StarGan

### Get UniversalPert With Celeba

```bash
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 --sample_dir D:/lkq/disrupting-deepfakes-master/stargan/stargan_celeba/samples --log_dir D:/lkq/disrupting-deepfakes-master/stargan/stargan_celeba/logs --model_save_dir D:/lkq/disrupting-deepfakes-master/stargan/stargan_celeba/models --result_dir D:/lkq/disrupting-deepfakes-master/stargan/stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

### AttGan

D:/lkq/UniversalPert_Gan/AttGAN

