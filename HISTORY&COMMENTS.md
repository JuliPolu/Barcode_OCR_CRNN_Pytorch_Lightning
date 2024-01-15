

### Input images preprocessing
Запускаем скрипт чтобы вырезать кропы баркодов чтобы ускорить обучение
crop_barcodes.py

изменила страйд в резнете и взяла 3 слой

резнет34 большого улучшения не дал



### Архитектура CRNN

#### Бэкбоны
  Пробовала:
- resnet18  - итоговый выбор
- resnet34 (не дал прироста)

#### Conv feature extractor
- изменила stride в resnet18 и взяла более глубокий 3-й слой как аутпут конволюционного бэкбона (дало самый значимый прирост)
- соответствено изменила размерности 

#### RNN
- Пробовала уменьшить и увеличить rnn_hidden_size (32 64, 128)
- увеличивала rnn_num_layers до 4
- лучший результат: rnn_hidden_size=64, rnn_num_layers=4

### Pretrain
- imagenet

### Optimizers and learning rate
- Adam lr: 1e-3 weight_decay: 1e-5
- scheduler: 'ReduceLROnPlateau'

### Loss function
- CTCloss

### Метрики
 - StringMatchMetric - самая важная (полное соответствие строк), так как в баркодах каждая цифра критичная
 - EditDistanceMetric

### Аугментации

- В итого Использовались предлоденные в мини домашке:
  albu.RandomBrightnessContrast(p=1),
  albu.CLAHE(p=0.5),
  albu.Blur(blur_limit=3, p=0.3),
  albu.GaussNoise(p=0.3),
  albu.Downscale(scale_min=0.3, scale_max=0.9, p=0.5),
  albu.CoarseDropout(max_holes=8, min_holes=2, p=0.3),

- Пробовала добавить еще - но результат лучше не стал:
  albu.Rotate(limit=10, p=0.5),
  albu.ElasticTransform(p=0.3),
  albu.GridDistortion(p=0.3),

### Image size
- Подбор осуществлялся в [ноутбуке](notebooks/подбор параметров.ipynb)

  
  
