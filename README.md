# Barcode OCR

## Основная задача

Создание модели распознавания цифр на штрих-кодах (OCR) по фотографиям.

### Датасет

Трейн датасет включает 540 фото, предварительно собранных с помощью Tолоки.

[Ссылка](https://disk.yandex.ru/d/pRFNuxLQUZcDDg) на исходный датасет

Запускаем скрипт чтобы вырезать кропы баркодов чтобы ускорить обучение
crop_barcodes.py

Перед тем как запускать обучение, необходимо:
- запустить [скрипт](crop_barcodes.py), чтобы вырезать кропы баркодов для ускорения обучение 
- входной размер изображения и бекбон для CRNN. Для этого используется [эта](notebooks/подбор параметров.ipynb) тетрадка.

### Обучение

Запуск тренировки:

```
PYTHONPATH=. ./src/train.py ./configs/exp_8_resnet18_layer3_rnn64x4.yaml
```

### Логи финальной модели в ClearML

Перформанс модели можно посмотреть тут:

[ClearML](https://app.clear.ml/projects/d0622774127546c4820e6fc78dbfd129/experiments/1feb80cd1cd34cc6aa8f8c60f26f1fca/output/execution)


### Актуальная версия чекпойнта модели:

dvc pull models/checkpoint/epoch_epoch=48-valid_ctc_loss=0.218.ckpt.dvc

### Актуальная версия сохраненной torscript модели:

dvc pull models/ts_script_model/final_ocr.pt.dvc

### Инеренс

Посмотреть результаты работы обученной сети можно посмотреть в [тетрадке](notebooks/анализ результатов.ipynb)

А также запустить скрипт для конвертации чекпойнта в onnx
```
python src/convert_checkpoint.py --checkpoint ./models/checkpoint/epoch_epoch=48-valid_ctc_loss=0.218.ckpt
```

И запустить скрипт для инференса
```
PYTHONPATH=.  python ./src/infer.py --model_path ./models/ts_script_model/final_ocr.pt --image_path ./data/images/000a8eff-08fb-4907-8b34-7a13ca7e37ea--ru.8e3b8a9a-9090-46ba-9c6c-36f5214c606d.jpg
```

### Комментарии и история экспериментов 

Кратко подбор параметров и оставшиеся вопросы описаны в файле [HISTORY&COMMENTS.md](HISTORY&COMMENTS.md)

Очень бы хотела получить максимально развернутые коментарии по корректности используемых параметров и вообще по лучшим практикам, так как опыта пока маловато, и любой комментарий очень ценен!
