.PHONY: *

VENV=venv
PYTHON=$(VENV)/bin/python3
DEVICE=gpu
DATASET_FOLDER := data


# ================== LOCAL WORKSPACE SETUP ==================
venv:
	python3 -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'


pre_commit_install:
	@echo "=== Installing pre-commit ==="
	$(PYTHON) -m pre_commit install


# ========================= TRAINING ========================
run_training:
	$(PYTHON) -m ./src/train ./configs/exp_8_resnet18_layer3_rnn64x4.yaml

threshold_validation:
	$(PYTHON) -m  src.thresholds_validation.py --config_file /path/to/config --checkpoint /path/to/checkpoint


# ============================ DVC ==========================
dvc_checkpoint:
	dvc pull models/checkpoint/epoch_epoch=48-valid_ctc_loss=0.218.ckpt.dvc


dvc_pull:
	dvc pull models/ts_script_model/final_ocr.pt.dvc


# ========================= INFERENCE ========================
convert_checkpoint:

	$(PYTHON) ./src/convert_checkpoint.py --checkpoint ./models/checkpoint/epoch_epoch=48-valid_ctc_loss=0.218.ckpt

inference:
	$(PYTHON)  ./src/infer.py --model_path ./models/ts_script_model/final_ocr.pt --image_path ./data/images/000a8eff-08fb-4907-8b34-7a13ca7e37ea--ru.8e3b8a9a-9090-46ba-9c6c-36f5214c606d.jpg
