from torch import nn
import torch
from config import Config
from timm import create_model


class CRNN(nn.Module):
    """Реализует CRNN модель для OCR задачи.
    CNN-backbone берется из timm, в RNN части стоит GRU.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config.model_kwargs
        self.rnn_hidden_size: int = 64,
        self.rnn_dropout: float = 0.1,
        self.rnn_bidirectional: bool = True,
        self.rnn_num_layers: int = 2,
        self.num_classes: int = 11,
        
        self.backbone = create_model(
            self._config['backbone_name'],
            pretrained=self._config['pretrained'],
            features_only=True,
            out_indices=(self._config['cnn_out_index'],),
        )
                    
        layer_name = 'layer2'
        layer = getattr(self.backbone, layer_name)
        for block in layer:
            block.conv1.stride = (2, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (2, 1)
            break
                    
        self.gate = nn.Conv2d(self._config['cnn_output_size'], self._config['rnn_features_num'], kernel_size=1, bias=False)
        
        self.rnn_input_size = self._config['rnn_features_num'] * self._config['cnn_output_height']

        # Рекуррентная часть.
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size, # (48*6)
            hidden_size=self._config['rnn_hidden_size'],
            dropout=self._config['rnn_dropout'],
            bidirectional=self._config['rnn_bidirectional'],
            num_layers=self._config['rnn_num_layers'],
        )

        classifier_in_features = self._config['rnn_hidden_size']
        if self.rnn.bidirectional:
            classifier_in_features = 2 * self.rnn.hidden_size

        # Классификатор.
        self.fc = nn.Linear(classifier_in_features, self._config['num_classes'])
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        cnn_features = self.backbone(tensor)[0]
        cnn_features = self.gate(cnn_features)
        cnn_features = cnn_features.permute(3, 0, 2, 1)
        cnn_features = cnn_features.reshape(
            cnn_features.shape[0],
            cnn_features.shape[1],
            cnn_features.shape[2] * cnn_features.shape[3],
        )
        rnn_output, _ = self.rnn(cnn_features)
        logits = self.fc(rnn_output)
        return self.softmax(logits)
