import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,
                           bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.linear(recurrent)
        return output


class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.rnn_input_size = 512

        self.rnn1 = BidirectionalLSTM(
            self.rnn_input_size, hidden_size, hidden_size)
        self.rnn2 = BidirectionalLSTM(hidden_size, hidden_size, num_classes)

    def forward(self, x):
        conv = self.cnn(x)  # [batch, channels, height, width]

        # Prepare for RNN: [batch, width, channels*height]
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(0, 2, 1)  # [batch, width, channels*height]

        output = self.rnn1(conv)
        output = self.rnn2(output)  # [batch, seq_len, num_classes]

        return output


class CRNN_Small(nn.Module):
    """Smaller CRNN for faster training"""

    def __init__(self, img_height, num_channels, num_classes, hidden_size=128):
        super(CRNN_Small, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16
        )

        # After pooling: img_height=32 -> 32/16=2, so 256*2=512
        self.rnn_input_size = 256 * max(1, img_height // 16)
        self.rnn1 = BidirectionalLSTM(
            self.rnn_input_size, hidden_size, hidden_size)
        self.rnn2 = BidirectionalLSTM(hidden_size, hidden_size, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        batch, channel, height, width = conv.size()
        # Reshape for RNN: [batch, width, channels*height]
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(0, 2, 1)  # [batch, width, feature_dim]
        output = self.rnn1(conv)
        output = self.rnn2(output)
        return output
