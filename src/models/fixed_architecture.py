from torch import nn


class HumanActivityClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv_1 = nn.Conv1d(
            in_channels=52,
            out_channels=64,
            kernel_size = 7,
            stride = 1,
            padding = "same"
        )
        self.conv_2 = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size = 5,
            stride = 1,
            padding = "same"
        )
        self.pool_1 = nn.MaxPool1d(
            kernel_size = 2,
            stride = 2
        )
        self.conv_3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size = 3,
            stride = 1,
            padding = "same"
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=0.5)

        self.linear = nn.Linear(
            in_features=256,
            out_features=128
        )
        self.relu = nn.ReLU()

        self.classifier = nn.Linear(
            in_features=128,
            out_features=num_classes
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.pool_1(x)
        x = self.conv_3(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x