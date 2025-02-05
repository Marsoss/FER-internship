import torch.nn as nn

class DefaultModel(nn.Module):
        def __init__(self, num_classes:int=7, num_channels:int=1, im_width:int=300, im_height:int=300):
            super(DefaultModel, self).__init__()
            self.__im_width = im_width
            self.__im_height = im_height
            self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2)

            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2)

            self.flatten = nn.Flatten()

            self.fc1 = nn.Linear(in_features=64*(self.__im_width//4)*(self.__im_height//4), out_features=1024)

            self.dropout = nn.Dropout(p=0.5)

            self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
            

        def forward(self, x):
            x = self.conv1(x) 
            x = nn.functional.relu(x) 
            x = self.conv2(x) 
            x = nn.functional.relu(x)  
            x = self.pool1(x) 
            x = self.conv3(x) 
            x = nn.functional.relu(x) 
            x = self.pool2(x) 
            x = self.flatten(x)
            x = self.fc1(x)
            x = nn.functional.relu(x) 
            x = self.dropout(x)
            x = self.fc2(x)

            return x
