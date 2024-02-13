import torch
from torch import nn
import copy
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ValueNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNN, self).__init__()
        self.Linear_1 = nn.Linear(input_size, int(hidden_size/2))
        self.Linear_2 = nn.Linear(int(hidden_size/2), hidden_size)
        self.Linear_3 = nn.Linear(hidden_size, int(hidden_size/2))
        self.Linear_4 = nn.Linear(int(hidden_size/2), output_size)
        self.LLU = nn.LeakyReLU(0.1)


    def forward(self, input_element):
        output = self.Linear_1(input_element)
        output = self.LLU(output)
        output = self.Linear_2(output)
        output = self.LLU(output)
        output = self.Linear_3(output)
        output = self.LLU(output)
        output = self.Linear_4(output)
        return output



class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        self.Qlinear = nn.Linear(input_size, int(hidden_size/2))
        self.Klinear = nn.Linear(input_size, int(hidden_size/2))
        self.Vlinear = nn.Linear(input_size, int(hidden_size/2))
        self.Linear_1 = nn.Linear(int(hidden_size/2), hidden_size)
        self.Linear_2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.Linear_3 = nn.Linear(int(hidden_size/2), output_size)
        self.LLU = nn.LeakyReLU(0.1)
       
    def forward(self, input_element): # 1,6 형식으로 들어가야함
        Qvalue = self.Qlinear(input_element)
        Kvalue = self.Klinear(input_element)
        Vvalue = self.Vlinear(input_element)
        attention = self.LLU(Kvalue @ Qvalue).squeeze() 
        attention = (Vvalue @ attention).squeeze() 
        attention = self.LLU(attention)
        attention = self.Linear_1(attention)
        attention = self.LLU(attention)
        attention = self.Linear_2(attention)
        attention = self.LLU(attention)
        attention = self.Linear_3(attention)
        return attention
        


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Transformer, self).__init__()
        self.Qlinear = nn.Linear(input_size, int(hidden_size/2))
        self.Klinear = nn.Linear(input_size, int(hidden_size/2))
        self.Vlinear = nn.Linear(input_size, int(hidden_size/2))
        self.Linear_1 = nn.Linear(int(hidden_size/2), hidden_size)
        self.Linear_2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.Linear_3 = nn.Linear(int(hidden_size/2), output_size)
        self.LLU = nn.LeakyReLU(0.1)
       
    def forward(self, input_element): # 1,6 형식으로 들어가야함
        #torch.set_printoptions(threshold=10000000)
        #print(input_element)
        Qvalue = self.Qlinear(input_element).unsqueeze(1)
        Kvalue = self.Klinear(input_element).unsqueeze(-1)
        Vvalue = self.Vlinear(input_element).unsqueeze(1)
 
        attention = self.LLU(Kvalue @ Qvalue).squeeze() 
        attention = (Vvalue @ attention).squeeze() 
        attention = self.LLU(attention)
        attention = self.Linear_1(attention)
        attention = self.LLU(attention)
        attention = self.Linear_2(attention)
        attention = self.LLU(attention)
        attention = self.Linear_3(attention)
        return attention
    
class TransformerQ(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Transformer, self).__init__()
        self.Qlinear = nn.Linear(input_size, int(hidden_size/2))
        self.Klinear = nn.Linear(input_size, int(hidden_size/2))
        self.Vlinear = nn.Linear(input_size, int(hidden_size/2))
        self.Linear_1 = nn.Linear(int(hidden_size/2), hidden_size)
        self.Linear_2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.Linear_3 = nn.Linear(int(hidden_size/2), output_size)
        self.LLU = nn.LeakyReLU(0.1)
       
    def forward(self, input_element): # 1,6 형식으로 들어가야함
        #torch.set_printoptions(threshold=10000000)
        #print(input_element)
        Qvalue = self.Qlinear(input_element).unsqueeze(1)
        Kvalue = self.Klinear(input_element).unsqueeze(-1)
        Vvalue = self.Vlinear(input_element).unsqueeze(1)
 
        attention = self.LLU(Kvalue @ Qvalue).squeeze() 
        attention = (Vvalue @ attention).squeeze() 
        attention = self.LLU(attention)
        attention = self.Linear_1(attention)
        attention = self.LLU(attention)
        attention = self.Linear_2(attention)
        attention = self.LLU(attention)
        attention = self.Linear_3(attention)
        return attention
    
class CustomActivationF:
    
    def __init__(self):
        self.rate = 1
    def log_act(self, a):
        positive = torch.log(a + self.rate)
        negative = - torch.log(self.rate - a)
        return torch.where(a > 0, positive, negative)
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(10000, 2000)
        self.fc2 = nn.Linear(2000, 400)
        self.fc3 = nn.Linear(400, 100)
        self.fc4 = nn.Linear(100, 400)
        self.fc5 = nn.Linear(400, 2000)
        self.fc6 = nn.Linear(2000, 10000)
        self.LLU = nn.LeakyReLU(0.1)

    def reparameterize(self, mu):
        #std = torch.one_like(logvar)
        #std = torch.exp(0.5*logvar)
        eps = torch.rand_like(mu)*0.1
        #print(eps)
        return mu + eps

    def forward(self, x):
        
        x = x.view(-1, 10000)
        x = self.LLU(self.fc1(x))
        x = self.LLU(self.fc2(x))
        mu = self.LLU(self.fc3(x))
        mu = self.reparameterize(mu)
        x = self.LLU(self.fc4(mu))
        x = self.LLU(self.fc5(x))
        x = self.fc6(x)
 
        return x.view(-1, 100, 100),  mu # , logvar
    
class VAEcnn(nn.Module):
    def __init__(self):
        super(VAEcnn, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=10, stride=4, padding=2) # Output: 50x50
        self.conv2 = nn.Conv2d(10, 16, kernel_size=6, stride=3, padding=0) # Output: 25x25
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0) # Output: 12x12

        self.LLU = nn.LeakyReLU(0.1)
        # Fully connected layers for mean and log variance
        self.fc1 = nn.Linear(32 * 2 * 2, 100)
        # self.fc_logvar = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, 32 * 2 * 2)

        # Convolutional transpose layers

        self._conv1 = nn.ConvTranspose2d(32, 16, kernel_size=10, stride=2, padding=2)
        self._conv2 = nn.ConvTranspose2d(16, 10, kernel_size=6, stride=3, padding=3)
        self._conv3 = nn.ConvTranspose2d(10, 1, kernel_size=10, stride=5, padding=5)

    def reparameterize(self, mu):
        #std = torch.one_like(logvar)
        #std = torch.exp(0.5*logvar)
        eps = torch.rand_like(mu)
        return mu #+ eps

    def forward(self, x):
        
        x = self.LLU(self.conv1(x))
        x = self.LLU(self.conv2(x))
        x = self.LLU(self.conv3(x))
        x = x.view(-1, 32 * 2 * 2)
        mu = self.LLU(self.fc1(x))
        #logvar =  self.fc_logvar(x)

        z = self.reparameterize(mu)
        x = self.LLU(self.fc2(z))
        x = x.view(-1, 32, 2, 2)
        x = self.LLU(self._conv1(x))
        x = self.LLU(self._conv2(x))

        x = self._conv3(x) # Sigmoid activation for the final layer
        return x,  mu # , logvar
    