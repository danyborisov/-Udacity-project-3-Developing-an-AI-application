import torch
from torch import nn
import torch.nn.functional as F

#Will have to go for nn.Module because I couldn't find a solution for a configurabloe nn.Sequential (I wrote jupyter notebook in Sequential..) Very annoying
class Model(nn.Module):
    def __init__(self, output_size, hidden_layers, drop_rate):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(25088, hidden_layers[0])])

        layers_list = zip(hidden_layers[:-1], hidden_layers[1:]) #Using the rest of the hidden_layers input from the user
        self.hidden_layers.extend([nn.Linear(hl1, hl2) for hl1, hl2 in layers_list])
          
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_rate)
        
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x) 
        x = self.output(x)
        return F.log_softmax(x, dim=1)
    
def accuracy_check (model, validloader, criterion, gpu):
        if gpu == 'gpu': #letting the user chose GPU or CPU within the funcion and loading it on a device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = 'cpu'
        
        valid_loss = 0
        accuracy = 0
            
        #Run a validation 
        model.eval()
            
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                    
                valid_out = model.forward(images)
                batch_loss = criterion(valid_out, labels)
                    
                valid_loss += batch_loss.item()
                    
        #Validation based accuracy
                v_out = torch.exp(valid_out)
                top_p, top_class = v_out.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                step_accuracy = accuracy/len(validloader)
                step_loss = valid_loss/len(validloader)
                
        return step_accuracy, step_loss
    
    
def training_steps(model, trainloader, validloader, epochs, print_every, criterion, optimizer, gpu):
        if gpu == 'gpu': #letting the user chose GPU or CPU within the funcion and loading it on a device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = 'cpu'
        
        model.to(device)
        model.train()
        
        steps = 0

        #Report results parameters
        running_loss = 0
        print_every = 100
        print('Training starts')
        print(f"Running on {device}\n")
        for epoch in range(epochs):

            for images, labels in trainloader:
                steps +=1

                #Load data to the available GPU or CPU (if GPU is not available)
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                out = model.forward(images)
                losses = criterion(out, labels)
                losses.backward()
                optimizer.step()

                running_loss += losses.item()

                #using accuracy function to print out accuracy running validation
                if steps % print_every ==0:
                    step_accuracy, step_loss = accuracy_check(model, validloader,criterion, gpu)
                    
                    print(f"\nEpoch {epoch+1}/{epochs}")
                    print(f"{steps}")
                    print(f"Training loss {running_loss/print_every:.3f}")

                    #Reference to the validation
                    print(f"Validation loss: {step_loss:.3f}")
                    print(f"Validation accuracy: {step_accuracy:.3f}")
                   
                    running_loss = 0
                model.train()


    