import torch
import numpy as np


def train(model, train_loader, lr, epoch_update,learning_rate_change, weight_decay, num_epochs, gradclip=1):

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return optimizer
                    else:
                        return optimizer

    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        for idx, (X , Y) in enumerate(train_loader):

            model.train()
            loss = torch.tensor(0.0)
            Yt = Y[:,:,:,0:Y.shape[3]-1]
            #loss
            out = model(X,Yt)
            loss = criterion(out,Y)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip)
            optimizer.step()
        # schedule learning rate decay    
        lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)                
        #if (epoch) % 100 == 0:
         #   print('********** Epoche %s **********' %(epoch+1))
          #  print("loss : ", loss.item())
            
    return model