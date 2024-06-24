import numpy as np
from train import train_client, running_model_sum, scale_model_state, evaluate
from util import swap_losses
import zlib
import pickle

def fedlbs(global_model, client_train_loader, test_loader, label_sc, n_clients, num_clients_per_round, batch_size, num_local_epochs, lr, max_rounds, model_type, device, S):
    all_smape = [] #list of all rounds smape
    all_loss=[] #list of all rounds loss     
    all_mae = []
    all_rmse = []
    loss_lst = [0.0] * num_clients_per_round
    weight_lst = [0.0] * num_clients_per_round
    loss = 0.0

    clients = np.arange(num_clients_per_round)
    for t in range(max_rounds):
        # ratio = num_clients_per_round
        rnd= "average"
        weight_avg = None
        if t % S == 0 and t!=0:
            # swap= swap_weights(dict_idl)
            swap = swap_losses(loss_lst, weight_lst)
            rnd= "swapping"
        
        print("\nstarting {} round {}".format(rnd, t))
        # clients = list(range(0, num_clients_per_round))
        print("clients: ", clients)
        global_model.eval() # turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc. 
        global_model = global_model.to(device) #run on gpu

        for k,cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1,num_clients_per_round, cid))
            
            if rnd == "swapping":
                global_model.load_state_dict(pickle.loads(zlib.decompress(swap[k])))
                
            local_model, train_loss = train_client(
                train_loader = client_train_loader[k],
                global_model = global_model,
                lr = lr,
                batch_size = batch_size,
                num_local_epochs = num_local_epochs,
                model_type = model_type,
                device = device)

            weight_avg = running_model_sum(weight_avg, local_model.state_dict())
            
            # if the next round is swapping
            if (t+1) % S == 0:
                print('updating loss and weight')
                loss_lst[k] = train_loss
                weight_lst[k] = zlib.compress(pickle.dumps(local_model.state_dict()))
        
        weight_avg = scale_model_state(weight_avg, 1/num_clients_per_round)
        # set global model parameters for the next step
        global_model.load_state_dict(weight_avg)
        outputs, target, loss, smape, mae, rmse  = evaluate(global_model, test_loader, label_sc, model_type,device)       
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)
        
    return outputs, target, all_loss, all_smape, all_mae, all_rmse

def fedavg(global_model, client_train_loader, test_loader, label_sc, n_clients, num_clients_per_round, batch_size, num_local_epochs, lr, max_rounds, model_type, device):
    
    all_smape = [] #list of all rounds smape
    all_loss=[] #list of all rounds loss     
    all_mae = []
    all_rmse = []
    
    clients = np.arange(0, num_clients_per_round) # choose client ids w/out repetition
    for t in range(max_rounds):

        weight_avg = None
        print("\nstarting avg round {}".format(t))
        # clients = list(range(0, num_clients_per_round))
        print("clients: ", clients)
        global_model.eval() # turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc. 
        global_model = global_model.to(device) #run on gpu

        for k,cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1,num_clients_per_round, cid))
            
            local_model, _ = train_client(
                train_loader = client_train_loader[k],
                global_model = global_model,
                lr = lr,
                batch_size = batch_size,
                num_local_epochs = num_local_epochs,
                model_type = model_type,
                device = device)
            
            # do the average of local model weights with each client in a single round
            # weight_avg = running_model_avg(weight_avg, local_model.state_dict(), 1/num_clients_per_round)
            weight_avg = running_model_sum(weight_avg, local_model.state_dict())

        weight_avg = scale_model_state(weight_avg, 1/num_clients_per_round)
   
        # set global model parameters for the next step
        global_model.load_state_dict(weight_avg)
        outputs, target, loss, smape, mae, rmse = evaluate(global_model, test_loader, label_sc, model_type, device)       
        
        #save final round accuracy and loss in a list to return 
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)
        
    return outputs, target, all_loss, all_smape, all_mae, all_rmse