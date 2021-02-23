from torch import optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import utils
import visual


def train(model, train_loader, test_loader, epochs_per_task=10,
          batch_size=64, test_size=1024, consolidate=True,
          fisher_estimation_sample_size=1024,
          lr=1e-3, weight_decay=1e-5,
          loss_log_interval=30,
          eval_log_interval=50,
          cuda=False):
    # prepare the loss criteriton and the optimizer.
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          weight_decay=weight_decay)

    # set the model's mode to training mode.
    model.train()
    ce_l, total_l, ewc_l, acc = {}, {}, {}, {}
    for task in range(1,len(train_loader)+1):
        total_l[task] = []
        ce_l[task] = []
        ewc_l[task] = []
        acc[task] =[]
        
        data_loader = train_loader[task-1]
        
        for epoch in range(1, epochs_per_task+1):
            # prepare the data loaders.
            data_stream = tqdm(enumerate(data_loader, 1))

            for batch_index, (x, y) in data_stream:
                # where are we?
                data_size = len(x)
                dataset_size = len(data_loader.dataset)
                dataset_batches = len(data_loader)
    

                # run the model and backpropagate the errors.
                optimizer.zero_grad()
                scores = model(x)
                ce_loss = criterion(scores, y)
                ewc_loss = model.ewc_loss(cuda=cuda)
                loss = ce_loss + ewc_loss
                loss.backward()
                optimizer.step()

                # calculate the training precision.
                _, predicted = scores.max(1)
                precision = (predicted == y).sum().float() / len(x)

                data_stream.set_description((
                    '=> '
                    'task: {task}/{tasks} | '
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'prec: {prec:.4} | '
                    'loss => '
                    'ce: {ce_loss:.4} / '
                    'ewc: {ewc_loss:.4} / '
                    'total: {loss:.4}'
                ).format(
                    task=task,
                    tasks=len(train_loader),
                    epoch=epoch,
                    epochs=epochs_per_task,
                    trained=batch_index*batch_size,
                    total=dataset_size,
                    progress=(100.*batch_index/dataset_batches),
                    prec=float(precision),
                    ce_loss=float(ce_loss),
                    ewc_loss=float(ewc_loss),
                    loss=float(loss),
                ))

            for i in range(1,task + 1):
                acc[i].append(utils.validate(model, test_loader[i-1], cuda=cuda, verbose=False))
                
            '''
            acc[i].append(utils.validate(
                model, test_datasets[i], test_size=test_size,
                cuda=cuda, verbose=False,
            )) if i+1 <= task else 0 for i in
            range(len(train_datasets))'''
            
            
            #print('Test Accuracy', acc)
            '''title = (
                'precision (consolidated)' if consolidate else
                'precision'
            )
            visual.visualize_scalars(
                vis, precs, names, title,
                iteration
            )'''
            total_l[task].append(loss.item())
            ce_l[task].append(ce_loss.item())
            ewc_l[task].append(ewc_loss.item())
            # Send losses to the visdom server.
            #if iteration % loss_log_interval == 0:
                #title = 'loss (consolidated)' if consolidate else 'loss'
                #visual.visualize_scalars(
                 #   vis,
            #print('total loss', total_l, 'cross entropy', ce_l, 'ewc', ewc_l)#[loss, ce_loss, ewc_loss],
#                    ['total', 'cross entropy', 'ewc'],
                    
            #    )

        if consolidate and task < len(train_loader):
            # estimate the fisher information of the parameters and consolidate
            # them in the network.
            print(
                '=> Estimating diagonals of the fisher information matrix...',
                flush=True, end='',
            )
            model.consolidate(model.estimate_fisher(
                data_loader, fisher_estimation_sample_size
            ))
            print(' Done!')
    return acc, total_l, ce_l, ewc_l
