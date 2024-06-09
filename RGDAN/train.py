import math
import argparse
import utils
import model
import time
import datetime
import numpy as np
import torch

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='each time step represents 5 minutes')
parser.add_argument('--P', type=int, default=12,
                    help='number of historical steps')
parser.add_argument('--Q', type=int, default=6,
                    help='number of prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=4,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=4,
                    help='dimensionality of each attention head output')
parser.add_argument('--adjdata', type=str, default='data/adj_mx.pkl',
                    help='path to adjacency data')
parser.add_argument('--adjtype', type=str, default='doubletransition',
                    help='type of adjacency')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set ratio [default: 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set ratio [default: 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set ratio [default: 0.2]')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=50,
                    help='number of epochs to run')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stopping')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001,
                    help='weight decay')
parser.add_argument('--decay_epoch', type=int, default=20,
                    help='number of epochs after which to decay learning rate')
parser.add_argument('--path', default='./',
                    help='path to traffic file')
parser.add_argument('--dataset', default='PeMS',
                    help='name of traffic dataset')
parser.add_argument('--load_model', default="F",
                    help='load pretrained model (T/F)')

args = parser.parse_args()
LOG_FILE = f"{args.path}data/log({args.dataset})"
MODEL_FILE = f"{args.path}data/GMAN({args.dataset})"

start = time.time()  # Start timer

log = open(LOG_FILE, 'w')
utils.log_string(log, str(args)[10: -1])

# Load data
utils.log_string(log, 'loading data...')  # Log loading data
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE,
 mean, std) = utils.loadData(args)
utils.log_string(log, f'trainX: {trainX.shape}\ttrainY: {trainY.shape}')
utils.log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
utils.log_string(log, f'testX:  {testX.shape}\t\ttestY:  {testY.shape}')
utils.log_string(log, 'data loaded!')

# If GPU available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sensor_ids, sensor_id_to_ind, adj_mx = utils.load_adj(args.adjdata, args.adjtype)  # Load adjacency data
adj_mx = [torch.tensor(i).to(device) for i in adj_mx]  # Convert adjacency matrix to PyTorch tensors

num_nodes = adj_mx[0].shape[0]
print('num_nodes : %d' % num_nodes)

# Transform data to tensors
# Convert data to PyTorch tensors and move to device (GPU if available)
trainX, trainTE, trainY = torch.FloatTensor(trainX).to(device), torch.LongTensor(trainTE).to(device), torch.FloatTensor(trainY).to(device)
valX, valTE, valY = torch.FloatTensor(valX).to(device), torch.LongTensor(valTE).to(device), torch.FloatTensor(valY).to(device)
testX, testTE, testY = torch.FloatTensor(testX).to(device), torch.LongTensor(testTE).to(device), torch.FloatTensor(testY).to(device)
SE = torch.FloatTensor(SE).to(device)

TEmbsize = (24 * 60 // args.time_slot) + 7  # number of slots in a day + number of days in a week
RGDAN = model.RGDAN(args.K, args.d, SE.shape[1], TEmbsize, args.P, args.L, device, adj_mx, num_nodes).to(device)
optimizer = torch.optim.Adam(RGDAN.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # Define optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.3)  # Learning rate scheduler
print("initial learning rate:", optimizer.defaults['lr'])

utils.log_string(log, '**** training model ****')
if args.load_model == 'T':
    utils.log_string(log, f'loading pretrained model from {MODEL_FILE}')
    RGDAN.load_state_dict(torch.load(MODEL_FILE))
num_train, _, N = trainX.shape
num_val = valX.shape[0]
wait = 0
val_loss_min = np.inf

# Training loop
for epoch in range(args.max_epoch):
    if wait >= args.patience:
        utils.log_string(log, f'early stop at epoch: {epoch + 1:04d}')
        break  # early stop
    # Shuffle training data
    permutation = torch.randperm(num_train)
    trainX, trainTE, trainY = trainX[permutation], trainTE[permutation], trainY[permutation]
    # Train loss
    start_train = time.time()
    train_loss = 0
    num_batch = math.ceil(num_train / args.batch_size)

    for batch_idx in range(num_batch):
        RGDAN.train()
        optimizer.zero_grad()
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        batchX, batchTE, batchlabel = trainX[start_idx:end_idx], trainTE[start_idx:end_idx], trainY[start_idx:end_idx]
        batchpred = RGDAN(batchX, SE, batchTE)
        batchpred = batchpred * std + mean
        batchloss = model.mae_loss(batchpred, batchlabel, device)
        batchloss.backward()
        optimizer.step()
        train_loss += batchloss.item() * (end_idx - start_idx)
    train_loss /= num_train
    end_train = time.time()

    start_val = time.time()
    val_loss = 0
    num_batch = math.ceil(num_val / args.batch_size)

    # Loop over validation batches
    for batch_idx in range(num_batch):
        RGDAN.eval()
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        batchX, batchTE, batchlabel = valX[start_idx:end_idx], valTE[start_idx:end_idx], valY[start_idx:end_idx]
        batchpred = RGDAN(batchX, SE, batchTE)
        batchpred = batchpred * std + mean
        batchloss = model.mae_loss(batchpred, batchlabel, device)
        val_loss += batchloss.item() * (end_idx - start_idx)
    val_loss /= num_val
    end_val = time.time()
    utils.log_string(
        log,
        f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | epoch: {epoch + 1:04d}/{args.max_epoch}, training time: {end_train - start_train:.1f}s, inference time: {end_val - start_val:.1f}s')
    utils.log_string(
        log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
    if val_loss <= val_loss_min:  # Check if validation loss decreased
        utils.log_string(
            log,
            f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {MODEL_FILE}')
        wait = 0
        val_loss_min = val_loss
        torch.save(RGDAN.state_dict(), MODEL_FILE)
    else:
        wait += 1  # Increment patience counter
    scheduler.step()  # Update learning rate
    print(f"learning rate for epoch {epoch + 2}: {optimizer.param_groups[0]['lr']}")

# Test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, f'loading model from {MODEL_FILE}')
RGDAN.load_state_dict(torch.load(MODEL_FILE))
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')

# Evaluate on training data
trainPred = []
num_batch = math.ceil(num_train / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
    batchX, batchTE, batchlabel = trainX[start_idx:end_idx], trainTE[start_idx:end_idx], trainY[start_idx:end_idx]
    batchpred = RGDAN(batchX, SE, batchTE)
    batchpred = batchpred * std + mean
    trainPred.append(batchpred.detach().cpu().numpy())
trainPred = np.concatenate(trainPred, axis=0)

# Evaluate on validation data
valPred = []
num_batch = math.ceil(num_val / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
    batchX, batchTE, batchlabel = valX[start_idx:end_idx], valTE[start_idx:end_idx], valY[start_idx:end_idx]
    batchpred = RGDAN(batchX, SE, batchTE)
    batchpred = batchpred * std + mean
    valPred.append(batchpred.detach().cpu().numpy())
valPred = np.concatenate(valPred, axis=0)

# Evaluate on test data
testPred = []
num_test = testX.shape[0]
start_test = time.time()
num_batch = math.ceil(num_test / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    batchX, batchTE, batchlabel = testX[start_idx:end_idx], testTE[start_idx:end_idx], testY[start_idx:end_idx]
    batchpred = RGDAN(batchX, SE, batchTE)
    batchpred = batchpred * std + mean
    testPred.append(batchpred.detach().cpu().numpy())
end_test = time.time()
testPred = np.concatenate(testPred, axis=0)

# Convert true labels to numpy arrays
trainY, valY, testY = trainY.cpu().numpy(), valY.cpu().numpy(), testY.cpu().numpy()

# Save predictions
np.save(f'./{args.dataset}_true.npy', testPred)
np.save(f'./{args.dataset}_pred.npy', testY)

# Calculate evaluation metrics
train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
test_mae, test_rmse, test_mape = utils.metric(testPred, testY)
utils.log_string(log, f'testing time: {end_test - start_test:.1f}s')
utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
utils.log_string(log, f'train            {train_mae:.2f}\t\t{train_rmse:.2f}\t\t{train_mape * 100:.2f}%')
utils.log_string(log, f'val              {val_mae:.2f}\t\t{val_rmse:.2f}\t\t{val_mape * 100:.2f}%')
utils.log_string(log, f'test             {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%')

# Performance in each prediction step
MAE, RMSE, MAPE = [], [], []
for q in range(args.Q):
    mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q])
    MAE.append(mae)
    RMSE.append(rmse)
    MAPE.append(mape)
    utils.log_string(log, f'step: {q + 1:02d}         {mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%')
average_mae, average_rmse, average_mape = np.mean(MAE), np.mean(RMSE), np.mean(MAPE)
utils.log_string(
    log, f'average:         {average_mae:.2f}\t\t{average_rmse:.2f}\t\t{average_mape * 100:.2f}%')
end = time.time()
utils.log_string(log, f'total time: {(end - start) / 60:.1f}min')
log.close()
