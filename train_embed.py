import torch
from torch import nn
import torch.nn.functional as F
import sys
import os

numf = 7

with open(f'losses{numf}', 'w') as f:
    f.write(f'{""}')


# LOGFILE
def WriteLog(text, num):
    with open(f'log_file{num}', 'a') as f:
        f.write(f'{text}\n')


def WriteLoss(text, num):
    with open(f'losses{num}', 'a') as f:
        f.write(f'{text}\n')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# use ESM-embedding, classifier

class Predistal(nn.Module):

    def __init__(self, esmfold_config=None, **kwargs):
        super().__init__()
        # mdl, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # mdl.to(self.device)
        self.ln1 = nn.Linear(1280, 512)
        self.fc1_drop = nn.Dropout1d(p=0.1)
        self.ln2 = nn.Linear(512, 256)
        self.fc2_drop = nn.Dropout1d(p=0.1)
        self.ln3 = nn.Linear(256, 256)
        self.fx3_drop = nn.Dropout1d(p=0.1)
        self.ln4 = nn.Linear(256, 128)
        self.ln5 = nn.Linear(128, 64)
        self.ln6 = nn.Linear(64, 32)
        self.lno = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.ln1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.ln2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.ln3(x))
        x = self.fx3_drop(x)
        x = F.relu(self.ln4(x))
        x = F.relu(self.ln5(x))
        x = F.relu(self.ln6(x))
        # softmax in loss fn
        x = F.relu(self.lno(x))
        return x

# data loader and data conversion
class YABC():
    def __init__(self):
        self.yabc = {}

    def GetYs(self, dataf):

        batchy = []
        with open(dataf, 'r') as data:
            for row in data:
                header, seq, label = row.split(sep=",")
                if header == 'h':
                    continue
                label = label.strip()
                batchy.append(label)
                if label not in self.yabc.keys():
                    self.yabc[label] = len(self.yabc.keys())
            return torch.tensor([self.yabc[y] for y in batchy])

yabc = YABC()
def train_epoch(model, loss_f, optimizer, loader):
    total_loss = 0
    len_train = 0

    for dt in loader:

        X, y = dt
        X = X.to(device)
        y = y.to(device)
        # X.to(model.device)
        # y.to(model.device)
        optimizer.zero_grad()

        # batch prediction
        y_hat = model(X)

        # gradient calc + step
        loss = loss_f(y_hat, y)
        lambda_l1 = 0.01
        l1 = 0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
        loss = loss + lambda_l1 * l1
        loss.backward()
        optimizer.step()

        # batch loss value
        last_loss = loss.item()
        total_loss += last_loss * X.shape[0]
        len_train += len(X)

        # with open("loses.txt", "a") as f:
        #     f.write(f"last {last_loss}, total {total_loss}, len {len_train}\n")

    return total_loss / len_train


# train, test loop

def train(num_epochs, model, loss_f, optimizer, train_loader, val_loader, early_stop=0, out="losses4.txt"):
    last_val_loss = 1000000000000.
    num_same_val = 0

    if early_stop > 0:
        max_num_change = early_stop
    else:
        max_num_change = num_epochs

    for i in range(num_epochs):

        model.train(True)
        train_loss = train_epoch(model=model, loss_f=loss_f, optimizer=optimizer, loader=train_loader)
        model.train(False)

        total_val_loss = 0
        len_val = 0
        val_correct = 0

        for vdata in val_loader:
            vx, vy = vdata

            vx = vx.to(device)
            vy = vy.to(device)

            with torch.no_grad():
                y_hat = model(vx)
                val_loss = loss_f(y_hat, vy)

            total_val_loss += val_loss * vx.shape[0]
            len_val += len(vx)
            val_correct += (y_hat.argmax(1) == vy).float().sum()

        total_val_loss = total_val_loss / len_val
        val_accuracy = val_correct / len_val

        if i == 0:
            last_val_loss = total_val_loss

        WriteLoss(f"\nEPOCH{i}: train_loss = {train_loss}, val_loss = {val_loss}, accuracy = {val_accuracy}", numf)

        if last_val_loss > total_val_loss:
            last_val_loss = total_val_loss

        if early_stop > 0:

            if total_val_loss > last_val_loss:
                num_same_val += 1
                if num_same_val >= max_num_change:
                    break
            else:
                num_same_val = 0
                torch.save(model.state_dict(), f"model_lin7")


model = Predistal()
model.to(device)

# freeze ESM parameters
if len(sys.argv) > 6:
    generator1 = torch.Generator().manual_seed(420)
    model.load_state_dict(torch.load(sys.argv[2], weights_only=True))
else:
    torch.nn.init.normal_(model.ln1.weight)
    torch.nn.init.zeros_(model.ln1.bias)
    torch.nn.init.normal_(model.ln4.weight)
    torch.nn.init.zeros_(model.ln4.bias)
    torch.nn.init.normal_(model.ln3.weight)
    torch.nn.init.zeros_(model.ln3.bias)
    torch.nn.init.normal_(model.ln2.weight)
    torch.nn.init.zeros_(model.ln2.bias)
    torch.nn.init.normal_(model.ln5.weight)
    torch.nn.init.zeros_(model.ln5.bias)
    torch.nn.init.normal_(model.lno.weight)
    torch.nn.init.zeros_(model.lno.bias)

# freeze_layers = [model.embedding, model.first_layers]
# for lay in freeze_layers:
#     lay.to(device)
#     for param in lay.parameters():
#         param.requires_grad = False

# init the model

dat1 = open(sys.argv[1], 'r').readlines() #LONG
dat2 = open(sys.argv[2], 'r').readlines() #SHORT
ys_1 = yabc.GetYs(dat1)
ys_2 = yabc.GetYs(dat2)

Xs_1 = torch.load(sys.argv[3]) #LONG
Xs_2 = torch.load(sys.argv[4]) #SHORT

ys_ = torch.cat((ys_1, ys_2),0)
Xs_ = torch.cat((Xs_1, Xs_2), 0)


dataset = torch.utils.data.TensorDataset(Xs_, ys_)
generator1 = torch.Generator().manual_seed(420)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(dataset, [train_size, test_size], generator1)
batch_size = 128

train_data_loader = training_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                                  drop_last=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

for i in train_data_loader:
    x, y = i
    print("LOADER")
    print(x.shape)
    print(y.shape)
    break

loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters()) # SGD
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

train(num_epochs=18, model=model, loss_f=loss_fn, optimizer=optimizer, train_loader=train_data_loader,
      val_loader=val_data_loader, early_stop=4)

# TODO navrh FFNN, log file, #continue, embedding
