# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric
import dhg
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import logging
import pprint
import os
from sklearn.metrics import f1_score

# global variables
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_epoch_list = [1, 10, 50, 100]  # number of epochs to visualize

# %%
all_activations = {
    'relu': nn.ReLU(),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(),
    'leakyrelu': nn.LeakyReLU(negative_slope=0.1),  # hyperparameters
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(),
    'identity': nn.Identity()
}

# =========================== Model =========================== #


class UniEGNNConv(nn.Module):
    """
    This convolution utilizes the hypergraph structure.
    Message passes from vertex to (hyper)edge, then back to vertex.
    """

    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            bias,
            drop_rate
    ):
        super().__init__()
        self.args = args
        self.v_channels = in_channels
        self.e_channels = out_channels
        self.act = all_activations[args.activation]
        self.drop = nn.Dropout(drop_rate)
        self.theta_vertex = nn.Linear(
            self.v_channels, self.e_channels, bias=bias)
        self.theta_edge = nn.Linear(
            self.e_channels, self.v_channels, bias=bias)
        self.edge_merge = nn.Linear(
            self.e_channels * 2, self.e_channels, bias=bias)
        self.vertex_merge = nn.Linear(
            self.v_channels * 2, self.v_channels, bias=bias)
        self.first_aggregate = args.first_aggregate
        self.second_aggregate = args.second_aggregate
        self.fix_edge = args.fix_edge
        self.layer_norm_v = nn.LayerNorm(self.v_channels)
        self.layer_norm_e = nn.LayerNorm(self.e_channels)

    def forward(self, X, Y, graph):
        X_0 = X
        Y_0 = Y
        X = self.theta_vertex(X)

        if isinstance(graph, dhg.BiGraph):
            Y = self.edge_merge(
                torch.cat([Y_0, graph.u2v(X, aggr=self.first_aggregate)], dim=-1))
            Y = self.theta_edge(Y)
            X = self.vertex_merge(
                torch.cat([X_0, graph.v2u(Y, aggr=self.second_aggregate)], dim=-1))

        elif isinstance(graph, dhg.Hypergraph):
            Y = self.edge_merge(
                torch.cat([Y_0, graph.v2e_aggregation(X, aggr=self.first_aggregate)], dim=-1))
            Y = self.theta_edge(Y)
            X = self.vertex_merge(
                torch.cat([X_0, graph.e2v_aggregation(Y, aggr=self.second_aggregate)], dim=-1))
        else:
            raise TypeError(
                "graph should be either dgg.BiGraph or dhg.Hypergraph.")

        X = self.layer_norm_v(X)
        Y = self.layer_norm_e(Y)
        X = self.drop(self.act(X))
        Y = self.drop(self.act(Y))
        if self.fix_edge:
            return X, Y_0
        else:
            return X, Y


all_convs = {
    'UniEGNN': UniEGNNConv,
}

class HyperGraphModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        Conv = all_convs[args.model]

        self.convs = nn.ModuleList(
            [Conv(args, args.nhid, args.nhid, args.bias, args.drop_rate)
             for _ in range(args.nlayer)]
        )

        self.args = args
        self.act = all_activations[args.activation]
        self.vertex_input = nn.Sequential(
            nn.LayerNorm(args.nfeat),
            nn.Linear(args.nfeat, args.nhid, bias=args.bias),
            self.act,
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nhid, bias=args.bias)
        )

        self.edge_input = nn.Sequential(
            nn.LayerNorm(args.nedge),
            nn.Linear(args.nedge, args.nhid, bias=args.bias),
            self.act,
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nhid, bias=args.bias)
        )

        self.vertex_out = nn.Sequential(
            nn.LayerNorm(args.nhid),
            nn.Linear(args.nhid, args.nhid, bias=args.bias),
            self.act,
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nout, bias=args.bias)
        )

    def forward(self, G):
        X = self.vertex_input(G.X)
        Y = self.edge_input(G.Y)
        for i, conv in enumerate(self.convs):
            X, Y = conv(X, Y, G)
        X = self.act(X)
        X = self.vertex_out(X)
        return X.squeeze()


all_models = {
    'UniEGNN': HyperGraphModel
}


class HGData(dhg.Hypergraph):
    def __init__(self,
                 var_features: np.ndarray,
                 one_features: np.ndarray,
                 sqr_features: np.ndarray,
                 constr_features: np.ndarray,
                 obj_features: np.ndarray,
                 edge_features: np.ndarray,
                 edges: np.ndarray,
                 sol: np.ndarray
                 ):
        self.edges = torch.LongTensor(edges).T
        self.X = np.concatenate([var_features, one_features,
                                 sqr_features, constr_features, obj_features], axis=0)
        self.X = torch.FloatTensor(self.X)
        self.Y = torch.FloatTensor(edge_features).reshape(-1, 1)
        self.opt_sol = torch.FloatTensor(sol)
        super().__init__(num_v=self.X.shape[0], e_list=edges.tolist())

    def to(self, device: torch.cuda.device):
        super().to(device)
        self.edges = self.edges.to(device)
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.opt_sol = self.opt_sol.to(device)
        return self


class HGDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files):
        super().__init__()
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        name, graph = pickle.load(open(self.sample_files[idx], "rb"))
        return name, HGData(*graph)

all_data = {
    "HG": HGData,
}

all_datasets = {
    "HG": HGDataset,
}

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        if self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss
        else:
            raise NotImplementedError(f"{self.reduction} hasn't been implemented.")

def collate_fn(batch):
    [batch] = batch
    return batch


def setup_logger(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    f_handler = logging.FileHandler(log_path)
    f_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    f_handler.setFormatter(formatter)

    logger.addHandler(f_handler)
    return logger


def train(args):
    data_name = args.problem
    difficulty = args.difficulty
    model_name = args.model
    nlayer = args.nlayer
    device = args.device
    nepoch = args.nepoch
    batch_size = args.batch_size
    seed = args.seed
    if difficulty is not None:
        out_dir = f'{_root}/runs/train/{args.encoding}/{model_name}/{data_name}_{difficulty}_layer{nlayer}_nepoch{nepoch}_lr{args.lr}_wd{args.wd}_{args.loss}'
        data_dir = os.path.join(
            _root, 'data', 'train', args.encoding, f'{data_name}_train', difficulty)
    else:
        out_dir = f'{_root}/runs/train/{args.encoding}/{model_name}/{data_name}_layer{nlayer}_nepoch{nepoch}_lr{args.lr}_wd{args.wd}_{args.loss}'
        data_dir = os.path.join(
            _root, 'data', 'train', args.encoding, f'{data_name}_train')
    os.makedirs(out_dir, exist_ok=True)
    model_save_path = os.path.join(out_dir, 'trained_model.pkl')

    if os.path.exists(model_save_path):
        print(f"Model {model_save_path} exists.\nOverwrite? (y/n)")
        if input() not in ['Y', 'y']:
            print("Skip training.")
            return
        else:
            print("Overwriting trained model.")
            for f in os.listdir(out_dir):
                if os.path.isfile(os.path.join(out_dir, f)):
                    os.remove(os.path.join(out_dir, f))

    log_file = os.path.join(out_dir, 'training.log')
    logger = setup_logger(log_file)
    args_dict = vars(args)
    args_pretty = pprint.pformat(args_dict, indent=4)
    logger.info("Training arguments:\n %s", args_pretty)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model = all_models[args.model](args).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(alpha=0.7, gamma=0, reduction='mean')
    else:
        raise NotImplementedError(f"{args.loss} hasn't been implemented.")

    print("Loading data...")

    sample_files = [os.path.join(data_dir, file)
                    for file in os.listdir(data_dir)]

    dataset = all_datasets[args.encoding](sample_files)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"Start training {model_name} on {data_name} with {nlayer} layers.")

    f1s = []
    losses = []
    early_stop_count = 0
    best_val_loss = float('inf')

    for epoch in range(nepoch):

        model.train()
        train_losses = []
        train_f1s = []
        accumulated_loss = torch.tensor(0.0, device=device)
        count = 0

        for name, G in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            G = G.to(device)
            output = model(G)
            output = output[:len(G.opt_sol)]
            binary_output = (output > 0).long()
            loss = criterion(output, G.opt_sol)
            f1 = f1_score(G.opt_sol.cpu().numpy().astype(int), binary_output.cpu().numpy().astype(int), average='binary')

            accumulated_loss += loss
            count += 1

            if count % batch_size == 0:
                optimizer.zero_grad()
                (accumulated_loss / batch_size).backward()
                optimizer.step()
                train_losses.append(accumulated_loss.item() / batch_size)
                train_f1s.append(f1)
                accumulated_loss = torch.tensor(0.0, device=device)

        if count % batch_size != 0:
            optimizer.zero_grad()
            (accumulated_loss / (count % batch_size)).backward()
            optimizer.step()
            train_losses.append(accumulated_loss.item() / (count % batch_size))
            train_f1s.append(f1)

        mean_train_loss = np.mean(train_losses, axis=0)
        mean_train_f1 = np.mean(train_f1s, axis=0)
        logger.info(f"Epoch {epoch+1} training: loss: {mean_train_loss:.5f}")
        logger.info(f"Epoch {epoch+1} training: f1: {mean_train_f1:.5f}")
        print(f"Epoch {epoch+1} training: loss: {mean_train_loss:.5f}")
        print(f"Epoch {epoch+1} training: f1: {mean_train_f1:.5f}")

        model.eval()
        val_losses = []
        val_f1s = []
        with torch.no_grad():
            for _, G in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                G = G.to(device)
                output = model(G)
                output = output[:len(G.opt_sol)]
                binary_output = (output > 0).long()
                loss = criterion(output, G.opt_sol)
                f1 = f1_score(G.opt_sol.cpu().numpy().astype(int), binary_output.cpu().numpy(), average='binary')
                val_losses.append(loss.item())
                val_f1s.append(f1)

            mean_val_loss = np.mean(val_losses, axis=0)
            mean_val_f1 = np.mean(val_f1s, axis=0)
            logger.info(
                f"Epoch {epoch+1} validation: loss: {mean_val_loss:.5f}")
            logger.info(f"Epoch {epoch+1} validation: f1: {mean_val_f1:.5f}")
            print(f"Epoch {epoch+1} validation: loss: {mean_val_loss:.5f}")
            print(f"Epoch {epoch+1} validation: f1: {mean_val_f1:.5f}")

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                early_stop_count = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"Model {model_name} saved to {model_save_path}.")
                logger.info(f"Model {model_name} saved to {model_save_path}.")
            else:
                early_stop_count += 1
                if early_stop_count >= args.early_stop:
                    print(f"Early stop at epoch {epoch+1}.")
                    logger.info(f"Early stop at epoch {epoch+1}.")
                    break

        f1s.append(mean_val_f1)
        losses.append(mean_val_loss)
            
        if epoch+1 in _epoch_list and args.output:
            with torch.no_grad():
                outputs = []
                for name, G in tqdm(data_loader, desc=f"Epoch {epoch+1} Output"):

                    G = G.to(device)

                    output = model(G)
                    output = output[:len(G.opt_sol)]

                    outputs.append((name, output.cpu().numpy()))

                pickle.dump(outputs, open(
                    f"{out_dir}/epoch_{epoch+1}_outputs.pkl", 'wb'))
                print(f"Epoch {epoch+1} outputs saved.")
                logger.info(f'Epoch {epoch+1} outputs saved.')

    f1s = np.array(f1s)
    losses = np.array(losses)
    logger.info(f"Best valid f1: {max(f1s):.5f}")
    pickle.dump([f1s, losses], open(
        os.path.join(out_dir, "training_data.pkl"), "wb"))
    print(f"Training finished.\nBest validation f1: {max(f1s):.5f}")

# ==================== Visualize ==================== #
def visualize(model, encoding, dataset, difficulty, nlayer, name, nepoch, lr, wd, loss, vepoch):

    import matplotlib.pyplot as plt
    import re
    import pickle

    def _draw_single_epoch(outputs_dir, figure_dir, sol, name, vepoch):
        output_file = f'{outputs_dir}/epoch_{vepoch}_outputs.pkl'
        if not os.path.exists(output_file):
            print(f"Epoch {vepoch} output file not found.")
            return
        datum = pickle.load(
            open(output_file, 'rb'))
        for n, d in datum:
            if n == name:
                data = d
                break

        print(f"Visualizing epoch {vepoch} neural outputs of {name}...")
        scatter = plt.scatter([range(len(data))],
                              data.reshape(-1, 1), c=sol, cmap='viridis', s=2)
        plt.colorbar(scatter)
        plt.title(f"Epoch {vepoch} neural outputs of {name}")
        plt.savefig(f"{figure_dir}/{name}_{vepoch}_output.png",
                    bbox_inches='tight')
        print(f"Neural outputs of {name} visualized.\n")
        plt.close()
        return

    def _draw_all_epoch(outputs_dir, figure_dir, sol, name):
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()

        files = [f for f in os.listdir(outputs_dir) if f.startswith('epoch')]
        files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

        subplot_count = 0
        for f in files:
            epoch_ = int(re.findall(r'\d+', f)[0])

            if epoch_ not in _epoch_list:
                continue

            if not os.path.exists(f'{outputs_dir}/epoch_{epoch_}_outputs.pkl'):
                print(f"Epoch {epoch_} output file not found.")
                continue

            datum = pickle.load(
                open(f'{outputs_dir}/epoch_{epoch_}_outputs.pkl', 'rb'))
            for n, d in datum:
                if n == name:
                    data = d
                    break
            print(f"Visualizing epoch {epoch_} neural outputs of {name}...")
            scatter = axes[subplot_count].scatter(
                [range(len(data))], data.reshape(-1, 1), c=sol, cmap='viridis', s=5)

            axes[subplot_count].set_title(f"Epoch {epoch_}", fontsize=20)
            axes[subplot_count].set_xlabel('Variable Index', fontsize=20)
            axes[subplot_count].set_ylabel('Output Value', fontsize=20)
            axes[subplot_count].tick_params(axis='x', labelsize=16)
            axes[subplot_count].tick_params(axis='y', labelsize=16)

            subplot_count += 1

        cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
        cbar = plt.colorbar(scatter, cax=cbar_ax, ticks=[0, 1])
        cbar.set_label('Optimal Solution Value', fontsize=20)
        cbar.ax.tick_params(labelsize=16)
        fig.subplots_adjust(right=0.9, wspace=0.3, hspace=0.3)

        plt.savefig(f"{figure_dir}/{name}_all_epoch.png", bbox_inches='tight')
        plt.close()
        print(f"Neural outputs of {name} visualized.\n")
        return

    if name.endswith('.pkl'):
        name = name.rsplit('.', 1)[0]

    # load solution from graph data file
    _root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    outputs_dir = os.path.join(
        _root, "runs", "train", encoding, model, f"{dataset}_{difficulty}_layer{nlayer}_nepoch{nepoch}_lr{lr}_wd{wd}_{loss}")
    figure_dir = os.path.join(
        _root, "figures", "UniEGNN", f"{dataset}_{difficulty}_layer{nlayer}_nepoch{nepoch}_lr{lr}_wd{wd}_{loss}")

    os.makedirs(figure_dir, exist_ok=True)

    graph_file = os.path.join(
        _root, "data", "train", encoding, f"{dataset}_train", difficulty, name+".pkl")
    _, graph_data = pickle.load(open(graph_file, 'rb'))
    sol = graph_data[-1]

    if vepoch == 0:
        _draw_all_epoch(outputs_dir, figure_dir, sol, name)
    else:
        _draw_single_epoch(outputs_dir, figure_dir, sol, name, vepoch)

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-p', type=str, required=True)
    parser.add_argument('--difficulty', '-d', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, default='UniEGNN',
                        choices=['UniEGNN'])
    parser.add_argument('--encoding', '-e', type=str, default='HG',
                        choices=['HG'])
    parser.add_argument('--nlayer', type=int, default=6)
    parser.add_argument('--nout', type=int, default=1, choices=[1])
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--input_drop', type=float, default=0.0)
    parser.add_argument('--first_aggregate', type=str, default='sum',
                        choices=['sum', 'mean', 'softmax_then_sum'])
    parser.add_argument('--second_aggregate', type=str,
                        default='mean', choices=['sum', 'mean', 'softmax_then_sum'])
    parser.add_argument('--nobias', action='store_true', default=False)
    parser.add_argument('--fix_edge', '-f', action='store_true', default=False)
    parser.add_argument('--activation', type=str, default='leakyrelu')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['mse', 'bce', 'focal'])
    parser.add_argument('--visualize', '-v', default=None, type=str,
                        help="visualize the neural outputs of the given problem",)
    parser.add_argument('--vepoch', type=int, default=0,
                        help="0 for default epochs in one figure, -1 for default epochs in separate figures, integer for a specific epoch")
    parser.add_argument('--no_train', '-n', action='store_true',
                        default=False, help="train the model")
    parser.add_argument('--output', '-o', action='store_true',
                        default=False, help="output the neural outputs")
    args = parser.parse_args()
    args.bias = not args.nobias
    channels_dict = {
        'HG': (16, 1),
    }
    args.nfeat, args.nedge = channels_dict[args.encoding]

    return args


if __name__ == "__main__":
    args = parse_args()

    if not args.no_train:
        train(args=args)
    if args.visualize is not None:
        if args.vepoch == -1:
            for epoch in _epoch_list:
                visualize(args.model, args.encoding, args.problem, args.difficulty, args.nlayer, args.visualize,
                          args.nepoch, args.lr, args.wd, args.loss, epoch)
        elif args.vepoch >= 0 and isinstance(args.vepoch, int):
            visualize(args.model, args.encoding, args.problem, args.difficulty, args.nlayer, args.visualize,
                      args.nepoch, args.lr, args.wd, args.loss, args.vepoch)
        else:
            raise ValueError("Invalid epoch value.")

# %%
