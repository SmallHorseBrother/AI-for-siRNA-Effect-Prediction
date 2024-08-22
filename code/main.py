###2024 8 8 deal with numerical and categorical variables


###2024 8 10  add new loss function


###2024 8 12 add argparse and use bash script


###2024 8 15 add more prior

###2024 8 22 相比CNN_1的改动   加了随机种子 模型参数的初始化 以及去掉了clip。 多尺度增加了一个更大的kernel 13.  输出通道数增加到256.

###2024 8 22 提交到初赛的版本

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
from rich import print
from sklearn.metrics import precision_score, recall_score, mean_absolute_error
import argparse
import os
import math
from itertools import product
import RNA 
import random
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果您使用的是多个GPU
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 避免卷积算法在训练时的选择波动

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
     
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return self.relu(out)

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(in_channels, 32, kernel_size=13, padding=6)### 再增加一个比较大的尺度
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        out3 = self.relu(self.conv3(x))
        out4 = self.relu(self.conv4(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

    
def log_results(params, best_score, epoch_scores, filename='training_log.csv'):
    # Flatten the params dictionary to have individual keys as columns
    params_flat = {f'param_{k}': v for k, v in params.items()}
    epoch_columns = {f'epoch_{i+1}': score for i, score in enumerate(epoch_scores)}

    log_data = {**params_flat, 'best_score': best_score, **epoch_columns}

    log_df = pd.DataFrame([log_data])

    if not os.path.isfile(filename):
        log_df.to_csv(filename, index=False)
    else:
        log_df.to_csv(filename, mode='a', header=False, index=False)



class GenomicTokenizer:
    def __init__(self, ngram=5, stride=2):
        self.ngram = ngram
        self.stride = stride
        
    def tokenize(self, t):
        t = t.upper()
        if self.ngram == 1:
            toks = list(t)
        else:
            toks = [t[i:i+self.ngram] for i in range(0, len(t), self.stride) if len(t[i:i+self.ngram]) == self.ngram]
        if len(toks[-1]) < self.ngram:
            toks = toks[:-1]
        return toks


class GenomicVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {v:k for k,v in enumerate(self.itos)}
        
    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        freq = Counter(tokens)
        itos = ['<pad>'] + [o for o,c in freq.most_common(max_vocab-1) if c >= min_freq]
        return cls(itos)


class SiRNADataset(Dataset):
    def __init__(self, df, seq_columns, cat_columns, num_columns,prior_columns, vocab, tokenizer, max_len_sirna, max_len_mrna):
        self.df = df
        self.seq_columns = seq_columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.prior_columns = prior_columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len_sirna = max_len_sirna
        self.max_len_mrna = max_len_mrna

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        seqs = [self.tokenize_and_encode(row[col]) for col in self.seq_columns]
        cat_data = torch.tensor([row[col] for col in self.cat_columns]).unsqueeze(0)
        num_data = torch.tensor([row[col] for col in self.num_columns], dtype=torch.float)#.unsqueeze(0)  pay attention to dimension
        prior = torch.tensor(np.concatenate([row[col] for col in self.prior_columns]))
        target = torch.tensor(row['mRNA_remaining_pct'], dtype=torch.float)

        return seqs, cat_data, num_data,prior, target

    def tokenize_and_encode(self, seq):
        if ' ' in seq:  # Modified sequence
            tokens = seq.split()
        else:  # Regular sequence
            tokens = self.tokenizer.tokenize(seq)
        
        encoded = [self.vocab.stoi.get(token, 0) for token in tokens]  # Use 0 (pad) for unknown tokens
        
        if 'Extended_Sequence' in seq:  # Adjust this condition based on your column naming convention
            max_len = self.max_len_mrna
        else:
            max_len = self.max_len_sirna
        
        padded = encoded + [0] * (max_len - len(encoded))
        return torch.tensor(padded[:max_len], dtype=torch.long)


class SiRNAModel(nn.Module):
    def __init__(self, vocab_size, cat_mapping_len, embed_dim=200, cat_embed_dim=32, hidden_dim=256, n_layers=3, dropout=0.5, num_dim=2, device='cuda'):
        super(SiRNAModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.cat_embedding = nn.ModuleList([nn.Embedding(num, cat_embed_dim) for num in cat_mapping_len])
        
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim * 10, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        # Multi-scale CNN with residual connections
        self.convseq = nn.Sequential(
            ResidualBlock(in_channels=200, out_channels=200, kernel_size=3, padding=1),
            MultiScaleCNN(in_channels=200),
            nn.Conv1d(in_channels=128, out_channels= 256, kernel_size=3, padding=1),##应该增加通道数
            nn.AdaptiveAvgPool1d(1)
        )

        self.cat_net = nn.Sequential(
            nn.Linear(len(cat_mapping_len) * cat_embed_dim, hidden_dim*2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.num_net = nn.Sequential(
            nn.Linear(num_dim, hidden_dim*2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        
        self.fc = nn.Sequential(
            nn.Linear(1137, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, seqs, cat_data, num_data,prior):
        embedded = [self.embedding(seq) for seq in seqs]
        outputs = []
        gru_outputs=[]
        for embed in embedded:
            x, _ = self.gru(embed)
            gru_outputs.append(x)
            x = self.dropout(x[:, -1, :])  # Use last hidden state
            outputs.append(x)
        
        x = torch.cat(outputs, dim=1)
        x1 = self.fc1(x)

        # 将GRU输出的所有序列在序列维度上拼接
        combined_embedded = torch.cat(embedded, dim=1)

        #print(combined_embedded.shape)

        # 通过CNN进行卷积处理
        x_conv = self.convseq(combined_embedded.permute(0, 2, 1))  # 调整维度顺序 (batch, channels, sequence length)

        #print(x_conv.shape)
        x_conv = torch.flatten(x_conv, start_dim=1)
        #print(x_conv.shape)

        cat_data = torch.cat(cat_data, dim=0).to(dtype=torch.long)
        cat_data = [embed(cat_data[:, i]) for i, embed in enumerate(self.cat_embedding)]
        cat_data = torch.cat(cat_data, dim=1)
        x2 = self.cat_net(cat_data)

        x3 = self.num_net(num_data)

        # print(x1.shape)
        # print(prior.shape)
        x = self.fc(torch.cat((x1,x_conv, x2, x3,prior.float()),dim=-1))

        return x.squeeze()


def calculate_metrics(y_true, y_pred, threshold=30):
    mae = np.mean(np.abs(y_true - y_pred))

    y_true_binary = (y_true < threshold).astype(int) 
    y_pred_binary = (y_pred < threshold).astype(int)

    mask = (y_pred >= 0) & (y_pred <= threshold)
    range_mae = mean_absolute_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 100

    precision = precision_score(y_true_binary, y_pred_binary, average='binary')
    recall = recall_score(y_true_binary, y_pred_binary, average='binary')
    
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    return score
    


class CustomLoss(nn.Module):
    def __init__(self, weight_0_30=2, weight_above_30=1, penalty_above_100=1):
        super(CustomLoss, self).__init__()
        self.weight_0_30 = weight_0_30
        self.weight_above_30 = weight_above_30
        self.penalty_above_100 = penalty_above_100
        self.loss = nn.MSELoss(reduction='none') 

    def forward(self, predictions, targets):
        loss = torch.zeros_like(targets)
        mask_0_30 = targets < 30
        mask_above_30 = targets >= 30
        mask_above_100 = predictions > 100
        
        loss[mask_0_30] = self.weight_0_30 * self.loss(predictions[mask_0_30],targets[mask_0_30])
        loss[mask_above_30] = self.weight_above_30 * self.loss(predictions[mask_above_30],targets[mask_above_30])
        loss[mask_above_100] += self.penalty_above_100 * self.loss(predictions[mask_above_100],torch.ones_like(predictions[mask_above_100])*100)
        
        return loss.mean()



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=51, device='cuda'):
    model.to(device)
    best_score = -float('inf')
    best_model = None
    epoch_scores = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for seqs, cat_data, num_data, prior,targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            seqs = [x.to(device) for x in seqs]
            cat_data = [x.to(device) for x in cat_data]
            num_data = num_data.to(device)
            prior = prior.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs, cat_data, num_data,prior)
            loss = criterion(outputs, targets) 
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for seqs, cat_data, num_data, prior,targets in val_loader:
                seqs = [x.to(device) for x in seqs]
                cat_data = [x.to(device) for x in cat_data]
                num_data = num_data.to(device)
                prior = prior.to(device)
                targets = targets.to(device)
                outputs = model(seqs, cat_data, num_data,prior)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_preds = np.clip(val_preds,0,100)
        val_targets = np.clip(val_targets,0,100)
        score = calculate_metrics(val_targets, val_preds)
        epoch_scores.append(score)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Validation Score: {score:.4f}')

        if score > best_score:
            best_score = score
            best_model = model
            print(f'New best model found with score: {best_score:.4f}')
    
    torch.save(best_model.state_dict(), f'./m12_{best_score:.4f}.pth')
    return best_model, best_score, epoch_scores


def test_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for seqs, cat_data, num_data, prior,targets in test_loader:
            seqs = [x.to(device) for x in seqs]
            cat_data = [x.to(device) for x in cat_data]
            num_data = num_data.to(device)
            prior = prior.to(device)
            targets = targets.to(device)
            outputs = model(seqs, cat_data, num_data,prior)
            results.append(outputs.cpu().numpy())
            
    results = np.concatenate(results)
    results = np.clip(results, 0, 100)
    
    return results




def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for seqs, cat_data, num_data,prior, target in test_loader:
            seqs = [x.to(device) for x in seqs]
            cat_data = [x.to(device) for x in cat_data]
            num_data = num_data.to(device)
            prior = prior.to(device)
            outputs = model(seqs, cat_data, num_data,prior)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target.numpy())

    y_pred = np.array(predictions)
    y_pred = np.clip(y_pred, 0, 100)##not very reasonable
    y_test = np.array(targets)
    
    score = calculate_metrics(y_test, y_pred)
    print(f"Test Score: {score:.4f}")


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

######################################  prior function ####################################################

def gibbs_energy(seq):
    energy_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 
    'table': np.array(
        [[-0.93, -2.24, -2.08, -1.1],
         [-2.11, -3.26, -2.36, -2.08],
         [-2.35, -3.42, -3.26, -2.24],
         [-1.33, -2.35, -2.11, -0.93]])}

    result = []
    for i in range(len(seq)-1):
        index_1 = energy_dict.get(seq[i])
        index_2 = energy_dict.get(seq[i + 1])
        result.append(energy_dict['table'][index_1, index_2])

    result.append(np.array(result).sum().round(3))
    result.append((result[0] - result[-2]).round(3)) 
    result = result[:24] + (24-len(result))*[0]
    result = np.array(result)
    return result

def get_gc_sterch(seq):
    max_len, tem_len = 0, 0
    for i in range(len(seq)):
        if seq[i] == 'G' or seq[i] == 'C':
            tem_len += 1
            max_len = max(max_len, tem_len)
        else:
            tem_len = 0

    result = round((max_len / len(seq)), 3)
    return np.array([result])

def get_gc_percentage(seq):
    result = round(((seq.count('C') + seq.count('G')) / len(seq)), 3)
    return np.array([result])

def get_single_comp_percent(seq):
    nt_percent = []
    for base_i in list(['A', 'G', 'C', 'U']):
        nt_percent.append(round((seq.count(base_i) / len(seq)), 3))
    return np.array(nt_percent)

def get_di_comp_percent(seq):
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=2))
    di_nt_percent = []
    for pmt_i in pmt:
        di_nt = pmt_i[0] + pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt) / (len(seq) - 1)), 3))
    return np.array(di_nt_percent)

def get_tri_comp_percent(seq): 
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=3))
    tri_nt_percent = []
    for pmt_i in pmt:
        tri_nt = pmt_i[0] + pmt_i[1] + pmt_i[2]
        tri_nt_percent.append(round((seq.count(tri_nt) / (len(seq) - 2)), 3))
    return np.array(tri_nt_percent)

def secondary_struct(seq):
    def _percentage(if_paired):
        paired_percent = (if_paired.count('(') + if_paired.count(')')) / len(if_paired)
        unpaired_percent = (if_paired.count('.')) / len(if_paired)
        return [paired_percent, unpaired_percent]

    paired_seq, min_free_energy = RNA.fold(seq)
    return np.array(_percentage(paired_seq)+[min_free_energy])


def create_pssm(train_seq):
    train_seq = [list(seq.upper())[:19] for seq in train_seq]
    train_seq = np.array(train_seq)

    nr, nc = np.shape(train_seq)
    pseudocount = nr ** 0.5 
    pssm = []
    for c in range(0, nc):
        col_c = train_seq[:, c].tolist()
        f_A = round(((col_c.count('A') + pseudocount) / (nr + pseudocount)), 3)
        f_G = round(((col_c.count('G') + pseudocount) / (nr + pseudocount)), 3)
        f_C = round(((col_c.count('C') + pseudocount) / (nr + pseudocount)), 3)
        f_U = round(((col_c.count('U') + pseudocount) / (nr + pseudocount)), 3)
        pssm.append([f_A, f_G, f_C, f_U])
    pssm = np.array(pssm)
    pssm = pssm.transpose()
    return pssm


def score_seq_by_pssm(pssm, seq): 
    nt_order = {'A': 0, 'G': 1, 'C': 2, 'U': 3}
    ind_all = list(range(0, 19))
    scores = [pssm[nt_order[nt], i] for nt, i in zip(seq, ind_all)]
    log_score = sum([-math.log2(i) for i in scores])
    return np.array([log_score])



def get_priors(data):
    print('create prior gc_sterch')
    data['gc_sterch'] = data['siRNA_antisense_seq'].apply(get_gc_sterch)                    # 1

    print('create prior gc_content')
    data['gc_content'] = data['siRNA_antisense_seq'].apply(get_gc_percentage)               # 1

    print('create prior gibbs_energy_result')
    data['gibbs_energy_result'] = data['siRNA_antisense_seq'].apply(gibbs_energy)           # 24

    print('create prior single_nt_percent')
    data['single_nt_percent'] = data['siRNA_antisense_seq'].apply(get_single_comp_percent)  # 4

    print('create prior di_nt_percent')
    data['di_nt_percent'] = data['siRNA_antisense_seq'].apply(get_di_comp_percent)          # 16

    print('create prior tri_nt_percent')
    data['tri_nt_percent'] = data['siRNA_antisense_seq'].apply(get_tri_comp_percent)        # 64

    print('create prior secondary_struct')
    data['secondary_struct'] = data['siRNA_antisense_seq'].apply(secondary_struct)          # 3
    
    # print('create prior pssm_score')
    # pssm = create_pssm(np.array(data['siRNA_antisense_seq']))
    # data['pssm_score'] = data['siRNA_antisense_seq'].apply(lambda seq: score_seq_by_pssm(pssm,seq)) # 1

    return data



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train SiRNA Model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training and inference")
    parser.add_argument("--epoch", type=int, default=200, help="Epoch for training")
    parser.add_argument("--clip", type=str2bool, nargs='?', const=False, default=True, help="Clip mRNA_remaining_pct to [0, 120]")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test size for train/val split")
    parser.add_argument("--random_state", type=int, default=1234, help="Random state for train/val split")
    parser.add_argument("--weight_0_30", type=float, default= 1.2, help="Loss weight for mRNA_remaining_pct < 30")
    parser.add_argument("--weight_above_30", type=float, default=1, help="Loss weight for mRNA_remaining_pct >= 30")
    parser.add_argument("--penalty_above_100", type=float, default= 0, help="Loss penalty for predictions > 100")
    parser.add_argument("--seed", type=int, default= 1234, help="Seed for reproduction")
    
    args = parser.parse_args()


    # 设置随机种子
    set_seed(args.seed)
    
    
    device = args.device
    
    print(args.clip)
    
    
    # Load data
    train_data = pd.read_csv('../data/external_data/train_data_aug3.2.csv')
    test_data = pd.read_csv('../data/external_data/sample_submission_aug3.2.csv')


    seq_columns = ['siRNA_antisense_seq','siRNA_sense_seq','modified_siRNA_antisense_seq_list', 'modified_siRNA_sense_seq_list','Extended_Sequence']
    cat_columns = ['cell_line_donor', 'Transfection_method', 'gene_target_species', 'publication_id', 'gene_target_ncbi_id']
    num_columns = ['siRNA_concentration', 'Duration_after_transfection_h']
    prior_columns = ['gc_sterch','gc_content','gibbs_energy_result','single_nt_percent','di_nt_percent','tri_nt_percent','secondary_struct']
    
    data_columns = cat_columns + num_columns

    train_data.dropna(subset=data_columns + seq_columns + ['mRNA_remaining_pct'], inplace=True)### no imputation
    
    
    if args.clip:
        train_data['mRNA_remaining_pct'] = train_data['mRNA_remaining_pct'].clip(lower=0, upper=120)
        test_data['mRNA_remaining_pct'] = test_data['mRNA_remaining_pct'].clip(lower=0, upper=120)

    cat_mapping_set = []
    cat_mapping_len = []

    for col in cat_columns:
        train_data[col], unique = pd.factorize(train_data[col])
        mapping = {value: code for code, value in enumerate(unique)}
        cat_mapping_len.append(len(mapping))
        cat_mapping_set.append(mapping)
        

        
    test_data_final = test_data.copy()


    data_combined = pd.concat([train_data, test_data], ignore_index=True)

    data_combined = get_priors(data_combined) 

    train_data = data_combined.iloc[:len(train_data)].reset_index(drop=True)
    test_data = data_combined.iloc[len(train_data):].reset_index(drop=True)

    for i, col in enumerate(cat_columns):
        test_data[col] = test_data[col].map(cat_mapping_set[i])

    train_data, val_data = train_test_split(train_data, test_size=args.test_size, random_state=args.random_state)

    tokenizer = GenomicTokenizer(ngram=3, stride=3)

    all_tokens = []
    for col in seq_columns:
        for seq in train_data[col]:
            if ' ' in seq:  # Modified sequence
                all_tokens.extend(seq.split())
            else:
                all_tokens.extend(tokenizer.tokenize(seq))
    vocab = GenomicVocab.create(all_tokens, max_vocab=10000, min_freq=1)

    max_len_sirna = max(max(len(seq.split()) if ' ' in seq else len(tokenizer.tokenize(seq))
                            for seq in train_data[col]) for col in ['siRNA_antisense_seq','siRNA_sense_seq','modified_siRNA_antisense_seq_list', 'modified_siRNA_sense_seq_list'])
    max_len_mrna = max(max(len(seq.split()) if ' ' in seq else len(tokenizer.tokenize(seq))
                           for seq in train_data[col]) for col in ['Extended_Sequence'])
    
    train_dataset = SiRNADataset(train_data, seq_columns, cat_columns, num_columns, prior_columns, vocab, tokenizer, max_len_sirna, max_len_mrna)
    val_dataset = SiRNADataset(val_data, seq_columns, cat_columns, num_columns, prior_columns, vocab, tokenizer, max_len_sirna, max_len_mrna)
    test_dataset = SiRNADataset(test_data, seq_columns, cat_columns, num_columns, prior_columns, vocab, tokenizer, max_len_sirna, max_len_mrna)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = SiRNAModel(len(vocab.itos), cat_mapping_len, num_dim=len(num_columns),device= device)
    criterion = CustomLoss(weight_0_30=args.weight_0_30, weight_above_30=args.weight_above_30, penalty_above_100=args.penalty_above_100)###new loss
    optimizer = optim.AdamW(model.parameters())

    model, score, epoch_scores = train_model(model, train_loader, val_loader, criterion, optimizer, args.epoch, device)
    
    results = test_model(model, test_loader, device)
    test_data_final['mRNA_remaining_pct'] = results
    test_data_final.to_csv(("../submit/submit_" + time.strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)
    # params = vars(args)
    # log_results(params, score, epoch_scores, filename='training_log.csv')
    
    