
import argparse
import torch
import json 
import copy 
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_embed_path",
        type=str,
    )

    parser.add_argument(
        "--test_embed_path",
        type=str,
    )
    
    parser.add_argument(
        "--test_data_path",
        type=str,
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
    )

    parser.add_argument(
        "--save_path",
        type=str,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
    )
    return parser.parse_known_args()

def get_distmat(db, features):
    m, n = features.shape[0], db.shape[0]
    distmat = torch.zeros((m,n))

    q_norm = torch.norm(features, p=2, dim=1, keepdim=True)
    g_norm = torch.norm(db, p=2, dim=1, keepdim=True)
    qf = features.div(q_norm.expand_as(features))
    gf = db.div(g_norm.expand_as(db))
    for i in range(m):
        distmat[i] = - torch.mm(features[i:i+1], db.t())
    distmat = distmat.numpy()

    return distmat

if __name__ == '__main__':
    args, _ = parse_args()

    db_embeds = np.load(args.db_embed_path)
    test_embeds = np.load(args.test_embed_path)

    distmat = get_distmat(torch.tensor(db_embeds), torch.tensor(test_embeds))
    indexs = distmat.argsort(-1)
    with open(args.test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    with open(args.train_data_path, 'r') as f:
        train_data = [json.loads(line) for line in f]
    test_info = copy.copy(test_data)

    for i, info in enumerate(test_info):
        cidxs = indexs[i][:args.top_k]
        candidates = [train_data[i] for i in cidxs]
        test_info[i]['evolution'] = candidates

    print('Saving Test Data')
    with open(args.save_path ,'w') as f:
        json.dump(test_info, f, indent=4)