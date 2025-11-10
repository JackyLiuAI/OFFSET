import argparse
import os
import logging
import torch
import open_clip
import datasets
import test
import types


def build_params(batch_size: int = 16, local_rank: int = 0, max_eval_queries: int = 0, max_eval_targets: int = 0):
    p = types.SimpleNamespace()
    p.batch_size = batch_size
    p.local_rank = local_rank
    p.max_eval_queries = max_eval_queries
    p.max_eval_targets = max_eval_targets
    return p


def load_model(model_path: str, hidden_dim: int = 1024, dropout_rate: float = 0.5, P: int = 4, Q: int = 8, tau_: float = 0.1):
    # Prefer loading full pickled model; fallback to state_dict
    obj = torch.load(model_path, map_location='cuda')
    if isinstance(obj, torch.nn.Module):
        model = obj
        model.cuda()
        return model
    else:
        # Fallback: construct model and load state dict
        import model_OFFSET as model_def
        model = model_def.OFFSET(hidden_dim=hidden_dim, dropout=dropout_rate, local_token_num=Q, global_token_num=P, t=tau_)
        model.load_state_dict(obj)
        model.cuda()
        return model


def main():
    ap = argparse.ArgumentParser(description='Evaluate saved model on FashionIQ or CIRR')
    ap.add_argument('--dataset', required=True, choices=['dress', 'shirt', 'toptee', 'cirr'], help='Dataset to evaluate')
    ap.add_argument('--fashioniq_path', default='data/fashionIQ/', help='FashionIQ root path')
    ap.add_argument('--cirr_path', default='data/CIRR/', help='CIRR root path')
    ap.add_argument('--model_path', required=True, help='Path to .pt model file')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--local_rank', type=int, default=0)
    ap.add_argument('--max_eval_queries', type=int, default=0)
    ap.add_argument('--max_eval_targets', type=int, default=0)
    args = ap.parse_args()

    # Transforms
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')

    # Dataset
    if args.dataset in ['dress', 'shirt', 'toptee']:
        dset = datasets.FashionIQ_SavedSegment_all(path=args.fashioniq_path, transform=[preprocess_train, preprocess_val], split='val-split')
    elif args.dataset == 'cirr':
        dset = datasets.CIRR_SavedSegment(path=args.cirr_path, transform=[preprocess_train, preprocess_val])
    else:
        raise ValueError(f'Unsupported dataset {args.dataset}')

    # Model
    model = load_model(args.model_path)
    params = build_params(batch_size=args.batch_size, local_rank=args.local_rank, max_eval_queries=args.max_eval_queries, max_eval_targets=args.max_eval_targets)

    # Evaluate
    if args.dataset in ['dress', 'shirt', 'toptee']:
        results = test.test(params, model, dset, args.dataset)
    elif args.dataset == 'cirr':
        results = test.test_cirr_valset(params, model, dset)
    else:
        results = []

    # Print
    print('[EVAL] Metrics:')
    for k, v in results:
        print(f'{k}: {v:.4f}')


if __name__ == '__main__':
    main()