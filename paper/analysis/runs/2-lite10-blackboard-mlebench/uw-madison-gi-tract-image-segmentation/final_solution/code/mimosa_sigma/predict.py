import os, sys, json, argparse, time
import numpy as np, torch
import segmentation_models_pytorch as smp
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/mimosa_sigma/code')
from train import GIDataset
C = '/download_job_code/payload/sandbox/workspaces/mimosa_sigma/cache'
M = '/download_job_code/payload/sandbox/workspaces/mimosa_sigma/models'

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', default='pred')
    p.add_argument('--splits', default='val,test')
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--tta', type=int, default=1)
    return p.parse_args()

def predict(model, ds, dev, bs, workers, tta):
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    out = []
    with torch.no_grad():
        for x, _, _ in dl:
            x = x.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast():
                p = torch.sigmoid(model(x)).float()
                if tta >= 1:
                    p2 = torch.sigmoid(model(torch.flip(x, dims=[3]))).float()
                    p = (p + torch.flip(p2, dims=[3])) * 0.5
            out.append(p.cpu().numpy().astype(np.float16))
    return np.concatenate(out)

def main():
    a = get_args()
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    ck = torch.load(a.ckpt, map_location=dev)
    args = ck['args']
    model = getattr(smp, args['arch'])(encoder_name=args['encoder'], encoder_weights=None,
                                       in_channels=args['nc'], classes=3).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    print('loaded', a.ckpt, 'ep', ck.get('ep'), 'dice', ck.get('dice'), flush=True)
    foldarr = np.load(f'{C}/train_fold.npy')
    va = np.where(foldarr == args['fold'])[0]
    if 'val' in a.splits:
        ds = GIDataset('train', va, train=False, nc=args['nc'], size=args['size'])
        p = predict(model, ds, dev, a.bs, a.workers, a.tta)
        np.save(f'{M}/{a.tag}_valprobs.npy', p)
        print('val', p.shape, flush=True)
    if 'test' in a.splits:
        n = len(json.load(open(f'{C}/test_ids.json')))
        ds = GIDataset('test', np.arange(n), train=False, nc=args['nc'], size=args['size'])
        p = predict(model, ds, dev, a.bs, a.workers, a.tta)
        np.save(f'{M}/{a.tag}_testprobs.npy', p)
        print('test', p.shape, flush=True)

if __name__ == '__main__':
    main()
