import sys, os, time, argparse
sys.path.append('/usr/local/lib/python3.11/site-packages')
import numpy as np, pandas as pd, torch, cv2
from tokenizers import Tokenizer
from model import Model

cv2.setNumThreads(0)
WORK = '/download_job_code/payload/sandbox/workspaces/rigel_mu'
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location='cpu')
    a = ck.get('args', {})
    m = Model(369, enc_name=a.get('enc', 'convnext_tiny.fb_in22k_ft_in1k'),
              out_index=a.get('out_index', 2), d_model=a.get('d_model', 512),
              nhead=a.get('nhead', 8), layers=a.get('layers', 6),
              dim_ff=a.get('dim_ff', 2048), dropout=a.get('dropout', 0.1),
              max_len=320, pretrained=False)
    m.load_state_dict(ck['model'])
    m.to(device).eval()
    return m, a


def make_img(path, size):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return np.transpose(img, (2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ids', required=True, help='csv with image_id column')
    ap.add_argument('--imgroot', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max_len', type=int, default=320)
    ap.add_argument('--fast', type=int, default=1)
    ap.add_argument('--flush_batches', type=int, default=40)
    args = ap.parse_args()

    device = torch.device('cuda')
    torch.set_num_threads(4)
    tok = Tokenizer.from_file(f'{WORK}/bpe1024.json')
    BOS, EOS = tok.token_to_id('[BOS]'), tok.token_to_id('[EOS]')
    model, a = load_model(args.ckpt, device)
    size = a.get('img_size', 384)

    df = pd.read_csv(args.ids)
    df = df.iloc[args.shard::args.nshard].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit]
    ids = df.image_id.tolist()
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(ids), args.bs):
            chunk = ids[k:k + args.bs]
            imgs = [make_img(f'{args.imgroot}/{i[0]}/{i[1]}/{i[2]}/{i}.png', size) for i in chunk]
            x = torch.from_numpy(np.stack(imgs)).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if args.fast:
                    outs = model.generate_fast(x, BOS, EOS, max_len=args.max_len)
                else:
                    outs = model.generate(x, BOS, EOS, max_len=args.max_len)
            for i, o in zip(chunk, outs):
                rows.append((i, tok.decode(o)))
            if k and (k // args.bs) % 20 == 0:
                el = time.time() - t0
                print(f'{k}/{len(ids)} {k/el:.1f} img/s', flush=True)
            if args.flush_batches and k and (k // args.bs) % args.flush_batches == 0:
                tmp = args.out + '.tmp'
                pd.DataFrame(rows, columns=['image_id', 'InChI']).to_csv(tmp, index=False)
                os.replace(tmp, args.out)
                print(f'flush {len(rows)} -> {args.out}', flush=True)
    tmp = args.out + '.tmp'
    pd.DataFrame(rows, columns=['image_id', 'InChI']).to_csv(tmp, index=False)
    os.replace(tmp, args.out)
    el = time.time() - t0
    print(f'DONE {len(rows)} in {el:.1f}s {len(rows)/el:.1f} img/s -> {args.out}')


if __name__ == '__main__':
    main()
