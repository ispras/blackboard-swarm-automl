import os, sys, time, random, json, math, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup

TRAIN = "/download_job_code/payload/sandbox/workspaces/spica_psi/train_v2.txt"
HOLDOUT_START = 29998384 - 30000  # do not train on holdout tail

class LineDS(IterableDataset):
    def __init__(self, max_lines, seed, shuffle_buf=200000):
        self.max_lines = max_lines
        self.seed = seed
        self.shuffle_buf = shuffle_buf
    def __iter__(self):
        rnd = random.Random(self.seed + torch.initial_seed() % (2**31))
        buf = []
        with open(TRAIN, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                line = line.rstrip('\n')
                if not line:
                    continue
                words = line.split()
                if len(words) < 3:
                    continue
                k = rnd.randint(1, len(words) - 2)
                target = words[k]
                mod = words[:k] + ["<mask>"] + words[k+1:]
                buf.append((" ".join(mod), target))
                if len(buf) >= self.shuffle_buf:
                    rnd.shuffle(buf)
                    for x in buf:
                        yield x
                    buf = []
        rnd.shuffle(buf)
        for x in buf:
            yield x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--max_lines", type=int, default=4000000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="roberta_mlm_ft")
    ap.add_argument("--log", default="train_log.txt")
    ap.add_argument("--steps_limit", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(dev)
    model.train()
    mask_id = tok.mask_token_id

    ds = LineDS(args.max_lines, seed=17)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=True, prefetch_factor=4, drop_last=True)

    steps_per_epoch = max(1, args.max_lines // args.bs)
    total_steps = steps_per_epoch * args.epochs
    if args.steps_limit > 0:
        total_steps = min(total_steps, args.steps_limit)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.03*total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler()

    logf = open(args.log, 'a')
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"START model={args.model} max_lines={args.max_lines} bs={args.bs} lr={args.lr} total_steps={total_steps}")
    t0 = time.time(); step = 0; run_loss = 0.0; run_n = 0
    for texts, targets in dl:
        try:
            batch = tok(texts, padding="longest", truncation=True, max_length=args.maxlen, return_tensors="pt")
        except Exception as e:
            log("TOKERR " + str(e)); continue
        input_ids = batch["input_ids"].to(dev, non_blocking=True)
        attn = batch["attention_mask"].to(dev, non_blocking=True)
        labels = torch.full_like(input_ids, -100)
        mp = (input_ids == mask_id)
        tt = tok([" " + t for t in targets], add_special_tokens=False)["input_ids"]
        tt = [x[0] if len(x) > 0 else tok.unk_token_id for x in tt]
        tt = torch.tensor(tt, device=dev)
        valid = 0
        for bi in range(input_ids.size(0)):
            pos = (input_ids[bi] == mask_id).nonzero(as_tuple=True)[0]
            if len(pos) >= 1:
                labels[bi, pos[0]] = tt[bi]
                valid += 1
        if valid == 0:
            continue
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        run_loss += loss.item(); run_n += 1
        if step % 100 == 0:
            el = time.time() - t0
            log(f"step {step}/{total_steps} loss {run_loss/run_n:.4f} lr {sched.get_last_lr()[0]:.2e} {step/el:.2f} it/s elapsed {el/60:.1f}m")
            run_loss = 0.0; run_n = 0
        if step % 2000 == 0 or step == total_steps:
            ck = os.path.join(args.out, f"ckpt_{step}")
            os.makedirs(args.out, exist_ok=True)
            model.save_pretrained(ck); tok.save_pretrained(ck)
            log(f"SAVED {ck}")
        if step >= total_steps:
            break
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(os.path.join(args.out, "final")); tok.save_pretrained(os.path.join(args.out, "final"))
    log(f"DONE step={step}")

if __name__ == "__main__":
    main()
