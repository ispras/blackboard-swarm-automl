import argparse, os, time, math, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

ap=argparse.ArgumentParser()
ap.add_argument("--model",default="gpt2-medium")
ap.add_argument("--train",default="train_v2.txt")
ap.add_argument("--out",default="ft_lm")
ap.add_argument("--max_bytes",type=int,default=300_000_000)
ap.add_argument("--skip_bytes",type=int,default=0)
ap.add_argument("--block",type=int,default=256)
ap.add_argument("--epochs",type=float,default=1.0)
ap.add_argument("--batch",type=int,default=16)
ap.add_argument("--accum",type=int,default=4)
ap.add_argument("--lr",type=float,default=5e-5)
ap.add_argument("--max_steps",type=int,default=-1)
args=ap.parse_args()

t0=time.time()
tok=AutoTokenizer.from_pretrained(args.model)
tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(args.model)
model.gradient_checkpointing_enable()
print("loaded",args.model,round(time.time()-t0,1),flush=True)

# stream raw text up to max_bytes
texts=[]
nbytes=0; skipped=0
with open(args.train,encoding="utf-8",errors="replace") as f:
    for line in f:
        if skipped < args.skip_bytes:
            skipped += len(line)
            continue
        texts.append(line.rstrip("\n"))
        nbytes+=len(line)
        if nbytes>=args.max_bytes: break
print("lines",len(texts),"bytes",nbytes,flush=True)

# tokenize in chunks and pack into blocks
blocks=[]
buf=[]
BL=args.block
t1=time.time()
for i,t in enumerate(texts):
    ids=tok(t,add_special_tokens=False)["input_ids"]
    ids.append(tok.eos_token_id)
    buf.extend(ids)
    while len(buf)>=BL:
        blocks.append(buf[:BL]); buf=buf[BL:]
    if i%200000==0 and i>0:
        print("tok",i,"blocks",len(blocks),"elapsed",round(time.time()-t1,1),flush=True)
print("blocks",len(blocks),flush=True)
random.seed(1); random.shuffle(blocks)
ds=Dataset.from_dict({"input_ids":blocks})

args_t=TrainingArguments(
    output_dir=args.out+"_ckpt",
    overwrite_output_dir=True,
    per_device_train_batch_size=args.batch,
    gradient_accumulation_steps=args.accum,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    logging_steps=50,
    save_strategy="no",
    report_to=[],
    fp16=True,
    dataloader_num_workers=2,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)
coll=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
trainer=Trainer(model=model,args=args_t,train_dataset=ds,data_collator=coll)
trainer.train()
os.makedirs(args.out,exist_ok=True)
model.half().save_pretrained(args.out)
tok.save_pretrained(args.out)
print("saved",args.out,"total_elapsed",round(time.time()-t0,1),flush=True)
