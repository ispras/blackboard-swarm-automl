import os, json
import numpy as np, pandas as pd
ROOT='/download_job_code/payload/sandbox'; OUT=os.path.join(ROOT,'final_solution')
MEMBERS = {
 'bellatrix_efficientnet_b5_512':('workspaces/bellatrix_delta/run_b5_512/tta4/val_probs_tta4.npy','workspaces/bellatrix_delta/run_b5_512/tta4/test_probs_tta4.npy','workspaces/bellatrix_delta/run_b5_512/tta4/test_probs_ft_tta4.npy'),
 'bellatrix_efficientnet_b5_640':('workspaces/bellatrix_delta/run_b5_640/tta4ms/val_probs_tta4.npy','workspaces/bellatrix_delta/run_b5_640/tta4ms/test_probs_tta4.npy','workspaces/bellatrix_delta/run_b5_640/tta4ms/test_probs_ft_tta4.npy'),
 'bellatrix_convnext_base_384':('workspaces/bellatrix_delta/run_cnxt_384/tta4ms/val_probs_tta4.npy','workspaces/bellatrix_delta/run_cnxt_384/tta4ms/test_probs_tta4.npy','workspaces/bellatrix_delta/run_cnxt_384/tta4ms/test_probs_ft_tta4.npy'),
 'vega_eva02_base_448':('workspaces/vega_eta/runs/eva02b448_f0_s42/val_probs_tta.npy','workspaces/vega_eta/runs/eva02b448_f0_s42/test_probs.npy','workspaces/vega_eta/runs/eva02b448_f0_s42/full/test_probs.npy'),
 'vega_caformer_b36_384':('workspaces/vega_eta/runs/caformer384_f0_s42/val_probs_tta.npy','workspaces/vega_eta/runs/caformer384_f0_s42/test_probs.npy','workspaces/vega_eta/runs/caformer384_f0_s42/full/test_probs.npy'),
 'aldebaran_swin_base_384':('workspaces/aldebaran_epsilon/swin/tta4ms/val_probs_tta4ms.npy','workspaces/aldebaran_epsilon/swin/tta4ms/test_probs_tta4ms.npy','workspaces/aldebaran_epsilon/swin_ft/tta4ms/test_probs_ft_tta4ms.npy'),
 'aldebaran_vit_base_384':('workspaces/aldebaran_epsilon/vit/tta4/val_probs_tta4.npy','workspaces/aldebaran_epsilon/vit/tta4/test_probs_tta4.npy','workspaces/aldebaran_epsilon/vit_ft/tta4/test_probs_tta4.npy'),
 'vega_convnextv2_base_384':('workspaces/vega_eta/runs/cnv2b384_f0_s42/val_probs_tta.npy','workspaces/vega_eta/runs/cnv2b384_f0_s42/test_probs.npy','workspaces/vega_eta/runs/cnv2b384_f0_s42/full/test_probs.npy'),
}
def norm(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-9,None); return p/p.sum(1,keepdims=True)
def main():
    cv=pd.read_csv(os.path.join(ROOT,'workspaces/aldebaran_epsilon/cv_split.csv'))
    va=cv[cv.fold==0].reset_index(drop=True); y=va.label.values
    sample=pd.read_csv(os.path.join(ROOT,'data/sample_submission.csv'))
    V=[];T=[];meta={};names=[]
    for name,(vp,tp,ftp) in MEMBERS.items():
        v=norm(np.load(os.path.join(ROOT,vp))); h=norm(np.load(os.path.join(ROOT,tp))); f=norm(np.load(os.path.join(ROOT,ftp)))
        assert len(v)==len(va) and len(h)==len(sample) and len(f)==len(sample), name
        acc=float((v.argmax(1)==y).mean()); agree=float((h.argmax(1)==f.argmax(1)).mean())
        V.append(v); T.append(0.5*h+0.5*f); names.append(name)
        meta[name]={'val_acc':acc,'ft_used':True,'honest_vs_ft_test_agreement':agree}
        print(f'{name:34s} val={acc:.5f} agree={agree:.4f}')
    ensv=np.mean(V,0); enst=np.mean(T,0); fold0=float((ensv.argmax(1)==y).mean())
    print('UNIFORM',len(names),'MODEL HEDGED fold0',fold0)
    sub=pd.DataFrame({'image_id':sample.image_id,'label':enst.argmax(1)})
    sub.to_csv(os.path.join(OUT,'submission.csv'),index=False)
    np.save(os.path.join(OUT,'ens_val_probs.npy'),ensv); np.save(os.path.join(OUT,'ens_test_probs.npy'),enst)
    out={'n_models':len(names),'members':meta,'fold0_accuracy_uniform_hedged':fold0,'weights':'uniform',
         'test_label_distribution':{int(k):int(v) for k,v in sub.label.value_counts().sort_index().items()}}
    json.dump(out,open(os.path.join(OUT,'ensemble_meta.json'),'w'),indent=2)
    print('dist',out['test_label_distribution'])
if __name__=='__main__': main()
