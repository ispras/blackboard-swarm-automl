import os, sys, cv2, pandas as pd, numpy as np
from multiprocessing import Pool

ROOT="/download_job_code/payload/sandbox"
OUT=f"{ROOT}/workspaces/castor_alpha/frames"
FPS=59.94
OFF=float(os.environ.get("FRAME_OFFSET","290"))
FORCE=os.environ.get("FORCE","0")=="1"

def steps_for_play(gp, which):
    if which=="train":
        return None
    return None

def build_tasks():
    lab=pd.read_csv(f"{ROOT}/data/train_labels.csv",usecols=["game_play","step"])
    lsteps=lab.groupby("game_play").step.unique().to_dict()
    ss=pd.read_csv(f"{ROOT}/data/sample_submission.csv",usecols=["contact_id"])
    parts=ss.contact_id.str.split("_",n=4,expand=True)
    ss["game_play"]=parts[0]+"_"+parts[1]; ss["step"]=parts[2].astype(int)
    tsteps=ss.groupby("game_play").step.unique().to_dict()
    tasks=[]
    for gp,steps in lsteps.items():
        for view in ["Sideline","Endzone"]:
            tasks.append((gp,view,np.asarray(steps),"train"))
    for gp,steps in tsteps.items():
        for view in ["Sideline","Endzone"]:
            tasks.append((gp,view,np.asarray(steps),"test"))
    return tasks

def worker(task):
    gp,view,steps,which=task
    outdir=f"{OUT}/{which}"; os.makedirs(outdir,exist_ok=True)
    vpath=f"{ROOT}/data/{which}/{gp}_{view}.mp4"
    todo=[]
    for s in steps:
        fr=int(round(OFF+float(s)*FPS/10.0))
        path=f"{outdir}/{gp}_{view}_{int(s):04d}.jpg"
        if FORCE or not os.path.exists(path): todo.append((int(s),fr,path))
    if not todo: return (gp,view,0,0)
    if not os.path.exists(vpath): return (gp,view,-1,0)
    need={fr for _,fr,_ in todo}
    maxfr=max(need)
    cap=cv2.VideoCapture(vpath); n_saved=0; idx=0
    while idx<=maxfr:
        ok,frame=cap.read()
        if not ok: break
        if idx in need:
            for s,fr,path in todo:
                if fr==idx:
                    cv2.imwrite(path,frame,[cv2.IMWRITE_JPEG_QUALITY,90]); n_saved+=1
        idx+=1
    cap.release()
    return (gp,view,n_saved,len(todo))

if __name__=="__main__":
    tasks=build_tasks()
    print("tasks",len(tasks),flush=True)
    maxp=int(sys.argv[1]) if len(sys.argv)>1 else None
    if maxp: tasks=tasks[:maxp]
    with Pool(4) as p:
        for i,r in enumerate(p.imap_unordered(worker,tasks,chunksize=1)):
            if i%20==0 or r[2]!=r[3]:
                print(i,r,flush=True)
    print("DONE",flush=True)
