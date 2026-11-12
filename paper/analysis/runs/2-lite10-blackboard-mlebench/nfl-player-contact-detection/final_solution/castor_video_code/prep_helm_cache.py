import pandas as pd, os
R="/download_job_code/payload/sandbox"; W=f"{R}/workspaces/castor_alpha"
for split in ["train","test"]:
    out=f"{W}/helm_cache/{split}"; os.makedirs(out,exist_ok=True)
    h=pd.read_csv(f"{R}/data/{split}_baseline_helmets.csv",usecols=["game_play","view","frame","nfl_player_id","left","top","width","height"])
    h=h[h.view.isin(["Sideline","Endzone"])]
    n=0
    for gp,g in h.groupby("game_play",sort=False):
        g[["view","frame","nfl_player_id","left","top","width","height"]].to_pickle(f"{out}/{gp}.pkl"); n+=1
    print(split,"plays",n,flush=True)
print("DONE")
