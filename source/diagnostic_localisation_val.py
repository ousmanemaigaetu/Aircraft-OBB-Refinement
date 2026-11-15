#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic géométrique OBB sur VAL uniquement.

GT DOTA par image:
  x1 y1 x2 y2 x3 y3 x4 y4 plane difficult
Prédictions fusionnées par modèle:
  image_id score x1 y1 x2 y2 x3 y3 x4 y4

Le script compare S²A-Net et YOLO26l-OBB sur les mêmes GT et produit
CSV + figures pour comprendre l'écart d'AP75 avant toute amélioration.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
try:
    from scipy.stats import spearmanr
    HAVE_SPEARMAN = True
except Exception:
    HAVE_SPEARMAN = False


def norm_id(s): return Path(str(s)).stem

def polygon_area(poly):
    return abs(float(cv2.contourArea(poly.astype(np.float32))))

def poly_iou(a, b):
    a, b = cv2.convexHull(a.astype(np.float32)), cv2.convexHull(b.astype(np.float32))
    aa, bb = polygon_area(a), polygon_area(b)
    if aa <= 0 or bb <= 0: return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
        inter = float(inter)
    except cv2.error:
        return 0.0
    union = aa + bb - inter
    return max(0.0, min(1.0, inter / union)) if union > 0 else 0.0

def rect_features(poly):
    pts = poly.astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), _ = rect
    box = cv2.boxPoints(rect).astype(np.float32)
    edges = []
    for i in range(4):
        v = box[(i+1)%4] - box[i]
        L = float(np.linalg.norm(v))
        ang = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
        edges.append((L, ang))
    major, angle = max(edges, key=lambda x: x[0])
    minor = min(x[0] for x in edges)
    area = max(major * minor, 1e-12)
    return dict(cx=float(cx), cy=float(cy), major=major, minor=minor,
                area=area, angle=angle, aspect=major/max(minor,1e-12))

def angle_error_deg(a,b):
    d = abs((a-b) % 180.0)
    return float(min(d, 180.0-d))

def read_gt_dir(gt_dir, class_name='plane'):
    out = {}
    for fp in sorted(Path(gt_dir).glob('*.txt')):
        arr=[]
        for line in fp.read_text(encoding='utf-8', errors='ignore').splitlines():
            p=line.strip().split()
            if len(p)<9 or p[0].lower().startswith('imagesource') or p[0].lower().startswith('gsd'):
                continue
            try: coords=list(map(float,p[:8]))
            except ValueError: continue
            if p[8] != class_name: continue
            difficult=0
            if len(p)>=10:
                try: difficult=int(float(p[9]))
                except ValueError: difficult=0
            arr.append({'poly':np.asarray(coords,np.float32).reshape(4,2),'difficult':difficult})
        out[norm_id(fp.name)] = arr
    if not out: raise RuntimeError(f'Aucun GT lisible dans {gt_dir}')
    return out

def read_predictions(path):
    out=defaultdict(list); bad=0
    for line in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
        p=line.strip().split()
        if not p: continue
        if len(p)<10:
            bad+=1; continue
        try:
            image_id=norm_id(p[0]); score=float(p[1]); coords=list(map(float,p[2:10]))
        except Exception:
            bad+=1; continue
        out[image_id].append({'score':score,'poly':np.asarray(coords,np.float32).reshape(4,2)})
    if not out: raise RuntimeError(f'Aucune prédiction lisible dans {path}')
    if bad: print(f'[WARN] {bad} lignes ignorées dans {path}')
    return dict(out)

def match_image(gt_items, pred_items, min_iou=0.10):
    gts=[g for g in gt_items if int(g.get('difficult',0))==0]
    preds=list(pred_items)
    if not gts or not preds: return [], gts, preds
    M=np.zeros((len(gts),len(preds)),float)
    for i,g in enumerate(gts):
        for j,p in enumerate(preds): M[i,j]=poly_iou(g['poly'],p['poly'])
    pairs=[]
    if HAVE_SCIPY:
        rr,cc=linear_sum_assignment(1.0-M)
        for i,j in zip(rr.tolist(),cc.tolist()):
            if M[i,j]>=min_iou: pairs.append((i,j,float(M[i,j])))
    else:
        cand=sorted([(float(M[i,j]),i,j) for i in range(M.shape[0]) for j in range(M.shape[1])],reverse=True)
        ug,up=set(),set()
        for iou,i,j in cand:
            if iou<min_iou: break
            if i in ug or j in up: continue
            ug.add(i); up.add(j); pairs.append((i,j,iou))
    return pairs,gts,preds

def pair_row(image_id,gi,pj,g,p,iou,model):
    gf,pf=rect_features(g['poly']),rect_features(p['poly'])
    center_px=math.hypot(pf['cx']-gf['cx'],pf['cy']-gf['cy'])
    diag=math.hypot(gf['major'],gf['minor'])
    return {
      'image_id':image_id,'gt_idx':gi,'pred_idx':pj,'model':model,'score':float(p['score']),'iou':iou,
      'gt_area':gf['area'],'gt_major':gf['major'],'gt_minor':gf['minor'],'gt_angle_deg':gf['angle'],'gt_aspect':gf['aspect'],
      'pred_area':pf['area'],'pred_major':pf['major'],'pred_minor':pf['minor'],'pred_angle_deg':pf['angle'],
      'center_err_px':center_px,'center_err_norm':center_px/max(diag,1e-12),
      'angle_err_deg':angle_error_deg(pf['angle'],gf['angle']),
      'major_rel_err':(pf['major']-gf['major'])/max(gf['major'],1e-12),
      'minor_rel_err':(pf['minor']-gf['minor'])/max(gf['minor'],1e-12),
      'area_ratio':pf['area']/max(gf['area'],1e-12),
      'aspect_ratio_ratio':pf['aspect']/max(gf['aspect'],1e-12)}

def diagnose(gt,pred,model,min_iou):
    rows=[]
    for image_id,gt_items in gt.items():
        pairs,gts,preds=match_image(gt_items,pred.get(image_id,[]),min_iou)
        for i,j,iou in pairs: rows.append(pair_row(image_id,i,j,gts[i],preds[j],iou,model))
    return pd.DataFrame(rows)

def add_bins(df,edges=None):
    if df.empty: return df,edges
    out=df.copy()
    if edges is None:
        q1,q2=np.quantile(out.gt_area,[1/3,2/3]); q2=max(q2,q1+1e-6); edges=(float(q1),float(q2))
    q1,q2=edges
    out['size_bin']=pd.cut(out.gt_area,[-np.inf,q1,q2,np.inf],labels=['small','medium','large'])
    bins=np.arange(0,181,15); labels=[f'{a:03.0f}-{b:03.0f}' for a,b in zip(bins[:-1],bins[1:])]
    out['angle_bin']=pd.cut(np.mod(out.gt_angle_deg,180.0),bins=bins,labels=labels,right=False,include_lowest=True)
    return out,edges

def summarize(df):
    rows=[]
    for model,g in df.groupby('model'):
        r={'model':model,'n_matched':len(g),'iou_mean':g.iou.mean(),'iou_median':g.iou.median(),
           'iou_ge_050':(g.iou>=.5).mean(),'iou_ge_075':(g.iou>=.75).mean(),
           'angle_err_mean':g.angle_err_deg.mean(),'angle_err_median':g.angle_err_deg.median(),
           'center_err_norm_mean':g.center_err_norm.mean(),'center_err_norm_median':g.center_err_norm.median(),
           'abs_major_rel_err_median':g.major_rel_err.abs().median(),
           'abs_minor_rel_err_median':g.minor_rel_err.abs().median()}
        if HAVE_SPEARMAN and len(g)>=3:
            rho,p=spearmanr(g.score,g.iou); r['score_iou_spearman_rho']=rho; r['score_iou_spearman_p']=p
        rows.append(r)
    return pd.DataFrame(rows)

def paired(s2,yo):
    keys=['image_id','gt_idx']; cols=keys+['iou','score','center_err_norm','angle_err_deg','major_rel_err','minor_rel_err','area_ratio','gt_area','gt_angle_deg']
    a=s2[cols].rename(columns={c:'s2_'+c for c in cols if c not in keys})
    b=yo[cols].rename(columns={c:'yolo_'+c for c in cols if c not in keys})
    p=a.merge(b,on=keys,how='inner')
    if p.empty: return p
    p['delta_iou_yolo_minus_s2']=p.yolo_iou-p.s2_iou
    p['delta_angle_err_yolo_minus_s2']=p.yolo_angle_err_deg-p.s2_angle_err_deg
    p['delta_center_err_yolo_minus_s2']=p.yolo_center_err_norm-p.s2_center_err_norm
    p['winner_iou']=np.where(p.delta_iou_yolo_minus_s2>1e-9,'YOLO',np.where(p.delta_iou_yolo_minus_s2<-1e-9,'S2','tie'))
    return p

def boxplot(df,metric,ylabel,out):
    if df.empty:return
    models=[m for m in ['S2A-Net','YOLO26l-OBB'] if m in set(df.model)]
    vals=[df.loc[df.model==m,metric].dropna().values for m in models]
    if len(vals)<2:return
    plt.figure(figsize=(7,5)); plt.boxplot(vals,labels=models,showfliers=False); plt.ylabel(ylabel); plt.title(ylabel+' sur GT appariés - VAL'); plt.tight_layout(); plt.savefig(out,dpi=180); plt.close()

def line_by_angle(df,out):
    tab=df.groupby(['angle_bin','model'],observed=True).iou.mean().unstack()
    if tab.empty:return
    plt.figure(figsize=(11,5)); x=np.arange(len(tab.index))
    for c in tab.columns: plt.plot(x,tab[c].values,marker='o',label=c)
    plt.xticks(x,tab.index.astype(str),rotation=45,ha='right'); plt.ylabel('IoU moyen'); plt.xlabel('Orientation GT du grand axe (°)'); plt.title('IoU selon orientation GT - VAL'); plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=180); plt.close()

def bar_by_size(df,out):
    tab=df.groupby(['size_bin','model'],observed=True).iou.mean().unstack()
    if tab.empty:return
    ax=tab.plot(kind='bar',figsize=(7,5)); ax.set_ylabel('IoU moyen'); ax.set_xlabel('Taille GT'); ax.set_title('IoU selon taille - VAL'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(out,dpi=180); plt.close()

def delta_hist(p,out):
    if p.empty:return
    plt.figure(figsize=(8,5)); plt.hist(p.delta_iou_yolo_minus_s2,bins=35); plt.axvline(0,linewidth=1); plt.xlabel('Δ IoU = YOLO - S²A-Net'); plt.ylabel('Nombre de GT'); plt.title('Comparaison appariée sur les mêmes avions - VAL'); plt.tight_layout(); plt.savefig(out,dpi=180); plt.close()

def score_iou(df,out):
    plt.figure(figsize=(7,5))
    for m,g in df.groupby('model'): plt.scatter(g.score,g.iou,s=12,alpha=.45,label=m)
    plt.xlabel('Score de confiance'); plt.ylabel('IoU avec GT apparié'); plt.title('Score vs qualité de localisation - VAL'); plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=180); plt.close()

def guard(paths,allow_test):
    if allow_test:return
    bad=[]
    for p in paths:
        parts=[x for x in str(p).lower().replace('\\','/').split('/') if x]
        if any(x=='test' or x.startswith('test_') or x.endswith('_test') for x in parts): bad.append(str(p))
    if bad: raise SystemExit('[STOP] Utiliser VAL pour le diagnostic, pas TEST:\n  - '+'\n  - '.join(bad))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gt-dir',type=Path,required=True)
    ap.add_argument('--s2-pred',type=Path,required=True)
    ap.add_argument('--yolo-pred',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--class-name',default='plane')
    ap.add_argument('--match-iou',type=float,default=.10)
    ap.add_argument('--allow-test',action='store_true')
    a=ap.parse_args(); guard([a.gt_dir,a.s2_pred,a.yolo_pred],a.allow_test); a.out_dir.mkdir(parents=True,exist_ok=True)
    gt=read_gt_dir(a.gt_dir,a.class_name); s2=diagnose(gt,read_predictions(a.s2_pred),'S2A-Net',a.match_iou); yo=diagnose(gt,read_predictions(a.yolo_pred),'YOLO26l-OBB',a.match_iou)
    both=pd.concat([s2[['gt_area']],yo[['gt_area']]],ignore_index=True)
    edges=None
    if len(both):
        q1,q2=np.quantile(both.gt_area,[1/3,2/3]); edges=(float(q1),float(max(q2,q1+1e-6)))
    s2,_=add_bins(s2,edges); yo,_=add_bins(yo,edges); all_df=pd.concat([s2,yo],ignore_index=True)
    p=paired(s2,yo); summary=summarize(all_df)
    s2.to_csv(a.out_dir/'matches_s2.csv',index=False); yo.to_csv(a.out_dir/'matches_yolo.csv',index=False); all_df.to_csv(a.out_dir/'matches_all.csv',index=False); p.to_csv(a.out_dir/'paired_same_gt.csv',index=False); summary.to_csv(a.out_dir/'summary_metrics.csv',index=False)
    boxplot(all_df,'iou','IoU',a.out_dir/'01_iou_boxplot.png'); boxplot(all_df,'angle_err_deg','Erreur angle (°)',a.out_dir/'02_angle_error_boxplot.png'); boxplot(all_df,'center_err_norm','Erreur centre normalisée',a.out_dir/'03_center_error_boxplot.png'); line_by_angle(all_df,a.out_dir/'04_iou_by_gt_angle.png'); bar_by_size(all_df,a.out_dir/'05_iou_by_size.png'); delta_hist(p,a.out_dir/'06_paired_delta_iou.png'); score_iou(all_df,a.out_dir/'07_score_vs_iou.png')
    n_gt=sum(1 for items in gt.values() for g in items if int(g.get('difficult',0))==0)
    lines=['DIAGNOSTIC OBB - VAL','='*60,f'Images GT: {len(gt)}',f'GT utiles: {n_gt}',f'Appariements S2A-Net (IoU >= {a.match_iou:.2f}): {len(s2)}',f'Appariements YOLO26l-OBB (IoU >= {a.match_iou:.2f}): {len(yo)}',f'GT appariés par les deux modèles: {len(p)}','',summary.to_string(index=False)]
    if not p.empty:
        lines += ['', 'Gagnant IoU sur mêmes GT:', p.winner_iou.value_counts().to_string(), '', f"ΔIoU médian YOLO-S2: {p.delta_iou_yolo_minus_s2.median():.4f}", f"Δ erreur angle médiane YOLO-S2: {p.delta_angle_err_yolo_minus_s2.median():.3f}°", f"Δ erreur centre médiane YOLO-S2: {p.delta_center_err_yolo_minus_s2.median():.5f}"]
    report='\n'.join(lines); (a.out_dir/'diagnostic_summary.txt').write_text(report,encoding='utf-8'); print(report); print('\n[OK]',a.out_dir)
if __name__=='__main__': main()
