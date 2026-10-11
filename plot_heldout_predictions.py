"""Draw primary out-of-fold predictions. No model fitting or outcome-based selection."""
from pathlib import Path
import json,csv,hashlib
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;B=D.parent;S=D/'figures';Q=D/'checks';Q.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((D/'saved_results_core/heldout_predictions.csv').open()));rows=[r for r in rows if r['scheme']=='category_study_purged'];cohort={r['row_id']:r for r in json.loads((D/'COHORT.json').read_text())['rows'] if r['cohort']=='core'}
summary=json.loads((D/'saved_results_core/RESULT_SUMMARY.json').read_text())['summary']
models={'RandomForest':'Random forest','SVR':'Support vector regression','training_median':'Training median'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(8.4,3.35),sharex=True,sharey=True);proof=[]
for ax,(m,label),letter in zip(axes,models.items(),'ABC'):
 r=[x for x in rows if x['model']==m];assert len(r)==450 and {x['row_id'] for x in r}==set(cohort)
 assert all(float(x['target'])==cohort[x['row_id']]['target'] for x in r)
 y=np.array([float(x['target']) for x in r]);p=np.array([float(x['prediction']) for x in r]);groups=defaultdict(list)
 for x in r:groups[x['study_group']].append(abs(float(x['prediction'])-float(x['target'])))
 mae=np.mean([np.mean(x) for x in groups.values()]);r2=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
 expected=next(x for x in summary if x['scheme']=='category_study_purged' and x['model']==m)
 assert abs(mae-expected['study_MAE'])<1e-10 and abs(r2-expected['R2'])<1e-10
 assert np.all((y>=-5)&(y<=115)&(p>=-5)&(p<=115))
 ax.plot([0,100],[0,100],color='#666666',lw=1,linestyle='--',zorder=1)
 ax.scatter(y,p,s=12,alpha=.43,c='#26779c' if m!='training_median' else '#565656',edgecolors='none',zorder=2)
 ax.set(xlim=(-5,115),ylim=(-5,115),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100]);ax.set_aspect('equal');ax.grid(alpha=.12)
 ax.set_title(letter+'  '+label,loc='left',fontweight='bold',fontsize=10)
 ax.set_xlabel('Observed (%)');ax.text(.04,.96,f'Study MAE = {mae:.2f} pp\nR² = {r2:.3f}',ha='left',va='top',transform=ax.transAxes,fontsize=9)
 proof.append({'model':m,'rows':450,'studies':len(groups),'independent_study_MAE':mae,'independent_R2':r2,'all_targets_match_cohort':True})
axes[0].set_ylabel('Predicted (%)');fig.subplots_adjust(left=.075,right=.99,top=.87,bottom=.2,wspace=.22)
for ext in ['png','pdf','svg']:fig.savefig(S/f'Figure_S1.{ext}',dpi=400,facecolor='white',bbox_inches='tight')
plt.close(fig)
(Q/'PREDICTION_PLOT_QA.json').write_text(json.dumps({'status':'complete','scheme':'category_study_purged','prediction_file_sha256':hashlib.sha256((D/'saved_results_core/heldout_predictions.csv').read_bytes()).hexdigest(),'models':proof,'points_are_observations_not_independent_studies':True,'identity_line_only_no_fitted_calibration_line':True},indent=2));print(proof)
