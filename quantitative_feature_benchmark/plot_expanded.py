"""Plot both specified quantitative comparisons from saved results."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;F=B.parent/'figures';F.mkdir(exist_ok=True)
primary=json.loads((B/'results_primary/metrics.json').read_text());loading=json.loads((B/'results_loading_only/metrics.json').read_text())
def mae(m,k):return m['models'][k]['study_balanced_MAE']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(11,5.2),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
labels=['Median','RF categorical','RF + concentration','RF + concentration\nand times','SVR categorical','SVR + concentration','SVR + concentration\nand times']
values=[mae(primary,'weighted_median'),mae(primary,'RandomForest_categorical'),mae(loading,'RandomForest_quantitative'),mae(primary,'RandomForest_quantitative'),mae(primary,'SVR_categorical'),mae(loading,'SVR_quantitative'),mae(primary,'SVR_quantitative')]
axs[0].barh(np.arange(7),values,color=['#777777','#4575b4','#74add1','#b7d8e9','#984ea3','#be9ac6','#dbc8df'])
axs[0].set_yticks(np.arange(7),labels);axs[0].invert_yaxis();axs[0].set_xlabel('Equal-study MAE (percentage points)');axs[0].set_title('A  81 observations, 10 studies',loc='left',fontsize=11)
for i,value in enumerate(values):axs[0].text(value+.4,i,f'{value:.2f}',va='center',fontsize=9)
axs[0].set_xlim(0,max(values)*1.15)
study=list(csv.DictReader((B/'results_loading_only/study_errors.csv').open()))
names={'10.1002/fsn3.2150':'Microalgae, 2021','10.1002/jsfa.5768':'Fruit / vegetables, 2012','10.1016/j.foodchem.2012.07.035':'Carrot / fruit, 2012','10.1016/j.foodchem.2022.132595':'Squash products, 2022','10.1016/j.foodres.2014.06.016':'Dairy, 2014','10.1021/jf072751r':'Mango, 2008','10.1021/jf100413m':'Pepper, 2010','10.15835/buasvmcn-fst:2021.0004':'Egg yolk, 2021','10.3390/foods14173089':'Oils, 2025','10.3390/nu12010076':'Sea buckthorn oil, 2020'}
deltas=[float(r['RandomForest_quantitative'])-float(r['RandomForest_categorical']) for r in study]
axs[1].scatter(deltas,np.arange(len(study)),color='#2166ac',s=36);axs[1].axvline(0,color='#777777',lw=.8)
axs[1].set_yticks(np.arange(len(study)),[names[r['study_group'].removeprefix('doi:')] for r in study]);axs[1].invert_yaxis()
axs[1].set_xlabel('RF concentration-only minus categorical MAE\n(percentage points; negative favors concentration)');axs[1].set_title('B  Secondary comparison by held-out study',loc='left',fontsize=11)
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=9)
for ext in ['png','pdf','svg']:fig.savefig(F/f'Figure_S2.{ext}',dpi=350,facecolor='white',bbox_inches='tight')
plt.close(fig);print('Expanded Figure S2 saved')
