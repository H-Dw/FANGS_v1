#!/usr/bin/env python3
"""Plot AF3 CDR distributions from actual per-sample extraction TSVs."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,required=True)
    args=p.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
    for file,label,suffix in [('af3_results.tsv','All evaluated samples','all_samples'),
                              ('af3_best_ptm.tsv','Highest pTM sample per target','best_ptm')]:
        with (args.analysis_dir/file).open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
        values=[[float(r[f'CDR{i}_cRMSD']) for r in rows if r.get(f'CDR{i}_cRMSD','')!=''] for i in (1,2,3)]
        if not any(values): raise ValueError(f'No CDR values in {file}')
        fig,ax=plt.subplots(figsize=(6.4,4.5))
        box=ax.boxplot(values,patch_artist=True,widths=.45,showfliers=True,
                       medianprops={'color':'#222222','linewidth':1.3},
                       flierprops={'marker':'.','markersize':3,'markeredgecolor':'#666666','alpha':.45})
        for patch in box['boxes']: patch.set(facecolor='#2C73B9',linewidth=.8)
        ax.set_xticks([1,2,3],[f'CDR{i}\nn = {len(v):,}' for i,v in zip((1,2,3),values)])
        ax.set_ylabel('Cα RMSD (Å)')
        ax.set_title(f'AlphaFold 3 · {label}',loc='left',pad=18)
        ax.set_ylim(bottom=0);ax.yaxis.grid(True,color='#eeeeee',linewidth=.6);ax.set_axisbelow(True)
        fig.text(.12,.02,'Independent CDR superposition · whiskers: 1.5 × IQR · outliers retained',fontsize=8,color='#555555')
        fig.tight_layout(rect=(0,.06,1,1))
        for ext in ('png','pdf'): fig.savefig(args.analysis_dir/f'cdr_rmsd_{suffix}.{ext}')
        plt.close(fig)
        print(suffix,[(len(v),float(np.median(v))) for v in values])

if __name__=='__main__': main()
