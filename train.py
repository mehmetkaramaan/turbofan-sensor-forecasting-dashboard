import matplotlib
matplotlib.use("Agg")

import os, json, time, hashlib, platform, warnings
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from IPython.display import display, Markdown, HTML
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
import plotly.graph_objects as go
import requests
warnings.filterwarnings('ignore',message='X does not have valid feature names.*')
SEED=42
np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
ROOT=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
OUT=ROOT/'turbofan_outputs'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','axes.grid':True,'grid.alpha':.15,'font.size':10})
COLORS=['#167D9A','#EF8354','#5B5F97','#2C9B75','#BD5B80','#C89424','#5274B5','#7D9751','#934FC0','#D45E65','#35495E']
def finish(name):
    plt.tight_layout(); plt.savefig(OUT/(name+'.png'),bbox_inches='tight'); plt.show(); plt.close()
print('Python:',platform.python_version(),'TensorFlow:',tf.__version__)

REV='4d14a0bf6248235bce31cbdacb297e03e0f75d52'
BASE=f'https://huggingface.co/datasets/SoyVitou/NASA-C-MAPSS-Turbofan-Engine/resolve/{REV}/processed/FD001'
DATA=ROOT/'fd001_data'; DATA.mkdir(exist_ok=True)
checksums={}
for split in ['train','test']:
    path=DATA/f'{split}.parquet'
    if not path.exists():
        response=requests.get(f'{BASE}/{split}.parquet',timeout=90); response.raise_for_status()
        path.write_bytes(response.content)
    checksums[split]=hashlib.sha256(path.read_bytes()).hexdigest()
raw=pd.read_parquet(DATA/'train.parquet').sort_values(['unit_id','cycle']).reset_index(drop=True)
test=pd.read_parquet(DATA/'test.parquet').sort_values(['unit_id','cycle']).reset_index(drop=True)
for frame in [raw,test]:
    assert not frame.duplicated(['unit_id','cycle']).any()
    assert frame.groupby('unit_id').cycle.apply(lambda s: s.diff().dropna().eq(1).all()).all()
summary=pd.DataFrame([{'Partition':label,'Rows':len(d),'Engines':d.unit_id.nunique(),
 'Shortest trajectory':d.groupby('unit_id').size().min(),'Longest trajectory':d.groupby('unit_id').size().max()}
 for label,d in [('Official training',raw),('Official test',test)]])
display(summary); display(raw.head())

fig,ax=plt.subplots(figsize=(14,4)); ax.set_xlim(0,14); ax.set_ylim(0,4); ax.axis('off')
parts=[('Inlet','T2 / P2'),('Fan','Nf / NRf'),('LPC','T24'),('HPC','T30 / P30'),('Combustor','Fuel addition'),('HPT','W31'),('LPT','T50 / W32')]
for i,(part,sensors) in enumerate(parts):
    x=.2+i*1.96
    ax.add_patch(FancyBboxPatch((x,1.2),1.65,1.4,boxstyle='round,pad=.08',facecolor=COLORS[i%len(COLORS)],alpha=.9))
    ax.text(x+.825,2.1,part,ha='center',color='white',weight='bold'); ax.text(x+.825,1.55,sensors,ha='center',color='white',fontsize=9)
    if i<6: ax.annotate('',(x+1.94,1.9),(x+1.69,1.9),arrowprops={'arrowstyle':'->','lw':2})
ax.text(7,3.35,'Conceptual turbofan core flow and selected sensor locations',ha='center',size=17,weight='bold')
ax.text(7,.45,'Forecast target: T50 at cycle t+1  |  Inputs: observed sensors/settings through cycle t',ha='center',size=12)
finish('01_engine_sensor_map')

sensor_names=['T2','T24','T30','T50','P2','P15','P30','Nf','Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']
meaning=['Fan inlet temperature','LPC outlet temperature','HPC outlet temperature','LPT outlet temperature',
'Fan inlet pressure','Bypass duct pressure','HPC outlet pressure','Physical fan speed','Physical core speed',
'Engine pressure ratio','HPC outlet static pressure','Fuel-flow / Ps30 ratio','Corrected fan speed',
'Corrected core speed','Bypass ratio','Burner fuel-air ratio','Bleed enthalpy','Demanded fan speed',
'Demanded corrected fan speed','HPT coolant bleed','LPT coolant bleed']
display(pd.DataFrame({'Channel':sensor_names,'Meaning':meaning,'Training standard deviation':raw[sensor_names].std().values}))
fig,axs=plt.subplots(2,3,figsize=(15,8))
for ax,col in zip(axs.flat,['T50','T30','P30','Nc','BPR','W32']):
    for uid in [1,10,25,50,75]:
        d=raw[raw.unit_id==uid]; ax.plot(d.cycle,d[col],alpha=.65,lw=1,label=f'Engine {uid}')
    ax.set(title=col,xlabel='Operational cycle',ylabel='Source sensor units')
axs[0,0].legend(fontsize=8); fig.suptitle('Observed training-engine sensor trajectories',size=17)
finish('02_training_trajectories')
fig,axs=plt.subplots(1,2,figsize=(13,4))
axs[0].hist(raw.groupby('unit_id').size(),bins=16,color=COLORS[0]); axs[0].set(title='Training trajectory lengths',xlabel='Cycles per engine',ylabel='Number of engines')
axs[1].barh(sensor_names,raw[sensor_names].std(),color=COLORS[1]); axs[1].set(xscale='symlog',title='Sensor variability (original units)',xlabel='Standard deviation; units differ')
finish('03_dataset_profile')

TARGET='T50'; WINDOW=30
ids=np.sort(raw.unit_id.unique()); rng=np.random.default_rng(SEED)
val_ids=np.sort(rng.choice(ids,size=20,replace=False)); train_ids=np.setdiff1d(ids,val_ids)
train=raw[raw.unit_id.isin(train_ids)].copy(); valid=raw[raw.unit_id.isin(val_ids)].copy()
candidates=['setting_1','setting_2','setting_3']+sensor_names
features=[c for c in candidates if train[c].nunique()>1]
assert TARGET in features and 'RUL' not in features and set(train_ids).isdisjoint(val_ids)
scaler=StandardScaler().fit(train[features]); yscale=StandardScaler().fit(train[[TARGET]].to_numpy())
def windows(frame):
    xx=[]; yy=[]; meta=[]
    for uid,d in frame.groupby('unit_id',sort=True):
        z=scaler.transform(d[features]).astype('float32'); target=d[TARGET].to_numpy()
        cycles=d.cycle.to_numpy()
        for j in range(WINDOW,len(d)):
            xx.append(z[j-WINDOW:j]); yy.append(target[j])
            meta.append((int(uid),int(cycles[j]),float(target[j-1])))
    return np.asarray(xx),np.asarray(yy),pd.DataFrame(meta,columns=['engine','target_cycle','last_T50'])
Xtr,ytr,mtr=windows(train); Xva,yva,mva=windows(valid); Xte,yte,mte=windows(test)
Ftr=Xtr.reshape(len(Xtr),-1); Fva=Xva.reshape(len(Xva),-1); Fte=Xte.reshape(len(Xte),-1)
ytr_z=yscale.transform(ytr[:,None]).ravel(); yva_z=yscale.transform(yva[:,None]).ravel()
assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
print('Retained input channels:',features)
display(pd.DataFrame({'Partition':['Train','Validation','Official test'],'Windows':[len(ytr),len(yva),len(yte)],'Engines':[len(train_ids),len(val_ids),test.unit_id.nunique()]}))
fig,axs=plt.subplots(1,2,figsize=(14,4))
axs[0].barh(['Train engines','Validation engines','Official test engines'],[80,20,test.unit_id.nunique()],color=COLORS[:3]); axs[0].set(title='Whole-engine partitions',xlabel='Number of engines')
im=axs[1].imshow(Xtr[0].T,aspect='auto',cmap='coolwarm',vmin=-3,vmax=3)
axs[1].set(title='One input window: only observed history',xlabel='Window position (oldest → latest)',ylabel='Input channel')
axs[1].set_yticks(range(len(features)),features,fontsize=7); fig.colorbar(im,ax=axs[1],label='Training-standardized value')
finish('04_split_and_window')
fig,ax=plt.subplots(figsize=(10,8)); corr=train[features].corr()
im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1); ax.set_xticks(range(len(features)),features,rotation=90); ax.set_yticks(range(len(features)),features)
ax.set_title('Training-only channel correlations'); fig.colorbar(im,ax=ax,label='Pearson correlation'); finish('05_correlations')
                                                                                 
d=train[train.unit_id==train_ids[0]].reset_index(drop=True)
end=70; start=end-WINDOW; observed=d.iloc[start:end]; target=d.iloc[end]
fig,ax=plt.subplots(figsize=(13,4))
ax.plot(observed.cycle,observed.T50,'o-',color=COLORS[0],label='Input: previous 30 observed cycles')
ax.scatter([target.cycle],[target.T50],s=120,marker='*',color=COLORS[1],label='Target: next cycle',zorder=5)
ax.axvline(observed.cycle.iloc[-1]+.5,color='black',ls='--',label='Information boundary at forecast time')
ax.axvspan(observed.cycle.iloc[0]-.5,observed.cycle.iloc[-1]+.5,color=COLORS[0],alpha=.08)
ax.set(title='Sliding window: history to next-cycle temperature',xlabel='Operational cycle',ylabel='T50 (°R)')
ax.legend(loc='upper left',fontsize=9); finish('06_window_target_alignment')
fig,axs=plt.subplots(1,2,figsize=(13,4))
axs[0].plot(d.cycle,d.T50,color=COLORS[0],alpha=.5,lw=1,label='Measurement')
axs[0].plot(d.cycle,d.T50.rolling(15,min_periods=15).mean(),color=COLORS[1],lw=2,label='Trailing 15-cycle mean')
axs[0].set(title='Training-engine temperature behavior',xlabel='Operational cycle',ylabel='T50 (°R)'); axs[0].legend(fontsize=8)
lags=np.arange(1,31); acf=[d.T50.autocorr(lag=int(k)) for k in lags]
axs[1].stem(lags,acf); axs[1].axhline(0,color='black',lw=1)
axs[1].set(title='Within-engine lag dependence',xlabel='Lag (cycles)',ylabel='Autocorrelation')
finish('07_temperature_temporal_structure')

models={}; predictions={}; val_predictions={}; rows=[]; histories={}
def metrics(y,p):
    return {'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p)}
base_rmse=metrics(yte,mte.last_T50.values)['RMSE']
def record(name,p,pv,fit_s,predict_s):
    p=np.asarray(p).ravel(); pv=np.asarray(pv).ravel()
    assert len(p)==len(yte) and np.isfinite(p).all()
    predictions[name]=p; val_predictions[name]=pv
    r={'Model':name,**metrics(yte,p),'Validation RMSE':metrics(yva,pv)['RMSE'],'Fit seconds':fit_s,
       'Prediction ms/sample':1000*predict_s/len(p),'Persistence skill':1-metrics(yte,p)['RMSE']/base_rmse,
       'Macro-engine MAE':pd.DataFrame({'engine':mte.engine,'error':np.abs(yte-p)}).groupby('engine').error.mean().mean()}
    rows.append(r); display(pd.DataFrame([r]).round(4))
def diagram(name,steps):
    fig,ax=plt.subplots(figsize=(13,2.1)); ax.axis('off')
    for i,step in enumerate(steps):
        x=.02+i*.245
        ax.text(x+.105,.5,step,ha='center',va='center',transform=ax.transAxes,
          bbox={'boxstyle':'round,pad=.6','facecolor':COLORS[i],'edgecolor':'none'},color='white',fontsize=10)
        if i<3: ax.annotate('',xy=(x+.235,.5),xytext=(x+.205,.5),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2})
    ax.set_title(name+' · learning mechanism',pad=10); finish('diagram_'+name.replace(' ','_'))
def report(name):
    p=predictions[name]; e=yte-p; mask=mte.engine.eq(7).values
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    axs[0,0].plot(mte.loc[mask,'target_cycle'],yte[mask],color='#243746',label='Observed',lw=1.5)
    axs[0,0].plot(mte.loc[mask,'target_cycle'],p[mask],color=COLORS[1],label='One-step forecast',lw=1.2)
    axs[0,0].set(title='Unseen test engine 7',xlabel='Target cycle',ylabel='T50 (°R)'); axs[0,0].legend()
    axs[0,1].hexbin(yte,p,gridsize=35,mincnt=1,cmap='viridis'); lo=min(yte.min(),p.min()); hi=max(yte.max(),p.max())
    axs[0,1].plot([lo,hi],[lo,hi],'r--',lw=1); axs[0,1].set(title='All test windows: parity',xlabel='Observed T50 (°R)',ylabel='Predicted T50 (°R)')
    axs[1,0].hist(e,bins=45,color=COLORS[0],alpha=.8); axs[1,0].axvline(0,color='black',ls='--'); axs[1,0].set(title='Residual distribution',xlabel='Observed − predicted (°R)',ylabel='Windows')
    per=pd.DataFrame({'engine':mte.engine,'absolute_error':np.abs(e)}).groupby('engine').absolute_error.mean()
    axs[1,1].plot(per.index,per.values,'o',ms=3,color=COLORS[2]); axs[1,1].axhline(np.abs(e).mean(),ls='--',color=COLORS[1],label='Pooled MAE')
    axs[1,1].set(title='Per-engine forecasting error',xlabel='Official test engine ID',ylabel='MAE (°R)'); axs[1,1].legend()
    fig.suptitle(name+' · one-step forecasting diagnostics',size=17); finish('report_'+name.replace(' ','_'))
record('Persistence',mte.last_T50.values,mva.last_T50.values,0,0); report('Persistence')

diagram('KNN',['30-cycle window', 'Standardized distance', '15 nearby windows', 'Weighted T50'])
model=KNeighborsRegressor(n_neighbors=15,weights="distance",n_jobs=-1)
start=time.perf_counter(); model.fit(Ftr,ytr_z); fit_s=time.perf_counter()-start
pv=yscale.inverse_transform(model.predict(Fva).reshape(-1,1)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(model.predict(Fte).reshape(-1,1)).ravel(); pred_s=time.perf_counter()-start
models['KNN']=model
record('KNN',p,pv,fit_s,pred_s); report('KNN')

diagram('Decision Tree',['Lagged features', 'Threshold splits', 'Terminal region', 'Leaf mean'])
model=DecisionTreeRegressor(max_depth=10,min_samples_leaf=15,random_state=SEED)
start=time.perf_counter(); model.fit(Ftr,ytr_z); fit_s=time.perf_counter()-start
pv=yscale.inverse_transform(model.predict(Fva).reshape(-1,1)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(model.predict(Fte).reshape(-1,1)).ravel(); pred_s=time.perf_counter()-start
models['Decision Tree']=model
record('Decision Tree',p,pv,fit_s,pred_s); report('Decision Tree')

diagram('Random Forest',['Bootstrap samples', 'Random features', '120 diverse trees', 'Average forecasts'])
model=RandomForestRegressor(n_estimators=120,max_depth=15,min_samples_leaf=5,max_features=0.7,n_jobs=-1,random_state=SEED)
start=time.perf_counter(); model.fit(Ftr,ytr_z); fit_s=time.perf_counter()-start
pv=yscale.inverse_transform(model.predict(Fva).reshape(-1,1)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(model.predict(Fte).reshape(-1,1)).ravel(); pred_s=time.perf_counter()-start
models['Random Forest']=model
record('Random Forest',p,pv,fit_s,pred_s); report('Random Forest')

diagram('Extra Trees',['Training windows', 'Random thresholds', '120 randomized trees', 'Average forecasts'])
model=ExtraTreesRegressor(n_estimators=120,max_depth=18,min_samples_leaf=5,max_features=0.7,n_jobs=-1,random_state=SEED)
start=time.perf_counter(); model.fit(Ftr,ytr_z); fit_s=time.perf_counter()-start
pv=yscale.inverse_transform(model.predict(Fva).reshape(-1,1)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(model.predict(Fte).reshape(-1,1)).ravel(); pred_s=time.perf_counter()-start
models['Extra Trees']=model
record('Extra Trees',p,pv,fit_s,pred_s); report('Extra Trees')

diagram('Gradient Boosting',['Initial estimate', 'Residual errors', 'Sequential trees', 'Add corrections'])
model=GradientBoostingRegressor(n_estimators=120,learning_rate=0.05,max_depth=3,min_samples_leaf=15,random_state=SEED)
start=time.perf_counter(); model.fit(Ftr,ytr_z); fit_s=time.perf_counter()-start
pv=yscale.inverse_transform(model.predict(Fva).reshape(-1,1)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(model.predict(Fte).reshape(-1,1)).ravel(); pred_s=time.perf_counter()-start
models['Gradient Boosting']=model
record('Gradient Boosting',p,pv,fit_s,pred_s); report('Gradient Boosting')

diagram('XGBoost',['Lagged sensors', 'Gradient statistics', 'Regularized trees', 'Additive forecast'])
model=XGBRegressor(n_estimators=180,max_depth=4,learning_rate=0.04,subsample=0.8,colsample_bytree=0.8,reg_lambda=5,objective="reg:squarederror",tree_method="hist",n_jobs=4,random_state=SEED)
start=time.perf_counter(); model.fit(Ftr,ytr_z); fit_s=time.perf_counter()-start
pv=yscale.inverse_transform(model.predict(Fva).reshape(-1,1)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(model.predict(Fte).reshape(-1,1)).ravel(); pred_s=time.perf_counter()-start
models['XGBoost']=model
record('XGBoost',p,pv,fit_s,pred_s); report('XGBoost')

diagram('CatBoost',['Numeric histories', 'Symmetric tree splits', 'Boost corrections', 'T50 regression'])
model=CatBoostRegressor(iterations=180,depth=5,learning_rate=0.04,loss_function="RMSE",l2_leaf_reg=5,verbose=False,thread_count=4,random_seed=SEED,allow_writing_files=False)
start=time.perf_counter(); model.fit(Ftr,ytr_z); fit_s=time.perf_counter()-start
pv=yscale.inverse_transform(model.predict(Fva).reshape(-1,1)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(model.predict(Fte).reshape(-1,1)).ravel(); pred_s=time.perf_counter()-start
models['CatBoost']=model
record('CatBoost',p,pv,fit_s,pred_s); report('CatBoost')

diagram('LightGBM',['Binned features', 'Leaf-wise growth', 'Constrained leaves', 'Boosted forecast'])
model=LGBMRegressor(n_estimators=180,num_leaves=20,max_depth=6,learning_rate=0.04,min_child_samples=30,reg_lambda=5,n_jobs=4,random_state=SEED,verbosity=-1)
start=time.perf_counter(); model.fit(Ftr,ytr_z); fit_s=time.perf_counter()-start
pv=yscale.inverse_transform(model.predict(Fva).reshape(-1,1)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(model.predict(Fte).reshape(-1,1)).ravel(); pred_s=time.perf_counter()-start
models['LightGBM']=model
record('LightGBM',p,pv,fit_s,pred_s); report('LightGBM')

rows[:]=[row for row in rows if row['Model']!='MLP']
tf.keras.utils.set_random_seed(SEED)
diagram('MLP',['30 × channels', 'Flatten history', 'Dense 64 → 32', 'Next temperature'])
net=tf.keras.Sequential([tf.keras.layers.Input(shape=(WINDOW,len(features))), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64,activation="relu"), tf.keras.layers.Dense(32,activation="relu"), tf.keras.layers.Dense(1)])
net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mse')
display(pd.DataFrame([{'Layer':layer.__class__.__name__,'Output shape':str(tuple(layer.output.shape)),'Parameters':layer.count_params()} for layer in net.layers]))
start=time.perf_counter()
history=net.fit(Xtr,ytr_z,validation_data=(Xva,yva_z),epochs=35,batch_size=128,shuffle=True,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True)],verbose=0)
fit_s=time.perf_counter()-start
print('Completed epochs:',len(history.history['loss']))
pv=yscale.inverse_transform(net.predict(Xva,verbose=0)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(net.predict(Xte,verbose=0)).ravel(); pred_s=time.perf_counter()-start
models['MLP']=net; histories['MLP']=history.history
record('MLP',p,pv,fit_s,pred_s); report('MLP')
fig,ax=plt.subplots(figsize=(9,4)); ax.plot(history.history['loss'],label='Train'); ax.plot(history.history['val_loss'],label='Validation')
ax.set(title='MLP'+' learning curve',xlabel='Epoch (zero-based)',ylabel='MSE (standardized target)'); ax.legend(); finish('learning_'+'MLP')

rows[:]=[row for row in rows if row['Model']!='SimpleRNN']
tf.keras.utils.set_random_seed(SEED)
diagram('SimpleRNN',['Ordered history', 'Shared recurrence', 'Hidden state (32)', 'Next temperature'])
net=tf.keras.Sequential([tf.keras.layers.Input(shape=(WINDOW,len(features))), tf.keras.layers.SimpleRNN(32,activation="tanh"), tf.keras.layers.Dense(16,activation="relu"), tf.keras.layers.Dense(1)])
net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mse')
display(pd.DataFrame([{'Layer':layer.__class__.__name__,'Output shape':str(tuple(layer.output.shape)),'Parameters':layer.count_params()} for layer in net.layers]))
start=time.perf_counter()
history=net.fit(Xtr,ytr_z,validation_data=(Xva,yva_z),epochs=35,batch_size=128,shuffle=True,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True)],verbose=0)
fit_s=time.perf_counter()-start
print('Completed epochs:',len(history.history['loss']))
pv=yscale.inverse_transform(net.predict(Xva,verbose=0)).ravel()
start=time.perf_counter(); p=yscale.inverse_transform(net.predict(Xte,verbose=0)).ravel(); pred_s=time.perf_counter()-start
models['SimpleRNN']=net; histories['SimpleRNN']=history.history
record('SimpleRNN',p,pv,fit_s,pred_s); report('SimpleRNN')
fig,ax=plt.subplots(figsize=(9,4)); ax.plot(history.history['loss'],label='Train'); ax.plot(history.history['val_loss'],label='Validation')
ax.set(title='SimpleRNN'+' learning curve',xlabel='Epoch (zero-based)',ylabel='MSE (standardized target)'); ax.legend(); finish('learning_'+'SimpleRNN')

leaderboard=pd.DataFrame(rows).set_index('Model')
selected=leaderboard['Validation RMSE'].idxmin()
display(Markdown(f'**Candidate selected on validation engines: {selected}.**'))
display(leaderboard.sort_values('RMSE').round(4))
leaderboard.to_csv(OUT/'model_comparison.csv')
fig,axs=plt.subplots(1,2,figsize=(15,6))
r=leaderboard.sort_values('RMSE',ascending=False)
axs[0].barh(r.index,r.RMSE,color=[COLORS[1] if x==selected else COLORS[0] for x in r.index]); axs[0].axvline(base_rmse,color='black',ls='--',label='Persistence'); axs[0].set(title='Held-out next-cycle error',xlabel='RMSE (°R)'); axs[0].legend()
for name,row in leaderboard.iterrows():
    axs[1].scatter(max(row['Fit seconds'],.01),row.RMSE,s=60)
    axs[1].annotate(name,(max(row['Fit seconds'],.01),row.RMSE),xytext=(4,4),textcoords='offset points',fontsize=8)
axs[1].set(xscale='log',xlabel='Measured fit seconds (log scale)',ylabel='Test RMSE (°R)',title='Accuracy and computational cost')
finish('comparison_accuracy_cost')
fig,ax=plt.subplots(figsize=(12,5))
ordered=leaderboard.sort_values('RMSE').index
errors=np.array([np.abs(yte-predictions[name]) for name in ordered])
ax.boxplot(errors.T,showfliers=False); ax.set_xticks(range(1,len(ordered)+1),ordered); ax.tick_params(axis='x',rotation=45); ax.set(title='Absolute error distributions: all test windows',ylabel='Absolute error (°R)'); finish('comparison_error_distributions')
result=mte.copy(); result['observed_T50']=yte
for name,p in predictions.items(): result[name]=p
result.to_csv(OUT/'test_predictions.csv',index=False)
print('Validation-selected test RMSE:',leaderboard.loc[selected,'RMSE'])
print('Validation-selected skill vs persistence:',leaderboard.loc[selected,'Persistence skill'])

mask=mte.engine.eq(7).values; cyc=mte.loc[mask,'target_cycle']
fig=go.Figure(); fig.add_trace(go.Scatter(x=cyc,y=yte[mask],name='Observed T50',line={'color':'#243746'}))
for name,p in predictions.items():
    fig.add_trace(go.Scatter(x=cyc,y=p[mask],name=name,visible=(name==selected)))
buttons=[]
for i,name in enumerate(predictions):
    visible=[True]+[j==i for j in range(len(predictions))]
    buttons.append({'label':name,'method':'update','args':[{'visible':visible},{'title':f'Engine 7 | Observed vs {name}'}]})
fig.update_layout(title=f'Engine 7 | Observed vs {selected}',template='plotly_white',height=520,
    xaxis={'title':'Target operational cycle','rangeslider':{'visible':True}},yaxis_title='T50 (°R)',
    updatemenus=[{'buttons':buttons,'x':1,'y':1.18}],hovermode='x unified')
fig.write_html(OUT/'interactive_engine_replay.html',include_plotlyjs=True)
fig.show()

import importlib.metadata as metadata
packages=['numpy','pandas','scikit-learn','xgboost','catboost','lightgbm','tensorflow','matplotlib','plotly','pyarrow']
versions={p:metadata.version(p) for p in packages}
manifest={'dataset_revision':REV,'sha256':checksums,'seed':SEED,'target':TARGET,'window':WINDOW,
 'features':features,'train_engines':train_ids.tolist(),'validation_engines':val_ids.tolist(),
 'validation_selected_model':selected,'versions':versions,'task':'one-step observed T50 forecasting'}
(OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
for name,h in histories.items(): (OUT/f'{name}_history.json').write_text(json.dumps(h))
import shutil
shutil.make_archive(str(ROOT/'turbofan_results'),'zip',OUT)
print('Artifacts saved to:',OUT)
print('Complete: 10 learned models plus persistence, figures, predictions, manifest and interactive replay.')
display(pd.DataFrame(versions.items(),columns=['Package','Version']))
