# 目录

|      | 内容     | CONTENTS          |
| ---- | -------- | ----------------- |
| 1    | 调库     | IMPORTS           |
| 2    | 引入     | INTRODUCTION      |
| 3    | 数据处理 | DATA PROCESSING   |
| 4    | 模型训练 | MODEL TRAINING    |
| 5    | 模型推断 | MODEL INFERENCING |
| 6    | 展望     | OUTRO             |

# 调库 PACKAGE IMPORTS

## 常规库导入

ctyps.CDLL("libc.so.6") 这个东西貌似只能在unix上用（实测 windows本地用不了 kaggle可以）

```python
%%time 

# General library imports:-
from IPython.display import display_html, clear_output, Markdown;
from gc import collect;

from copy import deepcopy;
import pandas as pd;
import numpy as np;
import joblib;
from os import system, getpid, walk;
from psutil import Process;
import ctypes;
libc = ctypes.CDLL("libc.so.6");

from pprint import pprint;
from colorama import Fore, Style, init;
from warnings import filterwarnings;
filterwarnings('ignore');

from tqdm.notebook import tqdm;

print();
collect();
```

## 模型库

```python
%%time 

# Model development:-
from sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, 
                                     StratifiedKFold as SKF,
                                     KFold, 
                                     RepeatedKFold as RKF, 
                                     cross_val_score);

from lightgbm import log_evaluation, early_stopping, LGBMRegressor as LGBMR;
from xgboost import XGBRegressor as XGBR;
from catboost import CatBoostRegressor as CBR;
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR;
from sklearn.metrics import mean_absolute_error as mae, make_scorer;

print();
collect();
```

## 显示的一些设置

```python
%%time

# Defining global configurations and functions:-

# Color printing    
def PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string";
    print(style + color + text + Style.RESET_ALL); 
    
def GetMemUsage():
    """
    This function defines the memory usage across the kernel. 
    Source-
    https://stackoverflow.com/questions/61366458/how-to-find-memory-usage-of-kaggle-notebook
    """;
    
    pid = getpid();
    py = Process(pid);
    memory_use = py.memory_info()[0] / 2. ** 30;
    return f"RAM memory GB usage = {memory_use :.4}";

# Making sklearn pipeline outputs as dataframe:-
from sklearn import set_config; 
set_config(transform_output = "pandas");
pd.set_option('display.max_columns', 50);
pd.set_option('display.max_rows', 50);

print();
collect();
```

# 引入 Introduction

1. 这本笔记本是我第一次尝试Optiver挑战。这是一个时间序列回归问题，涉及当天收盘时的股市交易数据。此处使用平均绝对误差度量。
2. 本笔记本旨在使用简单的CV (Cross Validation)策略，从为挑战创建的内存减少的数据集中训练基线模型。
3. 这是我的基线数据管理笔记本和数据集的延续。我们在此继续分析，并训练模型以得出CV分数。然后，我们在这里使用这些模型进行推断，并提交。

## 版本细节 VERSION DETAILS

| 版本号<br />Version Number | 版本细节<br />Version Details                                | 准备日期<br />Preparation date | LGBMR CV | CBR CV   | XGBR CV | HGBR CV  | Best LB score | Single/ Ensemble   |
| -------------------------- | ------------------------------------------------------------ | ------------------------------ | -------- | -------- | ------- | -------- | ------------- | ------------------ |
| V1                         | 基线特征<br />Baseline features<br />无无效处理和缩放<br />No null treatments and scaling<br />无需调整的简单机器学习模型<br />Simple ML models without tuning<br />5x1 K-fold 交叉验证<br />5x1 K-fold CV <br />简单加权集合<br />Simple weighted ensemble | 22Sep2023                      | 6.248286 | 6.25538  | 6.27198 | 6.266826 | 5.3702        | Ensemble LGBMR CBR |
| V2                         | 基线特征<br />Baseline features<br />无无效处理和缩放<br />No null treatments and scaling<br />简单的机器学习模型 无需使用更改的参数进行调整<br />Simple ML models without tuning with altered parameters <br />5x1 K-fold 交叉验证 5x1<br />K-fold CV<br />简单加权集合<br />Simple weighted ensemble | 23Sep2023                      | 6.23334  | 6.2535   |         |          | 5.3728        | Ensemble LGBMR CBR |
| V3                         | 基线特征<br />Baseline features<br />无无效处理和缩放<br />No null treatments and scaling<br />用V1参数的机器学习模型<br />ML models with V1 parameters<br />5x3 Repeated K-fold 交叉验证<br />5x3 Repeated K-fold CV<br />简单加权集合<br />Simple weighted ensemble | 24Sep2023                      | 6.248288 | 6.25532  |         | 6.267036 | 5.3712        | Ensemble LGBMR CBR |
| V4                         | 基线特征+成交量新特征<br />Baseline features + **Median volume new feature**<br />无无效处理和缩放<br />No null treatments and scaling<br />用V1参数的机器学习模型<br />ML models with V1 parameters<br />5x1 K-fold 交叉验证<br />5x1 K-fold CV<br />具有goto转换的简单加权集合<br />Simple weighted ensemble **with goto conversion** | 01Oct2023                      | 6.241901 | 6.250738 |         |          | 5.3638        | Ensemble LGBMR CBR |
| V5                         | 使用我的数据集作为输入而不是内核输出<br />Used my dataset as input instead of kernel output<br />基线特征+成交量新特征<br />Baseline features + **Median volume new features** <br />用V1参数的机器学习模型<br />ML models with V1 parameters <br />5x1 K-fold 交叉验证<br />5x1 K-fold CV <br />具有goto转换的简单加权集合<br /> Simple weighted ensemble **with goto conversion** | 02Oct2023                      | 6.239849 | 6.250021 |         | 6.262478 | 5.3635        | Ensemble LGBMR CBR |

## 设置参数 CONFIGURATION PARAMETERS

| 参数<br />Parameter | 内容<br />Comments                                           | 样本值<br />Sample values                                    |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| version_nb          | 版本号<br />Version Number                                   | integer value                                                |
| test_req            | 是否检查代码<br />Are we testing the code?                   | Y/N                                                          |
| test_frac           | 用于采样和测试的测试分数放置小值以便于执行<br />Test fraction for sampling and testing Place small values for easy execution | float between 0 and 1                                        |
| load_tr_data        | 是否加载训练数据 如果只是推理就不需要<br />Are we loading the train data here? If we are inferring only, this is not required | Y/N                                                          |
| gpu_switch          | 是否需要GPU<br />Do we need a GPU here?                      | Y/N                                                          |
| state               | 随机种子<br />Random seed                                    | integer                                                      |
| target              | 目标列名<br />Target column name                             | string value                                                 |
| path                | 模型训练的数据路径 我将其指向我的基线数据管理内核<br />Data path for model training I point this to my baseline data curation kernel |                                                              |
| test_path           | 测试数据的相关路径<br />Relevant path for test data          | 竞赛人工制品<br />Competition artefacts                      |
| df_choice           | 我需要哪些数据进行分析？有关详细信息，请参阅基线数据准备内核<br />Which data do I need for analysis? Refer the baseline data prep kernel for details |                                                              |
| mdl_path            | 使用joblib转储经过训练的模型的路径<br />Path to dump trained models with joblib |                                                              |
| inf_path            | 提取用于推理的模型的适当路径我指向我的基线数据集 其中模型被训练为启动器<br />Appropriate path to extract the models for inference I point to my baseline dataset with models trained as a starter |                                                              |
| methods             | 所有训练的模型方法 根据记忆约束选择1个摩尔 对于推理 所有训练的方法都需要存在<br />All trained model methods, choose 1-more based on the memory constraints For inferencing, all trained methods need to be present | list                                                         |
| ML                  | 是否需要模型训练<br />Do we need to do model training here?  | Y/N                                                          |
| n_splits            | 交叉训练分割数<br />CV number of splits                      | integer value                                                |
| n_repeats           | 交叉训练重复数<br />CV number of repetitions                 | integer value                                                |
| nbrnd_erly_stp      | 提前停止轮次<br />Number of early stopping rounds            | integer value                                                |
| mdlcv_mthd          | 模型交叉验证选择<br />Model CV choice                        | KF, SKF, RSKF, RKF                                           |
| ensemble_req        | 是否需要集合吗 目前未使用<br />Do we need an ensemble here? Currently this is unused | Y/N                                                          |
| enscv_mthd          | 集合交叉验证选择-主要与Optuna一起使用<br />Ensemble CV choice- used mostly with Optuna | KF, SKF, RSKF, RKF                                           |
| metric_obj          | 基于度量 是否需要最大最小函数<br />Based on the metric, do we wish to maximize/ minimize the function? | maximize/ minimize                                           |
| ntrials             | Optuna试验数量<br />Number of Optuna trials                  | integer value                                                |
| ens_weights         | 权重（如果按此决定）<br />Weights if decided subjecively     | 列出适当的训练方法的数量<br />list apropos to number of trained methods |
| inference_req       | 是否需要推理<br />Do we need to infer here?                  | Y/N                                                          |

```python
%%time 

# Configuration class:-
class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Please use caps lock capital letters while filling in parameters
    """;
    
    # Data preparation:-   
    version_nb         = 5;
    test_req           = "N";
    test_frac          = 0.01;
    load_tr_data       = "N";
    gpu_switch         = "OFF"; 
    state              = 42;
    target             = 'target';
    
    path               = f"/kaggle/input/optiver-memoryreduceddatasets/";
    test_path          = f"/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv";
    df_choice          = f"XTrIntCmpNewFtre.parquet";
    mdl_path           = f'/kaggle/working/BaselineML/';
    inf_path           = f'/kaggle/input/optiverbaselinemodels/';
     
    # Model Training:-
    methods            = ["LGBMR", "CBR", "HGBR"];
    ML                 = "N";
    n_splits           = 5;
    n_repeats          = 1;
    nbrnd_erly_stp     = 100 ;
    mdlcv_mthd         = 'KF';
    
    # Ensemble:-    
    ensemble_req       = "N";
    enscv_mthd         = "KF";
    metric_obj         = 'minimize';
    ntrials            = 10 if test_req == "Y" else 200;
    ens_weights        = [0.54, 0.44, 0.02];
    
    # Inference:-
    inference_req      = "Y";
    
    # Global variables for plotting:-
    grid_specs = {'visible': True, 'which': 'both', 'linestyle': '--', 
                  'color': 'lightgrey', 'linewidth': 0.75
                 };
    title_specs = {'fontsize': 9, 'fontweight': 'bold', 'color': 'tab:blue'};

print();
PrintColor(f"--> Configuration done!\n");
collect();

PrintColor(f"\n" + GetMemUsage(), color = Fore.RED);
```

```python
%%time 

# Commonly used CV strategies for later usage:-
all_cv= {'KF'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),
         'RKF' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),
         'RSKF': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),
         'SKF' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)
        };

# Defining the competition metric:-
def ScoreMetric(ytrue, ypred)-> float:
    """
    This function calculates the metric for the competition. 
    ytrue- ground truth array
    ypred- predictions
    returns - metric value (float)
    """;
    
    return mae(ytrue, ypred);

# Designing a custom scorer to use in cross_val_predict and cross_val_score:-
myscorer = make_scorer(ScoreMetric, greater_is_better = False, needs_proba=False,);

print();
collect();

PrintColor(f"\n" + GetMemUsage(), color = Fore.RED);
```

```python
%%time

def goto_conversion(listOfOdds, total = 1, eps = 1e-6, isAmericanOdds = False):
    "Source - https://www.kaggle.com/code/kaito510/goto-conversion-optiver-baseline-models";

    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        for i in range(len(listOfOdds)):
            currOdds = listOfOdds[i];
            isNegativeAmericanOdds = currOdds < 0;
            if isNegativeAmericanOdds:
                currDecimalOdds = 1 + (100/(currOdds*-1));
            else: 
                #Is non-negative American Odds
                currDecimalOdds = 1 + (currOdds/100);
            listOfOdds[i] = currDecimalOdds;

    #Error Catchers
    if len(listOfOdds) < 2:
        raise ValueError('len(listOfOdds) must be >= 2');
    if any(x < 1 for x in listOfOdds):
        raise ValueError('All odds must be >= 1, set isAmericanOdds parameter to True if using American Odds');

    #Computation:-
    #initialize probabilities using inverse odds
    listOfProbabilities = [1/x for x in listOfOdds];
    
    #compute the standard error (SE) for each probability
    listOfSe = [pow((x-x**2)/x,0.5) for x in listOfProbabilities];
    
    #compute how many steps of SE the probabilities should step back by
    step = (sum(listOfProbabilities) - total)/sum(listOfSe) ;
    outputListOfProbabilities = [min(max(x - (y*step),eps),1) for x,y in zip(listOfProbabilities, listOfSe)];
    return outputListOfProbabilities;

def zero_sum(listOfPrices, listOfVolumes):
    """
    Source - https://www.kaggle.com/code/kaito510/goto-conversion-optiver-baseline-models
    """;
    
    #compute standard errors assuming standard deviation is same for all stocks
    listOfSe = [x**0.5 for x in listOfVolumes];
    step = sum(listOfPrices)/sum(listOfSe);
    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)];
    return outputListOfPrices;

collect();
```

## 如何使用这些内核 How to use these kernels

1. 使用内存减少内核输入来curate特征，减少数据集内存，并准备必要的数据集作为该内核的输入。否则，如果功能已经准备好，则使用启动器数据集作为输入。下面提供了链接。

   **Baseline input features:-** https://www.kaggle.com/code/ravi20076/optiver-memoryreduction

   **Baseline input dataset:-** https://www.kaggle.com/datasets/ravi20076/optiver-memoryreduceddatasets

2. 在这里为基线设计自己的模型框架并训练模型。建议一次训练1/2个模型，以防止内存溢出问题。

3. 将模型对象存储在工作文件夹的BaselineML目录中以进行推理

4. 建议分别推断和提交。这肯定不会造成数据内存过长的问题。在这种情况下，请关闭训练，不要加载训练数据集。在这种情况下，我已将模型训练人工制品存储在链接中\- https://www.kaggle.com/datasets/ravi20076/optiverbaselinemodels

5. 在进行推理时，确保curate与训练过程中使用的特征相同的特征。我将在这里进行改进，并很快更新内核。

# 数据处理 DATA PROCESSING

在这个版本中，我们根据参考笔记本选择了具有新功能的int-float压缩数据集

```python
%%time 

if (CFG.load_tr_data == "Y" or CFG.ML == "Y") and CFG.test_req == "Y":
    if isinstance(CFG.test_frac, float):
        X = pd.read_parquet(CFG.path + CFG.df_choice).sample(frac = CFG.test_frac);
    else:
        X = pd.read_parquet(CFG.path + CFG.df_choice).sample(n = CFG.test_frac);
        
    y = pd.read_parquet(CFG.path + f"Ytrain.parquet").loc[X.index].squeeze();
    PrintColor(f"---> Sampled train shapes for code testing = {X.shape} {y.shape}", 
               color = Fore.RED);
    X.index, y.index = range(len(X)), range(len(y));
    
    PrintColor(f"\n---> Train set columns for model development");
    pprint(X.columns, width = 100, depth = 1, indent = 5);
    print();

elif CFG.load_tr_data == "Y" or CFG.ML == "Y":
    X = pd.read_parquet(CFG.path + CFG.df_choice);
    y = pd.read_parquet(CFG.path + f"Ytrain.parquet").squeeze();  
    PrintColor(f"---> Train shapes for code testing = {X.shape} {y.shape}");

elif CFG.load_tr_data != "Y" or CFG.inference_req == "Y":
    PrintColor(f"---> Train data is not required as we are infering from the model");
    
print();
collect();
libc.malloc_trim(0);

PrintColor(f"\n" + GetMemUsage(), color = Fore.RED);
```

# 模型训练和交叉验证 MODEL TRAINING AND CV

```python
%%time 

# Initializing model I-O:-

if CFG.ML == "Y":
    Mdl_Master = \
    {'CBR': CBR(**{'task_type'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                   'objective'           : "MAE",
                   'eval_metric'         : "MAE",
                   'bagging_temperature' : 0.5,
                   'colsample_bylevel'   : 0.7,
                   'iterations'          : 500,
                   'learning_rate'       : 0.065,
                   'od_wait'             : 25,
                   'max_depth'           : 7,
                   'l2_leaf_reg'         : 1.5,
                   'min_data_in_leaf'    : 1000,
                   'random_strength'     : 0.65, 
                   'verbose'             : 0,
                   'use_best_model'      : True,
                  }
               ), 

      'LGBMR': LGBMR(**{'device'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                        'objective'         : 'regression_l1',
                        'boosting_type'     : 'gbdt',
                        'random_state'      : CFG.state,
                        'colsample_bytree'  : 0.7,
                        'subsample'         : 0.65,
                        'learning_rate'     : 0.065,
                        'max_depth'         : 6,
                        'n_estimators'      : 500,
                        'num_leaves'        : 150,  
                        'reg_alpha'         : 0.01,
                        'reg_lambda'        : 3.25,
                        'verbose'           : -1,
                       }
                    ),

      'XGBR': XGBR(**{'tree_method'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",
                      'objective'          : 'reg:absoluteerror',
                      'random_state'       : CFG.state,
                      'colsample_bytree'   : 0.7,
                      'learning_rate'      : 0.07,
                      'max_depth'          : 6,
                      'n_estimators'       : 500,                         
                      'reg_alpha'          : 0.025,
                      'reg_lambda'         : 1.75,
                      'min_child_weight'   : 1000,
                      'early_stopping_rounds' : CFG.nbrnd_erly_stp,
                     }
                  ),

      "HGBR" : HGBR(loss              = 'squared_error',
                    learning_rate     = 0.075,
                    early_stopping    = True,
                    max_iter          = 200,
                    max_depth         = 6,
                    min_samples_leaf  = 1500,
                    l2_regularization = 1.75,
                    scoring           = myscorer,
                    random_state      = CFG.state,
                   )
    };

print();
collect();

PrintColor(f"\n" + GetMemUsage(), color = Fore.RED);
```

```python
%%time 

if CFG.ML == "Y":
    # Initializing the models from configuration class:-
    methods = CFG.methods;

    # Initializing a folder to store the trained and fitted models:-
    system('mkdir BaselineML');

    # Initializing the model path for storage:-
    model_path = CFG.mdl_path;

    # Initializing the cv object:-
    cv = all_cv[CFG.mdlcv_mthd];
        
    # Initializing score dataframe:-
    Scores = pd.DataFrame(index = range(CFG.n_splits * CFG.n_repeats),
                          columns = methods).fillna(0).astype(np.float32);
    
    FtreImp = pd.DataFrame(index = X.columns, columns = [methods]).fillna(0);

print();
collect();
libc.malloc_trim(0);

PrintColor(f"\n" + GetMemUsage(), color = Fore.RED);
```

```python
%%time 

if CFG.ML == "Y":
    PrintColor(f"\n{'=' * 25} ML Training {'=' * 25}\n");
    
    # Initializing CV splitting:-       
    for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y)), 
                                              f"{CFG.mdlcv_mthd} CV {CFG.n_splits}x{CFG.n_repeats}"
                                             ): 
        # Creating the cv folds:-    
        Xtr  = X.iloc[train_idx];   
        Xdev = X.iloc[dev_idx];
        ytr  = y.iloc[train_idx];
        ydev = y.iloc[dev_idx];
        
        PrintColor(f"-------> Fold{fold_nb} <-------");
        # Fitting the models:- 
        for method in methods:
            model = Mdl_Master[method];
            if method == "LGBMR":
                model.fit(Xtr, ytr, 
                          eval_set = [(Xdev, ydev)], 
                          verbose = 0, 
                          eval_metric = "mae",
                          callbacks = [log_evaluation(0,), 
                                       early_stopping(CFG.nbrnd_erly_stp, verbose = False)], 
                         );

            elif method == "XGBR":
                model.fit(Xtr, ytr, 
                          eval_set = [(Xdev, ydev)], 
                          verbose = 0, 
                          eval_metric = "mae",
                         );  

            elif method == "CBR":
                model.fit(Xtr, ytr, 
                          eval_set = [(Xdev, ydev)], 
                          verbose = 0, 
                          early_stopping_rounds = CFG.nbrnd_erly_stp,
                         ); 

            else:
                model.fit(Xtr, ytr);

            #  Saving the model for later usage:-
            joblib.dump(model, CFG.mdl_path + f'{method}V{CFG.version_nb}Fold{fold_nb}.model');
            
            # Creating OOF scores:-
            score = ScoreMetric(ydev, model.predict(Xdev));
            Scores.at[fold_nb, method] = score;
            num_space = 6- len(method);
            PrintColor(f"---> {method} {' '* num_space} OOF = {score:.5f}", 
                       color = Fore.MAGENTA);  
            del num_space, score;
            
            # Collecting feature importances:-
            try:
                FtreImp[method] = \
                FtreImp[method].values + (model.feature_importances_ / (CFG.n_splits * CFG.n_repeats));
            except:
                pass;
            
            collect();
            
        PrintColor(GetMemUsage());
        print();
        del Xtr, ytr, Xdev, ydev;
        collect();
    
    clear_output();
    PrintColor(f"\n---> OOF scores across methods <---\n");
    Scores.index.name = "FoldNb";
    Scores.index = Scores.index + 1;
    display(Scores.style.format(precision = 5).\
            background_gradient(cmap = "Pastel1")
           );
    
    PrintColor(f"\n---> Mean OOF scores across methods <---\n");
    display(Scores.mean());
    
    try: FtreImp.to_csv(CFG.mdl_path + f"FtreImp_V{CFG.version_nb}.csv");
    except: pass;
        
collect();
print();
libc.malloc_trim(0);

PrintColor(f"\n" + GetMemUsage(), color = Fore.GREEN);
```

# 模型推理和提交 MODEL INFERENCING AND SUBMISSION

```python
%%time 

def MakeFtre(df : pd.DataFrame, prices: list) -> pd.DataFrame:
    """
    This function creates new features using the price columns. This was used in a baseline notebook as below-
    https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost
    
    Inputs-
    df:- pd.DataFrame -- input dataframe
    cols:- price columns for transformation
    
    Returns-
    df:- pd.DataFrame -- dataframe with extra columns
    """;
    
    features = ['overall_medvol', "first5min_medvol", "last5min_medvol",
                'seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
               ];
    
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32);
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32);
       
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            if i>j:
                df[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})');
                features.append(f'{a}_{b}_imb'); 
                    
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            for k,c in enumerate(prices):
                if i>j and j>k:
                    max_ = df[[a,b,c]].max(axis=1);
                    min_ = df[[a,b,c]].min(axis=1);
                    mid_ = df[[a,b,c]].sum(axis=1)-min_-max_;

                    df[f'{a}_{b}_{c}_imb2'] = ((max_-mid_)/(mid_-min_)).astype(np.float32);
                    features.append(f'{a}_{b}_{c}_imb2');
    
    return df[features];

print();
collect();
```

```python
%%time 

# Creating the testing environment:-
if CFG.inference_req == "Y":
    try: 
        del X, y;
    except: 
        pass;
        
    prices = ['reference_price', 'far_price', 'near_price', 'bid_price', 'ask_price', 'wap'];
    
    # Making the test environment for inferencing:-
    import optiver2023;
    try: 
        env = optiver2023.make_env();
        iter_test = env.iter_test();
        PrintColor(f"\n---> Curating the inference environment");
    except: 
        pass;
    
    # Collating a list of models to be used for inferencing:-
    models = [];

    # Loading the models for inferencing:-
    if CFG.ML != "Y": 
        model_path = CFG.inf_path;
        PrintColor(f"---> Loading models from the input data for the kernel - V{CFG.version_nb}\n", 
                  color = Fore.RED);
    elif CFG.ML == "Y": 
        model_path = CFG.mdl_path;
        PrintColor(f"---> Loading models from the working directory for the kernel\n");
    
    # Loading the models from the models dataframe:-
    mdl_lbl = [];
    for _, _, filename in walk(model_path):
        mdl_lbl.extend(filename);

    models = [];
    for filename in mdl_lbl:
        models.append(joblib.load(model_path + f"{filename}"));
        
    mdl_lbl    = [m.replace(r".model", "") for m in mdl_lbl];
    model_dict = {l:m for l,m in zip(mdl_lbl, models)};
    PrintColor(f"\n---> Trained models\n");    
    pprint(np.array(mdl_lbl), width = 100, indent = 10, depth = 1);  
       
print();
collect();  
libc.malloc_trim(0);
PrintColor(f"\n" + GetMemUsage(), color = Fore.RED); 
```

```python
%%time 

if CFG.inference_req == "Y":
    print();
    counter = 0;
    
    try:
        median_vol = pd.read_csv(CFG.path + f"MedianVolV2.csv", index_col = ['Unnamed: 0']);
    except:
        median_vol = pd.read_csv(CFG.path + f"MedianVolV2.csv"); 
    median_vol.index.name = "stock_id";
    median_vol = median_vol[['overall_medvol', "first5min_medvol", "last5min_medvol"]];
    
    for test, revealed_targets, sample_prediction in iter_test:
        if counter >= 99: num_space = 1;
        elif counter >= 9: num_space = 2;
        else: num_space = 3;
        
        PrintColor(f"{counter + 1}. {' ' * num_space} Inference", color = Fore.MAGENTA);
        test  = test.merge(median_vol, how = "left", left_on = "stock_id", right_index = True);
        Xtest = MakeFtre(test, prices = prices);
        del num_space;
        
        # Curating model predictions across methods and folds:-        
        preds = pd.DataFrame(columns = CFG.methods, index = Xtest.index).fillna(0);
        for method in CFG.methods:
            for mdl_lbl, mdl in model_dict.items():
                if mdl_lbl.startswith(f"{method}V{CFG.version_nb}"):
                    if CFG.test_req == "Y":
                        print(mdl_lbl);
                    else:
                        pass;
                    preds[method] = preds[method] + mdl.predict(Xtest)/ (CFG.n_splits * CFG.n_repeats);
        
        # Curating the weighted average model predictions:-       
        sample_prediction['target'] = \
        np.average(preds.values, weights= CFG.ens_weights, axis=1);
        
        # Source - https://www.kaggle.com/code/kaito510/goto-conversion-optiver-baseline-models     
        sample_prediction['target'] = \
        zero_sum(sample_prediction['target'], test.loc[:,'bid_size'] + test.loc[:,'ask_size'])
        
        try: 
            env.predict(sample_prediction);
        except: 
            PrintColor(f"---> Submission did not happen as we have the file already");
            pass;
        
        counter = counter+1;
        collect();
    
    PrintColor(f"\n---> Submission file\n");
    display(sample_prediction.head(10));
            
print();
collect();  
libc.malloc_trim(0);
PrintColor(f"\n" + GetMemUsage(), color = Fore.RED); 
```

# 展望 OUTRO

## 下一步

1. 探索更好的模型和集成策略
2. 从现有特征列表中清除冗余要素
3. 在公开讨论和核心的基础上促进现有流程的改进

## 引用

1. https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost
2. https://www.kaggle.com/code/renatoreggiani/optv-lightgbm -- Median volume column
3. https://www.kaggle.com/code/kaito510/goto-conversion-optiver-baseline-models -- goto conversion