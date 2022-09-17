from pyexpat import features
import time
from sklearn.linear_model import LinearRegression
from importlib import reload  # Not needed in Python 2
import logging
import numpy as np
import pandas as pd
from importlib import reload  # Not needed in Python 2
from sklearn.linear_model import ElasticNet


def mse(pred, target):
    return np.mean((pred - target) ** 2)

def rmse(pred, target):
    return np.sqrt(np.mean((pred.values - target.values) ** 2))

def sse(pred, target):
    return np.sum((pred - target) ** 2)

reload(logging)
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s %(levelname)s:    %(message)s",
    level=logging.DEBUG,
    datefmt="%I:%M:%S",
)
logger.setLevel(logging.DEBUG)

def target_encoding(data,categorical_features,target):
    mappings={}
    features=[]
    for feature in categorical_features:
        mappings[feature]=data.groupby(feature,as_index=False).Sales.mean().rename({target:f"{feature}_enc"},axis="columns")
        features.append(f"{feature}_enc")
    mappings["na_target_average"]=data[target].mean()
    return mappings,features
def apply_mapping(data,mappings):
    data=data.copy()
    for feature,mapping in mappings.items():
        if feature !="na_target_average":
            data=data.merge(mapping,on=feature,how="left")
    return data.fillna(mappings["na_target_average"])

def find_best_split_numeric(train, split_feature,target,step_size=30,max_bins=100):
    #logging.debug(f"Finding best split for feature {split_feature}")
    splits = np.sort(train.sample(n=min(int(max_bins*step_size),len(train)))[split_feature])
    mses_=[]
    for split in splits[range(1,len(splits)-1,step_size)]:
        preds=train.groupby(train[split_feature]>=split)[target].transform("mean")
        mses_.append((sse(train[target],preds),split))
    return min(mses_, key = lambda t: t[0])

def find_best_split_categorical_enc(train, split_feature,target,max_bins=100):
    return find_best_split_numeric(train, split_feature,target,step_size=1,max_bins=max_bins)

def find_best_split_categorical(train, split_feature,target):
    #logging.debug(f"Finding best split for feature {split_feature}")
    splits = train[split_feature].unique().tolist()
    #logging.debug(f"Finding best split for feature {split_feature}")
    mses_=[]
    for split in splits:
        preds=train.groupby(train[split_feature]==split)[target].transform("mean")
        mses_.append((sse(train[target],preds),split))
    return min(mses_, key = lambda t: t[0])

def sample_features(numerical_features,categorical_features,feature_fraction=1):
    features=np.random.choice(
        numerical_features+categorical_features,
        int(np.ceil(feature_fraction*len(numerical_features+categorical_features))),
        replace=False)
    new_cat=[f for f in features if f in categorical_features]
    new_num=[f for f in features if f in numerical_features]
    return new_num,new_cat

def filter_non_variable(train,categorical_features):
    new_categoricals=[]
    for feature in categorical_features:
        if train[feature].nunique()>0:
            new_categoricals.append(feature)
    return new_categoricals
def find_best_feature(train, numerical_features,categorical_features_enc,target,step_size=30,feature_fraction=1,max_bins=100):
    sses_=[]
    categorical_features_enc=filter_non_variable(train,categorical_features_enc)
    numerical_features,categorical_features=sample_features(numerical_features,categorical_features_enc,feature_fraction=feature_fraction)

    for feature in numerical_features:
        feature_sse=find_best_split_numeric(train, split_feature=feature,target=target,step_size=step_size,max_bins=max_bins)
        sses_.append((feature_sse[0],feature_sse[1],feature))
        
    for feature in categorical_features:
        feature_sse=find_best_split_categorical_enc(train, split_feature=feature,target=target,max_bins=max_bins)
        sses_.append((feature_sse[0],feature_sse[1],feature))
    return min(sses_, key = lambda t: t[0])

def split_data(data,split_value,split_feature,numerical_features,categorical_features_enc):
    data=data.copy()
    if split_feature in numerical_features or split_feature in categorical_features_enc:
        return data[data[split_feature]>=split_value],data[data[split_feature]<split_value]
    #elif split_feature in categorical_features:
    #    return data[data[split_feature]==split_value],data[data[split_feature]!=split_value]
    else:
        raise ValueError(f"{split_feature} is neither in numerical features nor in categorical features")

def decide_split(train, numerical_features,categorical_features_enc,target,best_split,min_data_node=100,node_sse=None,node_lm=None,l1_ratio=0.5):
    left_node_sse=None
    right_node_sse=None
    if node_sse is None:
        node_lm=ElasticNet(l1_ratio=l1_ratio,fit_intercept=True,selection="random")
        node_lm.fit(train[numerical_features+categorical_features_enc],train[target])
        node_sse=sse(node_lm.predict(train[numerical_features+categorical_features_enc]),train[target])
        
    right_train,left_train = split_data(train,split_value=best_split[1],split_feature=best_split[2],numerical_features=numerical_features,categorical_features_enc=categorical_features_enc)
    
    if len(right_train)>min_data_node:
        right_lm=ElasticNet(l1_ratio=l1_ratio,fit_intercept=True,selection="random")
        right_lm.fit(right_train[numerical_features+categorical_features_enc],right_train[target])
        right_node_sse=sse(right_lm.predict(right_train[numerical_features+categorical_features_enc]),right_train[target])  
        right_grow=right_node_sse<node_sse
        if not right_grow:
            right_lm=node_lm
    else: 
        right_lm=node_lm
        right_grow=False
    if len(left_train)>min_data_node:
        left_lm=ElasticNet(l1_ratio=l1_ratio,fit_intercept=True,selection="random")
        left_lm.fit(left_train[numerical_features+categorical_features_enc],left_train[target])
        left_node_sse=sse(left_lm.predict(left_train[numerical_features+categorical_features_enc]),left_train[target])  
        left_grow=left_node_sse<node_sse
        if not left_grow:
            left_lm=node_lm
    else:
        left_lm=node_lm
        left_grow=False
        
    return {
        "split_feature": best_split[2],
        "split_value": best_split[1],
        "right_grow":right_grow,
        "left_grow":left_grow,
        "right_lm":right_lm if right_grow else node_lm,
        "left_lm":left_lm if left_lm else node_lm,
        "right_train":right_train,
        "left_train":left_train,
        "left_node_sse":left_node_sse,
        "right_node_sse":right_node_sse,
    }

class Node():
    def __init__(self,numerical_features,categorical_features,categorical_features_enc,target,bin_size=30,max_depth=5,feature_fraction=1,initial_node=True,node_sse=None,node_lm=None,max_bins=100,l1_ratio=0.5):
        self.bin_size=bin_size
        self.numerical_features=numerical_features
        self.categorical_features=categorical_features
        self.categorical_features_enc=categorical_features_enc
        self.target=target
        self.max_depth=max_depth
        self.feature_fraction=feature_fraction
        self.right_node=None
        self.left_node=None
        self.fitted=False
        self.initial_node=initial_node
        self.node_sse=node_sse
        self.node_lm=node_lm
        self.max_bins=max_bins
        self.l1_ratio = l1_ratio
    def fit_one_depth(self,train):
        if self.max_depth>0:
            if not self.fitted:
                best=find_best_feature(train, 
                                       numerical_features=self.numerical_features,
                                       categorical_features_enc=self.categorical_features_enc,
                                       target=self.target,
                                       step_size=self.bin_size,
                                       feature_fraction=self.feature_fraction,
                                       max_bins=self.max_bins
                                       ) 
                self.decision=decide_split(train, numerical_features=self.numerical_features,categorical_features_enc=self.categorical_features_enc,target=self.target,best_split=best,node_lm=self.node_lm,node_sse=self.node_sse,l1_ratio=self.l1_ratio)

                if self.decision["right_grow"]:
                    self.right_node=Node(
                        numerical_features=self.numerical_features,
                        categorical_features=self.categorical_features,
                        categorical_features_enc=self.categorical_features_enc,
                        target=self.target,
                        bin_size=self.bin_size,
                        max_depth=self.max_depth-1,
                        initial_node=False,
                        max_bins=self.max_bins,
                        node_sse=self.decision["right_node_sse"],
                        node_lm=self.decision["right_lm"]
                        )

                if self.decision["left_grow"]:
                    self.left_node=Node(
                        numerical_features=self.numerical_features,
                        categorical_features=self.categorical_features,
                        categorical_features_enc=self.categorical_features_enc,
                        target=self.target,
                        bin_size=self.bin_size,
                        max_depth=self.max_depth-1,
                        initial_node=False,
                        max_bins=self.max_bins,
                        node_sse=self.decision["left_node_sse"],
                        node_lm=self.decision["left_lm"]
                        )
                self.fitted=True
            else:
                if self.right_node:
                    self.right_node.fit_one_depth(train=self.decision["right_train"])
                if self.left_node:
                    self.left_node.fit_one_depth(train=self.decision["left_train"])
                    
        else:
            logging.warning(f"Reached maximum depth of {self.max_depth}")
        
    def fit(self,train,validation):
        if self.initial_node:
            self.mappings,features=target_encoding(train,categorical_features=self.categorical_features,target=self.target)
            self.categorical_features_enc=features
            train_enc=apply_mapping(train,mappings=self.mappings)
        else:
            train_enc=train.copy()
        for depth in range(1,self.max_depth+1):
            logging.info(f"Training at depth {depth}")
            self.fit_one_depth(train_enc)
            logging.info(f"MSE on train: {rmse(train[self.target],self.predict(train))}")
            logging.info(f"MSE on validation: {rmse(validation[self.target],self.predict(validation))}")
             
    def predict(self,data):
        data=data.copy()
        if self.initial_node:
            data=apply_mapping(data,self.mappings)
        right_data,left_data = split_data(data,split_value=self.decision["split_value"],split_feature=self.decision["split_feature"],numerical_features=self.numerical_features,categorical_features_enc=self.categorical_features_enc)

        if len(right_data)>0:
            if self.decision["right_grow"] and self.right_node and self.right_node.fitted:
                right_predictions=self.right_node.predict(right_data)
            else:
                right_predictions=self.decision["right_lm"].predict(right_data[self.numerical_features+self.categorical_features_enc])
                right_data["prediction"]=right_predictions
                right_predictions=right_data["prediction"]
        else:
            right_predictions=pd.Series(name="prediction",dtype=np.float64)
        if len(left_data)>0:
            if self.decision["left_grow"] and self.left_node and self.left_node.fitted:
                left_predictions=self.left_node.predict(left_data)
            else:
                left_predictions=self.decision["left_lm"].predict(left_data[self.numerical_features+self.categorical_features_enc])
                left_data["prediction"]=left_predictions
                left_predictions=left_data["prediction"]
        else:
            left_predictions=pd.Series(name="prediction",dtype=np.float64)
        data=data.merge(pd.concat([right_predictions,left_predictions]), left_index=True, right_index=True,how="left")
        return data["prediction"]  
    
    def explain_prediction(self,data):
        linear_model=[]
        data=data.copy()
        right_data,left_data = split_data(data,split_value=self.decision["split_value"],split_feature=self.decision["split_feature"],numerical_features=self.numerical_features,categorical_features=self.categorical_features)
        if self.decision["split_feature"] in self.numerical_features:
            if data[self.decision["split_feature"]].values[0]>=self.decision["split_value"]:
                logging.info(f"Numerical Feature {self.decision['split_feature']} >= {self.decision['split_value']}. Using right node...")
            else: ##++
                logging.info(f"Numerical Feature {self.decision['split_feature']} < {self.decision['split_value']}. Using left node...")
                
        if self.decision["split_feature"] in self.categorical_features:
            if data[self.decision["split_feature"]].values[0]==self.decision["split_value"]:
                logging.info(f"Categorical Feature {self.decision['split_feature']} == {self.decision['split_value']}. Using right node...")
            else:#
                logging.info(f"Categorical Feature {self.decision['split_feature']} != {self.decision['split_value']}. Using left node...")
                
        if len(right_data)>0:
            if self.decision["right_grow"] and self.right_node and self.right_node.fitted:
                right_predictions,lm=self.right_node.explain_prediction(right_data)
                linear_model.extend(lm)
            else:##
                right_predictions=self.decision["right_lm"].predict(right_data[self.numerical_features])
                right_data["prediction"]=right_predictions
                right_predictions=right_data["prediction"]
                linear_model.append(self.decision["right_lm"])
        else:###
            right_predictions=pd.Series(name="prediction",dtype=np.float64)
        if len(left_data)>0:
            if self.decision["left_grow"] and self.left_node and self.left_node.fitted:
                left_predictions,lm=self.left_node.explain_prediction(left_data)
                linear_model.extend(lm)
            else:####
                left_predictions=self.decision["left_lm"].predict(left_data[self.numerical_features])
                left_data["prediction"]=left_predictions
                left_predictions=left_data["prediction"]
                linear_model.append(self.decision["left_lm"])
        else:#####
            left_predictions=pd.Series(name="prediction",dtype=np.float64)
        data=data.merge(pd.concat([right_predictions,left_predictions]), left_index=True, right_index=True)
        return data["prediction"],linear_model
    