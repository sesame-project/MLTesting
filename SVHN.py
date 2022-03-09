#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.datasets import mnist,cifar10,cifar100
from keras.models import load_model
import metrics
import SVNH_DatasetUtil
import time

def gen_data(use_adv=False,deepxplore=False):
    (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()
    Y_test=np.argmax(Y_test,axis=1)
    model_path='Networks/model_svhn.hdf5'
    if use_adv:
        attack_lst=['cw','fgsm','jsma','bim']
        adv_image_all=[]
        adv_label_all=[]
        for attack in attack_lst:
            adv_image_all.append(np.load('data/adv_image/{}_svhn_image.npy'.format(attack)))
            adv_label_all.append(np.load('data/adv_image/{}_svhn_label.npy'.format(attack)))
        adv_image_all=np.concatenate(adv_image_all,axis=0)
        adv_label_all=np.concatenate(adv_label_all,axis=0)
        test=np.concatenate([X_test,adv_image_all],axis=0)
        true_test=np.concatenate([Y_test,adv_label_all],axis=0)
    else:
        test=X_test
        true_test=Y_test
    train=X_train
    model=load_model(model_path)
    pred_test_prob=model.predict(test)
    pred_test=np.argmax(pred_test_prob,axis=1)
    input=model.layers[0].output
    if not deepxplore:
        layers=[model.layers[2].output,model.layers[3].output,model.layers[5].output,model.layers[6].output,model.layers[8].output,model.layers[9].output,model.layers[10].output]
    else:
        layers=[model.layers[1].output,model.layers[3].output,model.layers[4].output,model.layers[8].output,model.layers[8].output,model.layers[9].output,model.layers[10].output]


    layers=list(zip(4*['conv']+3*['dense'],layers))


    return input,layers,test,train,pred_test,true_test,pred_test_prob

def exp(coverage,use_adv,std=0,k=1,k_bins=1000):
    input,layers,test,train,pred_test,true_test,pred_test_prob=gen_data(use_adv)
    rank_lst2=None
    if coverage=='kmnc':

        km=metrics.kmnc(train,input,layers,k_bins=k_bins)
        rank_lst=km.rank_fast(test)
        rate=km.fit(test)
    elif coverage=='nbc':

        #0 0.5 1
        bc=metrics.nbc(train,input,layers,std=std)
        rank_lst=bc.rank_fast(test,use_lower=True)
        rank_lst2=bc.rank_2(test,use_lower=True)
        rate=bc.fit(test,use_lower=True)
    elif coverage=='snac':

        #0 0.5 1
        bc=metrics.nbc(train,input,layers,std=std)
        rank_lst=bc.rank_fast(test,use_lower=False)
        rank_lst2=bc.rank_2(test,use_lower=False)
        rate=bc.fit(test,use_lower=False)
    elif coverage=='tknc':

        #1 2 3
        tk=metrics.tknc(test,input,layers,k=k)
        rank_lst=tk.rank(test)
        rate=tk.fit(list(range(len(test))))

    df=pd.DataFrame([])
    df['right']=(pred_test==true_test).astype('int')
    df['cam']=0
    df['cam'].loc[rank_lst]=list(range(1,len(rank_lst)+1))
    if rank_lst2 is not None:
        df['ctm']=0
        df['ctm'].loc[rank_lst2]=list(range(1,len(rank_lst2)+1))
    if use_adv:
        dataset='svhn_adv'
    else:
        dataset='svhn'
    df['rate']=rate
    if coverage=='kmnc':
        df.to_csv('data/output_svhn/{}_{}_k_bins_{}.csv'.format(dataset,coverage,k_bins))
    elif coverage=='nbc':
        df.to_csv('data/output_svhn/{}_{}_std_{}.csv'.format(dataset,coverage,std))
    elif coverage=='snac':
        df.to_csv('data/output_svhn/{}_{}_std_{}.csv'.format(dataset,coverage,std))
    elif coverage=='tknc':
        df.to_csv('data/output_svhn/{}_{}_k_{}.csv'.format(dataset,coverage,k))

def exp_nac(use_adv,t):
    input,layers,test,train,pred_test,true_test,pred_test_prob=gen_data(use_adv,deepxplore=False)
    rank_lst2=None

    ac=metrics.nac(test,input,layers,t=t)
    rate=ac.fit()
    rank_lst=ac.rank_fast(test)
    rank_lst2=ac.rank_2(test)

    df=pd.DataFrame([])
    df['right']=(pred_test==true_test).astype('int')
    df['cam']=0
    df['cam'].loc[rank_lst]=list(range(1,len(rank_lst)+1))
    if rank_lst2 is not None:
        df['ctm']=0
        df['ctm'].loc[rank_lst2]=list(range(1,len(rank_lst2)+1))
    if use_adv:
        dataset='svhn_adv'
    else:
        dataset='svhn'
    df['rate']=rate

    df.to_csv('data/output_svhn/{}_nac_t_{}.csv'.format(dataset,t))

def exp_deep_metric(use_adv):
    input,layers,test,train,pred_test,true_test,pred_test_prob=gen_data(use_adv,deepxplore=False)
    rank_lst=metrics.deep_metric(pred_test_prob)
    df=pd.DataFrame([])
    df['right']=(pred_test==true_test).astype('int')
    df['cam']=0
    df['cam'].loc[rank_lst]=list(range(1,len(rank_lst)+1))
    df['rate']=0
    if use_adv:
        dataset='svhn_adv'
    else:
        dataset='svhn'
    df.to_csv('data/output_svhn/{}_deep_metric.csv'.format(dataset))

if __name__=='__main__':
    dic={}

    start=time.time()
    exp_nac(use_adv=False,t=0)
    end=time.time()
    dic['mnist_nac_t_0']=(start-end)


    start=time.time()
    exp_nac(use_adv=False,t=0)
    end=time.time()
    dic['mnist_adv_nac_t_0']=(start-end)

    start=time.time()
    exp_nac(use_adv=False,t=0.75)
    end=time.time()
    dic['mnist_nac_t_0.75']=(start-end)

    start=time.time()
    exp_nac(use_adv=False,t=0.75)
    end=time.time()
    dic['mnist_adv_nac_t_0.75']=(start-end)

    #exp(coverage='kmnc',use_adv=False,k_bins=1000)
    #exp(coverage='kmnc',use_adv=False,k_bins=10000)
    #exp(coverage='kmnc',use_adv=True,k_bins=1000)
    #exp(coverage='kmnc',use_adv=True,k_bins=10000)

    start=time.time()
    exp_deep_metric(use_adv=False)
    end=time.time()
    dic['mnist_ours']=(start-end)

    start=time.time()
    exp_deep_metric(use_adv=False)
    end=time.time()
    dic['mnist_adv_ours']=(start-end)

    start=time.time()
    exp(coverage='tknc',use_adv=False,k=1)
    end=time.time()
    dic['mnist_tknc_k_1']=(start-end)

    start=time.time()
    exp(coverage='tknc',use_adv=False,k=2)
    end=time.time()
    dic['mnist_tknc_k_2']=(start-end)

    start=time.time()
    exp(coverage='tknc',use_adv=False,k=3)
    end=time.time()
    dic['mnist_tknc_k_3']=(start-end)

    start=time.time()
    exp(coverage='tknc',use_adv=False,k=1)
    end=time.time()
    dic['mnist_adv_tknc_k_1']=(start-end)

    start=time.time()
    exp(coverage='tknc',use_adv=False,k=2)
    end=time.time()
    dic['mnist_adv_tknc_k_2']=(start-end)

    start=time.time()
    exp(coverage='tknc',use_adv=False,k=3)
    end=time.time()
    dic['mnist_adv_tknc_k_3']=(start-end)

    start=time.time()
    exp(coverage='nbc',use_adv=False,std=0.5)
    end=time.time()
    dic['mnist_nbc_std_0.5']=(start-end)


    start=time.time()
    exp(coverage='nbc',use_adv=False,std=1)
    end=time.time()
    dic['mnist_nbc_std_1']=(start-end)

    start=time.time()
    exp(coverage='nbc',use_adv=False,std=0)
    end=time.time()
    dic['mnist_nbc_std_0']=(start-end)



    start=time.time()
    exp(coverage='nbc',use_adv=False,std=0.5)
    end=time.time()
    dic['mnist_adv_nbc_std_0.5']=(start-end)

    start=time.time()
    exp(coverage='nbc',use_adv=False,std=1)
    end=time.time()
    dic['mnist_adv_nbc_std_1']=(start-end)

    start=time.time()
    exp(coverage='nbc',use_adv=False,std=0)
    end=time.time()
    dic['mnist_adv_nbc_std_0']=(start-end)


    start=time.time()
    exp(coverage='snac',use_adv=False,std=0.5)
    end=time.time()
    dic['mnist_snac_std_0.5']=(start-end)

    start=time.time()
    exp(coverage='snac',use_adv=False,std=1)
    end=time.time()
    dic['mnist_snac_std_1']=(start-end)

    start=time.time()
    exp(coverage='snac',use_adv=False,std=0)
    end=time.time()
    dic['mnist_snac_std_0']=(start-end)


    start=time.time()
    exp(coverage='snac',use_adv=False,std=0.5)
    end=time.time()
    dic['mnist_adv_snac_std_0.5']=(start-end)

    start=time.time()
    exp(coverage='snac',use_adv=False,std=1)
    end=time.time()
    dic['mnist_adv_snac_std_1']=(start-end)

    start=time.time()
    exp(coverage='snac',use_adv=False,std=0)
    end=time.time()
    dic['mnist_adv_snac_std_0']=(start-end)
    print(dic)