# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:20:21 2023

@author: tsharma2
"""

import argparse
from numpy.random import seed
import pandas as pd
import random as rn
import os
import numpy as np
import time
from DenseAE import DenseTied, WeightsOrthogonalityConstraint, UncorrelatedFeaturesConstraint
#from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.preprocessing import MinMaxScaler
# from autoencoder_model import AutoEncoders
import keras
from keras import layers
from keras.models import model_from_json
from keras.layers import Dense
from keras import Sequential
from keras.constraints import UnitNorm
from keras.models import Model

from preProcess import run_cv_gender_race_comb, get_dataset, get_MicroRNA, standarize_dataset, normalize_dataset, get_n_years, get_Methylation, get_independent_data_single
from classify_all import run_mixture_cv, run_one_race_cv, run_unsupervised_transfer_cv, run_CCSA_transfer, run_supervised_transfer_cv, run_naive_transfer_cv, FADA_classification 
from tensorflow import set_random_seed

seed(11111)
set_random_seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)

folderISAAC = 'Protein_100Features_Test/'
if os.path.exists(folderISAAC)!=True:
    folderISAAC = './'

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("cancer_type", type=str, help="Cancer Type")
    parser.add_argument("feature_type", type=str, help="Feature Type")
    parser.add_argument("target", type=str, help="Clinical Outcome Endpoint")
    parser.add_argument("years", type=int, help="Event Time Threhold (Years)")
    parser.add_argument("target_domain", type=str, help="Target Group")
    parser.add_argument("features_count", type=int, help="No. of Features")
    parser.add_argument("autoencoders_OutLoop_val", type=int, help="0 for False, 1 for True")
    parser.add_argument("Original_features_val", type=int, help="0 for False, 1 for True")
    parser.add_argument("data_sel_val", type=int, help="0 for False, 1 for True")
    parser.add_argument("AE_MLTask_val", type=int, help="0 for False, 1 for True")
    parser.add_argument("WT_AE_val", type=int, help="0 for False, 1 for True")
    args = parser.parse_args()
    print(args)
    features_count = args.features_count # no. of features to be selected
    autoencoders_OutLoop = True if args.autoencoders_OutLoop_val==1 else False # set this True if want to use AE for feature selection
    Original_features = True if args.Original_features_val==1 else False # set this False if want to apply feature selection/ dimensionality reduction
    data_sel = True if args.data_sel_val==1 else False # Multiomics case only: set this True if want to select equal features from the combination of two omics
    WT_AE = True if args.WT_AE_val==1 else False # set this True if want to use weight tied AE for feature selection
    AE_MLTask = True if args.AE_MLTask_val==1 else False # set this True if want to train separate AEs for each ML task
    print('autoencoders_OutLoop is '+str(autoencoders_OutLoop))
    print('Original_features is '+str(Original_features))
    print('data_sel is '+str(data_sel))
    print('WT_AE is '+str(WT_AE))
    print('AE_MLTask is '+str(AE_MLTask))
    
    cancer_type = args.cancer_type
    feature_type = args.feature_type
    target = args.target
    years = args.years
    source_domain = 'WHITE'
    genders = ("MALE","FEMALE")
    target_domain = args.target_domain
    
    groups = (source_domain,target_domain)
    data_Category = 'R' # 'R', 'GR' ; it is 'GR' if MGtoMGF (Or) MGtoMGM = True
    if data_Category=='GR':
        MGtoMGF = True
        MGtoMGM = True
    else:
        MGtoMGF = False
        MGtoMGM = False
    
    TaskName = 'TCGA-'+cancer_type+'-'+str(feature_type)+'-'+ groups[0]+'-'+groups[1]+'-'+target+'-'+str(years)+'YR'
    out_file_name = folderISAAC + 'Result/' + TaskName + '.xlsx'
    CCSA_path = folderISAAC +'CCSA_data/' + TaskName + '/CCSA_pairs'
    checkpt_path = folderISAAC+'ckpt/FADA_'+TaskName+'_checkpoint.pt'
    if Original_features:
        k = -1
        out_file_name = folderISAAC + 'Result/' + TaskName + '_OF.xlsx'
    if data_sel:
        out_file_name = folderISAAC + 'Result/' + TaskName + '_100Equal.xlsx'
    if autoencoders_OutLoop:
        out_file_name = folderISAAC + 'Result/' + TaskName + '_AE.xlsx'
    if WT_AE:
        out_file_name = folderISAAC + 'Result/' + TaskName + '_WT_AE.xlsx'
    print("===============================================")
    print(out_file_name)
    print("===============================================")
    
    if os.path.exists(out_file_name)!=True:
        if len(np.shape(feature_type))==0:
            print('Single Omics')
        else:
            print('Multi Omics')
            feat_1 = feature_type[0]
            feat_2 = feature_type[1]
            if feat_1=='Protein':
                f_num = 189
            elif feat_1=='mRNA':
                f_num = 17176
            elif feat_1=='MicroRNA':
                f_num = 662
            elif feat_1=='Methylation':
                f_num = 11882

        if autoencoders_OutLoop:
            AE_train = False
            if data_sel: # this can only be in multiomics case
                if AE_MLTask:
                    AE_ModelName_1 = 'TCGA-'+cancer_type+'-'+feat_1+'-'+ groups[0]+'-'+groups[1]+'-'+target+'-'+str(years)
                    AE_ModelName_2 = 'TCGA-'+cancer_type+'-'+feat_2+'-'+ groups[0]+'-'+groups[1]+'-'+target+'-'+str(years)
                else:
                    AE_ModelName_1 = 'TCGA-'+cancer_type+'-'+feat_1+'-'+ groups[0]+'-'+groups[1]
                    AE_ModelName_2 = 'TCGA-'+cancer_type+'-'+feat_2+'-'+ groups[0]+'-'+groups[1]
                AE_json_file_name_1 = folderISAAC+'AE_models/autoencoder_'+AE_ModelName_1+'.json'
                E_json_file_name_1 = folderISAAC+'AE_models/encoder_'+AE_ModelName_1+'.json'
                AE_json_file_name_2 = folderISAAC+'AE_models/autoencoder_'+AE_ModelName_2+'.json'
                E_json_file_name_2 = folderISAAC+'AE_models/encoder_'+AE_ModelName_2+'.json'
            else:
                if AE_MLTask:
                    AE_ModelName = 'TCGA-'+cancer_type+'-'+str(feature_type)+'-'+ groups[0]+'-'+groups[1]+'-'+target+'-'+str(years)
                else:
                    AE_ModelName = 'TCGA-'+cancer_type+'-'+str(feature_type)+'-'+ groups[0]+'-'+groups[1]
                AE_json_file_name = folderISAAC+'AE_models/autoencoder_'+AE_ModelName+'.json'
                E_json_file_name = folderISAAC+'AE_models/encoder_'+AE_ModelName+'.json'
            while(AE_train==False):
                if data_sel:
                    Condition_AE = (os.path.exists(AE_json_file_name_1) and os.path.exists(E_json_file_name_1) and os.path.exists(AE_json_file_name_2) and os.path.exists(E_json_file_name_2))
                else:
                    Condition_AE = (os.path.exists(AE_json_file_name) and os.path.exists(E_json_file_name))
                if Condition_AE==1:
                    # No AE training needed
                    AE_train = False
                    print('$$$$$$$$$$$$$$$$$$$$$$')
                    print('AE training is not needed.')
                    print('$$$$$$$$$$$$$$$$$$$$$$')
                    break
                else:
                    # AE training is needed
                    AE_train = True
                    print('$$$$$$$$$$$$$$$$$$$$$$')
                    print('AE training is needed.')
                    print('$$$$$$$$$$$$$$$$$$$$$$')
                    if feature_type=='mRNA':
                        k=features_count
                        dataset = get_dataset(cancer_type=cancer_type,feature_type=feature_type,target=target,groups=groups,Gender=genders,data_Category=data_Category,autoencoders_OutLoop=autoencoders_OutLoop)
                    elif feature_type=='MicroRNA':
                        k=features_count
                        dataset = get_MicroRNA(cancer_type=cancer_type,target=target,groups=groups,Gender=genders,data_Category=data_Category,autoencoders_OutLoop=autoencoders_OutLoop)
                    elif feature_type=='Protein':
                        dataset = get_dataset(cancer_type=cancer_type,feature_type=feature_type,target=target,groups=groups,Gender=genders,data_Category=data_Category,autoencoders_OutLoop=autoencoders_OutLoop)
                        if features_count<=(np.shape(dataset['X'])[1]):
                            k = features_count
                        else:
                            k = -1
                    elif feature_type=='Methylation':
                        k=features_count
                        dataset = get_Methylation(cancer_type=cancer_type,target=target,groups=groups,Gender=genders,data_Category=data_Category,autoencoders_OutLoop=autoencoders_OutLoop)
                    else:
                        k=features_count
                        dataset = run_cv_gender_race_comb(cancer_type=cancer_type,feature_type=feature_type,target=target,genders=genders,groups=groups,data_Category=data_Category,autoencoders_OutLoop=autoencoders_OutLoop)
                    
                    dataset = standarize_dataset(dataset)
                    
                    if data_sel==True:
                        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                        print('Train two different AEs for '+str(k//2)+'+'+str(k//2)+' features for the combination of two features')
                        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                        X = dataset['X']
                        encoding_dim = k//2
                        print(encoding_dim)
                        standard_scaler = MinMaxScaler()
                        X_1 = X[:,0:f_num]
                        X_2 = X[:,f_num:]
                        #X_AE = pd.DataFrame(standard_scaler.fit_transform(X))
                        X_AE_1 = pd.DataFrame(standard_scaler.fit_transform(X_1))
                        X_AE_2 = pd.DataFrame(standard_scaler.fit_transform(X_2))
                        #encoded_input = keras.Input(shape=(X_AE.shape[1],))
                        encoded_input_1 = keras.Input(shape=(X_AE_1.shape[1],))
                        encoded_input_2 = keras.Input(shape=(X_AE_2.shape[1],))
                        #encoded = layers.Dense(encoding_dim, activation='relu', name='encoder')(encoded_input)
                        encoded_1 = layers.Dense(encoding_dim, activation='relu', name='encoder_1')(encoded_input_1)
                        encoded_2 = layers.Dense(encoding_dim, activation='relu', name='encoder_2')(encoded_input_2)
                        #decoded = layers.Dense(np.shape(X_AE)[1], activation='sigmoid')(encoded)
                        decoded_1 = layers.Dense(np.shape(X_AE_1)[1], activation='sigmoid')(encoded_1)
                        decoded_2 = layers.Dense(np.shape(X_AE_2)[1], activation='sigmoid')(encoded_2)
                        #autoencoder = keras.Model(encoded_input, decoded)
                        autoencoder_1 = keras.Model(encoded_input_1, decoded_1)
                        autoencoder_2 = keras.Model(encoded_input_2, decoded_2)
                        #encoder = keras.Model(encoded_input, encoded)
                        encoder_1 = keras.Model(encoded_input_1, encoded_1)
                        encoder_2 = keras.Model(encoded_input_2, encoded_2)
                        #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
                        autoencoder_1.compile(optimizer='adam', loss='binary_crossentropy')
                        autoencoder_2.compile(optimizer='adam', loss='binary_crossentropy')
                        #autoencoder.fit(X_AE, X_AE,epochs=50,batch_size=20,shuffle=True)
                        autoencoder_1.fit(X_AE_1, X_AE_1,epochs=50,batch_size=20,shuffle=True)
                        autoencoder_2.fit(X_AE_2, X_AE_2,epochs=50,batch_size=20,shuffle=True)
                        #json_model = autoencoder.to_json()
                        json_model_1 = autoencoder_1.to_json()
                        json_model_2 = autoencoder_2.to_json()
                        #json_file = open(AE_json_file_name, 'w')
                        json_file_1 = open(AE_json_file_name_1, 'w')
                        json_file_2 = open(AE_json_file_name_2, 'w')
                        #json_file.write(json_model)
                        json_file_1.write(json_model_1)
                        json_file_2.write(json_model_2)
                        #json_model = encoder.to_json()
                        json_model_1 = encoder_1.to_json()
                        json_model_2 = encoder_2.to_json()
                        #json_file = open(E_json_file_name, 'w')
                        json_file_1 = open(E_json_file_name_1, 'w')
                        json_file_2 = open(E_json_file_name_2, 'w')
                        #json_file.write(json_model)
                        json_file_1.write(json_model_1)
                        json_file_2.write(json_model_2)
                        #json_file.close()
                        json_file_1.close()
                        json_file_2.close()
                        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                        print('Two different AE models have been trained and saved.')
                        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                    else:
                        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                        print('Train AE for '+str(k)+' features')
                        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                        X = dataset['X']
                        encoding_dim = k
                        standard_scaler = MinMaxScaler()
                        X_AE = pd.DataFrame(standard_scaler.fit_transform(X))
                        encoded_input = keras.Input(shape=(X_AE.shape[1],))
                        encoded = layers.Dense(encoding_dim, activation='relu', name='encoder')(encoded_input)
                        decoded = layers.Dense(np.shape(X_AE)[1], activation='sigmoid')(encoded)
                        autoencoder = keras.Model(encoded_input, decoded)
                        encoder = keras.Model(encoded_input, encoded)
                        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
                        autoencoder.fit(X_AE, X_AE,epochs=50,batch_size=20,shuffle=True)
                        json_model = autoencoder.to_json()
                        json_file = open(AE_json_file_name, 'w')
                        json_file.write(json_model)
                        json_model = encoder.to_json()
                        json_file = open(E_json_file_name, 'w')
                        json_file.write(json_model)
                        json_file.close()
                    
                    AE_train = False
    
            if data_sel:
                print('$$$$$$$$$$$$$$$$$$$$')
                print('Reading saved AE models...')
                print('$$$$$$$$$$$$$$$$$$$$')
                #json_file = open(AE_json_file_name, 'r')
                json_file_1 = open(AE_json_file_name_1, 'r')
                json_file_2 = open(AE_json_file_name_2, 'r')
                #loaded_model_json = json_file.read()
                loaded_model_json_1 = json_file_1.read()
                loaded_model_json_2 = json_file_2.read()
                #json_file.close()
                json_file_1.close()
                json_file_2.close()
                #autoencoder = model_from_json(loaded_model_json)
                autoencoder_1 = model_from_json(loaded_model_json_1)
                autoencoder_2 = model_from_json(loaded_model_json_2)
                #json_file = open(E_json_file_name, 'r')
                json_file_1 = open(E_json_file_name_1, 'r')
                json_file_2 = open(E_json_file_name_2, 'r')
                #loaded_model_json = json_file.read()
                loaded_model_json_1 = json_file_1.read()
                loaded_model_json_2 = json_file_2.read()
                #json_file.close()
                json_file_1.close()
                json_file_2.close()
                #encoder = model_from_json(loaded_model_json)
                encoder_1 = model_from_json(loaded_model_json_1)
                encoder_2 = model_from_json(loaded_model_json_2)
            else:
                json_file = open(AE_json_file_name, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                autoencoder = model_from_json(loaded_model_json)
                json_file = open(E_json_file_name, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                encoder = model_from_json(loaded_model_json)
            
        if feature_type=='mRNA':
            dataset = get_dataset(cancer_type=cancer_type,feature_type=feature_type,target=target,groups=groups,Gender=genders,data_Category=data_Category,autoencoders_OutLoop=False)
        elif feature_type=='MicroRNA':
            dataset = get_MicroRNA(cancer_type=cancer_type,target=target,groups=groups,Gender=genders,data_Category=data_Category,autoencoders_OutLoop=False)
        elif feature_type=='Protein':
            dataset = get_dataset(cancer_type=cancer_type,feature_type=feature_type,target=target,groups=groups,Gender=genders,data_Category=data_Category,autoencoders_OutLoop=False)
        elif feature_type=='Methylation':
            dataset = get_Methylation(cancer_type=cancer_type,target=target,groups=groups,Gender=genders,data_Category=data_Category,autoencoders_OutLoop=False)
        else:
            dataset = run_cv_gender_race_comb(cancer_type=cancer_type,feature_type=feature_type,target=target,genders=genders,groups=groups,data_Category=data_Category,autoencoders_OutLoop=False)
        if features_count<=(np.shape(dataset['X'])[1]):
            k = features_count
        else:
            k = -1
        dataset = standarize_dataset(dataset)
        
        if WT_AE:
            X = dataset['X']
            encoding_dim = k
            standard_scaler = MinMaxScaler()
            standard_scaler.fit(X)
            X_AE = pd.DataFrame(standard_scaler.transform(X))
            input_dim = np.shape(X_AE)[1]
            encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True, kernel_constraint=UnitNorm(axis=0),
                            activity_regularizer=UncorrelatedFeaturesConstraint(encoding_dim, weightage=1.),
                            kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=0)) 
            decoder = DenseTied(input_dim, activation="linear", tied_to=encoder, use_bias = True, kernel_constraint=UnitNorm(axis=1),
                            kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=1))
            autoencoder = Sequential()
            autoencoder.add(encoder)
            autoencoder.add(decoder)
            autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
            autoencoder.summary()
            autoencoder.fit(X_AE, X_AE, epochs=100, batch_size=16, shuffle=True, verbose=1)
            encoder = Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[0].output)
            reduced_df_X_AE = pd.DataFrame(encoder.predict(np.array(X_AE)))
            X_AE = np.array(reduced_df_X_AE)
        k = -1
        dataset['X'] = X_AE

        if autoencoders_OutLoop:
            X = dataset['X']
            standard_scaler = MinMaxScaler()
            if data_sel:
                X_1 = X[:,0:f_num]
                X_2 = X[:,f_num:]
                X_AE_1 = pd.DataFrame(standard_scaler.fit_transform(X_1))
                X_AE_2 = pd.DataFrame(standard_scaler.fit_transform(X_2))
                reduced_df_X_AE_1 = pd.DataFrame(encoder_1.predict(np.array(X_AE_1)))
                reduced_df_X_AE_2 = pd.DataFrame(encoder_2.predict(np.array(X_AE_2)))
                X_AE = np.concatenate((np.array(reduced_df_X_AE_1),np.array(reduced_df_X_AE_2)),axis=1)
                print('$$$$$$$$$$$$$$$$$$$$')
                print('Data has been concatenated...')
                print(np.shape(X_AE_1))
                print(np.shape(X_AE_2))
                print(np.shape(X_AE))
                print('$$$$$$$$$$$$$$$$$$$$')
            else:
                X_AE = pd.DataFrame(standard_scaler.fit_transform(X))
                reduced_df_X_AE = pd.DataFrame(encoder.predict(np.array(X_AE)))
                X_AE = np.array(reduced_df_X_AE)
            k = -1
            dataset['X'] = X_AE
                
        print (cancer_type, feature_type, target, years, source_domain, target_domain)
        print('The value of k is: '+str(k))
        print(np.shape(dataset['X']))
        
        dataset = standarize_dataset(dataset)
        
        ## Independent Learning datasets ##
        # Independent - WHITE
        data_w = get_independent_data_single(dataset, 'WHITE', groups, genders)
        data_w = get_n_years(data_w, years)
        # Independent - MG
        data_b = get_independent_data_single(dataset, 'MG', groups, genders)
        data_b = get_n_years(data_b, years)
        if data_Category=='GR':
            # Independent - WHITE-FEMALE
            data_wf = get_independent_data_single(dataset, 'WHITE-FEMALE', groups, genders)
            data_wf = get_n_years(data_wf, years)
            # Independent - WHITE-MALE
            data_wm = get_independent_data_single(dataset, 'WHITE-MALE', groups, genders)
            data_wm = get_n_years(data_wm, years)
            # Independent - BLACK-FEMALE
            data_bf = get_independent_data_single(dataset, 'BLACK-FEMALE', groups, genders)
            data_bf = get_n_years(data_bf, years)
            # Independent - BLACK-MALE
            data_bm = get_independent_data_single(dataset, 'BLACK-MALE', groups, genders)
            data_bm = get_n_years(data_bm, years)
        ## Mixture, Naive Transfer, and Transfer Learning dataset ##
        dataset_tl_ccsa = normalize_dataset(dataset)
        dataset_tl_ccsa = get_n_years(dataset_tl_ccsa, years)
        dataset = get_n_years(dataset, years)
        
        X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = dataset
        df = pd.DataFrame(y_strat, columns=['RY'])
        df['GRY'] = GRy_strat
        df['GY'] = Gy_strat
        df['R'] = R
        df['G'] = G
        df['Y'] = Y
        print(X.shape)
        print(df['GRY'].value_counts())#gender with prognosis counts
        print(df['GY'].value_counts())#gender with prognosis counts
        print(df['G'].value_counts())#gender counts
        print(df['RY'].value_counts())#race with prognosis counts
        print(df['R'].value_counts())#race counts
        print(df['Y'].value_counts())#progonsis counts
        
        ###############################
        # parameters #
        ###############################
        parametrs_mix = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,'momentum':0.9, 'learning_rate':0.01,
                        'lr_decay':0.03, 'dropout':0.5, 'L1_reg': 0.001,'L2_reg': 0.001, 'hiddenLayers': [128,64]}
        parameters_MAJ = {'fold':3, 'k':k, 'batch_size':20, 'lr_decay':0.03, 'val_size':0.0, 'learning_rate':0.01,
                        'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_MIN = {'fold':3, 'k':k, 'batch_size':4, 'lr_decay':0.03, 'val_size':0.0, 'learning_rate':0.01,
                        'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_NT  = {'fold':3, 'k':k, 'batch_size':20, 'momentum':0.9, 'lr_decay':0.03, 'val_size':0.0,
                        'learning_rate':0.01, 'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_TL1 = {'fold':3, 'k':k, 'batch_size':20, 'momentum':0.9, 'lr_decay':0.03, 'val_size':0.0,
                        'learning_rate':0.01, 'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64],
                        'train_epoch':100, 'tune_epoch':100, 'tune_lr':0.002, 'tune_batch':10}
        parameters_TL2 = {'fold':3, 'k':k, 'batch_size':10, 'lr_decay':0.03, 'val_size':0.0, 'learning_rate':0.002,
                        'n_epochs':100, 'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_TL3 = {'fold':3, 'n_features':k, 'alpha':0.3, 'batch_size':20, 'learning_rate':0.01, 'hiddenLayers':[100],
                        'dr':0.5, 'momentum':0.9, 'decay':0.03, 'sample_per_class':None, 'SourcePairs':False}
        parameters_TL4 = {'fold':3, 'n_features':k, 'alpha':0.25, 'batch_size':20, 'learning_rate':0.01, 'hiddenLayers':[128,64],
                        'dr':0.5, 'momentum':0.9, 'decay':0.03, 'sample_per_class':None, 'EarlyStop':False,
                        'L1_reg':0.001, 'L2_reg':0.001, 'patience':100, 'n_epochs':100}
        
        res = pd.DataFrame()
        
        for i in range(20):
            
            print('###########################')
            print('Interation no.: '+str(i+1))
            print('###########################')
            
            seed = i
            
            start_iter = time.time()
            
            df_mix_200 = run_mixture_cv(seed, dataset, groups, genders, data_Category, MGtoMGF, MGtoMGM, 
                                        feature_type, data_sel, **parametrs_mix)
            print('###########################')
            print('Mixture is done')
            print('###########################')
            
            df_w = run_one_race_cv(seed, data_w, feature_type, data_sel, **parameters_MAJ)
            df_w = df_w.rename(columns={"Auc": "W_ind"})
            print('###########################')
            print('Independent EA is done.')
            print('###########################')
            df_b = run_one_race_cv(seed, data_b, feature_type, data_sel, **parameters_MIN)
            df_b = df_b.rename(columns={"Auc": "B_ind"})
            print('###########################')
            print('Independent MG is done.')
            print('###########################')
            if data_Category=='GR':
                if MGtoMGF:
                    df_wf = run_one_race_cv(seed, data_wf, feature_type, data_sel, **parameters_MAJ)
                    df_wf = df_wf.rename(columns={"Auc": "WF_ind"})
                    print('Independent EA(F) is done.')
                    df_bf = run_one_race_cv(seed, data_bf, feature_type, data_sel, **parameters_MIN)
                    df_bf = df_bf.rename(columns={"Auc": "BF_ind"})
                    print('Independent MG(F) is done.')
                if MGtoMGM:
                    df_wm = run_one_race_cv(seed, data_wm, feature_type, data_sel, **parameters_MAJ)
                    df_wm = df_wm.rename(columns={"Auc": "WM_ind"})
                    print('Independent EA(M) is done.')
                    df_bm = run_one_race_cv(seed, data_bm, feature_type, data_sel, **parameters_MIN)
                    df_bm = df_bm.rename(columns={"Auc": "BM_ind"})
                    print('Independent MG(M) is done.')
                print('###########################')
                print('Independent for Gender Racial Compositions is done.')
                print('###########################')
                
            df_nt = run_naive_transfer_cv(seed, 'WHITE', 'MG', dataset, groups, genders, 
                                          feature_type, data_sel, **parameters_NT)
            df_nt = df_nt.rename(columns={"NT_Auc": "NT_Auc"})
            if MGtoMGF: 
                df_nt_f = run_naive_transfer_cv(seed, 'WHITE', 'MG-FEMALE', dataset_tl_ccsa, groups, genders, 
                                                feature_type, data_sel, **parameters_NT)
                df_nt_f = df_nt_f.rename(columns={"NT_Auc": "NT_Auc_MGF"})
                print('Naive Transfer is done for MG-FEMALE.')
            if MGtoMGM: 
                df_nt_m = run_naive_transfer_cv(seed, 'WHITE', 'MG-MALE', dataset_tl_ccsa, groups, genders, 
                                                feature_type, data_sel, **parameters_NT)
                df_nt_m = df_nt_m.rename(columns={"NT_Auc": "NT_Auc_MGM"})
                print('Naive Transfer is done for MG-MALE.')
            print('###########################')
            print('Naive Transfer is done.')
            print('###########################')
            
            if data_Category=='GR':
                df_tl_sup_EAF = run_supervised_transfer_cv(seed, 'WHITE-FEMALE', 'MG-FEMALE', dataset, groups, genders, False, False, 
                                                            feature_type, data_sel, **parameters_TL1)
                df_tl_sup_EAF = df_tl_sup_EAF.rename(columns={"TL_Auc": "TL_sup_EAF_MGF"})
                print('Supervised is done for EA(F).')
                df_tl_sup_EAM = run_supervised_transfer_cv(seed, 'WHITE-MALE', 'MG-MALE', dataset, groups, genders, False, False, 
                                                            feature_type, data_sel, **parameters_TL1)
                df_tl_sup_EAM = df_tl_sup_EAM.rename(columns={"TL_Auc": "TL_sup_EAM_MGM"})
                print('Supervised is done for EA(M).')
                df_tl_sup_EA_EAF = run_supervised_transfer_cv(seed, 'WHITE', 'MG-FEMALE', dataset, groups, genders, False, False, 
                                                              feature_type, data_sel, **parameters_TL1)
                df_tl_sup_EA_EAF = df_tl_sup_EA_EAF.rename(columns={"TL_Auc": "TL_sup_EA_MGF"})
                print('Supervised is done for EA--MG(F).')
            df_tl_sup = run_supervised_transfer_cv(seed, 'WHITE', 'MG', dataset, groups, genders, MGtoMGF, MGtoMGM, 
                                                    feature_type, data_sel, **parameters_TL1)
            if data_Category=='GR':
                if MGtoMGF and MGtoMGM:
                    df_tl_sup = df_tl_sup.rename(columns={"TL_Auc": "TL_sup","TL_Auc_MGF": "TL_sup_MGF","TL_Auc_MGM": "TL_sup_MGM"})
                if MGtoMGF:
                    df_tl_sup = df_tl_sup.rename(columns={"TL_Auc": "TL_sup","TL_Auc_MGF": "TL_sup_MGF"})
                if MGtoMGM:
                    df_tl_sup = df_tl_sup.rename(columns={"TL_Auc": "TL_sup","TL_Auc_MGM": "TL_sup_MGM"})
            elif data_Category=='R':
                df_tl_sup = df_tl_sup.rename(columns={"TL_Auc": "TL_sup"})
            print('###########################')
            print('Supervised is done.')
            print('###########################')
            
            if data_Category=='GR':
                df_tl_unsup_EAM = run_unsupervised_transfer_cv(seed, 'WHITE-MALE', 'MG-MALE', dataset_tl_ccsa, groups, genders, False, False, 
                                                                feature_type, data_sel, **parameters_TL2)
                df_tl_unsup_EAM = df_tl_unsup_EAM.rename(columns={"TL_Auc": "TL_unsup_EAM_MGM"})
                print('Unsupervised is done for EA(M).')
                df_tl_unsup_EAF = run_unsupervised_transfer_cv(seed, 'WHITE-FEMALE', 'MG-FEMALE', dataset_tl_ccsa, groups, genders, False, False, 
                                                                feature_type, data_sel, **parameters_TL2)
                df_tl_unsup_EAF = df_tl_unsup_EAF.rename(columns={"TL_Auc": "TL_unsup_EAF_MGF"})
                print('Unsupervised is done for EA(F).')
                df_tl_unsup_EA_EAF = run_unsupervised_transfer_cv(seed, 'WHITE', 'MG-FEMALE', dataset_tl_ccsa, groups, genders, False, False, 
                                                                  feature_type, data_sel, **parameters_TL2)
                df_tl_unsup_EA_EAF = df_tl_unsup_EA_EAF.rename(columns={"TL_Auc": "TL_unsup_EA_MGF"})
                print('Unsupervised is done for EA--MG(F).')
            df_tl_unsup = run_unsupervised_transfer_cv(seed, 'WHITE', 'MG', dataset, groups, genders, MGtoMGF, MGtoMGM, 
                                                        feature_type, data_sel, **parameters_TL2)
            if data_Category=='GR':
                if MGtoMGF and MGtoMGM:
                    df_tl_unsup = df_tl_unsup.rename(columns={"TL_Auc": "TL_unsup","TL_Auc_MGF": "TL_unsup_MGF","TL_Auc_MGM": "TL_unsup_MGM"})
                if MGtoMGF:
                    df_tl_unsup = df_tl_unsup.rename(columns={"TL_Auc": "TL_unsup","TL_Auc_MGF": "TL_unsup_MGF"})
                if MGtoMGM:
                    df_tl_unsup = df_tl_unsup.rename(columns={"TL_Auc": "TL_unsup","TL_Auc_MGM": "TL_unsup_MGM"})
            elif data_Category=='R':
                df_tl_unsup = df_tl_unsup.rename(columns={"TL_Auc": "TL_unsup"})
            print('###########################')
            print('Unsupervised is done.')
            print('###########################')
            
            if data_Category=='GR':
                df_tl_ccsa_EAM = run_CCSA_transfer(seed, 'WHITE-MALE', 'MG-MALE', dataset_tl_ccsa, groups, genders, False, False, CCSA_path,
                                                    feature_type, data_sel, **parameters_TL3)
                df_tl_ccsa_EAM = df_tl_ccsa_EAM.rename(columns={"TL_Auc": "TL_ccsa_EAM_MGM"})
                print('CCSA is done for EA(M).')
                df_tl_ccsa_EAF = run_CCSA_transfer(seed, 'WHITE-FEMALE', 'MG-FEMALE', dataset_tl_ccsa, groups, genders, False, False, CCSA_path, 
                                                    feature_type, data_sel, **parameters_TL3)
                df_tl_ccsa_EAF = df_tl_ccsa_EAF.rename(columns={"TL_Auc": "TL_ccsa_EAF_MGF"})
                print('CCSA is done for EA(F).')
                df_tl_ccsa_EA_EAF = run_CCSA_transfer(seed, 'WHITE', 'MG-FEMALE', dataset_tl_ccsa, groups, genders, False, False, CCSA_path, 
                                                        feature_type, data_sel, **parameters_TL3)
                df_tl_ccsa_EA_EAF = df_tl_ccsa_EA_EAF.rename(columns={"TL_Auc": "TL_ccsa_EA_MGF"})
                print('CCSA is done for EA--MG(F).')
            df_tl_ccsa = run_CCSA_transfer(seed, 'WHITE', 'MG', dataset, groups, genders, 
                                            MGtoMGF, MGtoMGM, CCSA_path, feature_type, data_sel, 
                                            **parameters_TL3)
            if data_Category=='GR':
                if MGtoMGF and MGtoMGM:
                    df_tl_ccsa = df_tl_ccsa.rename(columns={"TL_Auc": "TL_ccsa", "TL_Auc_MGF": "TL_ccsa_MGF", "TL_Auc_MGM": "TL_ccsa_MGM"})
                if MGtoMGF:
                    df_tl_ccsa = df_tl_ccsa.rename(columns={"TL_Auc": "TL_ccsa", "TL_Auc_MGF": "TL_ccsa_MGF"})
                if MGtoMGM:
                    df_tl_ccsa = df_tl_ccsa.rename(columns={"TL_Auc": "TL_ccsa", "TL_Auc_MGM": "TL_ccsa_MGM"})
            elif data_Category=='R':
                df_tl_ccsa = df_tl_ccsa.rename(columns={"TL_Auc": "TL_ccsa"})
            print('###########################')
            print('CCSA is done.')
            print('###########################')
            
            if data_Category=='GR':
                df_tl_fada_EAM = FADA_classification(seed, 'WHITE-MALE', 'MG-MALE', dataset, groups, genders, False, False, checkpt_path, 
                                                      feature_type, data_sel, **parameters_TL4)
                df_tl_fada_EAM = df_tl_fada_EAM.rename(columns={"TL_DCD_Auc":"TL_FADA_EAM_MGM"})
                print('FADA is done for EA(M).')
                df_tl_fada_EAF = FADA_classification(seed, 'WHITE-FEMALE', 'MG-FEMALE', dataset, groups, genders, False, False, checkpt_path, 
                                                      feature_type, data_sel, **parameters_TL4)
                df_tl_fada_EAF = df_tl_fada_EAF.rename(columns={"TL_DCD_Auc":"TL_FADA_EAF_MGF"})
                print('FADA is done for EA(F).')
                df_tl_fada_EA_EAF = FADA_classification(seed, 'WHITE', 'MG-FEMALE', dataset, groups, genders, False, False, checkpt_path, 
                                                        feature_type, data_sel, **parameters_TL4)
                df_tl_fada_EA_EAF = df_tl_fada_EA_EAF.rename(columns={"TL_DCD_Auc": "TL_FADA_EA_MGF"})
                print('FADA is done for EA--MG(F).')
            df_tl_fada = FADA_classification(seed, 'WHITE', 'MG', dataset, groups, genders, MGtoMGF, MGtoMGM, checkpt_path, 
                                              feature_type, data_sel, **parameters_TL4)
            if data_Category=='GR':
                if MGtoMGF and MGtoMGM:
                    df_tl_fada = df_tl_fada.rename(columns={"TL_DCD_Auc":"TL_FADA", "TL_Auc_MGF":"TL_FADA_MGF", "TL_Auc_MGM":"TL_FADA_MGM"})
                if MGtoMGF:
                    df_tl_fada = df_tl_fada.rename(columns={"TL_DCD_Auc":"TL_FADA", "TL_Auc_MGF":"TL_FADA_MGF"})
                if MGtoMGM:
                    df_tl_fada = df_tl_fada.rename(columns={"TL_DCD_Auc":"TL_FADA", "TL_Auc_MGM":"TL_FADA_MGM"})
            elif data_Category=='R':
                df_tl_fada = df_tl_fada.rename(columns={"TL_DCD_Auc":"TL_FADA"})
            print('###########################')
            print('FADA is done.')
            print('###########################')
            
            end_iter = time.time()
            print("The time of loop execution is :", end_iter-start_iter)
            
            timeFor_iter = pd.DataFrame({'Time':[end_iter - start_iter]},index=[seed])
            if data_Category=='R':
                df1 = pd.concat([timeFor_iter,
                                    df_mix_200,
                                    df_w, df_b, 
                                    df_nt,
                                    df_tl_sup, 
                                    df_tl_unsup,
                                    df_tl_ccsa, 
                                    df_tl_fada
                                    ], sort=False, axis=1)
            elif data_Category=='GR':
                if MGtoMGF:
                    df1 = pd.concat([timeFor_iter, 
                                    df_mix_200,
                                    df_w, df_b, df_wf, df_bf, 
                                    df_nt, df_nt_f,
                                    df_tl_sup, df_tl_sup_EAF, df_tl_sup_EAM, df_tl_sup_EA_EAF,
                                    df_tl_unsup, df_tl_unsup_EAF, df_tl_unsup_EAM, df_tl_unsup_EA_EAF,
                                    df_tl_ccsa, df_tl_ccsa_EAF, df_tl_ccsa_EAM, df_tl_ccsa_EA_EAF,
                                    df_tl_fada, df_tl_fada_EAF, df_tl_fada_EAM, df_tl_fada_EA_EAF
                                    ], sort=False, axis=1)
                if MGtoMGM:
                    df1 = pd.concat([timeFor_iter, 
                                    df_mix_200,
                                    df_w, df_b, df_wm, df_bm, 
                                    df_nt, df_nt_m,
                                    df_tl_sup, df_tl_sup_EAF, df_tl_sup_EAM, df_tl_sup_EA_EAF,
                                    df_tl_unsup, df_tl_unsup_EAF, df_tl_unsup_EAM, df_tl_unsup_EA_EAF,
                                    df_tl_ccsa, df_tl_ccsa_EAF, df_tl_ccsa_EAM, df_tl_ccsa_EA_EAF,
                                    df_tl_fada, df_tl_fada_EAF, df_tl_fada_EAM, df_tl_fada_EA_EAF
                                    ], sort=False, axis=1)
                if MGtoMGM and MGtoMGF:
                    df1 = pd.concat([timeFor_iter, 
                                    df_mix_200,
                                    df_w, df_b, df_wf, df_bf, df_wm, df_bm, 
                                    df_nt, df_nt_f, df_nt_m,
                                    df_tl_sup, df_tl_sup_EAF, df_tl_sup_EAM, df_tl_sup_EA_EAF,
                                    df_tl_unsup, df_tl_unsup_EAF, df_tl_unsup_EAM, df_tl_unsup_EA_EAF,
                                    df_tl_ccsa, df_tl_ccsa_EAF, df_tl_ccsa_EAM, df_tl_ccsa_EA_EAF,
                                    df_tl_fada, df_tl_fada_EAF, df_tl_fada_EAM, df_tl_fada_EA_EAF
                                    ], sort=False, axis=1)
            print (df1)
            res = res.append(df1)
        res.to_excel(out_file_name)

if __name__ == '__main__':
    main()




    