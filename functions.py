import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import copy
import re
import seaborn as sns
from scipy.stats import fisher_exact

# reading conllu file, removing explanation with #, only keeping sentences
def read_file(file):
    sentence = []
    with open(file) as f:
        for line in f:
            
            if line:
                if line[0] != '#':
                    sentence.append(line)
            else:
                line = '#'
    return sentence

def pd_data(file = 'fr_gsd-sud-train.conllu'):
    # reading train file
    train_data = read_file(file)


    # converting into pandas
    train_data = pd.DataFrame(train_data)
    train_data = train_data[0].str.split('\t',expand = True)
    train_data.columns = ['ID', 'FORM', 'LEMMA', 'UPOS','XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']

    # information of train data
    # print(train_data.info())
    # show part of data
    #print(train_data[:20])
    return train_data

# convert ID (object) into ID (int)
def convert_ID (value):
    try:
        value = int(value)
    except Exception as e:
        pass
        
    return value

def ID_integer(data):
    data['ID'] = data['ID'].apply(convert_ID)

    # finding all the indexes that ID is not integer
    index = []
    for ind,value in enumerate (data['ID']):
        if isinstance(value, int) != True:
            index.append(ind)

    # removing rows that ID is not integer
    data = data.drop(index= index, axis= 0)
    data = data.reset_index()
    return data


# only keeping gender and number and person features in the dataframe
def gender_number_person_feature(data):
    index_nonegender = []
    for ind in data.index:
        if data.iloc[ind]['FEATS']:
            #if 'Number[psor]' in str(data.iloc[ind]['FEATS']):
            if 'Gender' not in str(data.iloc[ind]['FEATS']) or 'Number' not in str(data.iloc[ind]['FEATS']) or 'Person' not in str(data.iloc[ind]['FEATS']):
                index_nonegender.append(ind)
    data = data.drop(index= index_nonegender, axis= 0)   
    return data

# only keeping  number features in the dataframe
def number_feature(data):
    index_nonegender = []
    for ind in data.index:
        if data.iloc[ind]['FEATS']:
            #if 'Number[psor]' in str(data.iloc[ind]['FEATS']):
            if 'Number' not in str(data.iloc[ind]['FEATS']):
                index_nonegender.append(ind)
    data = data.drop(index= index_nonegender, axis= 0)   
    return data


# only keeping  number features in the dataframe
def gender_feature(data):
    index_nonegender = []
    for ind in data.index:
        if data.iloc[ind]['FEATS']:
            #if 'Number[psor]' in str(data.iloc[ind]['FEATS']):
            if 'Gender' not in str(data.iloc[ind]['FEATS']):
                index_nonegender.append(ind)
    data = data.drop(index= index_nonegender, axis= 0)   
    return data

# only keeping  number features in the dataframe
def person_feature(data):
    index_nonegender = []
    for ind in data.index:
        if data.iloc[ind]['FEATS']:
            #if 'Number[psor]' in str(data.iloc[ind]['FEATS']):
            if 'Person' not in str(data.iloc[ind]['FEATS']):
                index_nonegender.append(ind)
    data = data.drop(index= index_nonegender, axis= 0)   
    return data

def gender_number_person_data(train_data,gender = True, number = True, person = True):
    i = 0
    #train_data_extraction = pd.DataFrame(columns = ['POS_MOD','POS_HEAD','FEATURE_MOD','FEATURE_HEAD','RELATION'])
    train_data_extraction = []
    for ind, ID_value in enumerate (train_data['ID']):
        if ID_value == '\n':
            sentence = train_data[i:ind]
            sentence = sentence.reset_index()
            # removing rows that ID is not integer
            sentence = ID_integer(sentence)
            sentence = sentence.drop(['level_0'], axis = 1)
            # only keeping gender and number features in the dataframe
            
            if gender == True and number == False and person == False:
                sentence = gender_feature(sentence)
            elif gender == False and number == True and person == False:
                sentence = number_feature(sentence)
            elif gender == False and number == False and person == True:
                sentence = person_feature(sentence)
            else:
                sentence = gender_number_person_feature(sentence)
                
            sentence = sentence.reset_index()
            # converting HEAD into integer
            sentence['HEAD'] = sentence['HEAD'].apply(convert_ID)
     
    
            # mapping index and ID into a dict
            index_ID_mapping = sentence['ID'].to_dict()
            index_ID_mapping = {v:k for k,v in index_ID_mapping.items()}
   
            # extracting the data into a dict 
            for ind_head, head in enumerate (sentence['HEAD']):
                if head in list(sentence['ID']):
                    #train_data_extraction.append(sentence.iloc[ind_head])
                    #train_data_extraction.append(sentence.iloc[index_ID_mapping[head]])
 
                    data_extraction = {}
                    data_extraction['POS_HEAD'] = sentence.iloc[index_ID_mapping[head]]['UPOS']
                    data_extraction['POS_MOD'] = sentence.iloc[ind_head]['UPOS']
                    data_extraction['FEATURE_HEAD'] = sentence.iloc[index_ID_mapping[head]]['FEATS']
                    data_extraction['FEATURE_MOD'] = sentence.iloc[ind_head]['FEATS']
                    data_extraction['RELATION'] = sentence.iloc[ind_head]['DEPREL']
                    
                    # putting the data into the train_data_extraction list
                    train_data_extraction.append(data_extraction)
                    
                    
            # moving to the next sentence
            i = ind + 1
    # converting all the extracted data into dataframe        
    train_data_final = pd.DataFrame(train_data_extraction)    
    #train_data_final = train_data_final.drop(['level_0','index'],axis = 1)        
    return train_data_final

# let every feature be in one column
def seperate_feature_in_one_column (train_data_final):
    train_data_feature_head = train_data_final['FEATURE_HEAD'].str.split('|', expand=True)
    train_data_feature_head.columns = ['FEATURE_HEAD'+str(i) for i in train_data_feature_head.columns]

    train_data_feature_mod = train_data_final['FEATURE_MOD'].str.split('|', expand=True)
    train_data_feature_mod.columns = ['FEATURE_MOD'+str(i) for i in train_data_feature_mod.columns]

    train_data_feature = pd.concat([train_data_final, train_data_feature_head,train_data_feature_mod], axis=1)

    train_data_feature = train_data_feature.drop(['FEATURE_HEAD'], axis = 1)
    train_data_feature = train_data_feature.drop(['FEATURE_MOD'], axis = 1)
    return train_data_feature

# finding unique features 
def unique_features(train_data_feature):
    feature = []

    for i in train_data_feature.columns[3:]:
        for j in range(len(train_data_feature)):
            feature.append(train_data_feature[i][j])
    feature = set(feature)
    return feature

# finding unique category among features
def unique_category(feature):
    category = []

    for i in feature:
        if (i != None) and (i != '_') :
            c = str(i).split('=')
            category.append(c[0])

    pd_category = pd.DataFrame(category,columns = ['type'])
    category_list =pd_category['type'].unique().tolist()

    return  category_list

def feature_dataframe(data, column, category, head_mod):
    
    feature = {}
    for index,i in enumerate(list(data[column])):
        if i != '_':

            v = str(i).split('|')
            v_former = []
            for j in v:
                former = str(j).split('=')[0]
                v_former.append(former)
                back = str(j).split('=')[1]
                feature.setdefault(former + '_' + head_mod,[]).append(back)
            for item in category:
                if item not in v_former:
                    feature.setdefault(item + '_' + head_mod,[]).append(np.nan)
    

    return pd.DataFrame(feature)

def final_train_feature(train_data_final,category):
    pd_train_feature_head = feature_dataframe(data = train_data_final,column = 'FEATURE_HEAD', category= category, head_mod = 'head')

    pd_train_feature_mod = feature_dataframe(data = train_data_final,column ='FEATURE_MOD', category= category, head_mod = 'mod')

    train_data_final_feature = pd.concat([train_data_final,pd_train_feature_head,pd_train_feature_mod],axis = 1)
    train_data_final_feature = train_data_final_feature.drop(['FEATURE_HEAD','FEATURE_MOD'], axis = 1)
    return train_data_final_feature

def value_count(train_data_final_feature):
    value_count = train_data_final_feature.iloc[:,3:].count()
    return value_count

def agreement(head_data,mod_data):
    agreement_list = [float(head_data[i] == mod_data[i]) for i in range(len(head_data))]
    return agreement_list

def all_agreement(data):
    all_agreement = []
    for i in range(len(data)):
      if data.iloc[i]['Number_head'] == data.iloc[i]['Number_mod'] and data.iloc[i]['Gender_head'] == data.iloc[i]['Gender_mod'] and data.iloc[i]['Person_head'] == data.iloc[i]['Person_mod']:
          all_agreement.append(1.0)
      else:
          all_agreement.append(0.0)
    all_agreement = torch.tensor(all_agreement)
    return all_agreement

def filter_feature(data, list = ['POS_HEAD', 'POS_MOD', 'RELATION', 'Gender_head','Gender_mod','Number_head','Number_mod','Person_head','Person_mod']):
    train_4_feature = data[list]
    train_4_feature = train_4_feature.dropna()
    train_4_feature = train_4_feature.reset_index()
    train_4_feature = train_4_feature.drop(['index'], axis = 1)
    return train_4_feature

def cleaning_relation_labels(train_4_feature):
    for i in range (len(train_4_feature['RELATION'])):
        if re.search('@',train_4_feature['RELATION'][i]):
            train_4_feature['RELATION'][i],_ = train_4_feature['RELATION'][i].split('@')
        elif train_4_feature['RELATION'][i] == 'conj:emb':
            train_4_feature.loc[i,'RELATION'] = 'conj'
        elif train_4_feature['RELATION'][i] == 'comp:cleft':
            train_4_feature.loc[i,'RELATION'] = 'comp'
        elif train_4_feature['RELATION'][i] == 'comp:arg':
            train_4_feature.loc[i,'RELATION'] = 'comp'
    return train_4_feature

def covert_label_to_onehot(data_fit, data_transform):
    label_encoder = LabelEncoder()
    label_encoder.fit(data_fit)
    label = label_encoder.transform(data_transform)

    return label, label_encoder.classes_

 
def achieve_input_dict(label,train_4_feature):
    re_classes = []
    discrete_dict = {}
    binary_dict = {}
    for key in label.keys():

        for i in train_4_feature.columns:

            if re.search(key, i):
                result, result_classes = covert_label_to_onehot(label[key], train_4_feature[i].astype('str'))
                re_classes.append(result_classes)

                if  len(label[key]) == 2 :
                     binary_dict[i] = result
                     
                else:
                    if max(result) == len(label[key])-1:
                        onehot_result = to_categorical(np.array(result))
                        discrete_dict[i] = torch.tensor(onehot_result)

                    else:
                        add_columns = len(label[key]) - max(result) - 1
                        np_add_columns = np.zeros((len(train_4_feature[i]),add_columns))
                        onehot_result = to_categorical(np.array(result))
                        onehot_result_new = np.c_[onehot_result, np_add_columns]
                        discrete_dict[i] = torch.tensor(onehot_result_new)

    return discrete_dict, binary_dict, re_classes
  
def achive_input_array(discrete_dict, binary_dict):
    len_bianry = len(binary_dict)
    len_discrete = []
    discrete_data = []
    bi_data = []
    for value in discrete_dict.values():
        len_discrete.append(value.shape[1])
        discrete_data.append(value)
    for value in binary_dict.values():
        bi_data.append(value)

    binary_data = torch.tensor((pd.DataFrame(bi_data).T).values.tolist())

    return len_discrete,discrete_data, len_bianry, binary_data

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def np_relu(v):
    return np.maximum(v, 0)

class AdaSBN(nn.Module):
    def __init__(self, n_latent = 5 , n_observed_binary = 3, n_observed_discrete = [3,4,5,5]):
        super().__init__()
        
        self.n_latent = n_latent
        self.n_observed_binary = n_observed_binary
        self.prior_weights = nn.Parameter(torch.zeros(self.n_latent))
        self.outproj_binary = nn.Linear(self.n_latent, self.n_observed_binary, bias = True)
        self.n_observed_discrete = n_observed_discrete
        self.outproj_discrete = nn.ModuleList()
        for i in range (len(self.n_observed_discrete)):
            self.outproj_discrete.append(nn.Linear(self.n_latent, self.n_observed_discrete[i], bias = True))
         
   
    def forward (self,x,ys):
       
        # shape (n_samples,1,n_observed)
        x = x.unsqueeze(1)
    

        # y, shape (n_samples, n_possible_discrete_values, n_observed_discrete)
         


        # shape (1, n_possible_assignments ,n_latent)
        z = binary(torch.tensor(np.arange(2**self.n_latent)), self.n_latent).unsqueeze(0).to(device)
       
 
        # log_p_z shape (1, n_possible_assignments)
        log_p_z = z * self.prior_weights - F.softplus(self.prior_weights)
        log_p_z = log_p_z.sum(2)
        #print(log_p_z.shape)
     

        # w_x = Az+c shape(1,n_possible_assignments,n_observed_binary)
        w_x= self.outproj_binary(z)

 
        # w_y = Cz+d shape(1,n_possible_assignments,n_observed_discrete )
        w_ys = []
        for i in range (len(self.n_observed_discrete)):
            w_ys.append(self.outproj_discrete[i](z))
     


   

        # log_p_x_given_z shape(n_samples,n_possible_assignments), 
        # now some of the observations are not binary anymore, log p(x,y|z)= log(p(x|z)·p(y|z))(x,y is independent)
        # log p(x,y|z) = log p(x|z) + log p(y|z)
        
        log_p_x_given_z = (x * w_x - F.softplus(w_x)).sum(dim = 2)
       
        
        log_p_y_given_z = torch.zeros((x.shape[0], 2**self.n_latent)).to(device)
        
        for w_y, y in zip(w_ys, ys):
            
            #print(w_y.device,y.device)
            log_p_y_given_z += (y.unsqueeze(1) * w_y).sum(dim = 2) - torch.logsumexp(w_y,dim = 2)
            
            

       
          
        # log_joint shape(n_samples,n_possible_assignments)
       
        log_joint = log_p_z + log_p_x_given_z + log_p_y_given_z
     
  
        # shape ()
        log_marginalization = log_joint.logsumexp(dim = 1)
     
    
    
        return log_marginalization
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training_process (reg_weight, proximal,binary_data,discrete_data,sbn_adaptive,valid_binary_data,valid_discrete_data):
    #reg_weight = 0.0009
    #proximal = True
    #n_latent = 3
    optimizer = torch.optim.SGD(params=sbn_adaptive.parameters(), lr=1e-2, momentum=0.)
    train_losses = list()
    valid_losses = list()
    likelihood = list()

    for epoch in range(2000):

        sbn_adaptive.train()

        optimizer.zero_grad()
    
        discrete_data = [dd.to(device) for dd in discrete_data]
    
        train_loss = -sbn_adaptive(binary_data.to(device),discrete_data).mean()
        train_losses.append(train_loss.item())
        likelihood.append(-train_loss.item())
    
        if not proximal:
            train_loss = train_loss + reg_weight * torch.norm(sbn_adaptive.outproj_binary.weight, 1)
            for i in range(len(sbn_adaptive.outproj_discrete)):
                train_loss = train_loss + reg_weight * torch.norm(sbn_adaptive.outproj_discrete[i].weight, 1)
    
        train_loss.backward()
        optimizer.step()
        
        #print(epoch, ":\t", -train_loss)
        
    
        if proximal and epoch > 1000:
            with torch.no_grad():
                for i in range(len(sbn_adaptive.outproj_discrete)):
                    sbn_adaptive.outproj_discrete[i].weight.data = F.relu(sbn_adaptive.outproj_discrete[i].weight.abs() - reg_weight) * torch.sign(sbn_adaptive.outproj_discrete[i].weight)
                
                sbn_adaptive.outproj_binary.weight.data = F.relu(sbn_adaptive.outproj_binary.weight.abs() - reg_weight) * torch.sign(sbn_adaptive.outproj_binary.weight)
    
        sbn_adaptive.eval()
        with torch.no_grad():
            valid_discrete_data = [dd.to(device) for dd in valid_discrete_data]
            valid_loss = -sbn_adaptive(valid_binary_data.to(device),valid_discrete_data).mean()
            valid_losses.append(valid_loss.item())
            valid_loss = valid_loss + reg_weight * torch.norm(sbn_adaptive.outproj_binary.weight, 1)
            for i in range(len(sbn_adaptive.outproj_discrete)):
                valid_loss = valid_loss + reg_weight * torch.norm(sbn_adaptive.outproj_discrete[i].weight, 1)

        
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.show()

    plt.plot(np.arange(len(valid_losses)), valid_losses)
    plt.show()

    outproj_discrete_weight = []
    outproj_discrete_bias = []
    for i in range(len(sbn_adaptive.outproj_discrete)):
        outproj_discrete_weight.append(sbn_adaptive.outproj_discrete[i].weight)
        outproj_discrete_bias.append(sbn_adaptive.outproj_discrete[i].bias)
  

    return sbn_adaptive.outproj_binary.weight, sbn_adaptive.outproj_binary.bias, outproj_discrete_weight, outproj_discrete_bias, likelihood



def visual_result(binary_weight, binary_bias, discrete_weight,discrete_bias, n_latent, index_binary, re_classes, sbn):
    
    pd_binary_weight = pd.DataFrame(binary_weight.detach().cpu().numpy(), index = index_binary)
    pd_binary_weight.columns = ['latent'+str(n) for n in range(n_latent)]
    pd_binary = pd.concat([pd_binary_weight, pd.DataFrame(binary_bias.detach().cpu().numpy(), columns = ['bias'],index = index_binary)], axis = 1)

    #print(pd_binary)
    
    pd_discrete_sparsity = pd.DataFrame()
    pd_discrete_full = pd.DataFrame()
    for j in range(len(discrete_weight)):
        if len(discrete_weight) != 1:
            pd_discrete_weight = pd.DataFrame(discrete_weight[j].detach().cpu().numpy(), index = re_classes[j])
            pd_discrete_weight.columns = ['latent'+str(n) for n in range(n_latent)]
            pd_discrete = pd.concat([pd_discrete_weight, pd.DataFrame(discrete_bias[j].detach().cpu().numpy(), columns = ['bias'], index = re_classes[j])], axis = 1)
        else:
            pd_discrete_weight = pd.DataFrame(discrete_weight[j].detach().cpu().numpy(), index = re_classes[2])
            pd_discrete_weight.columns = ['latent'+str(n) for n in range(n_latent)]
            pd_discrete = pd.concat([pd_discrete_weight, pd.DataFrame(discrete_bias[j].detach().cpu().numpy(), columns = ['bias'], index = re_classes[2])], axis = 1)
        pd_discrete_full = pd.concat([pd_discrete_full, pd_discrete])
        # remove rows are all zeros except bias column 
        #print(pd_discrete.loc[(pd_discrete.iloc[:,: n_latent]!=0).any(axis=1)])
        pd_discrete_sparsity = pd.concat([pd_discrete_sparsity,pd_discrete.loc[(pd_discrete.iloc[:,: n_latent]!=0).any(axis=1)], pd.DataFrame([[np.NaN]*(n_latent+1)], columns = ['latent'+str(n) for n in range(n_latent)]+['bias'])])
       
        #print(pd_discrete_final)
        #pd_discrete_final = pd_discrete_final.append(pd_discrete.loc[(pd_discrete.iloc[:,: n_latent]!=0).any(axis=1)])
       
        #print(pd_discrete_final)
        
    pd_z = pd.DataFrame(sbn.prior_weights.sigmoid().detach().cpu().numpy(), columns = ['p(z)'], index = ['latent'+str(n) for n in range(n_latent)]).T
        
    #print(pd_z)
    
    return pd_binary,pd_discrete_full, pd_discrete_sparsity, pd_z


def visual_result_new(binary_weight, binary_bias, discrete_weight,discrete_bias, n_latent, index_binary, re_classes, sbn):
    
    pd_binary_weight = pd.DataFrame(binary_weight.detach().cpu().numpy(), index = index_binary)
    pd_binary_weight.columns = ['latent'+str(n) for n in range(n_latent)]
    pd_binary = pd.concat([pd_binary_weight, pd.DataFrame(binary_bias.detach().cpu().numpy(), columns = ['bias'],index = index_binary)], axis = 1)

    #print(pd_binary)
   
    pd_discrete_sparsity = pd.DataFrame()
    pd_discrete_full = pd.DataFrame()
    for j in range(len(discrete_weight)):
        if len(discrete_weight) != 1:
            pd_discrete_weight = pd.DataFrame(discrete_weight[j].detach().cpu().numpy(), index = re_classes[j])
            pd_discrete_weight.columns = ['latent'+str(n) for n in range(n_latent)]
            pd_discrete = pd.concat([pd_discrete_weight, pd.DataFrame(discrete_bias[j].detach().cpu().numpy(), columns = ['bias'], index = re_classes[j])], axis = 1)
        else:
            pd_discrete_weight = pd.DataFrame(discrete_weight[j].detach().cpu().numpy(), index = re_classes[2])
            pd_discrete_weight.columns = ['latent'+str(n) for n in range(n_latent)]
            pd_discrete = pd.concat([pd_discrete_weight, pd.DataFrame(discrete_bias[j].detach().cpu().numpy(), columns = ['bias'], index = re_classes[2])], axis = 1)
        pd_discrete_full = pd.concat([pd_discrete_full, pd_discrete])
        # remove rows are all zeros except bias column 
        #print(pd_discrete.loc[(pd_discrete.iloc[:,: n_latent]!=0).any(axis=1)])
        pd_discrete_sparsity = pd.concat([pd_discrete_sparsity,pd_discrete.loc[(pd_discrete.iloc[:,: n_latent]!=0).any(axis=1)], pd.DataFrame([[np.NaN]*(n_latent+1)], columns = ['latent'+str(n) for n in range(n_latent)]+['bias'])])
       
        
        #print(pd_discrete_final)
        #pd_discrete_final = pd_discrete_final.append(pd_discrete.loc[(pd_discrete.iloc[:,: n_latent]!=0).any(axis=1)])
       
        #print(pd_discrete_final)
    #print(pd_discrete_sparsity) 
    pd_z = pd.DataFrame(sbn.prior_weights.sigmoid().detach().cpu().numpy(), columns = ['p(z)'], index = ['latent'+str(n) for n in range(n_latent)]).T
    
    
    re_index_dict = {0:'Head',1:'Modifier',2:'Relation'}
    re_index = []
    j = 0
    
    for i in pd_discrete_sparsity.index:
        if i != 0:
            re_index.append(i + '_' + re_index_dict.get(j))
    
        else:
            j += 1
        
    pd_discrete_sparsity = pd_discrete_sparsity.dropna()

    pd_discrete_sparsity = pd.concat([pd_discrete_sparsity, pd.DataFrame(re_index, columns = ['new_index'],index = pd_discrete_sparsity.index)], axis = 1)
 
    pd_discrete_sparsity = pd_discrete_sparsity.set_index(['new_index'])
    pd_discrete_sparsity.index.name = None
    #pd_discrete_sparsity = pd_discrete_sparsity.dropna()
    
    
    return pd_binary,pd_discrete_full, pd_discrete_sparsity, pd_z



def possible_rules(data_discrete, data_binary, n_latent):
    Head_all = {}
    Mod_all= {}
    Relation_all = {}
    
    for i in data_discrete.iloc[:,: n_latent].columns:
        for k in data_binary.index:
            if data_binary.loc[k,i] > 0:
                Head = {}
                Mod = {}
                Relation = {}
                for j in data_discrete.index:
                    if re.search('Head', j):
                        if data_discrete.loc[j,i] > 0:
                            Head[j] = data_discrete.loc[j,i]

                    elif re.search('Modifier', j):
                        if data_discrete.loc[j,i] > 0:
                            Mod[j] = data_discrete.loc[j,i]

                    elif re.search('Relation', j):
                        if data_discrete.loc[j,i] > 0:
                            Relation[j] = data_discrete.loc[j,i]
                Head_all[i] = Head
                Mod_all[i] = Mod
                Relation_all[i] = Relation

    gr = []
    for i in data_discrete.iloc[:,: n_latent].columns:
        if Head_all.get(i) != None and Mod_all.get(i) != None and Relation_all.get(i) != None:
        
            if list(Head_all[i].values()) != [] and list(Mod_all[i].values()) != [] and list(Relation_all[i].values()) != []:
                h = list(Head_all[i].keys())[np.argmax(list(Head_all[i].values()))]

                m = list(Mod_all[i].keys())[np.argmax(list(Mod_all[i].values()))]

                r = list(Relation_all[i].keys())[np.argmax(list(Relation_all[i].values()))]
                gr.append((h, m, r))
    
    return gr

def possible_rules_all_agreement(data_discrete, data_binary, n_latent):
    
    Head_all = {}
    Mod_all= {}
    Relation_all = {}


    for i in data_discrete.iloc[:,: n_latent].columns:
        Head_column = {}
        Mod_column = {}
        Relation_column = {}
        for k in data_binary.index:
            if data_binary.loc[k,i] > 0:
                H = {}
                M = {}
                R = {}
                for j in data_discrete.index:
                    if re.search('Head', j):
                        if data_discrete.loc[j,i] > 0:
                            H[j] = data_discrete.loc[j,i]

                    elif re.search('Modifier', j):
                        if data_discrete.loc[j,i] > 0:
                            M[j] = data_discrete.loc[j,i]

                    elif re.search('Relation', j):
                        if data_discrete.loc[j,i] > 0:
                            R[j] = data_discrete.loc[j,i]

                Head_column[k] = H
                Mod_column[k] = M
                Relation_column[k] = R
        Head_all[i] = Head_column
        Mod_all[i] = Mod_column
        Relation_all[i] = Relation_column
    print(Head_all)
    print(Mod_all)
    print(Relation_all)
    
    gr_number = []
    gr_gender = []
    gr_person = []
    for i in data_discrete.iloc[:,: n_latent].columns:
        if list(Head_all[i].values()) != [] and list(Mod_all[i].values()) != [] and list(Relation_all[i].values()) != []:

            for j in data_binary.index:
    
                if re.search('Number', j):
   
                    head_number = list(Head_all[i][j].keys())[np.argmax(list(Head_all[i][j].values()))]
                    mod_number = list(Mod_all[i][j].keys())[np.argmax(list(Mod_all[i][j].values()))]
                    relation_number = list(Relation_all[i][j].keys())[np.argmax(list(Relation_all[i][j].values()))]
                    gr_number.append((head_number, mod_number, relation_number))
                elif re.search('Gender', j):

                    head_gender = list(Head_all[i][j].keys())[np.argmax(list(Head_all[i][j].values()))]
                    mod_gender = list(Mod_all[i][j].keys())[np.argmax(list(Mod_all[i][j].values()))]
                    relation_gender = list(Relation_all[i][j].keys())[np.argmax(list(Relation_all[i][j].values()))]
                    gr_gender.append((head_gender, mod_gender, relation_gender))

                elif re.search('Person', j):
              
                    head_person = list(Head_all[i][j].keys())[np.argmax(list(Head_all[i][j].values()))]
                    mod_person = list(Mod_all[i][j].keys())[np.argmax(list(Mod_all[i][j].values()))]
                    relation_person = list(Relation_all[i][j].keys())[np.argmax(list(Relation_all[i][j].values()))]
                    gr_number.append((head_person, mod_person, relation_person))


    return gr_number, gr_gender, gr_person

# explore the frequency of pos_head and pos_mod in different agreement  eg. Noun_head and Det_mod occurs 47298 in number agreement
def pos_frequency(data, title):
    pos_groupby = data.groupby(['POS_HEAD','POS_MOD'], as_index = False).size()
    pos_groupby_sns = pos_groupby.pivot(index = 'POS_HEAD', columns = 'POS_MOD', values = 'size')
    pos_groupby_sns.astype(pd.Int64Dtype())
    
    # plot
    sns.set(palette="muted", color_codes=True)  
    plt.figure(figsize=(10,6))
    ax = sns.heatmap(pos_groupby_sns,  annot=True, linewidths= 1,  cmap="YlGnBu", fmt = 'g')
    plt.title(title)
    plt.show()
    
# Frequency of relation labels
def relation_frequency(pd_data, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    sns.countplot(data = pd_data, x = 'RELATION', order = pd_data['RELATION'].value_counts().index[:5], ax = ax)
    plt.title(title)
    plt.show()
    
def fisher_test(rule, data, head, mod):
    predict_rules = pd.DataFrame(rule,columns = ['POS_HEAD','POS_MOD','RELATION'])
    data_with_agreement = data[data[head] == data[mod]]
    data_with_agreement = data_with_agreement.reset_index().drop(['index'], axis = 1)
    data_with_nonagreement = data[data[head] != data[mod]]
    data_with_nonagreement = data_with_nonagreement.reset_index().drop(['index'], axis = 1)

    # No. of occurrence of agreement 
    data_count_pos_re_agree = data_with_agreement.groupby(['POS_HEAD','POS_MOD','RELATION']).size() #as_index = False
   
    data_count_pos_agree = data_with_agreement.groupby(['POS_HEAD','POS_MOD']).size()
   
    # No. of occurrence of nonagreement 
    data_count_pos_re_nonagree = data_with_nonagreement.groupby(['POS_HEAD','POS_MOD','RELATION']).size() #as_index = False
    data_count_pos_nonagree = data_with_nonagreement.groupby(['POS_HEAD','POS_MOD']).size()
    p_value = {}
    for i in range(len(predict_rules)):
        try:
            data_count_pos_re_agree.loc[(predict_rules.loc[i,'POS_HEAD'].split('_')[0], predict_rules.loc[i,'POS_MOD'].split('_')[0], predict_rules.loc[i,'RELATION'].split('_')[0])]
        except KeyError:
            b = 0
        else:
            b = data_count_pos_re_agree.loc[(predict_rules.loc[i,'POS_HEAD'].split('_')[0], predict_rules.loc[i,'POS_MOD'].split('_')[0], predict_rules.loc[i,'RELATION'].split('_')[0])]

        try:
            data_count_pos_agree.loc[(predict_rules.loc[i,'POS_HEAD'].split('_')[0], predict_rules.loc[i,'POS_MOD'].split('_')[0])]
        except KeyError:
            d = 0
        else:
            d = data_count_pos_agree.loc[(predict_rules.loc[i,'POS_HEAD'].split('_')[0], predict_rules.loc[i,'POS_MOD'].split('_')[0])] - b

        try:
            data_count_pos_re_nonagree.loc[(predict_rules.loc[i,'POS_HEAD'].split('_')[0], predict_rules.loc[i,'POS_MOD'].split('_')[0], predict_rules.loc[i,'RELATION'].split('_')[0])]
        except KeyError:
            a = 0
        else:
            a = data_count_pos_re_nonagree.loc[(predict_rules.loc[i,'POS_HEAD'].split('_')[0], predict_rules.loc[i,'POS_MOD'].split('_')[0], predict_rules.loc[i,'RELATION'].split('_')[0])]

        try:
            data_count_pos_nonagree.loc[(predict_rules.loc[i,'POS_HEAD'].split('_')[0], predict_rules.loc[i,'POS_MOD'].split('_')[0])]
        except KeyError:
            c = 0
        else:
            c = data_count_pos_nonagree.loc[(predict_rules.loc[i,'POS_HEAD'].split('_')[0], predict_rules.loc[i,'POS_MOD'].split('_')[0])] - a

        table = np.array([[b,d], [a,c]])
        #print(a,b,c,d)
        oddsr, p = fisher_exact(table, alternative='greater')
        #_,p,_,_ = chi2_contingency(table)
        #p_value[((predict_rules.loc[i,'POS_HEAD'], predict_rules.loc[i,'POS_MOD'], predict_rules.loc[i,'RELATION']))] = p
        p_value[((predict_rules.loc[i,'POS_HEAD'], predict_rules.loc[i,'POS_MOD'], predict_rules.loc[i,'RELATION']))] = (oddsr, p)
        

    return p_value

    
       
    
    