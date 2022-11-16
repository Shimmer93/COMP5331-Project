import pandas as pd
import os
import csv
import json

def pprint(str_, f):
    print(str_)
    print(str_, end='\n', file=f)


def filter_data(filePath):
    data = []
    ratings = pd.read_csv(filePath, delimiter=",", encoding="latin1")
    ratings.columns = ['userId', 'itemId', 'Rating', 'timesteamp']

    rate_size_dic_i = ratings.groupby('itemId').size()
    # choosed_index_del_i = rate_size_dic_i.index[rate_size_dic_i < 10]
    choosed_index_del_i = rate_size_dic_i.index[rate_size_dic_i < 10]
    ratings = ratings[~ratings['itemId'].isin(list(choosed_index_del_i))] # item freq more than 10

    user_unique = list(ratings['userId'].unique())
    movie_unique = list(ratings['itemId'].unique())

    u = len(user_unique)
    i = len(movie_unique)
    rating_num = len(ratings)
    return u, i, rating_num, user_unique, ratings


def get_min_group_size(ratings):
    rate_size_dic_u = ratings.groupby('userId').size()
    return min(rate_size_dic_u)


def my_reindex_data(ratings1, dic_u=None):
    data = []
    if dic_u is None:
        user_unique = list(ratings1['userId'].unique())
        user_index = list(range(0, len(user_unique)))
        dic_u = dict(zip(user_unique, user_index))
    movie_unique1 = list(ratings1['itemId'].unique())
    movie_index1 = list(range(0, len(movie_unique1)))
    dic_m1 = dict(zip(movie_unique1, movie_index1))
    for element in ratings1.values:
        data.append((dic_u[element[0]], dic_m1[element[1]], 1))
    data = sorted(data, key=lambda x: x[0])
    return data, dic_u

def reindex_data(ratings1, dic_u=None,dic_v=None):
    data = []
    if dic_u is None:
        user_unique = list(ratings1['userId'].unique())
        user_index = list(range(0, len(user_unique)))
        dic_u = dict(zip(user_unique, user_index))
    if dic_v is None:
        movie_unique1 = list(ratings1['itemId'].unique())
        movie_index1 = list(range(0, len(movie_unique1)))
        dic_v = dict(zip(movie_unique1, movie_index1))
    for element in ratings1.values:
        data.append((dic_u[element[0]], dic_v[element[1]], 1))
    data = sorted(data, key=lambda x: x[0])
    return data, dic_u


def get_common_data(data1, data2, common_user):
    rating_new_1 = data1[data1['userId'].isin(common_user)]
    rating_new_2 = data2[data2['userId'].isin(common_user)]
    return rating_new_1, rating_new_2


def get_unique_lenth(ratings):
    r_n = len(ratings)
    user_unique = list(ratings['userId'].unique())
    movie_unique = list(ratings['itemId'].unique())
    u = len(user_unique)
    i = len(movie_unique)
    return u, i, r_n


def filter_user(ratings1, ratings2):
    rate_size_dic_u1 = ratings1.groupby('userId').size()
    rate_size_dic_u2 = ratings2.groupby('userId').size()
    choosed_index_del_u1 = rate_size_dic_u1.index[rate_size_dic_u1 < 5]
    choosed_index_del_u2 = rate_size_dic_u2.index[rate_size_dic_u2 < 5]
    ratings1 = ratings1[~ratings1['userId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    ratings2 = ratings2[~ratings2['userId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    return ratings1, ratings2

def filter_item(ratings1, ratings2):
    rate_size_dic_u1 = ratings1.groupby('itemId').size()
    rate_size_dic_u2 = ratings2.groupby('itemId').size()
    choosed_index_del_u1 = rate_size_dic_u1.index[rate_size_dic_u1 < 5]
    choosed_index_del_u2 = rate_size_dic_u2.index[rate_size_dic_u2 < 5]
    ratings1 = ratings1[~ratings1['itemId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    ratings2 = ratings2[~ratings2['itemId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    return ratings1, ratings2


def read_from_json(file):
    dict={}
    count=0
    with open(file, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            if(count==0):
                print('title' in data)
                print(data.get('title'))
            
            count+=1
            dict[data['asin']]=data.get('title')
    return dict

    return data

def read_from_csv(file):
    with open(file) as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(reader)
    return mydict


def write_to_csv(data,file):
    with open(file, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in data.items():
            writer.writerow([key, value])


def write_to_txt(data, file):
    f = open(file, 'w+')
    for i in data:
        line = '\t'.join([str(x) for x in i]) + '\n'
        f.write(line)
    f.close


def get_common_user(data1, data2):
    common_user = list(set(data1).intersection(set(data2)))
    return len(common_user), common_user

def get_total_user(data1, data2):
    total_user = list(set(data1).union(set(data2)))
    return len(total_user), total_user

def get_total_item(data1, data2):
    total_item = list(set(data1).union(set(data2)))
    return len(total_item), total_item
# cloth sport
# cell electronic
# game video
# cd movie
# music instrument


# datapath = './Amazon/'
datapath = './'
data_name_s = 'cloth'
data_name_t = 'sport'
save_path = 'generate_dataset/'

save_path_s = save_path + data_name_s + '_' + data_name_t + '/'
save_path_t = save_path + data_name_t + '_' + data_name_s + '/'
if not os.path.exists(save_path_s):
    os.makedirs(save_path_s)
if not os.path.exists(save_path_t):
    os.makedirs(save_path_t)
data_dic = {'sport': 'meta_Sports_and_Outdoors', 'electronic': 'meta_Electronics',
            'cloth': 'meta_Clothing_Shoes_and_Jewelry', 'cell': 'meta_Cell_Phones_and_Accessories',
            'instrument': 'meta_Musical_Instruments', 'game': 'meta_Toys_and_Games', 'video': 'meta_Video_Games',
            'book':'meta_Books', 'movie':'meta_Movies_and_TV', 'cd': 'meta_CDs_and_Vinyl', 'music':'meta_Digital_Music'}


filepath1 = datapath+data_name_t + '_' + data_name_s+'_'+'index2ID.csv'
filepath4 = datapath+data_name_s + '_' + data_name_t+'_'+'index2ID.csv'
filepath2 = datapath + data_dic[data_name_s] + '.json'
filepath3 = datapath + data_dic[data_name_t] + '.json'


save_file9= data_name_t + '_' + data_name_s+'_'+'index2Title.csv'
save_file10= data_name_s + '_' + data_name_t+'_'+'index2Title.csv'

index2ID=read_from_csv(filepath1)
index2ID2=read_from_csv(filepath4)
meta1=read_from_json(filepath2)
meta2=read_from_json(filepath3)
dict={}
for key in index2ID:
    if(key in meta2):
        dict[index2ID[key]]=meta2[key]
    elif(key in meta1):
        dict[index2ID[key]]=meta1[key]
    else:
        dict[index2ID[key]]=''

write_to_csv(dict, save_file9)
dict2={}
for key in index2ID2:
    if(key in meta2):
        dict2[index2ID2[key]]=meta2[key]
    elif(key in meta1):
        dict2[index2ID2[key]]=meta1[key]
    else:
        dict2[index2ID2[key]]=''

write_to_csv(dict2, save_file10)





print('write data finished!')
