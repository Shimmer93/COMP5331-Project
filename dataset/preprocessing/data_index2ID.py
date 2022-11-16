import pandas as pd
import os
import csv

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
data_name_s = 'sport'
data_name_t = 'cloth'
save_path = 'generate_dataset/'

save_path_s = save_path + data_name_s + '_' + data_name_t + '/'
save_path_t = save_path + data_name_t + '_' + data_name_s + '/'
if not os.path.exists(save_path_s):
    os.makedirs(save_path_s)
if not os.path.exists(save_path_t):
    os.makedirs(save_path_t)
data_dic = {'sport': 'ratings_Sports_and_Outdoors', 'electronic': 'ratings_Electronics',
            'cloth': 'ratings_Clothing_Shoes_and_Jewelry', 'cell': 'ratings_Cell_Phones_and_Accessories',
            'instrument': 'ratings_Musical_Instruments', 'game': 'ratings_Toys_and_Games', 'video': 'ratings_Video_Games',
            'book':'ratings_Books', 'movie':'ratings_Movies_and_TV', 'cd': 'ratings_CDs_and_Vinyl', 'music':'ratings_Digital_Music'}


filepath1 = datapath + data_dic[data_name_s] + '.csv'
filepath2 = datapath + data_dic[data_name_t] + '.csv'

save_file7= save_path_s+'index2ID1.csv'
save_file8= save_path_t+'index2ID2.csv'
save_file9= data_name_t + '_' + data_name_s+'_'+'index2ID.csv'
save_file10= data_name_s + '_' + data_name_t+'_'+'index2ID.csv'

f_path = save_path_t + '%s_%s_data_info.txt' % (data_name_s, data_name_t)
f = open(f_path, 'w+')
u_num, i_num, r_num, user_unique, data = filter_data(filepath1) # item<10

u_num2, i_num2, r_num2, user_unique2, data2 = filter_data(filepath2) # item<10
# nn_data_1 = filter_item(new_data_1)
# nn_data_2 = filter_item(new_data_2)
# u,i ,r= get_unique_lenth(nn_data_1)
# u2,i2 ,r2= get_unique_lenth(nn_data_2)


data, data2 = filter_user(data, data2) # del overlap user < 5
data, data2 = filter_item(data, data2) # del overlap user < 5
user_unique = list(data['userId'].unique())
user_unique2 = list(data2['userId'].unique())
item_unique = list(data['itemId'].unique())
item_unique2 = list(data2['itemId'].unique())



c_n, common_user = get_common_user(user_unique, user_unique2)
t_n, total_user = get_total_user(user_unique, user_unique2)
user_index = list(range(0, len(total_user)))
dic_u = dict(zip(total_user, user_index))

item_index1 = list(range(0, len(item_unique)))
dic_v1 = dict(zip(item_unique, item_index1))

item_index2 = list(range(0, len(item_unique2)))
dic_v2 = dict(zip(item_unique2, item_index2))
dict={}
dict.update(dic_v1)
dict.update(dic_v2)
write_to_csv(dic_v2, save_file9)
write_to_csv(dic_v1, save_file10)
#write_to_csv(dic_v1, save_file7)
#write_to_csv(dic_v2,save_file8)


pprint('write data finished!', f)
