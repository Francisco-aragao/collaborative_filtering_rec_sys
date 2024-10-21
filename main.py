# execution: python3 main.py ratings.csv targets.csv > submission.csv 

"""" 

data = {
    'UserId': ['33ce7ee122', 'eab9e065e5', 'f785763291'],
    'ItemId': ['34cb28c370', '34cb28c370', '34cb28c370'],
    'Rating': [3, 2, 4]
}

df = pd.DataFrame(data)

# Create a new column that combines 'UserId' and 'ItemId' with a colon ':'
df['UserId:ItemId'] = df['UserId'] + ':' + df['ItemId']

# Drop the original 'UserId' and 'ItemId' columns
df = df[['UserId:ItemId', 'Rating']]

# Write the DataFrame to a CSV file
df.to_csv('output.csv', index=False)
"""

import pandas as pd
import numpy as np

import argparse

ITEMS_INFO: dict[str, dict[str, float]] = {}
USERS_MEANS: dict[str, float] = {}
SIMIL_ALL_ITEMS: dict[str, dict[str, float]] = {}

def calc_simil_all_items(): 

    print('Entrou calc simil')

    all_items = [key for key in ITEMS_INFO.keys()]
    
    for i in range(len(all_items)):

        if i not in SIMIL_ALL_ITEMS:
            SIMIL_ALL_ITEMS[all_items[i]] = {}

        for j in range(i, len(all_items)):

            if all_items[j] not in SIMIL_ALL_ITEMS:
                SIMIL_ALL_ITEMS[all_items[j]] = {}

            simil = calc_similarity(ITEMS_INFO[all_items[i]], ITEMS_INFO[all_items[j]])
            
            SIMIL_ALL_ITEMS[all_items[i]][all_items[j]] = simil

            SIMIL_ALL_ITEMS[all_items[j]][all_items[i]] = simil
    
    print('Saiu calc simil')

def percentage_simil(list1, list2):

    if len(list1) > len(list2):

        return len(list2) / len(list1)

    return len(list1) / len(list2)

def find_most_simil_itens(item_target, items_info_sorted: dict[str, dict[str, float]]):

    keys1 = ITEMS_INFO[item_target].keys()

    k = 5

    simils = {}
    min_val = -500
    min_item = ''

    for item, value in items_info_sorted.items():
        
        keys2 = value.keys()

        if len(simils) == k:
            if len(keys2) < len(keys1) * 0.6:
                return simils 

            continue 

        if len(keys1 & keys2) == 0:
            continue
        
        simil = calc_similarity(ITEMS_INFO[item_target], value)

        if simil > min_val:
            del simils[min_item]

            min_val = simil

            simils[item] = simil

    return 'ARRUMAR'




def receive_args() -> tuple[pd.DataFrame, pd.DataFrame]:

    # Create argument parser
    parser = argparse.ArgumentParser(description="Process two CSV files.")

    parser.add_argument('ratings', type=str, help="ratings CSV file")
    parser.add_argument('targets', type=str, help="targets CSV file")
    parser.add_argument('submission', type=str, help="submission CSV file")

    args = parser.parse_args()

    if not (args.ratings and args.targets and args.submission):
        raise Exception
    
    ratings = pd.read_csv(args.ratings)
    targets = pd.read_csv(args.targets)

    return ratings, targets

RECOMENDATION = {}

def recomendation_new():

    return 3

# recomend item mean normalized
def recomendation_new_user (item: str) -> float:

    if item in RECOMENDATION:
        return RECOMENDATION[item]
    
    sum_mean_ratings = 0
    for user, rating in ITEMS_INFO[item].items():
            
        sum_mean_ratings += rating + USERS_MEANS[user]
    
    recomendation = sum_mean_ratings / len(ITEMS_INFO[item])
    
    RECOMENDATION[item] = recomendation
    
    return recomendation

# recomend the user mean
def recomendation_new_item (user: str) -> float:

    if user in RECOMENDATION:
        return RECOMENDATION[user]
    
    recomendation = USERS_MEANS[user]

    RECOMENDATION[user] = recomendation
    
    return recomendation


def find_recomendation(items_info_sorted: dict[str, dict[str, float]],ratings: pd.DataFrame, targets: pd.DataFrame):
    
    #calc_simil_all_items()

    df = ratings.copy()

    # Split the 'UserId:ItemId' column into two separate columns
    df[['UserId', 'ItemId']] = df['UserId:ItemId'].str.split(':', expand=True)

    # Drop the original 'UserId:ItemId' column
    df = df.drop(columns=['UserId:ItemId'])
    
    recomendations = []
    
    for index, row in targets.iterrows():
        print(index)
        user_item = row.iloc[0]

        user, item = user_item.split(':')
        #print('User', user)   

        item_info = df[df['ItemId'] == item]
        
        user_info = df[df['UserId'] == user]

        if len(item_info) <= 3:
            
            recomendations.append(round(recomendation_new_item(user)))
        
        elif len(user_info) <= 3:

            recomendations.append(round(recomendation_new_user(item)))

        else:
            
            itens_choosed_by_user = list(list(user_info['ItemId']))
            rat_itens_choosed_by_user = list(list(user_info['Rating']))

            #print('itens choosed', itens_choosed_by_user)
            #print('rat ', rat_itens_choosed_by_user)


            sum_rec_simil = 0
            sum_simil_abs = 0

            for i in range(len(itens_choosed_by_user)):

                #print('comparing ', item, ' and ', itens_choosed_by_user[i])


                if item not in SIMIL_ALL_ITEMS or itens_choosed_by_user[i] not in SIMIL_ALL_ITEMS[item]:

                    simil = calc_similarity(ITEMS_INFO[item], ITEMS_INFO[itens_choosed_by_user[i]])

                    if simil == 0.0:
                        simil = recomendation_new_user(item)

                    if item not in SIMIL_ALL_ITEMS:
                    
                        SIMIL_ALL_ITEMS[item] = {}
                    
                    if itens_choosed_by_user[i] not in SIMIL_ALL_ITEMS:

                        SIMIL_ALL_ITEMS[itens_choosed_by_user[i]] = {}

                    SIMIL_ALL_ITEMS[item][itens_choosed_by_user[i]] = simil
                    SIMIL_ALL_ITEMS[itens_choosed_by_user[i]][item] = simil

                #print('simil ', SIMIL_ALL_ITEMS[item][itens_choosed_by_user[i]])

                sum_rec_simil += SIMIL_ALL_ITEMS[item][itens_choosed_by_user[i]] * ITEMS_INFO[itens_choosed_by_user[i]][user]

                sum_simil_abs += abs(SIMIL_ALL_ITEMS[item][itens_choosed_by_user[i]])  
                     
            recomendation = sum_rec_simil / sum_simil_abs if sum_simil_abs != 0 else 0.0

            recomendations.append(round(recomendation + USERS_MEANS[user]))                       


    return recomendations

def find_user_items_and_ratings(userID: int, ratings: pd.DataFrame) -> list[tuple]:

    user_items = []

    for _, row in ratings.iterrows():
        user_item = row.iloc[0]
        rating = row.iloc[1]

        user, item = user_item.split(':')

        if user != userID:
            continue

        user_items.append((item,rating))
    
    return user_items

import time
def calc_similarity(item_info1: dict[str, float], item_info2: dict[str, float]) -> float:

    keys1 = item_info1.keys()
    keys2 = item_info2.keys()

    """ lenk1 =len(keys1)
    lenk2 = len(keys2)


    if lenk1 > lenk2 and lenk2 / lenk1 < 0.8:
        return 0.0 

    if lenk2 > lenk1 and lenk1 / lenk2 < 0.8:
        return 0.0  """

    common_users = item_info1.keys() & item_info2.keys()

    if len(common_users) == 0:
        return 0.0


    sum_common_users = 0
    for user in common_users:
        rat1 = item_info1[user]
        rat2 = item_info2[user]

        sum_common_users += rat1 * rat2
    
    sum_item1 = np.sqrt(sum(value ** 2 for value in item_info1.values()))
    sum_item2 = np.sqrt(sum(value ** 2 for value in item_info2.values()))

    if sum_item1 * sum_item2 == 0:
        return 0.0

    return sum_common_users / (sum_item1 * sum_item2)

# Mean-centering normalization
def ratings_normalizations(ratings: pd.DataFrame):

    items_means = {}
    user_means = {}

    for _, row in ratings.iterrows():
        user_item = row.iloc[0]
        rating = row.iloc[1]

        user, item = user_item.split(':')

        if user not in user_means:
            user_means[user] = {}
            user_means[user] = {'count': 0, 'value': 0}

        if item not in ITEMS_INFO:
            ITEMS_INFO[item] = {}
            items_means[item] = {'count': 0, 'value': 0}

        ITEMS_INFO[item][user] = rating
        
        items_means[item]['count'] += 1
        items_means[item]['value'] += rating

        user_means[user]['count'] += 1
        user_means[user]['value'] += rating
    
    for user, value in user_means.items():

        USERS_MEANS[user] = value['value'] / value['count']

    for item, value in ITEMS_INFO.items():

        item_mean = items_means[item]['value'] / items_means[item]['count'] 

        for user, rating in value.items():
            
            value[user] = rating - item_mean

    with open('ITEMS_INFO.json', 'w') as json_file:
        json.dump(ITEMS_INFO, json_file, indent=4)

    with open('USERS_MEANS.json', 'w') as json_file:
        json.dump(USERS_MEANS, json_file, indent=4)
        

if __name__ == '__main__':

    import json
    # receive parameters
    ratings, targets = receive_args()

    print('norm')
    #ratings_normalizations(ratings)

    with open('USERS_MEANS.json', 'r') as json_file:
        USERS_MEANS = json.load(json_file)

    with open('ITEMS_INFO.json', 'r') as json_file:
        ITEMS_INFO = json.load(json_file)

    # store the 20 items with more recomendations
    # 20 is an arbitrary number
    k = 20
    items_info_sorted = sorted_data = dict(sorted(ITEMS_INFO.items(), key=lambda item: len(item[1]), reverse=True)[:k])

    print('Rec')
    rec = find_recomendation(items_info_sorted, ratings, targets)

    df = targets.copy()

    df['Rating'] = rec

    # Write the DataFrame to a CSV file
    df.to_csv('output.csv', index=False)