# execution: python3 main.py ratings.csv targets.csv > submission.csv

import pandas as pd
import numpy as np
import argparse

# Global variables used to store important values
# These variables are useful to avoid reading the input files multiple times during execution
ITEMS_INFO_NORM: dict[str, dict[str, float]] = {}  # group the data based on the items
USERS_INFO: dict[str, list[str]] = {}  # group the data based on the users
USERS_MEANS: dict[str, float] = {}
SIMIL_ITEMS: dict[str, dict[str, float]] = {}
ITEMS_AVERAGE_RATING: float = 0.0
RECOMENDATION = {}

MIN_ITEMS_NECESSARY = 7
MIN_USERS_NECESSARY = 7
MIN_SIMILARITY_NECESSARY = 0.3


def receive_args() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Receive the arguments from the command line
    """
    
    parser = argparse.ArgumentParser(description="Process two CSV files.")

    parser.add_argument("ratings", type=str, help="ratings CSV file")
    parser.add_argument("targets", type=str, help="targets CSV file")
    parser.add_argument("--storeOutput", type=str, help="flag to store output in a csv file")

    args = parser.parse_args()

    if not (args.ratings and args.targets):
        raise Exception

    ratings = pd.read_csv(args.ratings)
    targets = pd.read_csv(args.targets)

    store_output = args.storeOutput if args.storeOutput else None

    return ratings, targets, store_output


def recomendation_new_user_and_item() -> float:
    """
        Performs a recomendation to new user AND new item as the mean of all ratings
    """

    return ITEMS_AVERAGE_RATING


def recomendation_new_user(item: str) -> float:
    """
        Performs a recomendation to a new user as the mean of all ratings of the item
    """
    if item in RECOMENDATION:
        return RECOMENDATION[item]

    sum_mean_ratings = 0
    for user, rating in ITEMS_INFO_NORM[item].items():

        sum_mean_ratings += rating + USERS_MEANS[user]

    recomendation = sum_mean_ratings / len(ITEMS_INFO_NORM[item])

    RECOMENDATION[item] = recomendation

    return recomendation


def recomendation_new_item(user: str) -> float:
    """
        Performs a recomendation to a new item as the mean of all ratings of the user
    """
    if user in RECOMENDATION:
        return RECOMENDATION[user]

    recomendation = USERS_MEANS[user]

    RECOMENDATION[user] = recomendation

    return recomendation


def update_simil_items(simil: float, item: str, item_rated_by_user: str):
    """ 
        Function to update SIMIL_ITEMS dict with the recomendations
    """

    if item not in SIMIL_ITEMS:

        SIMIL_ITEMS[item] = {}

    if item_rated_by_user not in SIMIL_ITEMS:

        SIMIL_ITEMS[item_rated_by_user] = {}

    SIMIL_ITEMS[item][item_rated_by_user] = simil
    SIMIL_ITEMS[item_rated_by_user][item] = simil


def find_recomendation(ratings: pd.DataFrame, targets: pd.DataFrame):
    """
    Calculate the recomednation based on differente scenarios
        new user and item -> mean of all ratings
        new user -> mean of all ratings of the item
        new item -> mean of all ratings of the user
        else -> using itens with similarity >= threshold, calculate the recomendation as =
            sum(similarity * rating) / sum(abs(similarity))
    """

    all_recomendations = []

    for _, row in targets.iterrows():
        
        user_item = row.iloc[0]

        user, item = user_item.split(":")

        users_rated_item = ITEMS_INFO_NORM[item].keys()
        itens_rated_by_user = USERS_INFO[user]

        len_itens_rated_by_user = len(itens_rated_by_user)
        len_users_rated_item = len(users_rated_item)

        new_user_and_item = (
            len_users_rated_item <= MIN_ITEMS_NECESSARY
            and len_itens_rated_by_user <= MIN_USERS_NECESSARY
        )

        new_user = len_itens_rated_by_user <= MIN_USERS_NECESSARY

        new_item = len_users_rated_item <= MIN_ITEMS_NECESSARY

        if new_user_and_item:

            all_recomendations.append((recomendation_new_user_and_item()))

        elif new_item:

            all_recomendations.append((recomendation_new_item(user)))

        elif new_user:

            all_recomendations.append((recomendation_new_user(item)))

        else:  # user and item already have a significant amount of ratings

            sum_recomendation_simil = 0
            sum_simil_abs = 0

            # runs over all itens that the user has rated, calculating the similarity
            # and performing the recomendation
            for item_rated_by_user in itens_rated_by_user:

                if (
                    item not in SIMIL_ITEMS
                    or item_rated_by_user not in SIMIL_ITEMS[item]
                ):

                    simil = calc_similarity(
                        ITEMS_INFO_NORM[item], ITEMS_INFO_NORM[item_rated_by_user]
                    )

                    if simil <= MIN_SIMILARITY_NECESSARY:
                        continue

                    update_simil_items(simil, item, item_rated_by_user)

                # store info to make recomendation

                sum_recomendation_simil += (
                    SIMIL_ITEMS[item][item_rated_by_user]
                    * ITEMS_INFO_NORM[item_rated_by_user][user]
                )

                sum_simil_abs += abs(SIMIL_ITEMS[item][item_rated_by_user])

            recomendation = (
                sum_recomendation_simil / sum_simil_abs if sum_simil_abs != 0 else 0.0
            )

            # user info is normalized, so the recomendation is normalized too.
            # to get the real value, we need to add the mean of the respective user
            recomendation = recomendation + USERS_MEANS[user]

            all_recomendations.append(recomendation)

    return all_recomendations

def calc_similarity(
    item_info1: dict[str, float], item_info2: dict[str, float]
) -> float:
    """ 
        Calculates the similarity between two itens as follows using cosine similarity =

        sum (rat item 1 * rat item 2 -> for all common users) / square_root(sum (rat item 1)^2 * sum (rat item 2)^2)
    """

    common_users = item_info1.keys() & item_info2.keys()

    # if no common users, no similarity
    if len(common_users) == 0:
        return 0.0

    sum_common_users = 0
    for user in common_users:
        rat1 = item_info1[user]
        rat2 = item_info2[user]

        sum_common_users += rat1 * rat2

    sum_item1 = np.sqrt(sum(value**2 for value in item_info1.values()))
    sum_item2 = np.sqrt(sum(value**2 for value in item_info2.values()))

    if sum_item1 * sum_item2 == 0:
        return 0.0

    return sum_common_users / (sum_item1 * sum_item2)


def normalizing_ratings(ratings: pd.DataFrame):

    """
        Normalizing the ratings of the users.

        This function also calculate and store other useful information to be used in the recomendation, avoiding passing through the data multiple times.
    """

    items_means = {}
    user_means = {}

    for _, row in ratings.iterrows():
        user_item = row.iloc[0]
        rating = row.iloc[1]

        user, item = user_item.split(":")

        if user not in user_means:
            user_means[user] = {}
            user_means[user] = {"count": 0, "value": 0}
            USERS_INFO[user] = []
        
        if item not in USERS_INFO[user]:
            USERS_INFO[user].append(item)

        if item not in ITEMS_INFO_NORM:
            ITEMS_INFO_NORM[item] = {}
            items_means[item] = {"count": 0, "value": 0}

        ITEMS_INFO_NORM[item][user] = rating

        items_means[item]["count"] += 1
        items_means[item]["value"] += rating

        user_means[user]["count"] += 1
        user_means[user]["value"] += rating

    for user, value in user_means.items():

        USERS_MEANS[user] = value["value"] / value["count"]

    for item, value in ITEMS_INFO_NORM.items():

        item_mean = items_means[item]["value"] / items_means[item]["count"]

        for user, rating in value.items():

            value[user] = rating - item_mean
    


def store_print_output(key: str, targets: pd.DataFrame, recomendations: list):
    """
        Store output in a csv file or just print in the console based on the key received
    """

    # to store the results, just copy the target dataset and add a new column with the recomendations
    df = targets.copy()
    df["Rating"] = recomendations

    if key:

        df.to_csv('output_file.csv', index=False)
    
    print("UserId:ItemId,Rating")
    for _, row in df.iterrows():
        print(row["UserId:ItemId"] + "," + str(row["Rating"]))
        



if __name__ == "__main__":

    # receive parameters
    ratings, targets, output_flag = receive_args()

    ITEMS_AVERAGE_RATING = ratings["Rating"].mean()

    normalizing_ratings(ratings)

    rec = find_recomendation(ratings, targets)

    store_print_output(output_flag, targets, rec)
