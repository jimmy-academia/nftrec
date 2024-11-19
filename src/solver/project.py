import random
from utils import *
from debug import *
from arguments import nft_project_names, min_purchase

class NFTProject:
    def __init__(self, nft_project_data, setN, setM, nft_project_name):
        self.setN = setN 
        self.setM = setM 
        self.nft_project_name = nft_project_name
        self.trait_dict = nft_project_data['trait_system']
        self.numericalize(nft_project_data)

    def numericalize(self, nft_project_data):
        asset_traits, buyer_assets_ids, buyer_budgets, item_counts = nft_project_data['asset_traits'], nft_project_data['buyer_assets_ids'], nft_project_data['buyer_budgets'], nft_project_data['item_counts']

        max_item_len = max([len(x) for x in buyer_assets_ids])
        min_purchase_limit = min_purchase[nft_project_names.index(self.nft_project_name)] if self.nft_project_name in nft_project_names else 1

        ## collect NFT item set and user prefs
        alluser_prefs = []
        nft_set = set()
        num_trades = 0
        for _assest_list in buyer_assets_ids:
            if len(_assest_list) < min_purchase_limit:
                continue
            num_trades += len(_assest_list)
            nft_set |= set(_assest_list)
            user_prefs = self.trait2label_vec([asset_traits[aid] for aid in _assest_list])
            # pad until max length, then transpose
            if len(user_prefs) <= max_item_len:
                user_prefs = user_prefs + [user_prefs[-1]]*(max_item_len - len(user_prefs))
            else:
                user_prefs = user_prefs[:max_item_len]
            user_prefs = [list(x) for x in zip(*user_prefs)]
            alluser_prefs.append(user_prefs)

        self.N = self.setN if self.setN is not None else len(alluser_prefs)
        self.M = self.setM if self.setM is not None else len(nft_set)

        ## process buyer list
        nft_set = list(nft_set)
        nft_set = [nft_set[i%len(nft_set)] for i in range(self.M)]

        self.item_attributes = self.trait2label_vec([asset_traits[i] for i in nft_set])
        self.user_preferences = [alluser_prefs[i%len(alluser_prefs)] for i in range(self.N)]
        self.user_budgets = [buyer_budgets[bi% len(buyer_budgets)] for bi in range(self.N)]
        self.item_counts = [item_counts[i% len(item_counts)] for i in nft_set]
        self.num_trades = num_trades

    def trait2label_vec(self, asset_traits):
        item_vec_list = []
        for item in asset_traits:
            item_vec = []
            for (trait, options), choice in zip(self.trait_dict.items(), item):
                choice = 'none' if choice == 'None' else choice
                try:
                    item_vec.append(options.index(choice))
                except:
                    item_vec.append(random.randint(0, len(options)-1))
            item_vec_list.append(item_vec)
        return item_vec_list

    