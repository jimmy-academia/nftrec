import random
from utils import *
from debug import *
from arguments import nft_project_names
min_purchase = [10, 2, 2, 2, 1, 2, 1]

class NFTProject:
    def __init__(self, nft_project_data, setN, setM, nft_project_name):
        self.setN = setN 
        self.setM = setM 
        self.nft_project_name = nft_project_name
        self.trait_dict = nft_project_data['trait_system']
        self.numericalize(nft_project_data)

    def numericalize(self, nft_project_data):
        asset_traits, buyer_assets_ids, buyer_budgets, item_counts = nft_project_data['asset_traits'], nft_project_data['buyer_assets_ids'], nft_project_data['buyer_budgets'], nft_project_data['item_counts']

        max_item_len = min(max([len(x) for x in buyer_assets_ids]), 10)
        min_purchase_limit = min_purchase[nft_project_names.index(self.nft_project_name)] if self.nft_project_name in nft_project_names else 1
        alluser_prefs = []
        buyer_num = self.setN if self.setN is not None else len(buyer_assets_ids)

        ## collect NFT item set
        nft_set = set()
        num_trades = 0
        for i in range(min(buyer_num, len(buyer_assets_ids))):
            if len(buyer_assets_ids[i]) < min_purchase_limit:
                continue
            num_trades += len(buyer_assets_ids[i])
            nft_set |= set(buyer_assets_ids[i])
            user_prefs = self.trait2label_vec([asset_traits[aid] for aid in buyer_assets_ids[i]])
            # pad until max length, then change shape
            if len(user_prefs) <= max_item_len:
                user_prefs = user_prefs + [user_prefs[-1]]*(max_item_len - len(user_prefs))
            else:
                user_prefs = user_prefs[:max_item_len]

            user_prefs = [list(x) for x in zip(*user_prefs)]
            alluser_prefs.append(user_prefs)

        self.N = self.setN if self.setN is not None else len(buyer_assets_ids)
        if 'large' in self.nft_project_name:
            nft_set = set(range(self.N))            
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

    