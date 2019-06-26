import numpy as np
import pandas as pd
import googlemaps
import os

from environs import Env
from tqdm import tqdm

# parse env file for api key (python environs package)
env=Env()
env.read_env()
api_key = env("API_KEY")

class Map(object):
    """docstring for Map."""

    locs_path = './data/locations.csv'
    columns=['Location','Latitude','Longitude']

    def __init__(self):
        super(Map, self).__init__()
        self.init_locs()
        self.needed_locs = self.load_needed()
        self.locs_not_stored = self.ck_already_saved()
        if len(self.locs_not_stored) > 0:
            self.use_api(self.locs_not_stored)
        self.loc_dict = self.create_dict()

    def init_locs(self):
        if not os.path.exists(self.locs_path):
            print("Creating location file...")
            locs_df = pd.DataFrame(columns=self.columns)
            locs_df.to_csv(self.locs_path,index=False)
        return

    def load_needed(self):
        needed_path = '../golf_scraper/sched.csv'
        needed = pd.read_csv(needed_path)
        locations = needed.location.unique()
        print('There are ' + str(len(locations)) + ' unique locations')
        return locations

    def ck_already_saved(self):
        stored_locs = pd.read_csv(self.locs_path)
        stored_locations = stored_locs.Location.unique()
        print(str(len(stored_locations))+" locations have already been stored")

        locs_not_stored = list(set(self.needed_locs)-set(stored_locations))

        locs_to_skip = []
        for loc in locs_not_stored:
            skip = self.check_skip(loc)
            if skip:
                locs_to_skip.append(loc)

        if len(locs_to_skip) > 0:
            print("Skipping " +str(len(locs_to_skip))+" tournaments because they don't deserve to be looked up")

        locs_not_stored = list(set(locs_not_stored)-set(locs_to_skip))

        if len(locs_not_stored) > 0:
            print(str(len(locs_not_stored)) + ' of those are not stored, looking them up...')
            # print(locs_not_stored)
        return locs_not_stored

    def create_dict(self):
        loc_df = pd.read_csv(self.locs_path)
        print(loc_df.head())
        loc_df = loc_df.set_index('Location')
        loc_dict = loc_df.to_dict('index')
        print(loc_dict)
        return

    def check_skip(self,trn):
        skip = False
        skips = ['• Purse:$480,000']
        if trn in skips:
            skip = True
        return skip

    def check_clean(self,trn):
        new_trn = None
        loc_found = False
        possible_clean = ['The Taiheiyo Club, ,  ']
        if trn in possible_clean:
            if trn =='The Taiheiyo Club, ,  ':
                new_trn = 'The Taiheiyo Club, Omitama, Ibaraki, Japan'
            loc_found = True
        return new_trn

    def use_api(self, locs_not_stored):

        print("Using googlemaps...")
        # init googlemaps
        gmaps = googlemaps.Client(key=api_key)

        new_locations = []
        for loc in tqdm(locs_not_stored):

            geocode_result = gmaps.geocode(loc)
            loc_found = False
            try:
                lat = geocode_result[0]['geometry']['location']['lat']
                lng = geocode_result[0]['geometry']['location']['lng']
                loc_found = True
            except:
                # check if can alter so that it can find it
                clean = self.check_clean(loc)
                # try again
                try:
                    geocode_result = gmaps.geocode(clean)
                    lat = geocode_result[0]['geometry']['location']['lat']
                    lng = geocode_result[0]['geometry']['location']['lng']
                    loc_found = True
                except:
                    print(loc, "Still Not Found")
                    pass

            if loc_found:
                print(loc, lat, lng)
                new_locations.append([loc,lat,lng])

        if len(new_locations) > 0:
            print(str(len(new_locations))+" new locations were found...")
            new_df = pd.DataFrame(new_locations, columns=self.columns)
            old_df = pd.read_csv(self.locs_path)
            old_df = old_df[self.columns]
            df_to_save = pd.concat([old_df,new_df],ignore_index=True,axis=0)
            df_to_save.to_csv(self.locs_path,index=False)



        return

map = Map()





#
