import pandas as pd 
import numpy as np
FEATURE_COLUMNS = ['device_type','os_vendor','os_name','os_version','browser_name','browser_version','location_region','location_state','app_name',
                   'site_name','site_link','data_type','exchange','carrier','captured_time',
                   'device_screen_height','device_screen_width']
#list of important numerical features
num_feat = ['device_screen_height', 'device_screen_width']

#list of important categorical features
cat_feature = ['device_type','os_name','os_vendor','browser_name','location_region','location_state',
           'exchange','carrier','os_name_version', 'browser_name_version']
# Non important features.
top_attr = ['os_name_version', 'browser_name_version', 'inventory','location_state']

dropped_feat = ['app_name','site_name','site_link','captured_time','os_version','browser_version','os_name', 'browser_name']

def counts(data,column):
    print(data[column].astype('str').value_counts())
# checking for features with missing values greater than 200,000
def missing_column(data):
    print (data.columns[data.isnull().sum()>200000].tolist())
def groupby(data,feature,target='data_type'):
    df_grp = data.groupby([target,feature])
    return df_grp.size().reset_index(name='count').sort_values('count', ascending= False)
def split(data,column1='site_link',column2='site_name'):
    data[column2] = data[column1].str.split('/', expand = True)[2]
def inventory(data,column1= 'app_name', column2 = 'site_name'):
    data['inventory'] = data[column1].replace(np.nan,'')+ data[column2].replace(np.nan,'')
    data['inventory'] = data['inventory'].replace('',np.nan)
    data.dropna(subset=['inventory'], inplace=True)
def replace_region(data):
    region = data['location_state']
    if region in ['Benue','Kogi','Kwara','Nasarawa','Niger','Plateau','Abuja','Makurdi','Lokoja','Asokoro']:
        return 'North_Central'
    elif region in ['Adamawa','Bauchi','Borno','Gombe','Taraba','Yobe','Jos','Minna','Maiduguri','Yola']:
        return 'North_East'
    elif region in ['Jigawa','Kaduna','Kano','Katsina','Kebbi','Sokoto','Zamfara','Zaria','Dutse']:
        return 'North_West'
    elif region in ['Abia','Anambra','Ebonyi','Enugu','Imo','Abakaliki','Owerri','Umuahia']:
        return 'South_East'
    elif region in ['Akwa Ibom','Cross River','Bayelsa','Rivers','Delta','Edo','Benin City','Port Harcourt','Asaba',
                   'Warri','Nsukka','Calabar','Uyo','Yenagoa','Eket','Sagbama','Bonny','Effurun']:
        return 'South_South'
    elif region in ['Ekiti','Lagos','Ogun','Ondo','Osun','Oyo','Ikeja','Ikire','Badagri','Ikorodu','Ibadan',
                   'Suleja','Ilorin','Abeokuta','Osogbo','Akure','Ede','Ikotun','Lekki','Ikoyi','Ota','Ojota',
                   'Sagamu','Ogudu','Mowe','Agege','Omu-Aran','Aponri']:
        return 'South_West'

# concatenating name and version to form a new single column
def concat_feat(data):
    data['os_name_version'] = data['os_name'] + data['os_version'].astype(str)
    data['browser_name_version'] = data['browser_name'] + data['browser_version'].astype(str)
def pick_top_attr(data):
    for item in top_attr:
        top_val = list(data[item].value_counts()[:50].index)
        data[item] = data[item].apply(lambda x : x if x in top_val else "other")
def replace_vendor(os_name,os_vendor):
    if 'Macintosh' in str(os_name):
        return 'Apple'
    elif 'X11' in str(os_name):
        return 'Microsoft'
    else:
        return os_vendor
    
def missing_values(data,missing_list=cat_feature):
    for features in missing_list:
        if missing_list == cat_feature:
            data[features].fillna('others', inplace= True)
        else:
            data[features].fillna(data[features].mean(), inplace=True)

def pre_process(data):
    ## populates the site_name by site_link
    split(data)
    #returns a list of inventory by app_name + site_name
    inventory(data)
    concat_feat(data)
    pick_top_attr(data)
    ## populates the location_region by using location_site
    data['location_region'] = data.apply(replace_region, axis=1)
    data['os_vendor']=data.apply(lambda x: replace_vendor(x['os_name'],x['os_vendor']),axis=1)
    missing_values(data, num_feat)
    missing_values(data)
    data.drop(dropped_feat, axis=1, inplace = True)
