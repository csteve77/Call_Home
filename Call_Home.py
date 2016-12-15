import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import math
from scipy.stats import mode
from sklearn.feature_selection import VarianceThreshold as VT
from pandas import set_option

name = ['MODULE_TYPE','CDR_ORIGINATED_FILE_ID','SERVER_ID','CLEAR_CAUSE','RELEASE_CAUSE','PROTOCOL_ID','CHARGEABLE_SUBS_ID',
'CHARGEABLE_CUSTOMER_ID','PRODUCT_TYPE','EVENT_PARAMETER4' ,'EVENT_PARAMETER5' ,'EVENT_PARAMETER6'  ,'EVENT_PARAMETER7',
'EVENT_PARAMETER9' ,'EVENT_PARAMETER10','EVENT_PARAMETER14','EVENT_PARAMETER15','EVENT_PARAMETER16' ,'EVENT_PARAMETER17',
'EVENT_PARAMETER18','EVENT_PARAMETER19','EVENT_PARAMETER20','EVENT_PARAMETER23','EVENT_PARAMETER24' ,'EVENT_PARAMETER25',
'EVENT_PARAMETER26','EVENT_PARAMETER27','EVENT_PARAMETER28','EVENT_PARAMETER29','EVENT_PARAMETER30' ,'RATING_PARAMETER1',
'RATING_PARAMETER2','RATING_PARAMETER6','RATING_PARAMETER8','RATING_PARAMETER9','RATING_PARAMETER12','RATING_PARAMETER13',
'SLPI','SL_SEIZE_TIME','SL_INITIATION_TYPE','SL_CLUSTER_ID','CHARGEABLE_DURATION','CHARGEABLE_WALLET_TYPE_ID','ENDING_BALANCE',
'AMOUNT_CHARGED','AMOUNT_CHARGED1','AMOUNT_CHARGED2','AMOUNT_CHARGED3','AMOUNT_CHARGED4','PRICE_ITEM_ID1','PRICE_ITEM_ID2',
'PRICE_ITEM_ID3','PRICE_ITEM_ID4','LAYOUT_ID1','LAYOUT_ID2','LAYOUT_ID3','LAYOUT_ID4','LAYOUT_ROW_ID1','LAYOUT_ROW_ID2',
'LAYOUT_ROW_ID3','LAYOUT_ROW_ID4','ACCUMULATOR_ID1','ACCUMULATOR_ID2','ACCUMULATOR_ID3','ACCUMULATOR_ID4','PARENT','CHILD_INDEX',
'REFUND_FLAG','SPARE2','SPARE3','PRODUCT_LAYOUT_ROW_ID','WALLET_LAYOUT_ROW_ID','DAYS_OF_RC','UNIQUE_ID','PRICE_PER_ITEM',
'PRODUCTS_QUANTITY','PRODUCT_ID','TOTAL_PAYMENTS','PAYMENT_NUMBER','COMMENTS','VERSION_NUMBER','RELEASE_NUMBER','COUNTER_ID',
'COUNTER_CONSUMED_AMOUNT','COUNTER_ENDING_VALUE','PROMOTION_SET_ID','TOTAL_VOLUME','PAYMENT_ID','IS_RERATEABLE','RERATE_REQUEST_ID',
'REVENUE_TYPE','SUBS_REF_NUMBER','TOTAL_TAX','MAIN_ACCUMULATOR_ID','ZONE','DESTINATION','BUNDLE_ID','RESOURCE_TYPE','RESOURCE_NUMBER',
'TAX_BASE_AMOUNT1','TAX_BASE_AMOUNT2']

#dates = ['ANSWER_TIME','DISCONNECT_TIME','END_COVERAGE','ORIGINATE_TIME','SL_SEIZE_TIME','START_COVERAGE','TRANSACTION_TIME']           
    
chunk = pd.read_csv('C:/Users/scian/Call_Home/Voice_August/CH_01_082016.csv',
                  encoding = 'mbcs', 
                  iterator = True,
                  chunksize = 1000)
                  #delimiter = ';',
                  #nrows=1000)#,
                  #dtype = dtypes)
                 # parse_dates = dates) takes much lobger and uses a lot of memory

cdr_a = pd.concat(chunk, ignore_index = True)
cdr =cdr_a[name]
pd.set_option('display.width', 100)
pd.set_option('precision', 2)
pd.set_option('display.float_format', lambda x: '%14.2f' % x) #displays numeric floats to 2 decimal places 
#cdr = cdr[cdr['PRODUCT_TYPE'] != -1] # eliminates invalid cdrs
data = cdr[cdr['PRODUCT_TYPE'] == 21] # dataframe with data only
voice = cdr[cdr['PRODUCT_TYPE'] == 13] # dataframe wih voice only
sms = cdr[cdr['PRODUCT_TYPE'] == 2] # dataframe with sms only
rc = cdr[cdr['PRODUCT_TYPE'] == 22] # dataframe with RC only

#'CHARGEABLE_SUBS_ID', new_voice,'PRODUCT_ID' , value = received
def get_count(col1, df, col2=None,value=[], agg = None):
    count = dict()
    column1 = df[col1]
    dinstinct_column1 = column1.unique()
    if (col2 == None):
        for each in dinstinct_column1:
            count[each] = column1[column1 == each].count()
        return count        
    elif ((col2 != None) & (value != None) & (agg == None)):
        count2 = dict()
        column2 = df[col2]
        for each in dinstinct_column1:
            count = 0
            for x in value:
                count1 = column1[(column1 == each) & (column2 == x)].count()
                count += count1
            count2[each] = count    
        return count2
    elif (agg == 'sum'):
        count3 = dict()
        column2 = df[col2]
        for each in dinstinct_column1:
            count3[each] =column2[column1==each].sum()         
        return count3

def get_cols_with_nulls(df):
    return df[pd.isnull(df).any(axis=1)]
        
def fill_nulls(df,column, value_to_assign):
    df[column] = df[column].fillna(value_to_assign)
    return df

def num_missing(x):
    return sum(x.isnull())   

def get_all_null_counts(df):   
    columns = df.columns
    na_count = dict()
    for each in columns:
        count = df[each].isnull().sum()
        if count > 0:
            na_count[each] = count    
    return  na_count      
    
def map_values(column, df):
    val_maps = dict()
    unique_vals = df[column].unique()
    maps = 1
    for key in unique_vals:
        val_maps[key] =  maps
        maps += 1
    return val_maps  

def drop_all_null_cols(df):
    df = df.dropna(how = 'all', axis =1)    
    return df
#na_counts = pd.DataFrame(list(na_count.items()))


def bar_plot_nulls(df, xlabel, ylabel, title):
    figure = plt.figure(figsize=(7,5))
    ax = figure.add_subplot(1,1,1)
    null_counts = get_all_null_counts(df)
    max_y = df.shape[0]
    #these conditionals define the scale for each plot
    if max_y >10000:
        major_ticks = np.arange(0,max_y,math.ceil((max_y/20)/10000)*10000)
        minor_ticks = np.arange(0,max_y,(max_y/100))
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
    elif ((max_y > 500) and (max_y < 10000)): 
        major_ticks = np.arange(0,max_y,math.ceil((max_y/10)/100)*100)
        minor_ticks = np.arange(0,max_y,(max_y/50))
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
    elif ((max_y < 500) & (max_y > 0)) :    
        major_ticks = np.arange(0,max_y,math.ceil((max_y/20)/5)*5)
        minor_ticks = np.arange(0,max_y,(max_y/40))
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        
    ax.set_xlabel(xlabel, weight ='bold', fontsize = 13)
    ax.set_ylabel(ylabel, weight = 'bold', fontsize = 13)
    ax.set_title(title + 'Ratio = {}:{}'.format(len(null_counts), cdr.shape[1]), fontsize = 15, weight = 'bold' )
    plt.xticks(range(len(null_counts)), null_counts.keys(), rotation= 90, ha='left')
    plt.margins(0.01,0.02)
    #labels = plt.xticks()
    plt.bar(range(len(null_counts)),null_counts.values(), width =0.75,color = '0.5', alpha = 0.8)
    plt.show()

    
def remove_low_variance(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    non_numeric = df.select_dtypes(exclude=numerics)
    numeric = df._get_numeric_data()
    selector = VT()
    selector.fit_transform(numeric)
    selected = selector.get_support()#identifies the features chosen by the algorithm
    selected_text = np.where(selected == True)[0]
    hv = numeric.iloc[:,selected_text]
    final_df = hv.join(non_numeric, how='inner')
    return final_df    
    


####-----------------------------------To keep-------------------------------------------####
#############################################################################################

# the groupby size option of pandas tells us how distributed a variable is by size.
#class_counts = filtered_cdrs_cc0.groupby('EVENT_PARAMETER1').size()
#print(class_counts)
#this results in    #EVENT_PARAMETER1
                    #DATA     158095
                    #RC           23
                    #SMS        2628
                    #VOICE     20965
##############################################################################################

#correlations = filtered_cdrs_cc0.corr(method ='pearson')
#print(correlations)

#########################################################################################
#this defines the skwness of the distribution and whether it is left or right. close to 0 means lesser skew
#skew = filtered_cdrs_cc0.skew()
#print(skew)

#new_set = filtered_cdrs_cc0[['SUBS_ID','AMOUNT_CHARGED2']]
#new_set.hist()
#plt.show()
'''

numeric = cdr._get_numeric_data()
numeric2 = numeric.dropna(how = 'all', axis =1)
for each in numeric2.columns:
    numeric2[each] = numeric2[each].fillna(mode(numeric2[each]).mode[0])

print(non_numerics.shape)

#print(numeric.apply(num_missing,axis = 0 ))

numeric2 = numeric.dropna(how = 'all', axis =1)

for each in numeric2.columns:
    numeric2[each] = numeric2[each].fillna(mode(numeric2[each]).mode[0])

#print(mode(numeric2['AMOUNT_CHARGED']).mode[0])

#grouped = numeric3.groupby(['CHARGEABLE_SUBS_ID', 'PRODUCT_TYPE'], sort = False)


#print(grouped.get_group((200204400,1))['EVENT_PARAMETER1'])
#print(numeric3.columns)
'''
print (cdr.PRODUCT_TYPE.unique())
dataframes = {'Data':data,'SMS':sms,'Voice':voice,'Recurring_Charges':rc}
#plot before we remove columns with all values null
for k,v in dataframes.items():
    bar_plot_nulls(v,'Null Features','Null Count', k +' Features With Nulls. ')
    print(k, v.shape)
    dataframes[k] = drop_all_null_cols(v) #drops the columns where all values are null

print()
print('---------------Reduced Dimensions--------------')
print()    

#plot with columns who have a few null values    
for k,v in dataframes.items():    
    print(k, v.shape)
    bar_plot_nulls(v,'Null Features','Null Count', k +' Features With Nulls. ')
'''
# writes all cleaned recurring charges to csv file
#dataframes['Recurring_Charges'] = remove_low_variance(dataframes['Recurring_Charges'])
#shortcodes = ['SC100','SC121','SC1182','SC112','1' ]
dataframes['Voice'] = dataframes['Voice'].drop(['EVENT_PARAMETER14','EVENT_PARAMETER16','EVENT_PARAMETER25'], axis =1)
dataframes['Voice'] = dataframes['Voice'].fillna({'EVENT_PARAMETER7':'35600000000','EVENT_PARAMETER6' : 0})
dataframes['Voice']['EVENT_PARAMETER24'] = dataframes['Voice']['EVENT_PARAMETER24'].fillna(method = 'ffill')
dataframes['Voice'].loc[(dataframes['Voice']['CLEAR_CAUSE'] == 449) , ['DESTINATION']] ='CALL_FAILED_DUE_TO_NO_TARIFF'
#print()
#print('-----------------------------------------------')
#print()
dataframes['Voice']=remove_low_variance(dataframes['Voice'])
new_voice = dataframes['Voice']
#print(dataframes['Voice'].columns)
#dataframes['Voice'][dataframes['Voice']['EVENT_PARAMETER6'].isnull()].to_csv('EP6.csv', encoding = 'mbcs', index = False, float_format='{:20.2f}')#!f (explicit conversion to float)
#print(dataframes['Voice']['AMOUNT_CHARGED'])

#-----------------------Aggregation per Subscriber_id---------------------------
#-----------------------BUILD THE NEW DATAFRAME-----------------------------------

#Need to insert specific date to identify the records per day. useful to remove old values'

on_net = [9,10]; off_net = [11,12];international = [13]; roaming = [14]
freephone = [17];shortcodes = [186];received_id = [19,21]; success_cc= [0]; no_credit_cc =[407]

voice_dict = dict()
voice_dict['CHARGEABLE_SUBS_ID'] = pd.Series(new_voice['CHARGEABLE_SUBS_ID'].unique(), index = None)
voice_agg = pd.DataFrame(data = voice_dict)
voice_agg['CALL_COUNT'] = pd.Series()
voice_agg['RECEIVED_CALLS'] = pd.Series() 
voice_agg['ON_NET'] = pd.Series()
voice_agg['OFF_NET'] = pd.Series()
voice_agg['INTERNATIONAL'] = pd.Series()
voice_agg['ROAMING'] = pd.Series()
voice_agg['SHORT_CODE_CALLS'] = pd.Series()
voice_agg['FREEPHONE'] = pd.Series()
voice_agg['SUCCESSFUL'] = pd.Series()
voice_agg['FAIL_NO_CREDIT'] = pd.Series()
voice_agg['CHARGEABLE_DURATION'] = pd.Series()
voice_agg['AMOUNT_CHARGED'] = pd.Series()

call_counts     = get_count('CHARGEABLE_SUBS_ID', new_voice)
received        = get_count('CHARGEABLE_SUBS_ID', new_voice,'PRODUCT_ID' , value = received_id)
success_counts  = get_count('CHARGEABLE_SUBS_ID', new_voice,'CLEAR_CAUSE', value = success_cc)
no_credit       = get_count('CHARGEABLE_SUBS_ID', new_voice,'CLEAR_CAUSE', value = no_credit_cc)
on_net_calls    = get_count('CHARGEABLE_SUBS_ID', new_voice,'PRODUCT_ID' , value = on_net)
off_net_calls   = get_count('CHARGEABLE_SUBS_ID', new_voice,'PRODUCT_ID' , value = off_net)
int_calls       = get_count('CHARGEABLE_SUBS_ID', new_voice,'PRODUCT_ID' , value = international)
roaming_calls   = get_count('CHARGEABLE_SUBS_ID', new_voice,'PRODUCT_ID' , value = roaming)
sc_calls        = get_count('CHARGEABLE_SUBS_ID', new_voice,'PRODUCT_ID' , value = shortcodes)
freephone_calls = get_count('CHARGEABLE_SUBS_ID', new_voice,'PRODUCT_ID' , value = freephone)
charge_duration = get_count('CHARGEABLE_SUBS_ID', new_voice,'CHARGEABLE_DURATION',agg ='sum')
amount_charged  = get_count('CHARGEABLE_SUBS_ID', new_voice,'AMOUNT_CHARGED',agg ='sum') 

#--------------------FILL THE NEW DATAFRAME-----------------------------------

for k,v in call_counts.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['CALL_COUNT']] = v

for k,v in received.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['RECEIVED_CALLS']] = v

for k,v in success_counts.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['SUCCESSFUL']] = v

for k,v in no_credit.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['FAIL_NO_CREDIT']] = v

for k,v in on_net_calls.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['ON_NET']] = v

for k,v in off_net_calls.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['OFF_NET']] = v

for k,v in int_calls.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['INTERNATIONAL']] = v

for k,v in roaming_calls.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['ROAMING']] = v

for k,v in sc_calls.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['SHORT_CODE_CALLS']] = v

for k,v in freephone_calls.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['FREEPHONE']] = v
 
for k,v in charge_duration.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['CHARGEABLE_DURATION']] = v

for k,v in amount_charged.items():
    voice_agg.loc[(voice_agg['CHARGEABLE_SUBS_ID']== k),['AMOUNT_CHARGED']] = v


voice_agg['FAILED'] = (voice_agg['CALL_COUNT'] - voice_agg['SUCCESSFUL'])
voice_agg['FAIL_OTHER_REASON'] = (voice_agg['FAILED'] - voice_agg['FAIL_NO_CREDIT']) 

sample = voice_agg.head(50)    
    
#print(voice_agg)
plt.scatter(voice_agg['FAIL_NO_CREDIT'],voice_agg['FAILED'])

scatter_matrix(voice_agg, alpha = 0.1, figsize =(15,15), diagonal = 'kde')
voice_agg.hist(figsize =(10,10))
#voice_agg.plot(kind = 'box',layout = (4,4), figsize = (10,10),subplots = True, sharex = False, sharey = False)
#plt.show()


#print(voice_agg.head(50))

'''
