from db_connect import CONNECTION_ORA
from calendar import monthrange
#from db_connect import CONNECTION_MSSQL
#from timer import TIMER

import configparser as cp 


conf = cp.ConfigParser()

conf.read('credentials.ini')

username = conf['USAGE']['username']
passw= conf['USAGE']['pass']
host = conf['USAGE']['host']
port = conf['USAGE']['port']
sid = conf['USAGE']['sid']

product = int(input('Enter Product LOB: '))
lob = {13:'_Voice',2:'_SMS',21:'_Data', 22:'_RC'}  
#cdr_path = '//AMSTERDAM/billingops$/Amdocs/Operations/Billing Cycles/Cycle Files'
cdr_path = 'E:/Call_Home_Plan/CDRs/July/Voice'
year = int(input ('Enter Year: '))
month = int(input('Enter Month: '))
days = monthrange(year,month)[1]
print (days)

day =1
while day <= days:
    if (len(str(month)) == 1):
        month_b = '0'+str(month)
        if (len(str(day)) == 1):
            day_b = '0'+str(day)
            filename = "CH_"+ day_b + "_" + str(month_b) + str(year) +str(lob[product])+ ".csv"
            filename = filename.replace(" ", "")
            con = CONNECTION_ORA(username,passw,host,port,sid)
            oracle = con.connect_ora()
            gl_query =  ("Select * from USAGE.cdr_partitioned partition (P{}{}{}) where offer_id = 321 and product_type = {} and chargeable_subs_id != -1".format(str(year), str(month_b), day_b, product))
            prod = con.read_query(oracle, gl_query)
            gl = con.read_query(oracle, gl_query)
            gl.to_csv(filename, encoding = 'mbcs', sep =',', index = False)
            print (day_b, month_b, ' : Done!')
        else:
             
            filename = "CH_"+ str(day) + "_" + str(month_b) + str(year) +str(lob[product])+ ".csv"
            filename = filename.replace(" ", "")
            con = CONNECTION_ORA(username,passw,host,port,sid)
            oracle = con.connect_ora()
            gl_query =  ("Select * from USAGE.cdr_partitioned partition (P{}{}{}) where offer_id = 321 and product_type = {} and chargeable_subs_id != -1".format(str(year), str(month_b), str(day), product))
            prod = con.read_query(oracle, gl_query)
            gl = con.read_query(oracle, gl_query)
            gl.to_csv(filename, encoding = 'mbcs', sep =',', index = False)
            print (day, month_b, ' : Done!')
    else:
        if (len(str(day)) == 1):
            day_b = '0'+str(day)
            filename = "CH_"+ day_b + "_" + str(month) + str(year) + str(lob[product])+".csv"
            filename = filename.replace(" ", "")
            con = CONNECTION_ORA(username,passw,host,port,sid)
            oracle = con.connect_ora()
            gl_query =  ("Select * from USAGE.cdr_partitioned partition (P{}{}{}) where offer_id = 321 and product_type = {} and chargeable_subs_id != -1".format(str(year), str(month), day_b, product))
            prod = con.read_query(oracle, gl_query)
            gl = con.read_query(oracle, gl_query)
            gl.to_csv(filename, encoding = 'mbcs', sep =',', index = False)
            print (day_b, month, ' : Done!')
        else:
             
            filename = "CH_"+ str(day) + "_" + str(month_b) + str(year) +str(lob[product])+ ".csv"
            filename = filename.replace(" ", "")
            con = CONNECTION_ORA(username,passw,host,port,sid)
            oracle = con.connect_ora()
            gl_query =  ("Select * from USAGE.cdr_partitioned partition (P{}{}{}) where offer_id = 321 and product_type = {} and chargeable_subs_id != -1".format(str(year), str(month), str(day), product))
            prod = con.read_query(oracle, gl_query)
            gl = con.read_query(oracle, gl_query)
            gl.to_csv(filename, encoding = 'mbcs', sep =',', index = False)
            print (day, month, ' : Done!')            
        
    day+=1

