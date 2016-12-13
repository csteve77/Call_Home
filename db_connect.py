import cx_Oracle as oracle
import pyodbc
import pandas

class CONNECTION_ORA():
    def __init__(self,username,password,host,port,sid):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.sid = sid
        
    def connect_ora(self):    
        connection = oracle.connect(self.username+'/'+self.password+'@'+self.host+':'+self.port+'/'+self.sid)
        return connection

    def closeCon(self,connection):
        connection.close()
        
    def read_query(self,connection, query):
        cur = connection.cursor()
        try:
            cur.execute(query)
            names = [ x[0] for x in cur.description]
            rows = cur.fetchall()
            return pandas.DataFrame( rows, columns=names)
        finally:
            if cur is not None:
                cur.close()
        
class CONNECTION_MSSQL():
    def __init__(self,driver,server,database,tc):
        self.driver = driver
        self.server = server
        self.database = database
        self.tc = tc
        
    def connect_mssql(self):
        connection = pyodbc.connect(self.driver+';'+self.server+';'+self.database+';'+self.tc)
        return connection    
        
    def closeCon(self,connection):
        connection.close()
        
    def read_query(self,connection, query):
        cur = connection.cursor()
        try:
            cur.execute(query)
            names = [ x[0] for x in cur.description]
            rows = cur.fetchall()
            return pandas.DataFrame( rows, columns=names)
        finally:
            if cur is not None:
                cur.close()
        
    def write_mssql_query(self, connection, query):    
        cur = connection.cursor()
        cur.execute(query)        
                