import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

class DBConnection():


    def get_connection(self):
        """

        :return: db: Gets the connection for the db supernova
        """
        config = {
            'user': 'root',
            'password': 'root@12345',
            'host': 'localhost',
            'database': 'GE_CLUSTERS',
            'raise_on_warnings': True,
            'use_pure': False,
        }
        cnx = mysql.connector.connect(**config)

        return cnx

    def close_connection(self,db):
        db.close()

class DFToSQl():

    def save_df_to_sql(self,df=pd.DataFrame()):
        """

        :return:
        """

        engine = create_engine("mysql://root:root@12345@localhost/GE_CLUSTERS")
        con = engine.connect()
        data = {'A':['1','3','4','8'],'B':['2','4','7','9']}
        df = pd.DataFrame(data=data)
        df['id'] = df.index
        df.to_sql(name='tbl_cluster', con=con, if_exists='append')

if __name__ == '__main__':
    db = DFToSQl()
    db.save_df_to_sql()
