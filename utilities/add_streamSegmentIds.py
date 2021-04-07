     # -*- coding: utf-8 -*-
"""
Created 10-22-2020 by Rai Hann

"""
import pandas as pd
import urllib
from sqlalchemy import create_engine
import pandas as pd

#enter input source QC configuration file
source_file = "../sample_QA_metadata/QC_SC_PROD.csv"

#target QC configuration filename
#
target_file = "../sample_QA_metadata/QC_SC_PROD_merged.csv"

df_left = pd.read_csv(source_file)
df_left['MonitorId'] = df_left['MonitorId'].astype(str)
#df_left['Parameter'] = df_left['Parameter'].astype(str)

# assign queries to be executed
# resultset contains StreamSegmentId for each registered monitor
qry1="""
     select ss.DetectionUOMId, ss.StreamSegmentId, ss.ParameterId, m.ExternalMonitorId from Site s
     inner join Monitor m on  s.SiteId = m.SiteId
     inner join StreamSegment ss on m.ExternalMonitorId = ss.ExternalMonitorId where DataProviderId = 67
     """

#Connecting to SQL server
params = urllib.parse.quote_plus(r'DRIVER={SQL Server};Trusted_Connection=yes;Server=CAQM-DB;DATABASE=AQview')
c_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
engine = create_engine(c_str)
connected =  engine.connect()
#Testing for connection (Prints connection ID if connection is successful)
print(connected)

# execute query1 and query2 and obtain resultset dataframes
df_right = pd.read_sql_query(qry1, engine)

df_right.rename(columns={'ParameterId':'Parameter', 'ExternalMonitorId':'MonitorId', 'DetectionUOMId':'UOMId'}, inplace=True)

merged_df = df_left.merge(df_right, on = ['MonitorId', 'Parameter'], how = 'inner')    

print(merged_df)
# close database connection
connected.close()


merged_df.to_csv(target_file, index=None)


    

    



