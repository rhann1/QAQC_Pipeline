import requests as req
from pandas.io.json import json_normalize
import pandas as pd
from datetime import datetime, timedelta
import pytz
import urllib
from sqlalchemy import create_engine
import numpy as np

# Set date for data you want to pull back from to current date
StartTime = datetime.strptime('2021-03-28T00:00:00Z', "%Y-%m-%dT%H:00:00Z")
EndTime = datetime.strptime('2021-03-29T00:00:00Z', "%Y-%m-%dT%H:00:00Z")

#CurrentTime = datetime.now()
TimeDif = EndTime - StartTime
NumDays = TimeDif.days

StartTime_local = pytz.timezone('US/Pacific').localize(StartTime)
StartTime_UTC= StartTime_local.astimezone(pytz.utc) 

#Setup Clarity API GET Parameters
URLMeasurements = "https://clarity-data-api.clarity.io/v1/measurements"
APIHEADERS = {'x-api-key': 'kqP7ILa57DV3jqjqDJDXiLXuhPqqZaF7YoUIgak9'}

#Mapping of Clarity parameter names to AQview parameter codes
parameterdict={'vocConc': 43104, 
               'no2Conc': 42602,
               'pm10ConcMass': 85101,
               'pm10ConcNum': 187100,
               'pm1ConcMass': 189101,
               'pm1ConcNum': 187010,
               'pm2_5ConcMass': 88501,
               'pm2_5ConcNum': 187025,
               'relHumid': 68110,
               'temperature': 62101,
               None:0
               }
                
#Mapping of Clarity parameter names to AQview unit codes
unitdict = {'vocConc': '008', 
            'no2Conc': '008',
            'pm10ConcMass': '105',
            'pm10ConcNum': '132',
            'pm1ConcMass': '105',
            'pm1ConcNum': '132',
            'pm2_5ConcMass': '105',
            'pm2_5ConcNum': '132',
            'relHumid': '019',
            'temperature': '017'}
             
#AQview data format fields list
AQviewFields= ['SiteId',
               'MonitorId',
               'Parameter',
               'StartDateTime',
               'AObs',
               'BObs',
               'CObs',
               'AObsAdj',
               'BObsAdj',
               'CObsAdj',
               'UOM',
               'AdjCode',
               'Uncertainty',
               'Quality']

#Get JSON from Clarity and convert to dataframe dfin
def GetBatchTime(StartTime_UTC, day):
    
    StartTime_UTC = StartTime_UTC + timedelta(days=day)
    EndTime_UTC = StartTime_UTC + timedelta(days=1)
    StartTime_local = StartTime_UTC.astimezone(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")
    EndTime_local = EndTime_UTC.astimezone(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")    

    StartTime_UTC= StartTime_UTC.strftime("%Y-%m-%dT%H:00:00Z")
    EndTime_UTC= EndTime_UTC.strftime("%Y-%m-%dT%H:00:00Z")
    
    FileTime = StartTime_local
    
    APIPARAMS = {'startTime': StartTime_UTC, 'endTime': EndTime_UTC, 'limit': 20000}
    #FilePath =  r'\\ab617-web-dev\ftptest\dataproviders\ad_sacramento\ftp\files\aqview\ad_sacramento\\' +  str(FileTime) + '_SMAQMD.csv' #DEV
    #FilePath =  r'\\caqm-web-uat\IncomingMeasurementData\ad_sacramento\ftp\files\aqview\ad_sacramento\\' +  str(FileTime) + '_SMAQMD.csv' #UAT
    FilePath =  r'\\datavol\logs\ad_sacramento\ftp\files\aqview\ad_sacramento\\' +  str(FileTime) + '_SMAQMD.csv' #PROD
     
   
    print("Pulling data for " + StartTime_local + " to " + EndTime_local + "...")
    
    return(FilePath, APIPARAMS)    

def Getdf(URL, HEADERS, PARAMS):
    r = req.get(url=URL, headers=HEADERS, params=PARAMS)
    data = r.json()
    dfin = json_normalize(data)
    
    return(dfin)    
    
#Reformat dfin to AQview format dfout
def ConvertFormat(dfin): 
    
    def MeltMerge(df):
        
        AobsFields = df.columns[df.columns.str.endswith('raw')] #define non-calibrated value fields
        AobsAdjFields = df.columns[df.columns.str.endswith('calibratedValue')] #define calibrated value fields
        dfout = df[["_id", "deviceCode", 'location.coordinates', "time"]].copy() #copies ID fields from unformatted Clarity output into df
        
        #Melt dfs down to have parameter be record level instead of column header; Separate melt for calibrated values and raw values since these are separate fields in aqview format
        s1 = df.melt(id_vars=["_id"],
        value_vars=AobsFields, 
        var_name="parameter", 
        value_name="aobs").dropna()
        s1['parameter'] = s1['parameter'].str.replace(".raw","") #cleanup for join
    
        s2 = df.melt(id_vars=["_id"],
        value_vars=AobsAdjFields, 
        var_name="parameter", 
        value_name="aobsadj").dropna()   
        s2['parameter'] = s2['parameter'].str.replace(".calibratedValue","") #cleanup for join

        #merge dfs - Aobs and AobsAdj
        dfout = pd.merge(dfout, s1, how ='left', on=["_id"]) #join dfs on '_id'; note: '_id' represents unique record level id
        dfout = pd.merge(dfout, s2, how ='left', on=["_id", "parameter"]) #join dfs on '_id' and 'parameter'
        dfout['parameter'] = dfout['parameter'].str.replace("characteristics.","") #clean up JSON artifact for mapping dictionary in future functions
        
        return(dfout)
    
    #Perform all field level conversions on Clarity data to match AQview format    
    def FieldLevelConversions(dfout):
        
        #Convert clarity timestamps from UTC to local 
        def ConvertTime(dfin):
            
            dfin['time'] = pd.to_datetime(dfin['time'], utc=True, format='%Y-%m-%dT%H:%M:%S.%fZ')
            dfin['time'] = dfin.time.dt.tz_convert('US/Pacific').dt.strftime('%Y-%m-%d %H:%M:%S')

            return(GenerateSiteIds(dfin))          
        
        #Fills in SiteId column based on results from AQview database and/or assigning "New Site" to locations not in database
        def GenerateSiteIds(dfin):
            
            #query existing sites for data providers
            def GetExistingSites(): 
                
                #Define queries to be executed
                sitesqry="""
                     SELECT sc.Latitude, sc.Longitude, s.ExternalSiteId as SiteId 
                     from SiteConfig as sc
                     Inner Join Site as s ON s.SiteId = sc.SiteId
                     where s.DataProviderId = 65"""
                                       
                #Connecting to SQL server
                #params = urllib.parse.quote_plus(r'DRIVER={SQL Server};Trusted_Connection=yes;Server=AB617-DB-Dev;DATABASE=AQview') #Dev
                #params = urllib.parse.quote_plus(r'DRIVER={SQL Server};Trusted_Connection=yes;Server=CAQM-DB-UAT;DATABASE=AQview') #UAT
                params = urllib.parse.quote_plus(r'DRIVER={SQL Server};Trusted_Connection=yes;Server=CAQM-DB;DATABASE=aqview') #Production
                c_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
                engine = create_engine(c_str)
                connection =  engine.connect()
                trans = connection.begin()
                
                #execute query and obtain resulting dataframes
                try:
                    SQLresult = pd.read_sql_query(sitesqry, engine)    # pull list of site ids                    
               
                except:
                    trans.rollback()
                    raise 
                
                finally:
                    connection.close() #close database connection
                    
                return(SQLresult)
            
            #Create dicitonary of SiteIds for each location coordinate pair and assign SiteIds
            def AssignSiteIds(SQLresult):
                
                SQLresult['Longitude'] = SQLresult['Longitude'].astype(float).round(5)
                SQLresult['Latitude'] = SQLresult['Latitude'].astype(float).round(5)
                SQLresult['Location'] = "[" + SQLresult['Longitude'].astype(str) + ", " + SQLresult['Latitude'].astype(str) + "]" #format df columns to match clarity format; Note precision is also truncated to 5 decimals here to improve join accuracy                
                
                sitedict = pd.Series(SQLresult.SiteId.values,index=SQLresult.Location).to_dict() #create dictionary from SQLresult df
                 
                
                dfin['longitude'] = dfin['location.coordinates'].str[0].astype(float).round(5) #limit precision of clarity longitude to 5 digits
                dfin['latitude'] = dfin['location.coordinates'].str[1].astype(float).round(5) #limit precision of clarity latitude to 5 digits
                dfin['location'] = "[" + dfin['longitude'].astype(str) + ", " + dfin['latitude'].astype(str) + "]" #format df columns to match clarity format; Note precision is also truncated to 5 decimals here to improve join accuracy                
                dfin['SiteId'] = dfin['location'].astype(str).map(sitedict) #Map SiteIds for existing Sites and tempSiteIds for existing temp sites
      
                NewLocations = list(dfin.loc[(dfin.SiteId.isnull())]['location'].astype(str).unique()) #Create list of location coordinates for new locations that do not correspond to existing sites or tempsites
              
                newsitesdict=dict.fromkeys(NewLocations, "NewSite")
                sitedict.update(newsitesdict)                
                                
                dfin['SiteId'] = dfin['location'].astype(str).map(sitedict)  
                
                if "NewSite" in dfin.SiteId.values:
                   
                    print("")
                    print("Unregistered site found - MonitorId: "+ dfin.loc[(dfin['SiteId']=="NewSite")]['deviceCode'].unique() + "; location: " + dfin.loc[(dfin['SiteId']=="NewSite")]['location'].unique() + "; Date range: " + dfin.loc[(dfin['SiteId']=="NewSite")]['time'].min() + " - " + dfin.loc[(dfin['SiteId']=="NewSite")]['time'].max())                     
                    print("")
                
                dfin.drop(dfin.loc[dfin['SiteId']=='NewSite'].index, inplace=True)
                return(dfin)
                
            SQLresult = GetExistingSites()
            dfin = AssignSiteIds(SQLresult)
            return(MapFields(dfin))        
        
        #Map formatted clarity data to AQview fields
        def MapFields(dfin):
            
            dfout = pd.DataFrame(columns=AQviewFields)
            dfout['SiteId'] = dfin['SiteId']
            dfout['MonitorId'] = dfin['deviceCode']
            dfout['Parameter'] = dfin['parameter'].map(parameterdict).astype(int)
            dfout['StartDateTime'] = dfin['time']
            dfout['AObs'] = dfin['aobs']            
            dfout['AObsAdj'] = dfin['aobsadj']
            dfout['UOM'] = dfin['parameter'].map(unitdict)
            dfout['AdjCode'] = np.where(dfout.AObsAdj.isnull(), "", "claritycal")

            return(dfout)
            
        return(ConvertTime(dfout))    
        
    dfout = MeltMerge(dfin)
    return(FieldLevelConversions(dfout))

def Main():
    
    for day in range(0, NumDays):
        FilePath, APIPARAMS = GetBatchTime(StartTime_UTC, day)
        dfin = Getdf(URLMeasurements, APIHEADERS, APIPARAMS) #Get DF from Clarity API call
        dfout = ConvertFormat(dfin) #Convert format into AQview format
        dfout.to_csv(FilePath, index=False)
if __name__ == "__main__": #Calls main() when executing script
    Main()
