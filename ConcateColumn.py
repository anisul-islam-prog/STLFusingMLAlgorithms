import pandas as pd

mainconcat=pd.read_csv( '1. mainconcat.csv',
    delimiter=',',
    header=None,
    names=['Date', 'boiler','solar_thermal_pump','laptop','washing_machine','dishwasher','tv','kitchen_lights','htpc','kettle',
           'toaster','fridge','microwave','lcd_office','hifi_office','breadmaker','amp_livingroom','adsl_router','livingroom_s_lamp',
           'soldering_iron','gigE_&_USBhub','hoover','kitchen_dt_lamp','bedroom_ds_lamp','lighting_circuit','livingroom_s_lamp2',
           'iPad_charger','subwoofer_livingroom','livingroom_lamp_tv','DAB_radio_livingroom','kitchen_lamp2','kitchen_phone&stereo',
           'utilityrm_lamp','samsung_charger','bedroom_d_lamp','coffee_machine','kitchen_radio','bedroom_chargers','hair_dryer',
           'straighteners','iron','gas_oven','data_logger_pc','childs_table_lamp','childs_ds_lamp','baby_monitor_tx',
           'battery_charger','office_lamp1','office_lamp2','office_lamp3','office_pc','office_ds_fan','LED_printer',
           'Holiday', 'Average_Temp_in_Celsius', 'Average_Humidity_in_percentage'],

    index_col='Date',
    parse_dates= True)
df1=pd.read_csv( r'E:\Thesis Content\ukdale CSV\WeatherData_AverageWindSpeed.csv',
    delimiter=',',
    header=None,names=['Date', 'Average_WindSpeed_in_KMperH'],index_col='Date',
    parse_dates = True)
result =pd.concat([mainconcat, df1], axis=1, join_axes=[mainconcat.index])
result.fillna(0, inplace=True)# fill the NaN with 0  , to drop NaN use .dropna(inplace=True)
result.to_csv('1. mainconcat.csv')