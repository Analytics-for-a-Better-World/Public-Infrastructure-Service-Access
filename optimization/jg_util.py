import folium
import folium.plugins
import io
from PIL import Image
import numpy as np
import re
import geopy
import plotly.express as px
import matplotlib.pyplot as plt

import pandas as pd
import geopandas as gpd
import pickle

from pathlib import Path

def FreeLocator():
    return geopy.Photon(user_agent='myGeocoder')

def Locate(description):
    location = FreeLocator().geocode(description)
    if location:
        return location.latitude, location.longitude
    return None,None

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# extracted from http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
# under public domain terms
# (longitude, latitude, longitude, latitude)

country_bounding_boxes = {
    'AF': ('Afghanistan', (60.5284298033, 29.318572496, 75.1580277851, 38.4862816432)),
    'AO': ('Angola', (11.6400960629, -17.9306364885, 24.0799052263, -4.43802336998)),
    'AL': ('Albania', (19.3044861183, 39.624997667, 21.0200403175, 42.6882473822)),
    'AE': ('United Arab Emirates', (51.5795186705, 22.4969475367, 56.3968473651, 26.055464179)),
    'AR': ('Argentina', (-73.4154357571, -55.25, -53.628348965, -21.8323104794)),
    'AM': ('Armenia', (43.5827458026, 38.7412014837, 46.5057198423, 41.2481285671)),
    'AQ': ('Antarctica', (-180.0, -90.0, 180.0, -63.2706604895)),
    'TF': ('Fr. S. and Antarctic Lands', (68.72, -49.775, 70.56, -48.625)),
    'AU': ('Australia', (113.338953078, -43.6345972634, 153.569469029, -10.6681857235)),
    'AT': ('Austria', (9.47996951665, 46.4318173285, 16.9796667823, 49.0390742051)),
    'AZ': ('Azerbaijan', (44.7939896991, 38.2703775091, 50.3928210793, 41.8606751572)),
    'BI': ('Burundi', (29.0249263852, -4.49998341229, 30.752262811, -2.34848683025)),
    'BE': ('Belgium', (2.51357303225, 49.5294835476, 6.15665815596, 51.4750237087)),
    'BJ': ('Benin', (0.772335646171, 6.14215770103, 3.79711225751, 12.2356358912)),
    'BF': ('Burkina Faso', (-5.47056494793, 9.61083486576, 2.17710778159, 15.1161577418)),
    'BD': ('Bangladesh', (88.0844222351, 20.670883287, 92.6727209818, 26.4465255803)),
    'BG': ('Bulgaria', (22.3805257504, 41.2344859889, 28.5580814959, 44.2349230007)),
    'BS': ('Bahamas', (-78.98, 23.71, -77.0, 27.04)),
    'BA': ('Bosnia and Herz.', (15.7500260759, 42.65, 19.59976, 45.2337767604)),
    'BY': ('Belarus', (23.1994938494, 51.3195034857, 32.6936430193, 56.1691299506)),
    'BZ': ('Belize', (-89.2291216703, 15.8869375676, -88.1068129138, 18.4999822047)),
    'BO': ('Bolivia', (-69.5904237535, -22.8729187965, -57.4983711412, -9.76198780685)),
    'BR': ('Brazil', (-73.9872354804, -33.7683777809, -34.7299934555, 5.24448639569)),
    'BN': ('Brunei', (114.204016555, 4.007636827, 115.450710484, 5.44772980389)),
    'BT': ('Bhutan', (88.8142484883, 26.7194029811, 92.1037117859, 28.2964385035)),
    'BW': ('Botswana', (19.8954577979, -26.8285429827, 29.4321883481, -17.6618156877)),
    'CF': ('Central African Rep.', (14.4594071794, 2.2676396753, 27.3742261085, 11.1423951278)),
    'CA': ('Canada', (-140.99778, 41.6751050889, -52.6480987209, 83.23324)),
    'CH': ('Switzerland', (6.02260949059, 45.7769477403, 10.4427014502, 47.8308275417)),
    'CL': ('Chile', (-75.6443953112, -55.61183, -66.95992, -17.5800118954)),
    'CN': ('China', (73.6753792663, 18.197700914, 135.026311477, 53.4588044297)),
    'CI': ('Ivory Coast', (-8.60288021487, 4.33828847902, -2.56218950033, 10.5240607772)),
    'CM': ('Cameroon', (8.48881554529, 1.72767263428, 16.0128524106, 12.8593962671)),
    'CD': ('Congo (Kinshasa)', (12.1823368669, -13.2572266578, 31.1741492042, 5.25608775474)),
    'CG': ('Congo (Brazzaville)', (11.0937728207, -5.03798674888, 18.4530652198, 3.72819651938)),
    'CO': ('Colombia', (-78.9909352282, -4.29818694419, -66.8763258531, 12.4373031682)),
    'CR': ('Costa Rica', (-85.94172543, 8.22502798099, -82.5461962552, 11.2171192489)),
    'CU': ('Cuba', (-84.9749110583, 19.8554808619, -74.1780248685, 23.1886107447)),
    'CY': ('Cyprus', (32.2566671079, 34.5718694118, 34.0048808123, 35.1731247015)),
    'CZ': ('Czech Rep.', (12.2401111182, 48.5553052842, 18.8531441586, 51.1172677679)),
    'DE': ('Germany', (5.98865807458, 47.3024876979, 15.0169958839, 54.983104153)),
    'DJ': ('Djibouti', (41.66176, 10.9268785669, 43.3178524107, 12.6996385767)),
    'DK': ('Denmark', (8.08997684086, 54.8000145534, 12.6900061378, 57.730016588)),
    'DO': ('Dominican Rep.', (-71.9451120673, 17.598564358, -68.3179432848, 19.8849105901)),
    'DZ': ('Algeria', (-8.68439978681, 19.0573642034, 11.9995056495, 37.1183806422)),
    'EC': ('Ecuador', (-80.9677654691, -4.95912851321, -75.2337227037, 1.3809237736)),
    'EG': ('Egypt', (24.70007, 22.0, 36.86623, 31.58568)),
    'ER': ('Eritrea', (36.3231889178, 12.4554157577, 43.0812260272, 17.9983074)),
    'ES': ('Spain', (-9.39288367353, 35.946850084, 3.03948408368, 43.7483377142)),
    'EE': ('Estonia', (23.3397953631, 57.4745283067, 28.1316992531, 59.6110903998)),
    'ET': ('Ethiopia', (32.95418, 3.42206, 47.78942, 14.95943)),
    'FI': ('Finland', (20.6455928891, 59.846373196, 31.5160921567, 70.1641930203)),
    'FJ': ('Fiji', (-180.0, -18.28799, 180.0, -16.0208822567)),
    'FK': ('Falkland Is.', (-61.2, -52.3, -57.75, -51.1)),
    'FR': ('France', (-54.5247541978, 2.05338918702, 9.56001631027, 51.1485061713)),
    'GA': ('Gabon', (8.79799563969, -3.97882659263, 14.4254557634, 2.32675751384)),
    'GB': ('United Kingdom', (-7.57216793459, 49.959999905, 1.68153079591, 58.6350001085)),
    'GE': ('Georgia', (39.9550085793, 41.0644446885, 46.6379081561, 43.553104153)),
    'GH': ('Ghana', (-3.24437008301, 4.71046214438, 1.0601216976, 11.0983409693)),
    'GN': ('Guinea', (-15.1303112452, 7.3090373804, -7.83210038902, 12.5861829696)),
    'GM': ('Gambia', (-16.8415246241, 13.1302841252, -13.8449633448, 13.8764918075)),
    'GW': ('Guinea Bissau', (-16.6774519516, 11.0404116887, -13.7004760401, 12.6281700708)),
    'GQ': ('Eq. Guinea', (9.3056132341, 1.01011953369, 11.285078973, 2.28386607504)),
    'GR': ('Greece', (20.1500159034, 34.9199876979, 26.6041955909, 41.8269046087)),
    'GL': ('Greenland', (-73.297, 60.03676, -12.20855, 83.64513)),
    'GT': ('Guatemala', (-92.2292486234, 13.7353376327, -88.2250227526, 17.8193260767)),
    'GY': ('Guyana', (-61.4103029039, 1.26808828369, -56.5393857489, 8.36703481692)),
    'HN': ('Honduras', (-89.3533259753, 12.9846857772, -83.147219001, 16.0054057886)),
    'HR': ('Croatia', (13.6569755388, 42.47999136, 19.3904757016, 46.5037509222)),
    'HT': ('Haiti', (-74.4580336168, 18.0309927434, -71.6248732164, 19.9156839055)),
    'HU': ('Hungary', (16.2022982113, 45.7594811061, 22.710531447, 48.6238540716)),
    'ID': ('Indonesia', (95.2930261576, -10.3599874813, 141.03385176, 5.47982086834)),
    'IN': ('India', (68.1766451354, 7.96553477623, 97.4025614766, 35.4940095078)),
    'IE': ('Ireland', (-9.97708574059, 51.6693012559, -6.03298539878, 55.1316222195)),
    'IR': ('Iran', (44.1092252948, 25.0782370061, 63.3166317076, 39.7130026312)),
    'IQ': ('Iraq', (38.7923405291, 29.0990251735, 48.5679712258, 37.3852635768)),
    'IS': ('Iceland', (-24.3261840479, 63.4963829617, -13.609732225, 66.5267923041)),
    'IL': ('Israel', (34.2654333839, 29.5013261988, 35.8363969256, 33.2774264593)),
    'IT': ('Italy', (6.7499552751, 36.619987291, 18.4802470232, 47.1153931748)),
    'JM': ('Jamaica', (-78.3377192858, 17.7011162379, -76.1996585761, 18.5242184514)),
    'JO': ('Jordan', (34.9226025734, 29.1974946152, 39.1954683774, 33.3786864284)),
    'JP': ('Japan', (129.408463169, 31.0295791692, 145.543137242, 45.5514834662)),
    'KZ': ('Kazakhstan', (46.4664457538, 40.6623245306, 87.3599703308, 55.3852501491)),
    'KE': ('Kenya', (33.8935689697, -4.67677, 41.8550830926, 5.506)),
    'KG': ('Kyrgyzstan', (69.464886916, 39.2794632025, 80.2599902689, 43.2983393418)),
    'KH': ('Cambodia', (102.3480994, 10.4865436874, 107.614547968, 14.5705838078)),
    'KR': ('S. Korea', (126.117397903, 34.3900458847, 129.468304478, 38.6122429469)),
    'KW': ('Kuwait', (46.5687134133, 28.5260627304, 48.4160941913, 30.0590699326)),
    'LA': ('Laos', (100.115987583, 13.88109101, 107.564525181, 22.4647531194)),
    'LB': ('Lebanon', (35.1260526873, 33.0890400254, 36.6117501157, 34.6449140488)),
    'LR': ('Liberia', (-11.4387794662, 4.35575511313, -7.53971513511, 8.54105520267)),
    'LY': ('Libya', (9.31941084152, 19.58047, 25.16482, 33.1369957545)),
    'LK': ('Sri Lanka', (79.6951668639, 5.96836985923, 81.7879590189, 9.82407766361)),
    'LS': ('Lesotho', (26.9992619158, -30.6451058896, 29.3251664568, -28.6475017229)),
    'LT': ('Lithuania', (21.0558004086, 53.9057022162, 26.5882792498, 56.3725283881)),
    'LU': ('Luxembourg', (5.67405195478, 49.4426671413, 6.24275109216, 50.1280516628)),
    'LV': ('Latvia', (21.0558004086, 55.61510692, 28.1767094256, 57.9701569688)),
    'MA': ('Morocco', (-17.0204284327, 21.4207341578, -1.12455115397, 35.7599881048)),
    'MD': ('Moldova', (26.6193367856, 45.4882831895, 30.0246586443, 48.4671194525)),
    'MG': ('Madagascar', (43.2541870461, -25.6014344215, 50.4765368996, -12.0405567359)),
    'MX': ('Mexico', (-117.12776, 14.5388286402, -86.811982388, 32.72083)),
    'MK': ('Macedonia', (20.46315, 40.8427269557, 22.9523771502, 42.3202595078)),
    'ML': ('Mali', (-12.1707502914, 10.0963607854, 4.27020999514, 24.9745740829)),
    'MM': ('Myanmar', (92.3032344909, 9.93295990645, 101.180005324, 28.335945136)),
    'ME': ('Montenegro', (18.45, 41.87755, 20.3398, 43.52384)),
    'MN': ('Mongolia', (87.7512642761, 41.5974095729, 119.772823928, 52.0473660345)),
    'MZ': ('Mozambique', (30.1794812355, -26.7421916643, 40.7754752948, -10.3170960425)),
    'MR': ('Mauritania', (-17.0634232243, 14.6168342147, -4.92333736817, 27.3957441269)),
    'MW': ('Malawi', (32.6881653175, -16.8012997372, 35.7719047381, -9.23059905359)),
    'MY': ('Malaysia', (100.085756871, 0.773131415201, 119.181903925, 6.92805288332)),
    'NA': ('Namibia', (11.7341988461, -29.045461928, 25.0844433937, -16.9413428687)),
    'NC': ('New Caledonia', (164.029605748, -22.3999760881, 167.120011428, -20.1056458473)),
    'NE': ('Niger', (0.295646396495, 11.6601671412, 15.9032466977, 23.4716684026)),
    'NG': ('Nigeria', (2.69170169436, 4.24059418377, 14.5771777686, 13.8659239771)),
    'NI': ('Nicaragua', (-87.6684934151, 10.7268390975, -83.147219001, 15.0162671981)),
    'NL': ('Netherlands', (3.31497114423, 50.803721015, 7.09205325687, 53.5104033474)),
    'NO': ('Norway', (4.99207807783, 58.0788841824, 31.29341841, 80.6571442736)),
    'NP': ('Nepal', (80.0884245137, 26.3978980576, 88.1748043151, 30.4227169866)),
    'NZ': ('New Zealand', (166.509144322, -46.641235447, 178.517093541, -34.4506617165)),
    'OM': ('Oman', (52.0000098, 16.6510511337, 59.8080603372, 26.3959343531)),
    'PK': ('Pakistan', (60.8742484882, 23.6919650335, 77.8374507995, 37.1330309108)),
    'PA': ('Panama', (-82.9657830472, 7.2205414901, -77.2425664944, 9.61161001224)),
    'PE': ('Peru', (-81.4109425524, -18.3479753557, -68.6650797187, -0.0572054988649)),
    'PH': ('Philippines', (117.17427453, 5.58100332277, 126.537423944, 18.5052273625)),
    'PG': ('Papua New Guinea', (141.000210403, -10.6524760881, 156.019965448, -2.50000212973)),
    'PL': ('Poland', (14.0745211117, 49.0273953314, 24.0299857927, 54.8515359564)),
    'PR': ('Puerto Rico', (-67.2424275377, 17.946553453, -65.5910037909, 18.5206011011)),
    'KP': ('N. Korea', (124.265624628, 37.669070543, 130.780007359, 42.9853868678)),
    'PT': ('Portugal', (-9.52657060387, 36.838268541, -6.3890876937, 42.280468655)),
    'PY': ('Paraguay', (-62.6850571357, -27.5484990374, -54.2929595608, -19.3427466773)),
    'QA': ('Qatar', (50.7439107603, 24.5563308782, 51.6067004738, 26.1145820175)),
    'RO': ('Romania', (20.2201924985, 43.6884447292, 29.62654341, 48.2208812526)),
    'RU': ('Russia', (-180.0, 41.151416124, 180.0, 81.2504)),
    'RW': ('Rwanda', (29.0249263852, -2.91785776125, 30.8161348813, -1.13465911215)),
    'SA': ('Saudi Arabia', (34.6323360532, 16.3478913436, 55.6666593769, 32.161008816)),
    'SD': ('Sudan', (21.93681, 8.61972971293, 38.4100899595, 22.0)),
    'SS': ('S. Sudan', (23.8869795809, 3.50917, 35.2980071182, 12.2480077571)),
    'SN': ('Senegal', (-17.6250426905, 12.332089952, -11.4678991358, 16.5982636581)),
    'SB': ('Solomon Is.', (156.491357864, -10.8263672828, 162.398645868, -6.59933847415)),
    'SL': ('Sierra Leone', (-13.2465502588, 6.78591685631, -10.2300935531, 10.0469839543)),
    'SV': ('El Salvador', (-90.0955545723, 13.1490168319, -87.7235029772, 14.4241327987)),
    'SO': ('Somalia', (40.98105, -1.68325, 51.13387, 12.02464)),
    'RS': ('Serbia', (18.82982, 42.2452243971, 22.9860185076, 46.1717298447)),
    'SR': ('Suriname', (-58.0446943834, 1.81766714112, -53.9580446031, 6.0252914494)),
    'SK': ('Slovakia', (16.8799829444, 47.7584288601, 22.5581376482, 49.5715740017)),
    'SI': ('Slovenia', (13.6981099789, 45.4523163926, 16.5648083839, 46.8523859727)),
    'SE': ('Sweden', (11.0273686052, 55.3617373725, 23.9033785336, 69.1062472602)),
    'SZ': ('Swaziland', (30.6766085141, -27.2858794085, 32.0716654803, -25.660190525)),
    'SY': ('Syria', (35.7007979673, 32.312937527, 42.3495910988, 37.2298725449)),
    'TD': ('Chad', (13.5403935076, 7.42192454674, 23.88689, 23.40972)),
    'TG': ('Togo', (-0.0497847151599, 5.92883738853, 1.86524051271, 11.0186817489)),
    'TH': ('Thailand', (97.3758964376, 5.69138418215, 105.589038527, 20.4178496363)),
    'TJ': ('Tajikistan', (67.4422196796, 36.7381712916, 74.9800024759, 40.9602133245)),
    'TM': ('Turkmenistan', (52.5024597512, 35.2706639674, 66.5461503437, 42.7515510117)),
    'TL': ('East Timor', (124.968682489, -9.39317310958, 127.335928176, -8.27334482181)),
    'TT': ('Trinidad and Tobago', (-61.95, 10.0, -60.895, 10.89)),
    'TN': ('Tunisia', (7.52448164229, 30.3075560572, 11.4887874691, 37.3499944118)),
    'TR': ('Turkey', (26.0433512713, 35.8215347357, 44.7939896991, 42.1414848903)),
    'TW': ('Taiwan', (120.106188593, 21.9705713974, 121.951243931, 25.2954588893)),
    'TZ': ('Tanzania', (29.3399975929, -11.7209380022, 40.31659, -0.95)),
    'UG': ('Uganda', (29.5794661801, -1.44332244223, 35.03599, 4.24988494736)),
    'UA': ('Ukraine', (22.0856083513, 44.3614785833, 40.0807890155, 52.3350745713)),
    'UY': ('Uruguay', (-58.4270741441, -34.9526465797, -53.209588996, -30.1096863746)),
    'US': ('United States', (-171.791110603, 18.91619, -66.96466, 71.3577635769)),
    'UZ': ('Uzbekistan', (55.9289172707, 37.1449940049, 73.055417108, 45.5868043076)),
    'VE': ('Venezuela', (-73.3049515449, 0.724452215982, -59.7582848782, 12.1623070337)),
    'VN': ('Vietnam', (102.170435826, 8.59975962975, 109.33526981, 23.3520633001)),
    'VU': ('Vanuatu', (166.629136998, -16.5978496233, 167.844876744, -14.6264970842)),
    'PS': ('West Bank', (34.9274084816, 31.3534353704, 35.5456653175, 32.5325106878)),
    'YE': ('Yemen', (42.6048726743, 12.5859504257, 53.1085726255, 19.0000033635)),
    'ZA': ('South Africa', (16.3449768409, -34.8191663551, 32.830120477, -22.0913127581)),
    'ZM': ('Zambia', (21.887842645, -17.9612289364, 33.4856876971, -8.23825652429)),
    'ZW': ('Zimbabwe', (25.2642257016, -22.2716118303, 32.8498608742, -15.5077869605)),
}

def GetFoliumMapForCountry( country, zoom_start=5 ):
    a,b,c,d = country_bounding_boxes[country][1]
    start_coords = ((d-b)/2,(c-a)/2)
    folium_map = folium.Map(location=start_coords, zoom_start=zoom_start)
    folium_map.fit_bounds( ((b,a), (d,c)) )
    return folium_map

def FitAround( folium_map, lat, lon, delta_lat=.13, delta_lon=.11 ):
    folium_map.fit_bounds( ( (lat-delta_lat,lon-delta_lon), (lat+delta_lat,lon+delta_lon) ) )
    return folium_map

def FoliumToPng( folium_map, file_name, rendering_seconds=5,crop=(300, 123, 1068, 557) ):
    img_data = folium_map._to_png(rendering_seconds)
    img = Image.open(io.BytesIO(img_data))
    if crop:
        img = img.crop(crop)
    img.save(file_name+'.png')
    
    
def CoveragePerDay( merged, column, household, day ):
    return household[ np.unique( np.concatenate( merged[ merged.Date == day ][column].values ) ).astype(np.uint) ].sum() / household.sum()


def ShowIsoDistancePoints( current_hospitals, population, selected_hosp = 'Rapti Academy of Health Science, Dang', country='NP' ):

    folium_map = GetFoliumMapForCountry(country)

    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]

    folium.Marker([test_ids.iloc[0]['Latitude'], test_ids.iloc[0]['Longitude']],
                            color='blue',popup=test_ids.iloc[0]['L_NAME']).add_to(folium_map)

    test_ids = test_ids['ID_100km'].values[0]
    test_set = population[population['ID'].isin(test_ids)]

    for i in range(0,len(test_set)):
        folium.CircleMarker([test_set.iloc[i]['ycoord'], test_set.iloc[i]['xcoord']],
                            color='orange',fill=True, radius=2).add_to(folium_map)

    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]['ID_50km'].values[0]
    test_set = population[population['ID'].isin(test_ids)]

    for i in range(0,len(test_set)):
        folium.CircleMarker([test_set.iloc[i]['ycoord'], test_set.iloc[i]['xcoord']],
                            color='yellow',fill=True, radius=2).add_to(folium_map)

    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]['ID_10km'].values[0]
    test_set = population[population['ID'].isin(test_ids)]

    for i in range(0,len(test_set)):
        folium.CircleMarker([test_set.iloc[i]['ycoord'], test_set.iloc[i]['xcoord']],
                            color='green',fill=True, radius=2).add_to(folium_map)
        
    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]['ID_5km'].values[0]
    test_set = population[population['ID'].isin(test_ids)]

    for i in range(0,len(test_set)):
        folium.CircleMarker([test_set.iloc[i]['ycoord'], test_set.iloc[i]['xcoord']],
                            color='purple',fill=True, radius=2).add_to(folium_map)
    
    return folium_map

def ShowIsoDistance( current_hospitals, selected_hosp = 'Rapti Academy of Health Science, Dang', country='NP' ):

    folium_map = GetFoliumMapForCountry(country)

    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]


    for i in range(0,len(test_ids)):
        folium.Marker([test_ids.iloc[i]['Latitude'], test_ids.iloc[i]['Longitude']],
                            color='blue',popup=test_ids.iloc[i]['L_NAME']).add_to(folium_map)
        
        geo_j = folium.GeoJson(data=test_ids.iloc[i]['100km'],style_function=lambda x:{'color': 'orange'})
        folium.Popup(test_ids.iloc[i]['L_NAME']).add_to(geo_j)
        geo_j.add_to(folium_map)
        
        geo_j = folium.GeoJson(data=test_ids.iloc[i]['50km'],style_function=lambda x:{'color': 'yellow'})
        folium.Popup(test_ids.iloc[i]['L_NAME']).add_to(geo_j)
        geo_j.add_to(folium_map)
        
        geo_j = folium.GeoJson(data=test_ids.iloc[i]['10km'],style_function=lambda x:{'color': 'green'})
        folium.Popup(test_ids.iloc[i]['L_NAME']).add_to(geo_j)
        geo_j.add_to(folium_map)
        
        geo_j = folium.GeoJson(data=test_ids.iloc[i]['5km'],style_function=lambda x:{'color': 'purple'})
        folium.Popup(test_ids.iloc[i]['L_NAME']).add_to(geo_j)
        geo_j.add_to(folium_map)
        
    return folium_map

def ShowIsoChronesPoints( current_hospitals, population, selected_hosp = 'Rapti Academy of Health Science, Dang', country='NP' ):

    folium_map = GetFoliumMapForCountry(country)
    
    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]

    folium.Marker([test_ids.iloc[0]['Latitude'], test_ids.iloc[0]['Longitude']],
                            color='blue',popup=test_ids.iloc[0]['L_NAME']).add_to(folium_map)

    test_ids = test_ids['ID_60min_driving'].values[0]
    test_set = population[population['ID'].isin(test_ids)]

    for i in range(0,len(test_set)):
        folium.CircleMarker([test_set.iloc[i]['ycoord'], test_set.iloc[i]['xcoord']],
                            color='red',fill=True, radius=2).add_to(folium_map)

    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]['ID_30min_driving'].values[0]
    test_set = population[population['ID'].isin(test_ids)]

    for i in range(0,len(test_set)):
        folium.CircleMarker([test_set.iloc[i]['ycoord'], test_set.iloc[i]['xcoord']],
                            color='cyan',fill=True, radius=2).add_to(folium_map)

    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]['ID_60min_walking'].values[0]
    test_set = population[population['ID'].isin(test_ids)]

    for i in range(0,len(test_set)):
        folium.CircleMarker([test_set.iloc[i]['ycoord'], test_set.iloc[i]['xcoord']],
                            color='blue',fill=True, radius=2).add_to(folium_map)
        
    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]['ID_30min_walking'].values[0]
    test_set = population[population['ID'].isin(test_ids)]

    for i in range(0,len(test_set)):
        folium.CircleMarker([test_set.iloc[i]['ycoord'], test_set.iloc[i]['xcoord']],
                            color='green',fill=True, radius=2).add_to(folium_map)
        
    return folium_map


def ShowIsoChrones( current_hospitals, selected_hosp = 'Rapti Academy of Health Science, Dang', country='NP' ):

    folium_map = GetFoliumMapForCountry(country)

    test_ids = current_hospitals[current_hospitals['L_NAME']==selected_hosp]


    for i in range(0,len(test_ids)):
        folium.Marker([test_ids.iloc[i]['Latitude'], test_ids.iloc[i]['Longitude']],
                            color='blue',popup=test_ids.iloc[i]['L_NAME']).add_to(folium_map)
        
        geo_j = folium.GeoJson(data=test_ids.iloc[i]['60min_driving'],style_function=lambda x:{'color': 'red'})
        folium.Popup(test_ids.iloc[i]['L_NAME']).add_to(geo_j)
        geo_j.add_to(folium_map)
        
        geo_j = folium.GeoJson(data=test_ids.iloc[i]['30min_driving'],style_function=lambda x:{'color': 'cyan'})
        folium.Popup(test_ids.iloc[i]['L_NAME']).add_to(geo_j)
        geo_j.add_to(folium_map)
        
        geo_j = folium.GeoJson(data=test_ids.iloc[i]['60min_walking'],style_function=lambda x:{'color': 'blue'})
        folium.Popup(test_ids.iloc[i]['L_NAME']).add_to(geo_j)
        geo_j.add_to(folium_map)
        
        geo_j = folium.GeoJson(data=test_ids.iloc[i]['30min_walking'],style_function=lambda x:{'color': 'green'})
        folium.Popup(test_ids.iloc[i]['L_NAME']).add_to(geo_j)
        geo_j.add_to(folium_map)
        
    return folium_map

# https://stackoverflow.com/questions/53721079/python-folium-icon-list
# https://fontawesome.com/v4/icons/
def ShowPoints( locations, choices, country='NP' ):
    folium_map = GetFoliumMapForCountry(country)
    
    for color,selected_locs in choices.items():
        for i, lat, lon, name in locations.loc[selected_locs].values:
            folium.Marker((lat,lon),icon=folium.plugins.BeautifyIcon(icon_shape='marker',background_color=color,border_width=1,number=i),popup=name).add_to(folium_map)
            
    return folium_map

# https://deparkes.co.uk/2016/06/10/folium-map-tiles/
def ShowAccessibility( current, potential, selected_locs, pop,
                      accessibility, mode, 
                      color_new ='cyan', color_current = 'green',
                      radius_no_access = 1, radius_access = 1,
                      color_no_access = 'red', color_access = 'green',
                      min_opacity_no_access = .1, min_opacity_access = .1, delta_lat = 0, delta_lon = 0,
                      country='NP', tiles='cartodbpositron' ):
    
    a,b,c,d = country_bounding_boxes[country][1]
    start_coords = ((d-b)/2,(c-a)/2)
    folium_map = folium.Map(location=start_coords,tiles=tiles)
    
    column = 'ID_'+mode
    
    new_labs = potential[accessibility][potential[accessibility]['Cluster_ID'].isin(selected_locs)]
    
    if color_current:
        for i,lat,lon,name in current[accessibility][['Hosp_ID','Latitude','Longitude','L_NAME']].values:
            folium.Marker((lat,lon),
                        icon=folium.plugins.BeautifyIcon(icon_shape='marker',background_color=color_current,
                                                        border_width=1,number=i),popup=name).add_to(folium_map)
    if color_new:
        for i,lat,lon,name in new_labs[['Cluster_ID','Latitude','Longitude','Name']].values:
            folium.Marker((lat,lon),
                        icon=folium.plugins.BeautifyIcon(icon_shape='marker',background_color=color_new,
                                                        border_width=1,number=i),popup=name).add_to(folium_map)
    
    if new_labs.empty:
        tot_access = np.unique( np.concatenate( current[accessibility][column].values ) ).astype(np.uint) 
    else:
        covered_current = np.unique( np.concatenate( current[accessibility][column].values ) ).astype(np.uint) 
        covered_new = np.unique( np.concatenate( new_labs[column].values ) ).astype(np.uint) 
        tot_access = np.unique( np.concatenate( (covered_current, covered_new ) ) )
    
    real_pop = pop[pop.population > 0][['ID','ycoord','xcoord','population']]
    pop_with_access = real_pop['ID'].isin(set(tot_access))

    max_pop = real_pop[pop_with_access].population.max()
    for _,lat,lon,population in real_pop[pop_with_access].values:
        weight = (1-min_opacity_access)*( population / max_pop ) + min_opacity_access
        folium.Circle( (lat,lon), color=color_access,radius=radius_access,fill_opacity=weight,opacity=weight).add_to(folium_map)
        
    max_pop = real_pop[~pop_with_access].population.max()
    for _,lat,lon,population in real_pop[~pop_with_access].values:
        weight = (1-min_opacity_no_access)*( population / max_pop ) + min_opacity_no_access
        folium.Circle( (lat,lon), color=color_no_access,radius=radius_no_access,fill_opacity=weight,opacity=weight).add_to(folium_map)
        
    folium_map.fit_bounds( ((b-delta_lat,a-delta_lon), (d+delta_lat,c+delta_lon)) )

    return folium_map
    
    
def ShowRWIxAccess( accessibility_frame, 
                   width=800, height=500, 
                   font_family='lmodern',font_size=13,
                   title='',
                   file_name=None ):
    fig = px.scatter(accessibility_frame,x='Median RWI',y='%',
                 hover_name='District',size='Total_Pop',color='Province',
                 width=width,height=height,title=title)
    fig.update_xaxes(title='Median Relative Wealth Index')
    fig.update_yaxes(title='% of population with access')
    fig.update_layout(plot_bgcolor='white',
                      font_family=font_family,
                      font_size=font_size)
    if file_name:
        fig.write_image(file_name, engine='orca')
    return fig

def get_pop_with_district():
    pop = pickle.load(open('../Results/population.pkl', 'rb'))
    pop = gpd.GeoDataFrame(pop)
    pop = pop.set_crs('EPSG:4326')
    shapefile = pickle.load(open('../Data/shapefile_pickle.pkl','rb'))
    return gpd.sjoin(pop,shapefile)    
    
def get_rwi_district():
    rwi = pd.read_csv('../Data/npl_relative_wealth_index.csv')
    rwi = gpd.GeoDataFrame(rwi, geometry=gpd.points_from_xy(rwi.longitude, rwi.latitude))
    rwi = rwi.set_crs('EPSG:4326')
    shapefile = pickle.load(open('../Data/shapefile_pickle.pkl','rb'))
    rwi_with_district = gpd.sjoin(rwi,shapefile)
    rwi_district = rwi_with_district.groupby(['DISTRICT','Province'])['rwi'].median().reset_index()
    rwi_district.columns = ['District','Province','Median RWI']    
    return rwi_district

def get_pop_rwi():
    return pickle.load(open('../Data/pop_rwi.pkl','rb')).corrected_weights.values.astype(np.uint)

def get_headcount(data_path):
    return pickle.load(open(data_path+'/population.pkl', 'rb')).population.values.astype(np.uint)
    
def GetAccessibilityFromOptimization( accessibility, mode, selected_locs, current, potential, pop_with_district, rwi_district ):
    current_plot = current[accessibility]
    new_plot = potential[accessibility]
    list_current_hh = list(current_plot['ID_'+str(mode)].values)
    list_new_hh = list(new_plot['ID_'+str(mode)].values)
    list_current_hh = list(set([e for v in list_current_hh for e in v]))
    list_new_hh = list(set([e for v in list_new_hh for e in v]))
    
    new_labs = new_plot[new_plot['Cluster_ID'].isin(selected_locs)]
    
    list_hhs_new = list(new_labs[new_labs['Cluster_ID'].isin(selected_locs)]['ID_'+mode].values)
    array_hhs_new = []

    for each_val in list_hhs_new:
        for each_hh in each_val:
            array_hhs_new.append(each_hh)
    array_hhs_new = list(set(array_hhs_new))

    list_hhs_current = list(current[accessibility]['ID_'+mode].values)
    array_hhs_current = []

    for each_val in list_hhs_current:
        for each_hh in each_val:
            array_hhs_current.append(each_hh)
    array_hhs_current = list(set(array_hhs_current))
    
    tot_access_list = list(array_hhs_new+array_hhs_current)

    pop_current_access = pop_with_district[pop_with_district['ID'].isin(set(tot_access_list))]
    access_pop = pop_current_access.groupby(['DISTRICT','Province'])['population'].sum().reset_index()
    access_pop.columns = ['District','Province','People_Access']
    total_pop = pop_with_district.groupby(['DISTRICT','Province'])['population'].sum().reset_index()
    total_pop.columns = ['District','Province','Total_Pop']

    current_accessibility = pd.merge(access_pop,total_pop,on=['District','Province'])
    current_accessibility['%'] = round(current_accessibility['People_Access']*100/current_accessibility['Total_Pop'])
    return pd.merge(current_accessibility,rwi_district,on=['District','Province']).sort_values(by='Province')

    
def GenerateRWIPlots( results_path, scenario, result, current, potential, pop_with_district, rwi_district, width=600, height=350, file_type='pdf' ):
    pictures_path = results_path+f'Pics/RWI/{scenario}/'
    accessibilities = result.keys()
    for a in accessibilities:
        Path(pictures_path).mkdir(parents=True, exist_ok=True)
        for c in result[a][1].columns:
            readable = c.replace('min',' min').replace('km',' km').replace('_',' ')
            accessibility_frame = GetAccessibilityFromOptimization( a, c, [], current, potential, pop_with_district, rwi_district )
            ShowRWIxAccess( accessibility_frame, width=width, height=height, title=readable, file_name=pictures_path+f'{c}_0.{file_type}' )
            for budget in result[a][1].index:
                sol = result[a][1][c][budget]
                accessibility_frame = GetAccessibilityFromOptimization( a, c, sol, current, potential, pop_with_district, rwi_district )
                ShowRWIxAccess( accessibility_frame, width=width, height=height, title=readable + f' with {budget} additional labs', file_name=pictures_path+f'{c}_{budget}.{file_type}' )
                
def GetAdministrativeNepal():
    shapefile = pickle.load(open('../Data/shapefile_pickle.pkl','rb'))
    shapefile['representative'] = [ x.representative_point().coords[:][0] for x in shapefile.geometry.values ]
    shapefile['District'] = [ name.replace('_',' ').title() for name in shapefile.DISTRICT.values ]
    districts_and_provinces = shapefile[['District','Province']]
    data1 = districts_and_provinces[:int(np.ceil(len(districts_and_provinces)/2))].reset_index()
    data2 = districts_and_provinces[int(np.ceil(len(districts_and_provinces)/2)):].reset_index()
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    color_provinces = { '1' : 'azure',
                    '2' : 'burlywood',
                    '5' : 'wheat',
                    'Bagmati' : 'gold',
                    'Gandaki' : 'lemonchiffon',
                    'Karnali' : 'yellowgreen',
                    'Sudur Pashchim' : 'lightsalmon' }
    shapefile['color'] = [ color_provinces[p] for p in shapefile.Province.values ] 
    return shapefile, data1, data2

def DrawAdministrative( shapefile, file_name=None ):
    _,ax = plt.subplots()
    ax.axis('off')
    ax.set_rasterized(True)
    shapefile.plot(ax=ax,color=shapefile.color.values,ec='black')
    for i, (_, coords) in enumerate(shapefile[['DISTRICT','representative']].values):
        plt.annotate(text=str(i), xy=coords, fontsize=3, color='blue',
                    horizontalalignment='center',verticalalignment='center')
    ax.margins(0)
    ax.margins(0)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0,dpi=2400)
        
def GetAccessibilityData(data_path):
    accessibilities = [ 'Time', 'Distance' ]
    current         = dict()
    potential       = dict()
    for a in accessibilities:
        current[a]   = pickle.load(open(data_path+'Travel {}/current_hospitals.pkl'.format(a), 'rb'))
        potential[a] = pickle.load(open(data_path+'Travel {}/new_hospitals.pkl'.format(a), 'rb'))
    return accessibilities, current, potential

def SplitPotentialSites(potential):
    potential_with_names = potential['Time'][['Cluster_ID','Name']].copy()
    potential_with_names['Name'] = [lab.replace('&','\&') if lab.isascii() else '' for lab in potential_with_names.Name.values]
    potential_with_names = potential_with_names[potential_with_names.Name != ''][['Cluster_ID','Name']]
    data1 = potential_with_names[:int(np.ceil(len(potential_with_names)/2))].reset_index(drop=True)
    data2 = potential_with_names[int(np.ceil(len(potential_with_names)/2)):].reset_index(drop=True)
    return data1, data2

def draw_lines( df, x=None, y=None, bgcolor='white', 
               x_title=None, y_title=None, 
               legend_title=None, font_family='lmodern',
               font_size=13,
               width=1000,height=500,line_width=3,
               file_name=None):
    fig = px.line(df,x=x,y=y,width=width,height=height)
    fig.update_layout(plot_bgcolor=bgcolor,
                      legend_title=legend_title,
                      font_family=font_family,
                      legend_font_family = font_family+',monospace',
                      font_size = font_size
                      )
    fig.update_xaxes(title=x_title,showgrid=False)
    fig.update_yaxes(title=y_title,showgrid=False)
    fig.update_traces(line=dict(width=line_width))
    if file_name:
        fig.write_image(file_name, engine='orca')
    return fig

def show_pareto( result, current, household, accessibility,
                bgcolor='white', 
                legend_title=None, font_family='lmodern',
                font_size=13,
                width=800,height=500,line_width=3,
                file_name=None):
    df_plot = (round(result[accessibility][0]*10000)/100).reset_index()
    
    population = household.sum()
    initial_values = [0]
    for c in result[accessibility][0].columns:
        covered = np.unique( np.concatenate( current[accessibility]['ID_'+c] ) ).astype(np.uint) 
        initial_values.append( round(household[covered].sum()/population*10000)/100 )
    df2 = pd.DataFrame([initial_values])
    df2.columns = df_plot.columns
    df_plot = pd.concat([df2, df_plot])
    df_plot = df_plot.rename(columns={'index':'new_labs'})
    df_plot.set_index('new_labs',inplace=True)
    
    def format_name( name ):
        d,_,what = name.replace('km', ' km').replace( 'min', ' min' ).partition(' ')
        return '{:4d} {}'.format(int(d),what.replace('_',' '))

    def range( df, column ):
        return( '{:4.1f} to {:4.1f}'.format( df[column].min(), df[column].max() ) )

    df_plot.rename(columns = { c : range( df_plot, c ) + format_name(c) for c in df_plot.columns }, inplace = True)
    
    df_plot = df_plot.stack().reset_index()
    df_plot.columns = ['new_labs','mode','%']
    df_plot['total_labs']= df_plot['new_labs']+current[accessibility]['Hosp_ID'].nunique()
    
    fig = px.line(df_plot,x='total_labs',y='%',color='mode',width=width,height=height)
    fig.update_xaxes(title='Total number of labs')
    fig.update_yaxes(title='% of population with <br> access to a testing laboratory')
    fig.update_layout(plot_bgcolor=bgcolor,
                      legend_title=legend_title,
                      font_family=font_family,
                      legend_font_family = font_family+',monospace',
                      font_size = font_size
                      )
    
    fig.update_traces(line=dict(width=line_width))
    
    top = 1.1*df_plot['%'].max()
    left = current[accessibility]['Hosp_ID'].nunique()

    fig.add_shape(type="line",
            x0=left, 
            y0=0, 
            x1=left, 
            y1=top,
            line=dict(color="grey",width=2)
        )

    fig.add_annotation(x=1.8*left, y=top,
                    text="Current laboratories:"+str(current[accessibility]['Hosp_ID'].nunique()),
                    showarrow=False,
                    align="right",
                    arrowhead=1)
    
    if file_name:
        fig.write_image(file_name, engine='orca')
        
    return fig