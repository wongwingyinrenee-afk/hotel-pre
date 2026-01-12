import pandas as pd
from datetime import timedelta
from sklearn.impute import KNNImputer
import holidays
import datetime


df = pd.read_csv('/Users/noonele/Desktop/hotel_bookings.csv')

print(df.info())



knn_features = df[['adults', 'babies', 'children', 'stays_in_weekend_nights', 'stays_in_week_nights']]
imputed = KNNImputer(n_neighbors=10).fit_transform(knn_features)
df['children'] = imputed[:, 2]
1
df['arrival_date_month'] = df['arrival_date_month'].str.capitalize()
df['arrival_date_month'] = pd.to_datetime(df['arrival_date_month'], format='%B').dt.month
df['arrival_date'] = pd.to_datetime({ 'year': df['arrival_date_year'], 'month': df['arrival_date_month'], 'day': df['arrival_date_day_of_month'] })

us_holidays = holidays.US()
def count_events(date):
    count = 0

    if date in us_holidays:
        count += 1

    for i in range(1, 8):
        if (date - timedelta(days=i)) in us_holidays:
            count += 1
        if (date + timedelta(days=i)) in us_holidays:
            count += 1

    return count
df['event_count'] = df['arrival_date'].apply(count_events)

df['country'] = df['country'].fillna(df['country'].mode()[0])

country_to_region = {
    'PRT': 'Europe', 'GBR': 'Europe', 'USA': 'North America', 'ESP': 'Europe', 'IRL': 'Europe', 'FRA': 'Europe',
    'ROU': 'Europe', 'NOR': 'Europe', 'OMN': 'Asia', 'ARG': 'South America', 'POL': 'Europe', 'DEU': 'Europe',
    'BEL': 'Europe', 'CHE': 'Europe', 'CN': 'Asia', 'GRC': 'Europe', 'ITA': 'Europe', 'NLD': 'Europe', 'DNK': 'Europe',
    'RUS': 'Europe', 'SWE': 'Europe', 'AUS': 'Oceania', 'EST': 'Europe', 'CZE': 'Europe', 'BRA': 'South America',
    'FIN': 'Europe', 'MOZ': 'Africa', 'BWA': 'Africa', 'LUX': 'Europe', 'SVN': 'Europe', 'ALB': 'Europe', 'IND': 'Asia',
    'CHN': 'Asia', 'MEX': 'North America', 'MAR': 'Africa', 'UKR': 'Europe', 'SMR': 'Europe', 'LVA': 'Europe',
    'PRI': 'North America', 'SRB': 'Europe', 'CHL': 'South America', 'AUT': 'Europe', 'BLR': 'Europe', 'LTU': 'Europe',
    'TUR': 'Asia', 'ZAF': 'Africa', 'AGO': 'Africa', 'ISR': 'Asia', 'CYM': 'North America', 'ZMB': 'Africa',
    'CPV': 'Africa', 'ZWE': 'Africa', 'DZA': 'Africa', 'KOR': 'Asia', 'CRI': 'North America', 'HUN': 'Europe',
    'ARE': 'Asia', 'TUN': 'Africa', 'JAM': 'North America', 'HRV': 'Europe', 'HKG': 'Asia', 'IRN': 'Asia',
    'GEO': 'Asia', 'AND': 'Europe', 'GIB': 'Europe', 'URY': 'South America', 'JEY': 'Europe', 'CAF': 'Africa',
    'CYP': 'Asia', 'COL': 'South America', 'GGY': 'Europe', 'KWT': 'Asia', 'NGA': 'Africa', 'MDV': 'Asia',
    'VEN': 'South America', 'SVK': 'Europe', 'FJI': 'Oceania', 'KAZ': 'Asia', 'PAK': 'Asia', 'IDN': 'Asia',
    'LBN': 'Asia', 'PHL': 'Asia', 'SEN': 'Africa', 'SYC': 'Africa', 'AZE': 'Asia', 'BHR': 'Asia', 'NZL': 'Oceania',
    'THA': 'Asia', 'DOM': 'North America', 'MKD': 'Europe', 'MYS': 'Asia', 'ARM': 'Asia', 'JPN': 'Asia', 'LKA': 'Asia',
    'CUB': 'North America', 'CMR': 'Africa', 'BIH': 'Europe', 'MUS': 'Africa', 'COM': 'Africa', 'SUR': 'South America',
    'UGA': 'Africa', 'BGR': 'Europe', 'CIV': 'Africa', 'JOR': 'Asia', 'SYR': 'Asia', 'SGP': 'Asia', 'BDI': 'Africa',
    'SAU': 'Asia', 'VNM': 'Asia', 'PLW': 'Oceania', 'QAT': 'Asia', 'EGY': 'Africa', 'PER': 'South America', 'MLT': 'Europe',
    'MWI': 'Africa', 'ECU': 'South America', 'MDG': 'Africa', 'ISL': 'Europe', 'UZB': 'Asia', 'NPL': 'Asia', 'BHS': 'North America',
    'MAC': 'Asia', 'TGO': 'Africa', 'TWN': 'Asia', 'DJI': 'Africa', 'STP': 'Africa', 'KNA': 'North America', 'ETH': 'Africa',
    'IRQ': 'Asia', 'HND': 'North America', 'RWA': 'Africa', 'KHM': 'Asia', 'MCO': 'Europe', 'BGD': 'Asia', 'IMN': 'Europe',
    'TJK': 'Asia', 'NIC': 'North America', 'BEN': 'Africa', 'VGB': 'North America', 'TZA': 'Africa', 'GAB': 'Africa', 'GHA': 'Africa',
    'TMP': 'Asia', 'GLP': 'North America', 'KEN': 'Africa', 'LIE': 'Europe', 'GNB': 'Africa', 'MNE': 'Europe', 'UMI': 'Oceania',
    'MYT': 'Africa', 'FRO': 'Europe', 'MMR': 'Asia', 'PAN': 'North America', 'BFA': 'Africa', 'LBY': 'Africa', 'MLI': 'Africa',
    'NAM': 'Africa', 'BOL': 'South America', 'PRY': 'South America', 'BRB': 'North America', 'ABW': 'North America', 'AIA': 'North America',
    'SLV': 'North America', 'DMA': 'North America', 'PYF': 'Oceania', 'GUY': 'South America', 'LCA': 'North America', 'ATA': 'Antarctica',
    'GTM': 'North America', 'ASM': 'Oceania', 'MRT': 'Africa', 'NCL': 'Oceania', 'KIR': 'Oceania', 'SDN': 'Africa', 'ATF': 'Antarctica',
    'SLE': 'Africa', 'LAO': 'Asia'
}

df['region'] = df['country'].map(country_to_region)


def check_festival(row):
    continent = row['region']
    arrival_date = row['arrival_date']
    festival_count = 0
    festivals = {
        'Asia': [
            {'name': 'Chinese New Year', 'year': 2016, 'month': 2, 'day': 8},
            {'name': 'Chinese New Year', 'year': 2017, 'month': 1, 'day': 28},
            {'name': 'Diwali', 'year': 2015, 'month': 11, 'day': 11},
            {'name': 'Diwali', 'year': 2016, 'month': 10, 'day': 30},
            {'name': 'Diwali', 'year': 2017, 'month': 10, 'day': 19},
            {'name': 'New Year', 'month': 1, 'day': 1},
        ],
        'Europe': [
            {'name': 'Christmas', 'month': 12, 'day': 25},
            {'name': 'New Year', 'month': 1, 'day': 1},
            {'name': 'Easter Monday', 'year': 2015, 'month': 4, 'day': 6},
            {'name': 'Easter Monday', 'year': 2016, 'month': 3, 'day': 28},
            {'name': 'Easter Monday', 'year': 2017, 'month': 4, 'day': 17}
        ],
        'North America': [
            {'name': 'Thanksgiving', 'year': 2015, 'month': 11, 'day': 26},
            {'name': 'Thanksgiving', 'year': 2016, 'month': 11, 'day': 24},
            {'name': 'Thanksgiving', 'year': 2017, 'month': 11, 'day': 23},
            {'name': 'Christmas', 'month': 12, 'day': 25},
            {'name': 'New Year', 'month': 1, 'day': 1},
            {'name': 'Independence Day', 'month': 7, 'day': 4}
        ],
        'South America': [
            {'name': 'Carnival', 'year': 2016, 'month': 2, 'day': 8},
            {'name': 'Carnival', 'year': 2017, 'month': 2, 'day': 24},
            {'name': 'Carnival', 'year': 2015, 'month': 2, 'day': 16},
        ],
        'Oceania': [
            {'name': 'Australia Day', 'month': 1, 'day': 26},
            {'name': 'New Year', 'month': 1, 'day': 1},
            {'name': 'ANZAC Day', 'month': 4, 'day': 25}
        ],
        'Antarctica': [
            {'name': 'New Year', 'month': 1, 'day': 1},
        ]
    }
    if continent in festivals:
        for festival in festivals[continent]:
            if 'year' in festival:
                if festival['year'] == arrival_date.year:
                    festival_date = datetime.datetime(arrival_date.year, festival['month'], festival['day'])
                    if (festival_date - datetime.timedelta(days=7)) <= arrival_date.to_pydatetime() <= (
                            festival_date + datetime.timedelta(days=7)):
                        festival_count += 1
            else:
                for year in [arrival_date.year - 1, arrival_date.year, arrival_date.year + 1]:
                    try:
                        festival_date = datetime.datetime(year, festival['month'], festival['day'])
                        if (festival_date - datetime.timedelta(days=7)) <= arrival_date.to_pydatetime() <= (
                                festival_date + datetime.timedelta(days=7)):
                            festival_count += 1
                    except ValueError:
                        # Handle cases where the day doesn't exist in the given month (e.g., Feb 30)
                        pass

    return festival_count


df['festival_count'] = df.apply(check_festival, axis=1)

df = pd.get_dummies(df, columns=['meal', 'market_segment', 'hotel','distribution_channel','deposit_type','customer_type'], drop_first=True)
df = pd.get_dummies(df, columns=['region'], drop_first=True)
df = df.drop('country', axis=1)
df = df.drop('company', axis=1)
df= df.drop(['reservation_status','reservation_status_date'], axis=1)

def room_status(row):
    if row['reserved_room_type'] == row['assigned_room_type']:
        return 0
    elif row['assigned_room_type'] > row['reserved_room_type']:
        return 1
    else:
        return -1

df['room_type_change'] = df.apply(room_status, axis=1)

room_type_map = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
    'L': 9,
    'P': 10
}
df['reserved_room_type'] = df['reserved_room_type'].map(room_type_map)
df = df.drop('assigned_room_type', axis=1)

df['arrival_quarter'] = df['arrival_date'].dt.quarter
df['arrival_is_high_season'] = df['arrival_date'].dt.month.isin([6,7,8]).astype(int)
df = df.drop('arrival_date', axis=1)

df['agent'] = df['agent'].fillna(0).astype(int)
df = df.drop(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month','arrival_date_week_number'], axis=1)



print(df.info())
or_df = df.copy()
or_df.to_csv('hotel_or_modeling_data.csv', index=False)
print(f" Saved {or_df.shape[0]} rows and {or_df.shape[1]} columns to 'hotel_or_modeling_data.csv'")


#divided
city_df =  df.copy()
city_df = df[df['hotel_Resort Hotel'] == 0]
city_df.to_csv('city_df.csv', index=False)
print(f" Saved {city_df.shape[0]} rows and {city_df.shape[1]} columns to 'city_df.csv'")

resort_df = df.copy()
resort_df = df[df['hotel_Resort Hotel'] == 1]
resort_df.to_csv('resort_df.csv', index=False)
print(f" Saved {resort_df.shape[0]} rows and {resort_df.shape[1]} columns to 'resort_df.csv'")


