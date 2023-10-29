#!/usr/bin/env python
# coding: utf-8

# # Note: This notebook is used in the following four videos:
# 
# [**Work with missing data in a Python notebook**](https://www.coursera.org/teach/go-beyond-the-numbers-translate-data-into-insight/mjPtRyfIEe2GwRLfSI7mvQ/content/item/lecture/rUXcJ/video-subtitles) 🠚 [jump to notebook section](#work_with_missing_data)
# 
# [**Identify and deal with outliers in Python**](https://www.coursera.org/teach/go-beyond-the-numbers-translate-data-into-insight/mjPtRyfIEe2GwRLfSI7mvQ/content/item/lecture/jadID/video-subtitles) 🠚 [jump to notebook section](#outliers)  
# 
# [**Label encoding in Python**](https://www.coursera.org/teach/go-beyond-the-numbers-translate-data-into-insight/mjPtRyfIEe2GwRLfSI7mvQ/content/item/lecture/fLMxl/video-subtitles) 🠚 [jump to notebook section](#encoding) 
# 
# [**Input validation with Python**](https://www.coursera.org/teach/go-beyond-the-numbers-translate-data-into-insight/mjPtRyfIEe2GwRLfSI7mvQ/content/item/lecture/C6Mok/video-subtitles) 🠚 [jump to notebook section](#input_validation)

# <a id='work_with_missing_data'></a>
# # Work with missing data in a Python notebook

# Throughout the following exercises, you will be discovering and working with missing data on a dataset.  Before starting on this programming exercise, we strongly recommend watching the video lecture and completing the IVQ for the associated topics.

# All the information you need for solving this assignment is in this notebook, and all the code you will be implementing will take place within this notebook. 

# As we move forward, you can find instructions on how to install required libraries as they arise in this notebook. Before we begin with the exercises and analyzing the data, we need to import all libraries and extensions required for this programming exercise. Throughout the course, we will be using pandas, numpy, datetime, for operations, and matplotlib, pyplot and seaborn for plotting.

# ## Objective
# 
# We will be examining lightning strike data collected by the National Oceanic and Atmospheric Association (NOAA) for the month of August 2018. There are two datasets. The first includes five columns:  
# 
# |date|center_point_geom|longitude|latitude|number_of_strikes|
# |---|---|---|---|---|
# 
# The second dataset contains seven columns:
# 
# |date|zip_code|city|state|state_code|center_point_geom|number_of_strikes|
# |---|---|---|---|---|---|---|  
# 
# The first dataset has two unique colums: `longitude` and `latitude`.  
# The second dataset has four unique columns: `zip_code`, `city`, `state`, and `state_code`.  
# There are three columns that are common between them: `date`, `center_point_geom`, and `number_of_strikes`.
# 
# We want to combine the two datasets into a single dataframe that has all of the information from both datasets. Ideally, both datasets will have the same number of entries for the same locations on the same dates. If they don't, we'll investigate which data is missing.

# In[1]:


# Import statements
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from matplotlib import pyplot as plt


# In[2]:


# Read in first dataset
df = pd.read_csv('eda_missing_data_dataset1.csv')


# In[3]:


# Print the first 5 rows of dataset 1
df.head()


# Let's check on our dataset shape to determine number of columns and rows. 

# In[4]:


df.shape


# Now we'll read in the second dataset.

# In[5]:


# Read in second dataset
df_zip = pd.read_csv('eda_missing_data_dataset2.csv')


# In[6]:


# Print the first 5 rows of dataset 2
df_zip.head()


# And check the shape...

# In[7]:


df_zip.shape


# Hmmm... This dataset has less than half the number of rows as the first one. But which ones are they?  
# 
# The first thing we'll do to explore this discrepancy is join the two datasets into a single dataframe. We can do this using the `merge()` method of the `DataFrame` class. For more information about the `merge()` method, refer to the [merge() pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html).  
# 
# Begin with the first dataframe (`df`) and call the `merge()` method on it. The first argument is a positional argument that specifies the dataframe we want to merge with, known as the `right` dataframe. (The dataframe you're calling the method on is always the `left` dataframe.) The `how` argument specifies which dataframe's keys we'll use to match to, and the `on` argument lets us define the columns to use as keys. 
# 
# A demonstration will make this easiest to understand. Refer to the **[BONUS CONTENT](#merge_bonus)** at the end of the notebook for different examples of the `merge()` method.
#   

# In[8]:


# Left-join the two datasets
df_joined = df.merge(df_zip, how='left', on=['date','center_point_geom'])


# In[9]:


# Print the first 5 rows of the merged data
df_joined.head()


# Notice that the new dataframe has all of the columns of both original dataframes, and it has two `number_of_strikes` columns that are suffixed with `_x` and `_y`. This is because the key columns from both dataframes were the same, so they appear once in the merged dataframe. The unique columns of each original dataframe also appear in the merged dataframe. But both original dataframes had another column&mdash;`number_of_strikes`&mdash;that had the same name in both dataframes and was not indicated as a key. Pandas handles this by adding both columns to the new dataframe. 
# 
# Now we'll check the summary on this joined dataset.  

# In[10]:


# Get descriptive statistics of the joined dataframe
df_joined.describe()


# The count information confirms that the new dataframe is missing some data. 

# Now let's check how many missing state locations we have by using `isnull()` to create a Boolean mask that we'll apply to `df_joined`. The mask is a pandas `Series` object that contains `True` for every row with a missing `state_code` value and `False` for every row that is not missing data in this column. When the mask is applied to `df_joined`, it filters out the rows that are not missing `state_code` data. (Note that using the `state_code` column to create this mask is an arbitrary decision. We could have selected `zip_code`, `city`, or `state` instead and gotten the same results.)

# In[11]:


# Create a new df of just the rows that are missing data
df_null_geo = df_joined[pd.isnull(df_joined.state_code)]
df_null_geo.shape


# We can confirm that `df_null_geo` contains only the rows with the missing `state_code` values by using the `info()` method on `df_joined` and comparing. 

# In[12]:


# Get non-null counts on merged dataframe
df_joined.info()


# If we subtract the 323,700 non-null rows in columns 5-9 of `df_joined` from the 717,530 non-null rows in columns 0-4 of `df_joined`, we're left with 393,830 rows that contain missing data&mdash;the same number of rows contained in `df_null_geo`.

# In[13]:


# Print the first 5 rows
df_null_geo.head()


# Now that we've merged all of our data together and isolated the rows with missing data, we can better understand what data is missing by plotting the longitude and latitude of locations that are missing city, state, and zip code data.

# In[14]:


# Create new df of just latitude, longitude, and number of strikes and group by latitude and longitude
top_missing = df_null_geo[['latitude','longitude','number_of_strikes_x']
            ].groupby(['latitude','longitude']
                      ).sum().sort_values('number_of_strikes_x',ascending=False).reset_index()
top_missing.head(10)


# Let's import plotly to reduce the size of the data frame as we create a geographic scatter plot. 

# In[15]:


import plotly.express as px  # Be sure to import express
# reduce size of db otherwise it could break
fig = px.scatter_geo(top_missing[top_missing.number_of_strikes_x>=300],  # Input Pandas DataFrame
                    lat="latitude",  # DataFrame column with latitude
                    lon="longitude",  # DataFrame column with latitude
                    size="number_of_strikes_x") # Set to plot size as number of strikes
fig.update_layout(
    title_text = 'Missing data', # Create a Title
)

fig.show()


# It’s a nice geographic visualization, but we really don’t need the global scale. Let’s scale it down to only the geographic area we are interested in - the United States.

# In[16]:


import plotly.express as px  # Be sure to import express
fig = px.scatter_geo(top_missing[top_missing.number_of_strikes_x>=300],  # Input Pandas DataFrame
                    lat="latitude",  # DataFrame column with latitude
                    lon="longitude",  # DataFrame column with latitude
                    size="number_of_strikes_x") # Set to plot size as number of strikes
fig.update_layout(
    title_text = 'Missing data', # Create a Title
    geo_scope='usa',  # Plot only the USA instead of globe
)

fig.show()


# This explains why so many rows were missing state and zip code data! Most of these lightning strikes occurred over water&mdash;the Atlantic Ocean, the Sea of Cortez, the Gulf of Mexico, the Caribbean Sea, and the Great Lakes. Of the strikes that occurred over land, most of those were in Mexico, the Bahamas, and Cuba&mdash;places outside of the U.S. and without U.S. zip codes. Nonetheless, some of the missing data is from Florida and elsewhere within the United States, and we might want to ask the database owner about this.

# If you have successfully completed the material above, congratulations! You now understand handling missing data in Python and should be able to start using it on your own datasets. 

# <a id='merge_bonus'></a>
# ### Bonus (not in video): `df.merge()` demonstration:

# Begin with two dataframes:

# In[17]:


# Define df1
data = {'planet': ['Mercury', 'Venus', 'Earth', 'Mars',
                    'Jupiter', 'Saturn', 'Uranus', 'Neptune'],
        'radius_km': [2440, 6052, 6371, 3390, 69911, 58232,
                      25362, 24622],
        'moons': [0, 0, 1, 2, 80, 83, 27, 14]
         }
df1 = pd.DataFrame(data)
df1


# In[18]:


# Define df2
data = {'planet': ['Mercury', 'Venus', 'Earth', 'Meztli', 'Janssen'],
        'radius_km': [2440, 6052, 6371, 48654, 11959],
        'life?': ['no', 'no', 'yes', 'no', 'yes'],
         }
df2 = pd.DataFrame(data)
df2


# Now we'll merge the two dataframes on the `['planet', 'radius_km']` columns. Try running the below cell with each of the following arguments for the **`how`** keyword: `'left'`, `'right'`, `'inner'`, and `'outer'`. Notice how each argument changes the result.  
# 
# Feel free to change the columns specified by the **`on`** argument too!

# In[19]:


merged = df1.merge(df2, how='left', on=['planet', 'radius_km'])
merged


# <a id='outliers'></a>
# # Identify and deal with outliers

# Throughout the following exercises, you will learn to find and deal with outliers in a dataset. Before starting on this programming exercise, we strongly recommend watching the video lecture and completing the IVQ for the associated topics.

# All the information you need for solving this assignment is in this notebook, and all the code you will be implementing will take place within this notebook.

# As we move forward, you can find instructions on how to install required libraries as they arise in this notebook. Before we begin with the exercises and analyzing the data, we need to import all libraries and extensions required for this programming exercise. Throughout the course, we will be using pandas, numpy, datetime, for operations, and matplotlib, pyplot and seaborn for plotting.

# ## Objective
# 
# We will be examining lightning strike data collected by the National Oceanic and Atmospheric Association (NOAA) from 1987 through 2020. Because this would be many millions of rows to read into the notebook, we've preprocessed the data so it contains just the year and the number of strikes.
# 
# We will examine the range of total lightning strike counts for each year and identify outliers. Then we will plot the yearly totals on a scatterplot.

# In[20]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[21]:


# Read in data
df = pd.read_csv('eda_outliers_dataset1.csv')


# In[22]:


# Print first 10 rows
df.head(10)


# Next, let's convert the number of strikes value to a more readable format on the graph (e.g., converting 100,000 to 100K, 3,000,000 to 3M, and so on).

# In[23]:


def readable_numbers(x):
    """takes a large number and formats it into K,M to make it more readable"""
    if x >= 1e6:
        s = '{:1.1f}M'.format(x*1e-6)
    else:
        s = '{:1.0f}K'.format(x*1e-3)
    return s

# Use the readable_numbers() function to create a new column 
df['number_of_strikes_readable']=df['number_of_strikes'].apply(readable_numbers)


# In[24]:


df.head(10)


# In[25]:


print("Mean:" + readable_numbers(np.mean(df['number_of_strikes'])))
print("Median:" + readable_numbers(np.median(df['number_of_strikes'])))


# A boxplot can help to visually break down the data into percentiles / quartiles, which are important summary statistics. The shaded center of the box represents the middle 50th percentile of the data points. This is the interquartile range, or IQR. 
# 
# The boxplot "whiskers" extend 1.5x the IQR by default.

# In[26]:


# Create boxplot
box = sns.boxplot(x=df['number_of_strikes'])
g = plt.gca()
box.set_xticklabels(np.array([readable_numbers(x) for x in g.get_xticks()]))
plt.xlabel('Number of strikes')
plt.title('Yearly number of lightning strikes');


# The points to the left of the left whisker are outliers. Any observations that are more than 1.5 IQR below Q1 or more than 1.5 IQR above Q3 are considered outliers.
# 
# One important point for every data professional: do not assume an outlier is erroneous unless there is an explanation or reason to do so.
# 
# Let's define our IQR, upper, and lower limit.

# In[27]:


# Calculate 25th percentile of annual strikes
percentile25 = df['number_of_strikes'].quantile(0.25)

# Calculate 75th percentile of annual strikes
percentile75 = df['number_of_strikes'].quantile(0.75)

# Calculate interquartile range
iqr = percentile75 - percentile25

# Calculate upper and lower thresholds for outliers
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

print('Lower limit is: '+ readable_numbers(lower_limit))


# Now we can use a Boolean mask to select only the rows of the dataframe where the number of strikes is less than the lower limit we calculated above. These rows are the outliers on the low end.

# In[28]:


# Isolate outliers on low end
df[df['number_of_strikes'] < lower_limit]


# Let's get a visual of all of the data points with the outlier values colored red.

# In[29]:


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(x[i]-0.5, y[i]+500000, s=readable_numbers(y[i]))

colors = np.where(df['number_of_strikes'] < lower_limit, 'r', 'b')

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['year'], df['number_of_strikes'],c=colors)
ax.set_xlabel('Year')
ax.set_ylabel('Number of strikes')
ax.set_title('Number of lightning strikes by year')
addlabels(df['year'], df['number_of_strikes'])
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
plt.show()


# ### Investigating the outliers 2019 and 1987
# 
# Let's examine the two outlier years a bit more closely. In the section above, we used a preprocessed dataset that didn't include a lot of the information that we're accustomed to having in this data. In order to further investigate the outlier years, we'll need more information, so we're going to import data from these years specifically.

# #### Import data for 2019

# In[30]:


df_2019 = pd.read_csv('eda_outliers_dataset2.csv')


# In[31]:


df_2019.head()


# First, we'll convert the `date` column to datetime. This will enable us to extract two new columns: `month` and `month_txt`. Then, we'll sort the data by `month` and `month_txt`, sum it, and sort the values. 

# In[32]:


# Convert `date` column to datetime
df_2019['date']= pd.to_datetime(df_2019['date'])

# Create 2 new columns
df_2019['month'] = df_2019['date'].dt.month
df_2019['month_txt'] = df_2019['date'].dt.month_name().str.slice(stop=3)

# Group by `month` and `month_txt`, sum it, and sort. Assign result to new df
df_2019_by_month = df_2019.groupby(['month','month_txt']).sum().sort_values('month', ascending=True).head(12).reset_index()
df_2019_by_month


# 2019 appears to have data only for the month of December. The likelihood of there not being any lightning from January to November 2019 is ~0. This appears to be a case of missing data. We should probably exclude 2019 from the analysis (for most use cases).

# #### Import data for 1987
# 
# Now let's inspect the data from the other outlier year, 1987.
# 

# In[33]:


# Read in 1987 data
df_1987 = pd.read_csv('eda_outliers_dataset3.csv')


# In this code block we will do the same datetime conversions and groupings we did for the other datasets. 

# In[34]:


# Convert `date` column to datetime
df_1987['date'] = pd.to_datetime(df_1987['date'])

# Create 2 new columns
df_1987['month'] = df_1987['date'].dt.month
df_1987['month_txt'] = df_1987['date'].dt.month_name().str.slice(stop=3)

# Group by `month` and `month_txt`, sum it, and sort. Assign result to new df
df_1987_by_month = df_1987.groupby(['month','month_txt']).sum().sort_values('month', ascending=True).head(12).reset_index()
df_1987_by_month


# 1987 has data for every month of the year. Hence, this outlier should be treated differently than 2019, which is missing data. 
# 
# Finally, let's re-run the mean and median after removing the outliers. Our final takeaway from our lesson on outliers is that outliers significantly affect the dataset's mean, but do not significantly affect the median. 
# 
# To remove the outliers, we'll use a Boolean mask to create a new dataframe that contains only the rows in the original dataframe where the number of strikes >= the lower limit we calculated above.

# In[35]:


# Create new df that removes outliers
df_without_outliers = df[df['number_of_strikes'] >= lower_limit]

# Recalculate mean and median values on data without outliers
print("Mean:" + readable_numbers(np.mean(df_without_outliers['number_of_strikes'])))
print("Median:" + readable_numbers(np.median(df_without_outliers['number_of_strikes'])))


# Both the mean and the median changed, but the mean much more so. It is clear that outlier values can affect the distributions of the data and the conclusions that can be drawn from them.

# If you have successfully completed the material above, congratulations! You now understand discovering in Python and should be able to start using it on your own datasets. 

# <a id='encoding'></a>
# # Label Encoding

# Throughout the following exercises, you will practice label encoding in Python.  Before starting on this programming exercise, we strongly recommend watching the video lecture and completing the IVQ for the associated topics. 

# As we move forward, you can find instructions on how to install required libraries as they arise in this notebook. Before we begin with the exercises and analyzing the data, we need to import all libraries and extensions required for this programming exercise. Throughout the course, we will be using pandas for operations, and matplotlib and seaborn for plotting.

# ## Objective
# 
# We will be examining monthly lightning strike data collected by the National Oceanic and Atmospheric Association (NOAA) for 2016&ndash;2018. The dataset includes three columns:  
# 
# |date|number_of_strikes|center_point_geom|
# |---|---|---|  
# 
# The objective is to assign the monthly number of strikes to the following categories: mild, scattered, heavy, or severe. Then we will create a heatmap of the three years so we can get a high-level understanding of monthly lightning severity from a simple diagram.   

# In[36]:


import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[37]:


# Read in the data
df = pd.read_csv('eda_label_encoding_dataset.csv')


# In[38]:


df.info()


# ### Create a categorical variable `strike_level`
# 
# Begin by converting the `date` column to datetime. Then we'll create a new `month` column that contains the first three letters of each month.  

# In[39]:


# Convert `date` column to datetime
df['date'] = pd.to_datetime(df['date'])

# Create new `month` column
df['month'] = df['date'].dt.month_name().str.slice(stop=3)


# In[40]:


df.head()


# Next, we'll encode the months as categorical information. This allows us to specifically designate them as categories that adhere to a specific order, which is helpful when we plot them later. We'll also create a new `year` column. Then we'll group the data by year and month, sum the remaining columns, and assign the results to a new dataframe.

# In[41]:


# Create categorical designations
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Encode `month` column as categoricals 
df['month'] = pd.Categorical(df['month'], categories=months, ordered=True)

# Create `year` column by extracting the year info from the datetime object
df['year'] = df['date'].dt.strftime('%Y')

# Create a new df of month, year, total strikes
df_by_month = df.groupby(['year', 'month']).sum().reset_index()
df_by_month.head()


# Now we'll create a new column called `strike_level` that contains a categorical variable representing the lightning strikes for each month as mild, scattered, heavy, or severe. The `pd.qcut` pandas function makes this easy. We just input the column to be categorized, the number of quantiles to sort the data into, and how we want to name each quantile. For more information on this function, refer to the [pandas qcut() documentation](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html).

# In[42]:


# Create a new column that categorizes number_of_strikes into 1 of 4 categories
df_by_month['strike_level'] = pd.qcut(
    df_by_month['number_of_strikes'],
    4,
    labels = ['Mild', 'Scattered', 'Heavy', 'Severe'])
df_by_month.head()


# ### Encode `strike_level` into numerical values

# Now that we have a categorical `strike_level` column, we can extract a numerical code from it using `.cat.codes` and assign this number to a new column. 

# In[43]:


# Create new column representing numerical value of strike level
df_by_month['strike_level_code'] = df_by_month['strike_level'].cat.codes
df_by_month.head()


# We can also create binary "dummy" variables from the `strike_level` column. This is a useful tool if we'd like to pass the categorical variable into a model. To do this, we could use the function `pd.get_dummies()`. Note that this is just to demonstrate the functionality of `pd.get_dummies()`. Simply calling the function as we do below will not convert the data unless we reassigned the result back to a dataframe. 
# 
# `pd.get_dummies(df['column'])` 🠚 **df unchanged**  
# `df = pd.get_dummies(df['column'])` 🠚 **df changed**

# In[44]:


pd.get_dummies(df_by_month['strike_level'])


# We don't need to create dummy variables for our heatmap, so let's continue without converting the dataframe.

# ### Create a heatmap of number of strikes per month

# We want our heatmap to have the months on the x-axis and the years on the y-axis, and the color gradient should represent the severity (mild, scattered, heavy, severe) of lightning for each month. A simple way of preparing the data for the heatmap is to pivot it so the rows are years, columns are months, and the values are the numeric code of the lightning severity. 
# 
# We can do this with the `df.pivot()` method. It accepts arguments for `index`, `columns`, and `values`, which we'll specify as described. For more information on the `df.pivot()` method, refer to the [pandas pivot() method documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html).

# In[45]:


# Create new df that pivots the data
df_by_month_plot = df_by_month.pivot(index='year', columns='month', values='strike_level_code')
df_by_month_plot.head()


# At last we can plot the heatmap! We'll use seaborn's `heatmap()` function for this.

# In[46]:


ax = sns.heatmap(df_by_month_plot, cmap = 'Blues')
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0, 1, 2, 3])
colorbar.set_ticklabels(['Mild', 'Scattered', 'Heavy', 'Severe'])
plt.show()


# The heatmap indicates that for all three years, the most lightning strikes occurred during the summer months. A heatmap is an easily digestable way to understand a lot of data in a single graphic.  

# If you have successfully completed the material above, congratulations! You now understand how to perform label encoding in Python and should be able to start using these skills on your own datasets.

# <a id='input_validation'></a>
# # Input Validation

# Throughout the following exercises, you will be practicing input validation in Python. Before starting on this programming exercise, we strongly recommend watching the video lecture and completing the IVQ for the associated topics.

# As we move forward, you can find instructions on how to install required libraries as they arise in this notebook. Before we begin with the exercises and analyzing the data, we need to import all libraries and extensions required for this programming exercise. Throughout the course, we will be using pandas for operations, and matplotlib and seaborn for plotting.

# ## Objective
# 
# We will be examining monthly lightning strike data collected by the National Oceanic and Atmospheric Association (NOAA) for 2018. The dataset includes five columns:  
# 
# |date|number_of_strikes|center_point_geom|longitude|latitude|
# |---|---|---|---|---|  
# 
# The objective is to inspect the data and validate the quality of its contents. We will check for:
#   
# * Null values
# * Missing dates
# * A plausible range of daily lightning strikes in a location
# * A geographical range that aligns with expectation

# In[47]:


import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px 
import seaborn as sns


# In[48]:


df = pd.read_csv('eda_input_validation_joining_dataset1.csv')


# In[49]:


df.head()


# In[50]:


# Display the data types of the columns
print(df.dtypes)


# The `date` column is currently a string. Let's parse it into a datetime column.

# In[51]:


# Convert `date` column to datetime
df['date'] = pd.to_datetime(df['date'])


# Now we'll do some data validation. We begin by counting the number of missing values in each column.

# In[52]:


df.isnull().sum()


# Check ranges for all variables.

# In[53]:


df.describe(include = 'all')


# Notice that the number of unique dates in the `date` column is 357. This means that eight days of 2018 are missing from the data, because 2018 had 365 days.

# ### Validate `date` column

# We need a way to easily determine which dates are missing. We can do this by comparing all of the actual dates in 2018 to the dates we have in our `date` column. The function `pd.date_range()` will create a datetime index of all dates between a start and end date (inclusive) that we'll give as arguments. This is a very useful function that can be used for more than just days. For more information about `pd.date_range()`, refer to the [pandas date_range() function documentation](https://pandas.pydata.org/docs/reference/api/pandas.date_range.html). 
# 
# Once we have the datetime index object of all dates in 2018, we'll compare its contents to the dates we have in the `date` column. The `index.difference()` method is used on index objects. Its argument is an index or array that you want to compare with the one the method is being applied to. It returns the set difference of the two indices&mdash;the values that are in the original index but not in the one given in the argument. 

# In[54]:


# Create datetime index of every date in 2018
full_date_range = pd.date_range(start='2018-01-01', end='2018-12-31')

# Determine which values are in `full_date_range` but not in `df['date']`
full_date_range.difference(df['date'])


# We knew that the data was missing eight dates, but now we know which specific dates they are. 

# ### Validate `number_of_strikes` column

# Let's make a boxplot to better understand the range of values in the data.

# In[55]:


sns.boxplot(y = df['number_of_strikes'])


# This is not a very useful visualization because the box of the interquartile range is squished at the very bottom. This is because the upper outliers are taking up all the space. Let's do it again, only this time we'll set `showfliers=False` so outliers are not included. 

# In[56]:


sns.boxplot(y = df['number_of_strikes'], showfliers=False)


# Much better! The interquartile range is approximately 2&ndash;12 strikes. But we know from the previous boxplot that there are many outlier days that have hundreds or even thousands of strikes. This exercise just helped us make sure that most of the dates in our data had plausible values for number of strikes. 

# ### Validate `latitude` and `longitude` columns

# Finally, we'll create a scatterplot of all the geographical coordinates that had lightning strikes in 2018. We'll plot the points on a map to make sure the points in the data are relevant and not in unexpected locations. Because this can be a computationally intensive process, we'll prevent redundant computation by dropping rows that have the same values in their `latitude` and `longitude` columns. We can do this because the purpose here is to examine locations that had lightning strikes, but it doesn't matter how many strikes they had or when.

# In[57]:


# Create new df only of unique latitude and longitude combinations
df_points = df[['latitude', 'longitude']].drop_duplicates() 
df_points.head()


# **Note:** The following cell's output is viewable in two ways: You can re-run this cell, or manually convert the notebook to "Trusted." 

# In[58]:


p = px.scatter_geo(df_points, lat = 'latitude', lon = 'longitude')
p.show()


# The plot indicates that the lightning strikes occurred primarily in the United States, but there were also many strikes in southern Canada, Mexico, and the Caribbean. We can click and move the map, and also zoom in for better resolution of the strike points.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
# 
# You now have a better understanding of different ways to examine a dataset and validate the quality of its contents.
