#!/usr/bin/env python
# coding: utf-8

# Name: Muhammad Hamza Azam

# Roll Number: 335323

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[42]:


data = pd.read_csv("Emirates Airways Reviews.csv")
data


# In[43]:


data.info()


# In[44]:


data.describe()


# In[45]:


data.isna().sum()


# In[46]:


data.duplicated().sum()


# In[47]:


cols_to_convert = ['Seating Comfort', 'Staff Service', 'Food Quality', 'Entertainment', 'WiFi', 'Ground Service']

for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')


# In[48]:


data.info()


# In[49]:


data.groupby('Travel Class')['Overall Rating'].mean().sort_values(ascending=False)


# In[50]:


business_class = data[data['Travel Class'] == 'Business Class']
recommended_ratio = business_class['Recommended'].value_counts(normalize=True) * 100
recommended_ratio


# In[51]:


correlation = data[['Seating Comfort', 'Overall Rating']].corr()
correlation


# In[52]:


sns.scatterplot(data=data, x='Seating Comfort', y='Overall Rating')
plt.title('Seating Comfort vs Overall Rating')
plt.show()


# In[53]:


rating_cols = ['Seating Comfort', 'Staff Service', 'Food Quality', 'Entertainment', 'WiFi', 'Ground Service']


# In[54]:


correlations = data[rating_cols + ['Value for Money']].corr()['Value for Money'].drop('Value for Money').sort_values(ascending=False)
print(correlations)


# In[55]:


sns.heatmap(data[rating_cols + ['Value for Money']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation with Value for Money')
plt.show()


# In[56]:


top_routes = data.groupby('Route')['Overall Rating'].mean().sort_values(ascending=False).head(10)
print("Top 10 Highest Rated Routes:\n", top_routes)


# In[57]:


bottom_routes = data.groupby('Route')['Overall Rating'].mean().sort_values().head(10)
print("\nBottom 10 Lowest Rated Routes:\n", bottom_routes)


# In[58]:


filtered_aircraft = data[data['Aircraft'].isin(['Boeing 777', 'A380'])]


# In[59]:


aircraft_rating = filtered_aircraft.groupby('Aircraft')['Overall Rating'].mean()
print("Average Overall Rating:\n", aircraft_rating)


# In[60]:


data['Is_Multi_Leg'] = data['Route'].str.contains('via', case=False, na=False)
data.groupby('Is_Multi_Leg')['Overall Rating'].mean()


# In[61]:


country_rating = data.groupby('Country')['Overall Rating'].mean().sort_values(ascending=False)
print("Top 10 Countries by Average Overall Rating:\n", country_rating.head(10))
print("\n Bottom 10 Countries by Average Overall Rating:\n", country_rating.tail(10))


# In[62]:


data['Recommended'] = data['Recommended'].astype(str).str.lower()
not_recommended_ratio = (
    data[data['Recommended'] == 'no']
    .groupby('Country')
    .size() / data.groupby('Country').size()
).sort_values(ascending=False) * 100

print("Top 10 Countries with Highest % of Not Recommended:\n", not_recommended_ratio.head(10))


# In[63]:


country_rating.head(10).plot(kind='barh', title='Top 10 Countries by Avg Overall Rating', color='green')
plt.xlabel('Average Overall Rating')
plt.show()


# In[64]:


not_recommended_ratio.head(10).plot(kind='barh', title='% Not Recommended by Country', color='red')
plt.xlabel('% Not Recommended')
plt.show()


# In[65]:


features = ['Seating Comfort', 'Staff Service', 'Food Quality', 'Entertainment', 'WiFi', 'Ground Service']
mean_ratings = data[features].mean().sort_values()
print("Features with Lowest Average Ratings:\n")
print(mean_ratings)


# In[66]:


wifi_missing_rating = data[data['WiFi'].isna()]['Overall Rating'].mean()
wifi_available_rating = data[data['WiFi'].notna()]['Overall Rating'].mean()
print(f"WiFi Missing - Avg Overall Rating: {wifi_missing_rating:.2f}")
print(f"WiFi Available - Avg Overall Rating: {wifi_available_rating:.2f}")


# In[67]:


ent_missing_rating = data[data['Entertainment'].isna()]['Overall Rating'].mean()
ent_available_rating = data[data['Entertainment'].notna()]['Overall Rating'].mean()
print(f"Entertainment Missing - Avg Overall Rating: {ent_missing_rating:.2f}")
print(f" Entertainment Available - Avg Overall Rating: {ent_available_rating:.2f}")


# In[68]:


sns.barplot(data=data, x='Travel Class', y='Overall Rating', estimator='mean', palette='Blues_d')
plt.title('Avg Overall Rating by Travel Class')
plt.ylabel('Average Rating')
plt.xticks(rotation=15)
plt.show()


# In[69]:


rating_cols = ['Seating Comfort', 'Staff Service', 'Food Quality', 'Entertainment', 'WiFi', 'Ground Service', 'Overall Rating']
corr = data[rating_cols].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Between Rating Features')
plt.show()


# In[70]:


rec_counts = data['Recommended'].astype(str).str.lower().value_counts()

plt.pie(rec_counts, labels=rec_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
plt.title('Recommendation Distribution')
plt.axis('equal')  # Equal aspect ratio
plt.show()

