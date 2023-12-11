import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

file_path1 = "data/modified_file_musk.csv"
file_path2 = "data/modified_file_others.csv"
elon = pd.read_csv(file_path1)
other = pd.read_csv(file_path2)

# Removing date from the data
elon = elon[['user_name', 'text', 'favorites']]
other = other[['author', 'content', 'number_of_likes']]

# Rename columns in 'elon_tweets' DataFrame
elon = elon.rename(columns={
    'user_name': 'author',  # Change 'user_name' to 'author'
    'text': 'content',      # Change 'text' to 'content'
    'favorites': 'number_of_likes'  # Change 'favorites' to 'number_of_likes'
})

# -----------------Remove links from tweets------------

# Function to remove links using regular expressions


def remove_links(text):
    # Regular expression pattern to match URLs
    url_pattern = r'https?://\S+|www\.\S+'
    # Replace URLs with an empty string
    return re.sub(url_pattern, '', text)


# Remove links from the 'content' column in elon DataFrame
elon['content'] = elon['content'].apply(remove_links)

# Remove links from the 'content' column in other DataFrame
other['content'] = other['content'].apply(remove_links)

# -------------Compare length----------------

elon_size = len(elon)
other_size = len(other)

print(elon_size)
print(other_size)

if other_size > elon_size:
    # Sample 'other' DataFrame to match the length of 'elon' DataFrame
    # Adjust random_state as needed for reproducibility
    other = other.sample(n=elon_size, random_state=42)

# Check if the sizes are now equal
print("Size of elon DataFrame:", len(elon))
print("Size of other DataFrame:", len(other))
# -------------Combine Tweets-------------------

combined_tweets = pd.concat([other, elon])

# Assign labels to tweets (assuming 'Elon Musk' and 'Other Author')
combined_tweets['label'] = combined_tweets['author'].apply(
    lambda x: 'Elon Musk' if x == 'Elon Musk' else 'Other Author')


# ------------Export-------------
combined_tweets.to_csv('combined.csv', index=False)
