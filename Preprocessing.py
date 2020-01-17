# Categorization of Amazon Product reviews
# CS 594 - Deep Learning for NLP - Fall'19
# sthiru5@uic.edu, pchand34@uic.edu, rkrish25@uic.edu

# Download the 5-core json.gz small subset review file for different categories from https://nijianmo.github.io/amazon/index.html

#Download and import the following libraries
import pandas as pd
import gzip
import json

# Function to open the compressed gz file
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

# Function to extract the json.gz file and store it as a dataframe
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# Read the file into a dataframe
df = getDF('filename.json.gz')

# Only select reviews that has the overall rating < 2. The reviewText is the main customer review and summary contains review keywords
reviews = df[df.overall<2]
reviews = reviews[['reviewText', 'summary']]

# Store 200 random reviews to a csv file across each category for manual tagging - to generate the test set
reviews_testset = reviews.sample(n=200)
reviews_testset.to_csv("testset_filename.csv", sep=",",index=False)

# Store the dataframe in a csv file, add a new attribute class with values based on the keywords present in the customer reviews and finally remove reviews with empty class values - to generate the training set
# Repeat this for all the categories
# The first 200 reviews in all categories are manually tagged to find the 'keywords' and then the below method is applied to tag the other reviews in that category.
# Below are the example set of keywords for each output class. This might vary depending on the product category.
reviews['Class'][reviews.reviewText.str.contains("cheaply made | low quality | worst quality | terrible quality", na=False,regex=True)] = "Quality"
reviews['Class'][(reviews.reviewText.str.contains("fit", na=False,regex=True)) & (reviews.reviewText.str.contains("compatible", na=False,regex=True))] = "Product Specifications"
reviews['Class'][(reviews.reviewText.str.contains("size", na=False,regex=True)) & (reviews.reviewText.str.contains("compatible", na=False,regex=True))] = "Product Specifications"
reviews['Class'][reviews.reviewText.str.contains("not compatible | misleading | advertised", na=False,regex=True)] = "Product Specifications"
reviews['Class'][reviews.reviewText.str.contains("wrong product | never worked | bad packaging | arrived defective | came defective | bad customer service | terrible service | from the beginning", na=False,regex=True)] = "Operations"
reviews['Class'][reviews.reviewText.str.contains("arrived late | came late | delivered broken | arrived damage", na=False,regex=True)] = "Logistics"
reviews['Class'][reviews.reviewText.str.contains("useless | worthless | not worth the price | over priced | high price", na=False,regex=True)] = "Others"
reviews_final = reviews.drop_duplicates(subset=['reviewText'], keep=False)
final_csv = reviews_final[reviews_final.Class!=""]
final_csv.to_csv("trainset_filename.csv", sep=",",index=False)