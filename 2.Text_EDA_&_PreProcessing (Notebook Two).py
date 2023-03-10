#!/usr/bin/env python
# coding: utf-8

# # TEXT DATA EXPLORATION AND PRE-PROCESSING
# 
# (Hussain et al., 2020) It has been observed from the literature that Spam Review detection using linguistic method uses only review text for spotting the spam review [37], [38]. It is usually performed binary classifcation in which the review is classifed as``spam or not spam``, and in our case studying the verified and unverified reviews for the detection of fake reviews.

# ## SUMMARY
# 
# In this notebook, we have done another EDA, however this time we focused on the input variable review_text itself, rather than the other attributes. Firstly columns depicting the character, word, stopcount, punctuation and the capital letter counts were added to guage the frequency of each of them, which then we later cleaned them during the text processing stage. 
# 
# It is important to note that before that, the necessary duplicates and NULL values were also taken care of, and then the reviews were then saved into a new file, where in the next notebook it is going to be utlized for model building.

# ----------------------------------------------------------------------

# ## INITIALIZATION

# In[30]:


#LIBRARIES 
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import  PorterStemmer 
import string
import re
import warnings
warnings.simplefilter("ignore")


# In[31]:


#lOADING DATASET 
df = pd.read_csv("updated_data.csv",encoding="latin1") #due to special charas should be encoded as latin 1
#REMOVE MAX
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[32]:


df.head()


# In[33]:


#DROPPING UNWANTED COLUMN
df.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[34]:


df.info()


# ## RE-CHECK NULL AND DUPLICATES
# 
# In our first EDA with the entire dataset, the duplicates within the reviews were not detected, which could be due to the other columns having slightly different values. Since we have removed the other columns and all we have left are the review_centric values, we need to double check on whether there are duplicated reviews within the dataset, which we can remove accordingly to remove potential bias.

# ### Removing Duplicates
# 
# Initially, in the first data exploration in the old cvs file, we have tried finding out the duplicates within this dataset. However, initially it did not yield any results. To double check on whether this dataset really has no duplicates, the selected columns were added to aid see if there are actually any duplicates within this dataset. 

# In[35]:


#CHECK TOTAL DUPLICATE OCCURENCES
dup = df.duplicated().sum()
print("Number of duplicates in dataset: ", dup)


# In[36]:


df = df.drop_duplicates().reset_index(drop=True)
df.info()


# ### NULL Values
# re-checking for NULL values to check if any needs to be filled up or dropped.

# In[37]:


df.isnull().sum()


# In[38]:


#DROP review_title
df.drop(["review_title","review_date"], axis=1, 
        inplace=True)
df.head()


# Titles are actually not mandatory in amazon reviews, and hence there are multiple missing values within the review_title. For this project, we are not going to be utlizing this dataset, and hence it is going to be dropped.

# ## EDA ON THE REVIEW TEXT
# 
# We have conducted an in-depth review surrounding the background of the Amazon dataset, and this time the ``review_text`` itself is going to be taken a further look. To aid in our pre-processing, certain columns will be added to understand certain instances the sentences have. Those include the counts of:
# 1. Word
# 2. Characters (with spaces)
# 3. Stopwords
# 4. Punctuations
# 5. Uppercase characters
# 
# After the columns are added, necessary ``summary statistics`` will be conducted to get an idea on how the pre-processing will take place.

# In[39]:


#WORD COUNT
df['total words'] = df['review_text'].apply(lambda i: len(str(i).split(" ")))

#CHARACTER COUNT
df['total characters'] = df['review_text'].str.len() #spaces are included

#STOPWORDS COUNT
sw = set(stopwords.words('english'))
df['total stopwords'] = df['review_text'].str.split().apply(lambda i: len(set(i) & sw))

#PUNCTUATION AND SPECIAL CHARA COUNT
count_p = lambda p1,p2: sum([1 for i in p1 if i in p2])
df['total punctuations'] = df.review_text.apply(lambda p: count_p(p, string.punctuation))

#UPPERCASE CHARA COUNT
df['total uppercases'] = df['review_text'].str.findall(r'[A-Z]').str.len() #findall - finds all


# In[40]:


df.head() #UPDATED 


# ### Summary Statistics

# In[41]:


df.describe()


# > Readings
# 1. The mean characters within the entire dataset happens to be at 177, averaging at about 33 words per review.
# 2. On average, there are about 9 stop words, and within the sentences there are about 4 punctuations.
# 3. As for the uppercase letters, from the mean value it is safe to assume that most of the reviews utlized their uppercases as Sentence Case.

# In[42]:


print(df.groupby("verified_purchase").describe())


# > Findings
# 1. Overall, we can see that False reviews have more words per character than True Values, where False values have an average of 50 words and 268 characters, while True values have about 14 words on average and 77 characters per review.
# 2. Witin the Fake reviews, it can be observed that there are more stopwords as well, than True reviews.
# 3. Since they are longer sentences in False values, it can be seen that there are more punctuations and Sentence case than True values.
# 

# In[43]:


#PIE CHART ON VERFIED PURCHASES -two
colors = ['#FED8B1','#79BAEC']
plt.figure(figsize=(4,4))
label = df['verified_purchase'].value_counts()
plt.pie(label.values,colors = colors, labels=label.index, autopct= '%1.1f%%', startangle=90)
plt.title('True and False Reviews Count', fontsize=15)
plt.show()


# After dropping the duplicates, we can see that the percentages of the True and False values are still near equal, and hence we can say that the dataset is balanced. Taking a closer look into the graph, there are more False values and True values within the dataset now.

# In[44]:


sns.catplot(x ='review_rating',kind="count", hue="verified_purchase", data=df)
plt.xlabel("review_rating")
plt.ylabel("count of reviews")
plt.title("Review_Rating Grouped by Verified_Purchase")


# In[45]:


cols = ["verified_purchase", "review_text"]
vprt = df[cols] #making a subset of the dataframe-

#FILTERING BASED ON TRUE AND FALSE VP
checkTrue = vprt["verified_purchase"] == True
filtered_true = vprt[checkTrue]

checkFalse = vprt["verified_purchase"] == False
filtered_false = vprt[checkFalse]


#AVERAGE REVIEW LENGTH BASED ON TRUE AND FALSE VP
false_average_length = filtered_false["review_text"].apply(len).mean()
true_average_length = filtered_true["review_text"].apply(len).mean()

#PLOTTING THE GRAPH
x = [true_average_length,false_average_length]
y = ["True", "False"]
sns.barplot(x=x, y=y,data=vprt)
plt.xlabel("average length of reviews")
plt.ylabel("verified_purchases")
plt.title("Average Length of Reviews based on Verified Purchases")
plt.show()


# From above we can see that 5 star rating is still the highest, and that true reviews still are more than false values within 5 star. Sentiment is still highly positive within this dataset.

# ## PRE-PROCESSING
# 
# Text preprocessing is a technique for cleaning text data and preparing it for use in a model. Text data comprises noise in the form of emotions, punctuation, and text in a different case, among other things. When it comes to Human Language, there are many different ways to communicate the same thing, and this is only the beginning of the difficulty. Machines cannot comprehend words; they want numbers, thus we must convert text to numbers efficiently.
# 
# 
# From the summary statistics conducted, we can see that the noise mentioned are having occurences within the review text, and hence the pre-processing will be conducted accordingly.
# 
# > To Do
# 1. Drop unwanted columns
# 2. Lowercasing
# 3. Remove Stopwords
# 4. Remove Punctuations and Special charas
# 5. Stemming

# In[46]:


#DROP UNNECESSARY COLUMNS
df.drop(["total words","total characters",
         "total stopwords","total punctuations",
         "total uppercases","review_rating"], axis=1, inplace=True)
df.head()


# For now, we are going to be only utlizing review_text and verified_purchase for our classifiers.

# ### Text Pre-Processing 
# 
# The ``review_text`` is going to be cleaned and standardized so that when implemented within the model, the model can be optimized at its best. This step takes the longest since it is in base of trial and error.
# 
# DONE IN THIS STAGE:
# 1. Spelling is corrected
# 2. tokenization,
# 3. removing stopwords, punctuations, special charas
# 4. lowercasing
# 5. stemming
# 6. removing top 3 common and rare words
# 

# In[47]:


#CORRECT SPELLING
df.review_text.apply(lambda i: ''.join(TextBlob(i).correct()))


# In[48]:


#REMOVING THE STOPWORDS,PUNCTUATIONS, LOWERCASING, AND STEMMING OF THE SENTENCES
def text_preprocessing(text):
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(text))
    tokens = removed_special_characters.lower().split()
    
    stemmer = PorterStemmer()
    cleaned = []
    stemmed = []
    
    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)


# In[49]:


df['review_text'] = df['review_text'].apply(text_preprocessing)


# In[50]:


df['review_text'].head()


# In[51]:


#CHECK RARE WORDS
r = pd.Series(' '.join(df['review_text']).split()).value_counts()[-10:]
print("RARE WORDS:")
print(r)


# In[52]:


#CHECK TOP COMMON WORDS
words = '' 
for i in df["review_text"]: 
    tokens = i.split()   
    words += " ".join(tokens)+" "

    
word_cloud = WordCloud(width = 700, height = 700, 
                       background_color ='white', 
                       min_font_size = 10).generate(words) 
plt.figure(figsize = (5, 5)) 
plt.imshow(word_cloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# In[53]:


#removing common and rare words
common = pd.Series(' '.join(df['review_text']).split()).value_counts()[:3]
common = list(common.index)
df['review_text'] = df['review_text'].apply(lambda i: " ".join(i for i in i.split() if i not in common))

rare = pd.Series(' '.join(df['review_text']).split()).value_counts()[-3:]
rare = list(rare.index)
df['review_text'] = df['review_text'].apply(lambda i: " ".join(i for i in i.split() if i not in rare))


# In[54]:


#WORDCLOUD - UPDATED TOP WORDS
words = '' 
for i in df["review_text"]: 
    tokens = i.split()   
    words += " ".join(tokens)+" "

    
word_cloud = WordCloud(width = 700, height = 700, background_color ='white', min_font_size = 10).generate(words) 
plt.figure(figsize = (5, 5)) 
plt.imshow(word_cloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# After removing the top 3 common word (it was removed since it would remove its meaning from the entire thing), we are left with the current top 10 words. As seen from above, we can see that the sentiment of it is quite positive, meaning that this dataset is dealing with many positive-centric reviews. The general polarity is thus, positive, and needs to be kept in mind for analysis later. It is to be noted that due to the lack of negative reviews in this case can cause for there to be discrepencies when, for instance a negative value is set to be identified as "fake" or "real", and thus can be added as a limitation to this study.

# In[55]:


df['review_text'].apply(word_tokenize).head()


# In[56]:


#SAVING UPDATED DATAFRAME AS .csv FILE
df.to_csv('cleaned_data.csv')

