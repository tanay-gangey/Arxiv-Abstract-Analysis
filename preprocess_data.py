import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk

path = "C:/Users/Tanay/CMU/Fall-1/IDS-05-839/HW-2/streamlit-example/arxiv_data/"

df_full = pd.read_csv(path+"arxiv_data.csv")

df = df_full.sample(frac=0.2, random_state=42)
df = df.reset_index()
df  = df.drop(columns=['index'])

termdict = dict()
for i in df.terms:
    for t in (eval(i)):
        if t in termdict:
            termdict[t]+=1
        else:
            termdict[t]=1

top10_dict = dict(sorted(termdict.items(),reverse=True,key=lambda item: item[1])[0:10])
top10_items = set(top10_dict.keys())

count = 0
top_terms_list = list()
for i in df.terms:
    l = set(eval(i))
    inter = list(l.intersection(top10_items))
    if(len(inter)==0):
        count+=1
    top_terms_list.append(inter)
print(top_terms_list[0:10], count)

df['top_terms'] = top_terms_list

puncts = string.punctuation
stop_words = stopwords.words('english')
transtab_pun = str.maketrans("","",puncts)
nltk.download("stopwords", quiet = True)
nltk.download("wordnet", quiet = True)
nltk.download("punkt", quiet = True)
nltk.download('averaged_perceptron_tagger', quiet = True)
lemmatizer = WordNetLemmatizer()

cleaned_abstracts= list()

i=0
for abstr in df.summaries:
    i+=1
    l_abs = abstr.lower()
    no_links = re.sub(r'http\S+', '', l_abs)
    no_puncts = no_links.translate(transtab_pun) #Ref:https://stackoverflow.com/a/266162
    tokens = word_tokenize(no_puncts)
    no_stops_tokens = [x for x in tokens if x not in stop_words]
    lemmatized = list()
    #Ref: for lemmatization I used my own logic from 11-637
    for w in no_stops_tokens:
        pos = "n"
        x = nltk.pos_tag([w])
        if(x[0][1][0]=="J"):
            pos = "a"
        elif(x[0][1][0]=="R"):
            pos = "r"
        elif(x[0][1][0]=="V"):
            pos = "v"
        word = lemmatizer.lemmatize(w,pos=pos)
        if(len(word)>2):
            lemmatized.append(word)
    final = [q for q in lemmatized if q not in stop_words]
    cleaned_abstracts.append(final)
    print("Row ",i)

df['cleaned_abstracts'] = cleaned_abstracts
df.to_csv(path+'cleaned_arxiv_data.csv',index=False)