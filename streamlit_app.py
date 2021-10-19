from collections import OrderedDict
from plotly.express import scatter, scatter_3d
import pandas as pd
import streamlit as st
import streamlit_wordcloud as wordcloud
#from sklearn.manifold import TSNE
import time
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from operator import itemgetter

"""
# Arxiv Paper Abstracts Analysis
"""

"""
    ### Abstract Cluster Analysis
    t-SNE is a technique that can be used to reduce the dimensionality of the vector representations of the abstracts.
    Changing perplexity will change the number of neighbors used to find this latent representation.
    Use the slider to choose perplexity, the radio button to choose the number of latent dimensions and the drop down menu to choose Arxiv research domains to compare.
    You can also zoom in to the scatterplot and hover over points for more information.

    Please wait after changing perplexity or dimension as TSNE may take up to 1 minute to calculate for 2D and about 5-6 minutes for 3D!  

"""

#Please wait as TSNE may take up to 1 minute to calculate for 2D and about 5-6 minutes for 3D

default_color = "#b5de2b"

@st.cache(suppress_st_warning=True)
def get_word_dict(df, domain):  
    
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    my_bar.empty()
    
    words = OrderedDict()
    for idx, row in df.iterrows():
        i = row["cleaned_abstracts"]
        dom = row["top_terms"]
        if(domain in eval(dom)):
            for w in eval(i):
                if w not in words:
                    words[w]=1
                else:
                    words[w]+=1
    sortedwords = OrderedDict(sorted(words.items(), key=lambda t: t[1], reverse=True))
    return sortedwords
    #sentence_list = list()
    #for idx, row in df.iterrows():
    #    i = row["cleaned_abstracts"]
    #    dom = row["top_terms"]
    #    if(domain in eval(dom)):
    #        sentence_list.append(eval(i))
    #print("Done")
    #return sentence_list

@st.cache(suppress_st_warning=True)
def calculate_TSNE(vectors,perplexity, dimensions):
    print("____________START___________")
    tsne = TSNE(n_jobs=4, n_components=dimensions,perplexity=perplexity, verbose=1, n_iter=400)
    reduced = tsne.fit_transform(vectors)
    return reduced
    pass



path = "./arxiv_data/"
cleaned_df = pd.read_csv(path+"cleaned_arxiv_data.csv")
vector_df = pd.read_csv(path+"doc_vectors.csv")

perplexity = st.slider("t-SNE Perplexity: ", 1, 49, 20)
dimensions = int(st.radio("t-SNE dimensions: ", ('2','3')))
print(perplexity, dimensions)
vecs = np.array(list(map(eval,vector_df.vectors)))

reduced = calculate_TSNE(vecs, perplexity, dimensions)
labs = list(map(itemgetter(0),list(map(eval,vector_df.top_terms))))
titles = list(vector_df.titles)
options = st.multiselect('Choose research fields to compare',['cs.AI','cs.CL', 'cs.CR', 'cs.CV','cs.LG','cs.NE','cs.RO','cs.SI', 'eess.IV','stat.ML'],['cs.AI','stat.ML'])
termset = set(['cs.AI','cs.CL', 'cs.CR', 'cs.CV','cs.LG','cs.NE','cs.RO','cs.SI', 'eess.IV','stat.ML'])
optionset = set(options)
diffs = list(termset.difference(optionset))

if(dimensions==2):
    dim_1 = reduced[:,0]
    dim_2 = reduced[:,1]
    full_plot_df = pd.DataFrame(
        {'Dimension-1': dim_1,
            'Dimension-2': dim_2,
            'label': labs,
            'titles':titles
    })
    
    
    plot_df_sample = full_plot_df.sample(2500)
    filtered_df = plot_df_sample
    for op in diffs:
        filtered_df = filtered_df[filtered_df.label!=op]
    
    fig = scatter(filtered_df,
            x='Dimension-1',
            y='Dimension-2',
            color='label',
            hover_name='titles',
            opacity=0.3,
            title=f't-SNE Dimension-1 vs Dimension-2')
    st.plotly_chart(fig)
    
else:
    dim_1 = reduced[:,0]
    dim_2 = reduced[:,1]
    dim_3 = reduced[:,2]
    full_plot_df = pd.DataFrame(
        {'Dimension-1': dim_1,
            'Dimension-2': dim_2,
            'Dimension-3': dim_3,
            'label': labs,
            'titles':titles
    })
    
    plot_df_sample = full_plot_df.sample(2000)
    
    filtered_df = plot_df_sample
    for op in diffs:
        filtered_df = filtered_df[filtered_df.label!=op]
    
    fig = scatter_3d(filtered_df,
            x='Dimension-1',
            y='Dimension-2',
            z='Dimension-3',
            color='label',
            hover_name='titles',
            opacity=0.3,
            title=f't-SNE Dimension-1 vs Dimension-2 vs Dimension-3')
    st.plotly_chart(fig)

"""
### Research Domain-wise Wordcloud
Wordclouds are a great tool to visualize popular terms in a document. Use this visualization to see what terms are commonly used in each Arxiv reserach domain.
You can use the drop-down menu to choose the Arxiv research domain and the slider to choose the number of words in the wordcloud.
You can also hover over each word for more information. Please hover slowly and wait for the wordcloud to finish refreshing.  

"""

domain = st.selectbox('Choose research domain: ',('cs.AI','cs.CL', 'cs.CR', 'cs.CV','cs.LG','cs.NE','cs.RO','cs.SI', 'eess.IV','stat.ML'))
number_of_words = st.slider("Number of words in cloud", 1, 50, 30)

words = get_word_dict(cleaned_df, domain)
top_words = list(words.items())[0:number_of_words]
#sentence_list = get_word_dict(cleaned_df, domain)
#vectorizer = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False)
#X = vectorizer.fit_transform(sentence_list)
#feature_array = np.array(vectorizer.get_feature_names())
#sorted_ids = np.argsort(X.data)[:-(number_of_words+1):-1]
#top_words = list(feature_array[X.indices[sorted_ids]])
#print(top_words)

top_word_dict_list = list()
for i in top_words:
    top_word_dict_list.append(dict(text=i[0], value=i[1], color=default_color))
print(domain)
print(top_word_dict_list)
return_obj = wordcloud.visualize(top_word_dict_list, tooltip_data_fields={'text':'Term', 'value':'Mentions'}, per_word_coloring=False)

"""
The code and report for this page can be found over [here](https://github.com/tanay-gangey/streamlit-example/tree/deployment). 
Refer to the README.md file for the report!
"""
