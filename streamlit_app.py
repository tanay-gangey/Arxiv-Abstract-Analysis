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
"""

#PLease wait as TSNE may take up to 1 minute to calculate for 2D and about 4 minutes for 3D

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

@st.cache(suppress_st_warning=True)
def calculate_TSNE(vectors,perplexity, dimensions):
    print("____________START___________")
    tsne = TSNE(n_jobs=4, n_components=dimensions,perplexity=perplexity, verbose=1, n_iter=400)
    reduced = tsne.fit_transform(vectors)
    return reduced
    pass


with st.echo(code_location='below'):
    path = "C:/Users/Tanay/CMU/Fall-1/IDS-05-839/HW-2/streamlit-example/arxiv_data/"
    cleaned_df = pd.read_csv(path+"cleaned_arxiv_data.csv")
    vector_df = pd.read_csv(path+"doc_vectors.csv")
    
    perplexity = st.slider("TSNE Perplexity: ", 1, 49, 20)
    dimensions = int(st.radio("TSNE dimensions: ", ('2','3')))
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
                title=f'TSNE dimension-1 vs dimension-2')
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
                title=f'TSNE dimension-1 vs dimension-2 vs Dimension-3')
        st.plotly_chart(fig)
    
    """
    ### Research Domain-wise Wordcloud
    """
   
    domain = st.selectbox('Choose research domain: ',('cs.AI','cs.CL', 'cs.CR', 'cs.CV','cs.LG','cs.NE','cs.RO','cs.SI', 'eess.IV','stat.ML'))
    number_of_words = st.slider("Number of words in cloud", 1, 50, 30)
    
    words = get_word_dict(cleaned_df, domain)
    top_words = list(words.items())[0:number_of_words]
    top_word_dict_list = list()
    for i in top_words:
        top_word_dict_list.append(dict(text=i[0], value=i[1], color=default_color))
    print(domain)
    print(top_word_dict_list)
    return_obj = wordcloud.visualize(top_word_dict_list, tooltip_data_fields={'text':'Term', 'value':'Mentions'}, per_word_coloring=False)