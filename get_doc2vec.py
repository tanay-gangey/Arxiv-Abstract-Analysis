import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

path = "C:/Users/Tanay/CMU/Fall-1/IDS-05-839/HW-2/streamlit-example/arxiv_data/"

df = pd.read_csv(path+"cleaned_arxiv_data.csv")

word2vec_list = list()

#Ref: https://radimrehurek.com/gensim/models/doc2vec.html

tagged_data = [TaggedDocument(eval(sentence),[idx]) for idx, sentence in enumerate(df.cleaned_abstracts)]

model = Doc2Vec(vector_size=50, min_count=1, epochs=50)
model.build_vocab(tagged_data)

model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

i=0
for sent in df.cleaned_abstracts:
    i+=1
    vec = list(model.infer_vector(eval(sent)))
    word2vec_list.append(vec)
    if(i%100==0):
        print(i, end=" ")

new_df = pd.DataFrame()
new_df["titles"] = df.titles
new_df["top_terms"] = df.top_terms
new_df["vectors"] = word2vec_list

new_df.to_csv(path+'doc_vectors.csv',index=False)