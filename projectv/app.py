from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/")
@app.route("/home",methods=['GET','POST'])
def home():
    return render_template("mainpage.html")

@app.route("/results",methods=['GET','POST'])
def results():
    output = request.form.to_dict()
    url = output["search"]
    results.param = output["search"]
    
    
    #!/usr/bin/env python
    # coding: utf-8

    # In[ ]:


    # Libraries for text preprocessing
    import re
    import nltk
    #nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import RegexpTokenizer
    #nltk.download('wordnet') 
    from nltk.stem.wordnet import WordNetLemmatizer


    # In[ ]:


    # importing the libraries
    from bs4 import BeautifulSoup
    import requests
    import nltk
    from nltk import word_tokenize
    import string
    import re
    #nltk.download('punkt')


    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(url,verify=False).text

    # Parse the html content
    soup = BeautifulSoup(html_content, "lxml")
    lst=[]
    for k in soup.find_all("body"): 
        b=k.text

        #Remove punctuations
        #b = re.sub('[^a-zA-Z]', ' ',b) -
        b = b.lower()
        b=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",b)
        # remove special characters and digits
        #b=re.sub("(\\d|\\W)+"," ",b) -
        ##Convert to list from string
        #b = b.split()

        #printable = set(string.printable)
        #text = filter(lambda x: x in printable, b)
        lst.append(b)
        text = word_tokenize(b)

    #print(text)
    #len(text)


    # In[ ]:


    #nltk.download('averaged_perceptron_tagger') 
    POS_tag = nltk.pos_tag(text)
    #print("Tokenized Text with POS tags: \n")
    #print(POS_tag)


    # In[ ]:


    #nltk.download('wordnet')

    from nltk.stem import WordNetLemmatizer

    wordnet_lemmatizer = WordNetLemmatizer()

    adjective_tags = ['JJ','JJR','JJS']
    verb_tags=['VBG','VBN']

    lemmatized_text = []

    for word in POS_tag:
        if word[1] in adjective_tags:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
        elif word[1] in verb_tags:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="v"))) 
        else:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun

    #print("Text tokens after lemmatization of adjectives and nouns: \n")
    #print(lemmatized_text)


    # In[ ]:


    POS_tag = nltk.pos_tag(lemmatized_text)

    #print("Lemmatized text with POS tags: \n")
    #print(POS_tag)


    # Pre-liminary Stop word generation

    # In[ ]:


    stoplist = []

    wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW','VBN'] 

    for word in POS_tag:
        if word[1] not in wanted_POS:
            stoplist.append(word[0])

    punctuations = list(str(string.punctuation))

    stoplist = stoplist + punctuations


    # Addition of stop words

    # In[ ]:


    #nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words=  list(set(stopwords.words("english")))
    punc = '''!()-[]{};:'",<>./?@#$%^&*_~–•©'''

    stopplus=[]
    mega_stop=[]
    mega_stop=stop_words+stoplist


    # In[ ]:


    example1="stoplist.txt"
    stopword_file=open(example1,'r')
    lots_of_stopwords = []

    for line in stopword_file.readlines():
        lots_of_stopwords.append(str(line.strip()))

    mega_stop = []
    mega_stop = stoplist + lots_of_stopwords
    mega_stop = set(mega_stop)


    # In[ ]:


    processed_text = []
    for word in lemmatized_text:
        if word not in mega_stop and word not in punc:
            processed_text.append(word)
    #print(processed_text)
    #print(len(processed_text))


    # In[ ]:


    vocabulary = list(set(processed_text))
    #print(vocabulary)
    #print(len(vocabulary))


    # In[ ]:


    import numpy as np
    import math
    vocab_len = len(vocabulary)

    weighted_edge = np.zeros((vocab_len,vocab_len),dtype=np.float32)

    score = np.zeros((vocab_len),dtype=np.float32)
    window_size = 3
    covered_coocurrences = []

    for i in range(0,vocab_len):
        score[i]=1
        for j in range(0,vocab_len):
            if j==i:
                weighted_edge[i][j]=0
            else:
                for window_start in range(0,(len(processed_text)-window_size+1)):

                    window_end = window_start+window_size

                    window = processed_text[window_start:window_end]

                    if (vocabulary[i] in window) and (vocabulary[j] in window):

                        index_of_i = window_start + window.index(vocabulary[i])
                        index_of_j = window_start + window.index(vocabulary[j])

                        # index_of_x is the absolute position of the xth term in the window 
                        # (counting from 0) 
                        # in the processed_text

                        if [index_of_i,index_of_j] not in covered_coocurrences:
                            weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                            covered_coocurrences.append([index_of_i,index_of_j])


    # In[ ]:


    inout = np.zeros((vocab_len),dtype=np.float32)

    for i in range(0,vocab_len):
        for j in range(0,vocab_len):
            inout[i]+=weighted_edge[i][j]


    # In[ ]:


    MAX_ITERATIONS = 50
    d=0.85
    threshold = 0.0001 #convergence threshold

    for iter in range(0,MAX_ITERATIONS):
        prev_score = np.copy(score)

        for i in range(0,vocab_len):

            summation = 0
            for j in range(0,vocab_len):
                if weighted_edge[i][j] != 0:
                    summation += (weighted_edge[i][j]/inout[j])*score[j]

            score[i] = (1-d) + d*(summation)

        if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
            #print("Converging at iteration "+str(iter)+"....")
            break


    # In[ ]:


    #for i in range(0,vocab_len):
    #    print (vocabulary[i]+": "+str(score[i]))


    # In[ ]:


    dictionary=sorted(zip(score,vocabulary),reverse=True)


    # In[ ]:


    #dictionary


    # In[ ]:


    text_rank=dictionary[0:10]
    #print(text_rank)


    # In[ ]:


    actual_list=[]
    for k in soup.find_all("title"): 
        b=soup.title.text
        actual_list.append(b)

    for k in soup.find_all("h2"): 
        b=k.text
        actual_list.append(b)

    train_data=[]
    for i in range(0,len(actual_list)):
      train_data.append(actual_list[i].split())
    flat_list = [item for sublist in train_data for item in sublist]
    #print(flat_list)

    train_list=[]
    for i in flat_list:
      j=i.lower()
      if j not in train_list and j not in stop_words and j not in punc:
        train_list.append(j)
    #print(train_list)


    # In[ ]:


    tp=0
    fp=0
    fn=0
    for i in dictionary:
      if(i[1] in train_list):
        tp=tp+1
      else:
        fp=fp+1

    fn=len(train_list)-tp

    precision_txtrank=0.0
    recall_txtrank=0.0

    #Calculation of Precision value
    precision_txtrank=tp/(tp+fn)

    #Calculation of Recall value
    recall_txtrank=tp/(tp+fp)

    #Calculation of F1-Measure Value
    f1_txtrank=2*(precision_txtrank*recall_txtrank/(precision_txtrank + recall_txtrank ))

    accuracy=()
    #print("Performance metrics of Text Rank Algorithm:")
    #print("Precision: ",precision_txtrank)
    #print("Recall: ",recall_txtrank)
    #print("F1-measure: ",f1_txtrank)


    # Tf-Idf Algorithm

    # In[ ]:


    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import numpy as np
    import pandas as pd

    import re
    cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(0,1))
    X=cv.fit_transform(processed_text)


    # In[ ]:


    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(X)
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': cv.get_feature_names(), 'weight': weights})


    # In[ ]:


    sorted_df=weights_df.sort_values(by='weight', ascending=False).head(50)
    #sorted_df


    # In[ ]:


    #print("Top 10 keywords of Tf-Idf Algorithm:")
    tf_idf=sorted_df.head(10)
    #print(tf_idf)


    # In[ ]:


    tp=0
    fp=0
    fn=0
    for i in sorted_df['term']:
      if(i in train_list):
        tp=tp+1
      else:
        fp=fp+1

    fn=len(train_list)-tp

    precision_tfidf=0.0
    recall_tfidf=0.0

    #Calculation of Precision value
    precision_tfidf=tp/(tp+fn)

    #Calculation of Recall value
    recall_tfidf=tp/(tp+fp)

    #Calculation of F1-Measure Value
    f1_tfidf=2*(precision_tfidf*recall_tfidf/(precision_tfidf + recall_tfidf ))

    accuracy=()
    #print("Performance metrics of Tf-Idf Algorithm:")
    #print("Precision: ",precision_tfidf)
    #print("Recall: ",recall_tfidf)
    #print("F1-measure: ",f1_tfidf)


    # YAKE! Algorithm

    # In[ ]:


    #pip install git+https://github.com/boudinfl/pke.git


    # In[ ]:


    def convert(lst):

        return ' '.join(lst)

    text2=convert(processed_text)

    #print(text2)


    # In[ ]:


    from pke.unsupervised import YAKE
    extractor = YAKE()
    extractor.load_document(input=text2,language='en',normalization=None)


    # In[ ]:


    extractor.candidate_selection(n=1)


    # In[ ]:


    extractor.candidate_weighting(window=2,use_stems=False)


    # In[ ]:


    #print("Top 10 keywords of Keybert Algorithm:")
    key_phrases = extractor.get_n_best(n=10, threshold=0.8)
    #print(key_phrases)


    # In[ ]:


    tp=0
    fp=0
    fn=0
    for i in key_phrases:
      if i[0] in train_list:
        tp=tp+1
      else:
        fp=fp+1

    fn=len(train_list)-tp

    precision_yake=0.0
    recall_yake=0.0

    #Calculation of Precision value
    precision_yake=tp/(tp+fn)

    #Calculation of Recall value
    recall_yake=tp/(tp+fp)

    #Calculation of F1-Measure Value
    f1_yake=2*(precision_yake*recall_yake/(precision_yake + recall_yake ))

    accuracy=()
    #print("Performance metrics of KeyBert Algorithm:")
    #print("Precision: ",precision_yake)
    #print("Recall: ",recall_yake)
   # print("F1-measure: ",f1_yake)


    # In[ ]:


    #Word cloud
    #from os import path
    #from PIL import Image
    #from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    #import matplotlib.pyplot as plt
    #get_ipython().run_line_magic('matplotlib', 'inline')
    #wordcloud = WordCloud(
    #                          background_color='white',
    #                          stopwords=stop_words,
    #                         max_words=100,
    #                          max_font_size=50, 
    #                          random_state=42
    #                         ).generate(str(processed_text))
    #print(wordcloud)
    #fig = plt.figure(1)
    #plt.imshow(wordcloud)
    #plt.axis('off')
    #plt.show()


    
    
    
    
    
   
    return render_template("results.html",url=url,text_rank=text_rank,precision_txtrank=precision_txtrank,recall_txtrank=recall_txtrank,f1_txtrank=f1_txtrank,tf_idf=tf_idf.values.tolist(),precision_tfidf=precision_tfidf,recall_tfidf=recall_tfidf,f1_tfidf=f1_tfidf,key_phrases=key_phrases,precision_yake=precision_yake,recall_yake=recall_yake,f1_yake=f1_yake)

if __name__=="__main__":
    app.run(debug=True)
