import nltk
from nltk.tokenize import sent_tokenize, word_tokenize,PunktSentenceTokenizer
from nltk.corpus import stopwords,state_union
from nltk.stem import PorterStemmer

def tok_demo():
    #tokenizing -word tokenizers....sentence tokenizers

    #print(sent_tokenize(exp_txt))

    #print(word_tokenize(exp_txt))

    for i in word_tokenize(exp_txt):
        print(i)

def stp_demo():
    stop_words=set(stopwords.words("english"))
    #print(stop_words)
    words=word_tokenize(exp_txt)
   #method 1
   # filter_sent=[]
    #for i in words:
     #   if i not in stop_words:
      #      filter_sent.append(i)
    #print(filter_sent)"""

    #method 2
    filtered_sent=[i for i in words if not i in stop_words]
    print(filtered_sent)

def stem_demo():
    ps=PorterStemmer()
#    for w in exp_words:
#        print(ps.stem(w))
    #deal with sentences
    stem_txt="It is important to be pythonly for pythoner while pyhoner are pythoning with python. All pythomers have pythoned at least at once."
    words=word_tokenize(stem_txt)
    for w in words:
        print(ps.stem(w))

def process_content():
    try:
        for i in tok:
            words= nltk.word_tokenize(i)
            tagged =nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))



if __name__=="__main__":
    exp_words=["python","pythoner","pythoning","pythoned","pythones","pythonation","pythonly"]
    exp_txt = "My life is brilliant. My love is pure. I saw an angle of that I am sure. She smiled at me on the subway."
    #tok_demo()
    #stp_demo()
    #stem_demo()
    #process content
    train_text = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")
    custom_sent_tok = PunktSentenceTokenizer(train_text)
    tok = custom_sent_tok.tokenize(sample_text)
    #process_content()





































































