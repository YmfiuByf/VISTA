import spacy
nlp = spacy.load("en_core_web_lg")
def get_sim(str1,str2):
    doc1,doc2 = nlp(str1),nlp(str2)
    return doc1.similarity(doc2)
doc1 = nlp("I like salty fries and hamburgers.")
doc2 = nlp("Fast food tastes very good.")
doc3 = nlp('i am stuipid')
doc4 = nlp('i am awkward')

# Similarity of two documents
print(doc1, "<->", doc2, doc1.similarity(doc2),doc4.similarity(doc3))
# Similarity of tokens and spans
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))