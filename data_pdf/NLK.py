import nltk
from pdfminer.high_level import extract_text
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk import ne_chunk
text=extract_text("Lease 5.pdf")
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
tokenized_sentences = [word_tokenize(sent) for sent in sent_tokenize(text)]
pos_tagged_sentences = [pos_tag(tokens) for tokens in tokenized_sentences]
chunked_text = [ne_chunk(pos_tags) for pos_tags in pos_tagged_sentences]
unique_person_names = set()
for sentence in chunked_text:
    for subtree in sentence.subtrees():
        if subtree.label() == 'PERSON':
            person_name = " ".join([leaf[0] for leaf in subtree.leaves()])
            if person_name not in unique_person_names:
                unique_person_names.add(person_name)
                print("Person Name:", person_name)


