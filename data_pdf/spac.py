import nltk
from pdfminer.high_level import extract_text
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk import ne_chunk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
text = extract_text("Lease 7.pdf")
tokenized_sentences = [word_tokenize(sent) for sent in sent_tokenize(text)]
pos_tagged_sentences = [pos_tag(tokens) for tokens in tokenized_sentences]
chunked_text = [ne_chunk(pos_tags) for pos_tags in pos_tagged_sentences]
unique_human_names = set()
unique_organization_names = set()
def extract_names(tree, entity_type):
    for subtree in tree.subtrees():
        if subtree.label() == entity_type:
            name = " ".join([leaf[0] for leaf in subtree.leaves()])
            return name
for sentence in chunked_text:
    human_name = extract_names(sentence, 'PERSON')
    if human_name and human_name not in unique_human_names:
        unique_human_names.add(human_name)
        print("Human Name:", human_name)

    org_name = extract_names(sentence, 'ORGANIZATION')
    if org_name and org_name not in unique_organization_names:
        unique_organization_names.add(org_name)
        print("Organization Name:", org_name)
