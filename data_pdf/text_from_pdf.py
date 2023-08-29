from pdfminer.high_level import extract_text, extract_pages, extract_text_to_fp
from flask import Flask, request, render_template, jsonify
import json
import pyap
import PyPDF2
import pandas as pd
import numpy as np
from dateutil import parser
import datefinder
import dateparser
import nltk
import re
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk import ne_chunk
from nltk.corpus import stopwords
import pdfplumber
import spacy
from dateparser.search import search_dates
from nameparser import HumanName
import os
import addressparser
from collections import OrderedDict
app = Flask(__name__)
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
@app.route("/ml-service/health/v1/ping",methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/text-extraction", methods=["POST"])
def text_from_pdf():
    if request.method == 'POST':
        #print(request.files)
        file = request.files["files"]
        selected_options = request.form["extractOptions"]
        file_path = "temp.pdf"
        file.save(file_path)
        with open(file_path, 'rb') as f:
            text = extract_text(f)
            #print(text)
        data = {}
        if "addresses" in selected_options:
            address_pattern = (
                r'(?:(?:PO BOX|Po Box|P\.O\. BOX)\s+\d+\s*[•-]*\s*[A-Za-z\s,]+\s*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)|'
                r'\d+\s[A-Za-z\s]+(?: - [A-Za-z\s]+)?,\s[A-Za-z\s]+\,\s[A-Z]{2}\s\d{5}|'
                r'\d+\s[A-Za-z\s]+\,\sSuite\s[A-Za-z\d]+\s•\s[A-Za-z\s]+\,\s[A-Z]{2}\s\d{5}|'
                r'\d+\s[A-Za-z\s]+\s•\sSuite\s[A-Za-z\s]+\s•\s[A-Za-z\s]+\,\s[A-Z]{2}\s\d{5}|'
                r'\d+\s[A-Za-z\s]+\s•\s[A-Za-z\s]+\,\s[A-Z]{2}\s\d{5}|'
                r'PO BOX\s\d+\s•\s[A-Za-z\s]+\,\s[A-Z]{2}\s\d{5}|'
                r'PO Box\s\d+\s•\s[A-Za-z\s]+\,\s[A-Z]{2}\s\d{5}(?:-\d{4})?|'
                r'(?:PO BOX|Po Box|P.O. BOX)\s+\d+\s*[•-]*\s*[A-Za-z\s,]+\s*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?|'                
                r"\b\d+\s[\w\s.-]+,\s\w+\s\d+\b"
            )
            addresses1 = re.findall(address_pattern, text)
            addresse1 = pyap.parse(text, country="US")
            addresse2 = pyap.parse(text, country="GB")  # uk
            addresse3 = pyap.parse(text, country="CA")  # canada
            addresses = addresse1
            filtered_addresses = [address.full_address for address in addresses if
                                  "1 RESIDENT IS RESPONSIBLE FOR CHAR" not in address.full_address
                                  and "invitees.11. THERE IS NO WARRANTY OF A SMOK" not in address.full_address]
            all_addresses = addresses1+filtered_addresses
            if all_addresses:
                unique_addresses = list(set(all_addresses))
                data['addresses'] = {f"address_{idx}": address for idx, address in enumerate(unique_addresses, start=1)}
                data['address_count'] = len(unique_addresses)
            else:
                data["address_response"] = "No addresses found"

        if "dates" in selected_options:
            date_pattern = r'(?i)\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|' \
                           r'\d{1,2}(?:st|nd|rd|th)? \w+ \d{2,4}|' \
                           r'\d{1,2} \w+ \d{2,4}|' \
                           r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}|' \
                           r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?) \d{1,2}, \d{4}|' \
                           r'[a-zA-Z]{3} \d{1,2}, \d{4}|' \
                           r'[a-zA-Z]{3} \d{1,2},\d{4})\b'

            matches = re.findall(date_pattern, text, flags=re.IGNORECASE)
            dates = [parser.parse(match, fuzzy=True) for match in matches]
            unique_dates = list(set(date.strftime("%Y-%m-%d") for date in dates))
            # Filter out dates with starting year "0"
            valid_dates = [date for date in unique_dates if not date.startswith("0")]
            ordered_dates_dict = OrderedDict()
            for idx, date in enumerate(valid_dates, start=1):
                ordered_dates_dict[f"date_{idx}"] = date
            data['dates'] = ordered_dates_dict
            data['date_count'] = len(ordered_dates_dict)

        if "names" in selected_options:
            tokenized_sentences = [word_tokenize(text) for text in sent_tokenize(text)]
            pos_tagged_sentences = [pos_tag(tokens) for tokens in tokenized_sentences]
            chunked_text = [ne_chunk(pos_tags) for pos_tags in pos_tagged_sentences]
            unique_person_names = set()
            extracted_names = []
            for sentence in chunked_text:
                for subtree in sentence.subtrees():
                    if subtree.label() == 'PERSON':
                        person_name = " ".join([leaf[0] for leaf in subtree.leaves()])
                        if person_name not in unique_person_names:
                            unique_person_names.add(person_name)
                            extracted_names.append(person_name)
            words_to_remove = ["Tenant", "Walk", "Please", "Emergency Service", "Lease","Front","Make","Emergency","LANDLORD","Rent Income","Rent","Landlord","Fire","Gas","Lot","Lot Rent","Follows","Monthly","Resident","Fist","Items","Unit","Family Alliance",
                               "Lessee Admin Fee Total","Fees","Time","Date","Term","Pet","Cc50270","Accept","Accept Release","Persons","Inventory Form","Good Cause","Related Activity Banned","Good Cause","Pets","Residents","Pet Rent","Bedrooms","Lessee Daynesha",
                               "Lease Agreement","Occupants","Broken Per","Lease Agreement","Accept Keveoni Simmons","Termination Paragraph","Per","Keveoni Simmons Children","Agent","Policy Number","Lessee","Lessor","Pet Fees","Paint","Lessees","Anyone","Regulations Addendum"
                               "Required Insurance","Lessor Insurance","Lessee Date","Pet Fee","Renters Insurance","Protect Your","Use","Carpet Replacement","Carpet Stains","Window","Mail Key","Lock Replacement","Clean Please","Rules","Start Date","Sex","Lessee IP Address",
                               "Admin Fee","Moveout Notice","Landlord Insurance","Move","Civil Relief Act","Savage Lessor IP Address","Lessee IP Address","Mail","Pool Key","Auto Pay","Pet Addendum","Renewal Term","Incorrect","Regulations Addendum","Mr. Larry Savage Date","Daynesha D. Glover Date",
                               "Entire","Screen","First Month","Paint Regulation","CARRY AND","Deposit Refund","Lease Contract","Paint Addendum","Buy Out","Center Point","Buy Out","Bed Bugs","Paint Addendum","None Lessee","Owner","Originals","Signs","Default Has Occurred","Apartment Lease",
                               "Water","Apartment No","Final Bill Fee","Disease Control","Behind", "Countless","Remove","Default","Barnes755","Permission Required","Rental Unit","Class Action Waiver","Etc","Request","Apartment Lease Form","Mold Information","Casualty Loss","Apartment Lease",
                               "Default Has Occurred","Account Fee","Between","A Breach","Property Left","Lysol Disinfectant®",

                               ]
            filtered_names = [name for name in extracted_names if name not in words_to_remove]
            name_length_threshold = 25
            filtered_names = [name for name in filtered_names if
                              (len(name) <= name_length_threshold and len(name) > 2)]
            formatted_names = {f"name_{idx}": name for idx, name in enumerate(filtered_names, start=1)}
            data['names'] = formatted_names
            data['name_count'] = len(formatted_names)

        if "organization" in selected_options:
            tokenized_sentences = [word_tokenize(text) for text in sent_tokenize(text)]
            pos_tagged_sentences = [pos_tag(tokens) for tokens in tokenized_sentences]
            chunked_text = [ne_chunk(pos_tags) for pos_tags in pos_tagged_sentences]
            unique_person_names = set()
            extracted_names = []
            for sentence in chunked_text:
                for subtree in sentence.subtrees():
                    if subtree.label() == 'ORGANIZATION':
                        person_name = " ".join([leaf[0] for leaf in subtree.leaves()])
                        if person_name not in unique_person_names:
                            unique_person_names.add(person_name)
                            extracted_names.append(person_name)
            words_to_remove1=["Tenant", "Walk", "Please","LANDLORD","_________","___","Landlord","RENT Rent","TENANT","Lease Term","Term","Tenant Portal",]
            filtered_names = [name for name in extracted_names if name not in words_to_remove1]
            name_length_threshold = 25
            filtered_names = [name for name in filtered_names if
                              (len(name) <= name_length_threshold and len(name) > 2)]
            formatted_names = {f"name_{idx}": name for idx, name in enumerate(filtered_names[:3], start=1)}
            data['names'] = formatted_names
            data['name_count'] = len(formatted_names)

        if "full_text" in selected_options:
            data['full_text'] = text

        def find_usd_amount(text):
            usd_pattern = r'\$\s*(\d+(\.\d+)?)'
            matches = re.findall(usd_pattern, text)
            usd_amounts = [float(match[0]) for match in matches]
            return usd_amounts
        if "monetary_amounts" in selected_options:
            usd_amounts = find_usd_amount(text)
            formatted_amounts = {f"amount_{idx}": amount for idx, amount in enumerate(usd_amounts, start=1)}
            data['monetary_amounts'] = formatted_amounts
            data['monetary_amount_count'] = len(formatted_amounts)

        os.remove(file_path)
        return jsonify(data)
    #return render_template("che.html", **locals())
    #host="10.244.0.7",port=8080
    #http://localhost:8080/ml-service/text-extraction
    #https://india.yoroflow.com/ml-service/text-extraction
if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8080)