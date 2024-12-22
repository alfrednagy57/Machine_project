import pickle
import pandas as pd
import tldextract
import urllib.parse
import requests
import re
import whois
from datetime import datetime
import streamlit as st
import torch   
from transformers import pipeline


prediction = 0
def extract_domain(url):
    return urllib.parse.urlparse(url).netloc

def url_length(url):
    return len(url)

def subdomain_count(url):
    domain_parts = extract_domain(url).split('.')
    return len(domain_parts[:-2])

def is_https(url):
    return 1 if urllib.parse.urlparse(url).scheme == 'https' else 0

def has_redirects(url):
    try:
        response = requests.get(url, allow_redirects=False, timeout=3)
        return 1 if response.status_code in [301, 302, 303, 307, 308] else 0
    except:
        return -1

def count_suspicious_chars(url):
    return len(re.findall('[@!$#%^&*()_+|~=`{}[\]:/;<>?,.]', url))

def path_length(url):
    path = urllib.parse.urlparse(url).path
    return len(path)

def num_digits(url):
    return sum(1 for c in url if c.isdigit())

def num_digits_in_domain(url):
    domain = extract_domain(url)
    return sum(1 for c in domain if c.isdigit())

def num_question_marks(url):
    return url.count('?')

def num_hyphen_in_domain(url):
    domain = extract_domain(url)
    return domain.count('-')

def tld_in_subdomain(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    tld = extracted.suffix
    return 1 if tld in subdomain else 0

def num_at_symbols(url):
    return url.count('@')

def num_equals_symbols(url):
    return url.count('=')

def num_ampersand_symbols(url):
    return url.count('&')

def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        now = datetime.now()
        age = (now - creation_date).days
        return age
    except Exception as e:
        return -1

def rfb_model(x):
    loaded_model = pickle.load(open("new_random_forest_model.sav", 'rb'))
    prediction = loaded_model.predict(x)
    return prediction

def logistic_model(x):
    loaded_model = pickle.load(open("new_logistic_regression_model.sav", 'rb'))
    prediction = loaded_model.predict(x)
    return prediction

def svc_model(x):
    loaded_model = pickle.load(open("new_SVC_model.sav", 'rb'))
    prediction = loaded_model.predict(x)
    return prediction


# Bert Model 

if torch.backends.mps.is_available():
    device = torch.device('mps')  # Use MPS backend for Apple Silicon
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_path = "ealvaradob/bert-finetuned-phishing"

pipe = pipeline("text-classification", model=model_path,device=device)     





# Streamlit UI
st.title("URL Detection System")

url = st.text_input("Enter URL", placeholder="https://example.com")
model_choice = st.selectbox("Select Model", ["Random forest model", "Logistic regression model", "Support vector classifier", "Bert model"])

if st.button("Check URL"):
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        domain = extract_domain(url)
        tld_subdomain = tld_in_subdomain(url)
        url_len = url_length(url)
        redirect = has_redirects(url)
        suspicious_chars = count_suspicious_chars(url)
        num_digits_url = num_digits(url)
        num_digits_domain = num_digits_in_domain(url)
        num_question_marks_url = num_question_marks(url)
        path_len = path_length(url)
        domain_age = get_domain_age(domain)
        num_hyphen = num_hyphen_in_domain(domain)
        num_at = num_at_symbols(url)
        num_equal = num_equals_symbols(url)
        num_and = num_ampersand_symbols(url)

        data = [{
            'url_length': url_len, 'num_digits': num_digits_url, 'domain_age': domain_age,
            'count_suspicious_chars': suspicious_chars, 'path_length': path_len,
            'num_digits_in_domain': num_digits_domain, 'num_?': num_question_marks_url,
            'has_redirects': redirect, 'num_hyphen_domain': num_hyphen,
            'tld_in_subdomain': tld_subdomain, 'num_@': num_at,
            'num_=': num_equal, 'num_&': num_and
        }]

        x = pd.DataFrame(data)


        if model_choice == "Random forest model":
            prediction = rfb_model(x)[0]

            if prediction == 1:
                st.success("The entered URL is real.")
            else:
                st.error("The entered URL is fake.")
        elif model_choice == "Logistic regression model":
            prediction = logistic_model(x)[0]

            if prediction == 1:
                st.success("The entered URL is real.")
            else:
                st.error("The entered URL is fake.")

        elif model_choice == "Support vector classifier":
            prediction = svc_model(x)[0]

            if prediction == 1:
                st.success("The entered URL is real.")
            else:
                st.error("The entered URL is fake.")

        elif model_choice == "Bert model":
            result = pipe(url)
            result = result[0]["label"]
            print(result)

            if result == "benign":

                st.success("The entered URL is real.")

            else :
                st.success("The entered URL is fake.")




        # print(prediction)

        # if prediction == 1:
        #     st.success("The entered URL is real.")
        # else:
        #     st.error("The entered URL is fake.")
