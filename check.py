import pandas as pd
import re
from urllib.parse import urlparse
import tldextract


# Load the blacklist file into memory when the application starts
try:
    df = pd.read_csv("blist.csv", usecols=["Obfuscated Domain"])
    domain_set = set(df["Obfuscated Domain"])  # Convert to a set for O(1) lookups
except FileNotFoundError:
    raise RuntimeError(f"File 'list.csv' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading CSV file: {e}")



def mix_checker(s):
    comma = "."
    swi = 0
    eflag = True
    
    for i in range(len(s)):
        if s[i] != comma:
            if eflag:
                if re.match(r'[^a-zA-Z]', s[i]):
                    swi += 1
                    eflag = False
            else:
                if re.match(r'[^0-9]', s[i]):
                    swi += 1
                    eflag = True
    
    # Calculate the ratio of switches to the length of the string (excluding commas)
    return (swi / (len(s) - 1)) > 0.2


def has_tld_in_list(url_or_domain):
    
    tld_list = ['.tk', '.buzz', '.xyz', '.top', '.ga', '.ml', 'pl', '.info', '.cf', '.gq', '.icu', '.wang', '.cn', '.io', '.fund', '.host',
                ' country', '.stream', '.download', '.xin', '.gdn', '.racing', '.jetzt', '.win', '.bid', '.biz', '.vip', 
                '.ren', '.kim', '.loan', '.mom', '.party', '.review', '.trade', '.trading','.lol', '.date', '.accountants', '.cfd', '.cyou', '.sbs', '.rest']

    # Extract domain components
    extracted = tldextract.extract(url_or_domain)
    
    # Get the full TLD (handles multi-level TLDs like .co.uk)
    domain_tld = f".{extracted.suffix.lower()}" if extracted.suffix else ""
    
    return domain_tld in tld_list



def check_domain(domain):
    # Remove the "www." prefix if it exists
    if domain.split(".")[0] == "www":
        domain = ".".join(domain.split(".")[1:])

    # Check if the domain exists in the set
    result = domain in domain_set
    mixCheck = mix_checker(domain)
    tldCheck = has_tld_in_list(domain)

    return result or mixCheck or tldCheck