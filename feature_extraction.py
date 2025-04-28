import dns.resolver
import tldextract
import whois
import requests
from scapy.all import IP, UDP, DNS, DNSQR, sr1
import math
import numpy as np
from collections import Counter, defaultdict
import datetime
import ipaddress
from datetime import datetime
from typing import Optional
import re
# from collections import Counter

def load_unethical_words(file_path):
    """Load unethical words from a text file (one word per line)."""
    with open(file_path, 'r') as file:
        unethical_words = [line.strip().lower() for line in file if line.strip()]
    return unethical_words

unethical_words = load_unethical_words("unethical-words.txt")

common_tlds = {
    'com', 'org', 'net', 'edu', 'gov', 'co', 'uk','in','us'
    }

homoglyphs = {
    'a': ['а', 'α', '⍺', 'ａ'],  # Cyrillic 'а', Greek 'alpha'
    'b': ['Ь', 'ｂ'],            # Cyrillic 'Ь'
    'c': ['с', 'ϲ', 'ｃ'],       # Cyrillic 'с'
    'd': ['ԁ', 'ｄ'],
    'e': ['е', 'є', 'ｅ'],       # Cyrillic 'е'
    'f': ['ｆ'],
    'g': ['ɡ', 'ｇ'],
    'h': ['һ', 'ｈ'],
    'i': ['і', 'ｉ'],            # Cyrillic 'і'
    'j': ['ј', 'ｊ'],
    'k': ['κ', 'ｋ'],
    'l': ['ⅼ', 'ｌ'],
    'm': ['ｍ'],
    'n': ['ո', 'ｎ'],
    'o': ['о', 'ο', 'ｏ'],       # Cyrillic 'о'
    'p': ['р', 'ｐ'],            # Cyrillic 'р'
    'q': ['ｑ'],
    'r': ['г', 'ｒ'],            # Cyrillic 'г'
    's': ['ѕ', 'ｓ'],            # Cyrillic 'ѕ'
    't': ['т', 'ｔ'],            # Cyrillic 'т'
    'u': ['υ', 'ｕ'],
    'v': ['ν', 'ｖ'],
    'w': ['ω', 'ｗ'],
    'x': ['х', 'ｘ'],            # Cyrillic 'х'
    'y': ['у', 'ｙ'],            # Cyrillic 'у'
    'z': ['ｚ']
}


def count_dots(domain):
    return domain.count('.')

def count_hyphens(domain):
    return domain.count('-')

def contains_unethical(domain):
    return 1 if any(word in domain.lower() for word in unethical_words) else 0

def has_uncommon_tld(domain):
    tld = domain.split('.')[-1].lower()
    return 1 if tld not in common_tlds else 0

def has_duplicate_chars(domain):
    clean = domain.replace('.', '').replace('-', '')
    return 1 if any(count > 1 for char, count in Counter(clean).items() if char.isalpha()) else 0

def has_repeated_numbers(domain):
    return 1 if re.search(r'(\d)\1', domain) else 0

def contains_homoglyphs(domain):
    for char in domain.lower():
        for eng_char, glyphs in homoglyphs.items():
            if char in glyphs:
                return 1
    return 0

# english_words = set(line.strip() for line in open('C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12/words_alpha.txt'))
# Load wordlist (ensure it contains "boost", "dam", etc.)
with open('words_alpha.txt', 'r') as f:
    WORDLIST = set(word.strip().lower() for word in f)

def is_meaningful(word: str) -> bool:
    """Check if word (or its plural) is in the wordlist."""
    return (word in WORDLIST) or (word.endswith('s') and word[:-1] in WORDLIST)

def extract_subwords(segment: str) -> list:
    """Extract all possible meaningful subwords from a segment."""
    subwords = []
    # Check all possible substrings (min 3 letters)
    for i in range(len(segment) - 2):
        for j in range(i + 3, len(segment) + 1):
            chunk = segment[i:j]
            if is_meaningful(chunk):
                subwords.append(chunk)
    return subwords

def longest_meaningful_word(domain: str) -> Optional[str]:
    """Find the longest meaningful word, prioritizing full segments."""
    domain = re.sub(r'\.[a-z]{2,}$', '', domain, flags=re.IGNORECASE)
    segments = re.split(r'[-0-9]+', domain.lower())
    
    candidates = []
    for seg in segments:
        # 1. Check full segment first (e.g., "damboost")
        if is_meaningful(seg):
            candidates.append(seg)
        # 2. Check all possible subwords (e.g., "boost" in "damboost")
        candidates.extend(extract_subwords(seg))
    
    return max(candidates, key=len) if candidates else 0

def extract_days(value):
    """Extract only the numeric days from Domain_Age column"""
    # Find the first number followed by 'days'
    match = re.search(r'(\d+)\s*days', str(value))
    if match:
        return int(match.group(1))
    return 0

def calculate_entropy(string):
    """Calculate the Shannon entropy of a string."""
    probabilities = [float(string.count(c)) / len(string) for c in set(string)]
    return -sum([p * math.log2(p) for p in probabilities])

def get_geolocation(ip):
    """Get geolocation and ASN details for an IP address."""
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json", timeout=5)
        data = response.json()
        # return data.get("country", "0"), data.get("org", "0").split()[-1]
         # Extract the country and ASN details
        country = data.get("country", "0")
        org = data.get("org", "0")
        
        # Extract the ASN number (assuming it starts with 'AS' followed by digits)  
        asn = org.split()[0][2:] if org.startswith("AS") else "0"
        # print("org", org)
        # print("asn", asn)

        
        return country, asn
    except Exception as e:
        print(f"Error fetching geolocation: {e}")
        return "0", "0"

def extract_features(domain):
    features = {
        "Country": 0, "ASN": 0, "TTL": 0, "IP": 0, "Domain": domain, "TTL.1": 0,
        "subdomain": 0, "tld": 0, "sld": 0, "len": 0, 
        "numeric_percentage": 0, "char_distribution": 0, "entropy": 0, 
        "1gram": [], "2gram": [], "3gram": [], "longest_word": 0, 
        "typos": 0, "obfuscate_at_sign": 0, "dec_8": 0, "dec_32": 0, 
        "oc_8": 0, "oc_32": 0, "hex_8": 0, "hex_32": 0, 
        "Domain_Name": 0, "Registrar": 0, "Registrant_Name": 0, 
        "Creation_Date_Time": 0, "Emails": 0, "Domain_Age": 0, 
        "Organization": 0, "State": 0, "Country.1": 0, 
        "Name_Server_Count": 0, "Page_Rank": 0
    }

    cols = [
        "ASN", "TTL", "IP", "subdomain", "tld", "sld", "len",
        'entropy', "avg_char_distribution", "avg_ngram_frequency", "numeric_percentage",
        'dots_count', 'hyphens_count', 'has_unethical_word', 'has_uncommon_tld', 
        'has_duplicate_chars', 'has_repeated_numbers', 'has_homoglyphs',
        "longest_meaningful_word",  "Domain_Name", "Registrar","Registrant_Name", 
        "Creation_Date_Time","Domain_Age", "Emails", "Organization", "State", 
        "Country.1", "Name_Server_Count", "Page_Rank"
    ]

    
    try:
        # Extract TLD, SLD, and subdomain
        ext = tldextract.extract(domain)
        features["tld"] = ext.suffix
        features["sld"] = ext.domain
        sld = features["sld"] # created by me
        features["subdomain"] = 1 if ext.subdomain else 0
        
        # Character distribution and entropy
        char_dist = Counter(sld)
        features["char_distribution"] = dict(char_dist)

        # bdomain = "b'"+domain+".'"
        # features["numeric_percentage"] = sum(c.isdigit() for c in sld) / len(sld) * 100
        # features["entropy"] = calculate_entropy(sld)

        #######################

        features["dots_count"] = count_dots(domain)
        features["hyphens_count"] = count_hyphens(domain)
        features["has_unethical_word"] = contains_unethical(domain)
        features["has_uncommon_tld"] = has_uncommon_tld(domain)
        features["has_duplicate_chars"] = has_duplicate_chars(domain)
        features["has_repeated_numbers"] = has_repeated_numbers(domain)
        features["has_homoglyphs"] = contains_homoglyphs(domain)
        features["longest_meaningful_word"] = longest_meaningful_word(domain)
        

        #######################

        def calculate_domain_metrics(domain):
            """
            Calculate entropy and numeric percentage for a domain string.
            Args:
                domain: String (e.g., "example123.com")
            Returns:
                Tuple of (entropy, numeric_percentage)
            """
            # Remove TLD (everything after first dot)
            # main_domain = domain.split('.')[0] if '.' in domain else domain

            main_domain = '.'.join(domain.split('.')[:-1]) if '.' in domain else domain
            # print("main domain", main_domain)
            
            if not main_domain:
                return 0.0, 0.0
            
            # Calculate entropy
            entropy = 0.0
            length = len(main_domain)
            for char in set(main_domain):
                p = main_domain.count(char) / length
                entropy -= p * math.log2(p) if p > 0 else 0
            
            # Calculate numeric percentage
            numeric_count = sum(c.isdigit() for c in main_domain)
            numeric_percent = (numeric_count / length) * 100 if length > 0 else 0
            
            return entropy, numeric_percent
        
        features["entropy"], features["numeric_percentage"] = calculate_domain_metrics(domain)
        # print("entropy: ", features["entropy"])
        # print("numeric percentage: ", features["numeric_percentage"])
        
        ########################
        # N-grams
        def calculate_avg_ngram_frequency(domain):
            """Calculate average n-gram frequency for a domain (1-gram, 2-gram, 3-gram combined)"""
            # Remove TLD if present and convert to lowercase
            clean_domain = domain.split('.')[0].lower()
            
            if not clean_domain:
                return 0.0
            
            total_ngrams = 0
            total_unique = 0
            
            # Calculate for 1-gram, 2-gram, and 3-gram
            for n in [1, 2, 3]:
                # Extract n-grams
                ngrams = [clean_domain[i:i+n] for i in range(len(clean_domain)-n+1)]
                
                if not ngrams:
                    continue
                    
                unique_ngrams = len(set(ngrams))
                total_ngrams += len(ngrams)
                total_unique += unique_ngrams
            
            # Calculate average frequency (avoid division by zero)
            return total_ngrams / total_unique if total_unique > 0 else 0.0
        
        features["avg_ngram_frequency"] = calculate_avg_ngram_frequency(domain)
        # print("avg_ngram_frequency", features["avg_ngram_frequency"])

        ######################

        # Average character distribution

        def calculate_avg_char_distribution(domain):
            """
            Calculate average character distribution for a domain after removing TLD.
            Returns: float (average frequency)
            """
            # Remove TLD (everything after first dot)
            main_domain = domain.split('.')[0]
            
            if not main_domain:  # Handle empty string case
                return 0.0
            
            # Calculate character frequencies
            char_counts = defaultdict(int)
            for char in main_domain:
                char_counts[char] += 1
            
            # Calculate average
            total_chars = sum(char_counts.values())
            unique_chars = len(char_counts)
            
            return total_chars / unique_chars if unique_chars > 0 else 0.0
        
        features["avg_char_distribution"] = calculate_avg_char_distribution(domain)
        # print("avg_char_distribution", features["avg_char_distribution"])

        # print("numeric percentage: ", features["numeric_percentage"])

        ######################
        
        # # Longest word
        # words = domain.split('.')
        # features["longest_word"] = max(words, key=len)
        
        # DNS query
        answers = dns.resolver.resolve(domain, 'A')
        ip_addresses = [str(rdata) for rdata in answers]
        features['IP'] = ip_addresses[0]
        # features['IP'] = ipp.replace('.', '', regex=False)

        # print("IP", features["IP"])

        # features["TTL"] = answers.rrset.ttl
         # Resolve A, AAAA, and CNAME records
        query_types = ["A", "AAAA", "CNAME"]
        resolver = dns.resolver.Resolver()

        for qtype in query_types:
            try:
                answers = resolver.resolve(domain, qtype)
                for answer in answers:
                    if qtype in ["A", "AAAA"]:  # IP address types
                        ip = str(answer)
                        if not ipaddress.ip_address(ip).is_private:  # Filter private IPs
                            # features["IP"] = ip
                            features["TTL"] = answers.rrset.ttl

                        # Get geolocation and ASN details
                        country, asn = get_geolocation(ip)
                        features["Country"] = country
                        features["ASN"] = asn
                    elif qtype == "CNAME":  # CNAME
                        features["Domain"] = str(answer).rstrip(".")
            except dns.resolver.NoAnswer:
                continue  # Skip if no answer for this type
            except dns.resolver.NXDOMAIN:
                print(f"Domain {domain} does not exist.")
                break
            except Exception as e:
                print(f"Error resolving {qtype} for {domain}: {e}")

        def domain_length(domain):
            # Split into parts
            parts = domain.split('.')
            # print(parts)
            
            # Exclude the SLD (last part) and join with dots
            domain_without_sld = '.'.join(parts[:-1])
            # print(domain_without_sld)
            # print(parts[:-1])
            
            return len("".join(parts[:-1]))+1
        
        # features["len"] = domain_length(domain)
        # print("len: ", domain_length(features["Domain"]))
        features["len"] = domain_length(features["Domain"])
        
        # WHOIS information
        whois_data = whois.whois(domain)
        features.update({
            "Domain_Name": whois_data.domain_name,
            "Registrar": whois_data.registrar,
            "Registrant_Name": whois_data.name,
            # "Creation_Date_Time": whois_data.creation_date[0].strftime('%d-%m-%Y %I.%M.%S %p'),
            "Emails": whois_data.emails,
            "Organization": whois_data.org,
            "State": whois_data.state,
            "Country.1": whois_data.country,
            # "Country" : whois_data.country,
            "Name_Server_Count": len(whois_data.name_servers) if whois_data.name_servers else 0
        })
        if(features["Domain_Name"] is None):
            features["Domain_Name"] = 0
        if(features["Registrar"] is None):
            features["Registrar"] = 0
        if(features["Emails"] is None):
            features["Emails"] = 0
        if(features["Registrant_Name"] is None):
            features["Registrant_Name"] = 0
        if(features["Organization"] is None):
            features["Organization"] = 0
        if(features["State"] is None):
            features["State"] = 0
        if(features["Country.1"] is None):
            features["Country.1"] = 0
        # Check and handle creation_date
        if whois_data.creation_date:
            if isinstance(whois_data.creation_date, list):  # If it's a list of dates
                creation_date = whois_data.creation_date[0]
            else:  # If it's a single datetime object
                creation_date = whois_data.creation_date
            
            features["Creation_Date_Time"] = creation_date.strftime('%d-%m-%Y %I.%M.%S %p')
            domain_age = datetime.now() - creation_date
            # features["Domain_Age"] = str(domain_age)
            features["Domain_Age"] = extract_days(str(domain_age))
        else:
            features["Creation_Date_Time"] = 0
            features["Domain_Age"] = 0

        def parse_creation_date_time(date_str):
            """Convert date string to Unix timestamp (seconds). Returns 0 if invalid."""
            if str(features["Creation_Date_Time"]).strip() in ("0", "0000-00-00 00:00:00"):
                return 0
            formats = [
                "%d-%m-%Y %I.%M.%S %p",  # e.g., "15-09-1997 4.00.00 AM"
                "%m-%d-%Y %I.%M.%S %p",  # e.g., "09-15-1997 4.00.00 AM"
                "%Y-%m-%d %H:%M:%S",      # e.g., "1997-09-15 04:00:00"
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(str(features["Creation_Date_Time"]).strip(), fmt)
                    return int(dt.timestamp())
                except ValueError:
                    continue
            return 0  # Fallback

        features["Creation_Date_Time"] = parse_creation_date_time(features["Creation_Date_Time"])
        
        # DNS packet capturing
        packet = IP(dst=domain) / UDP(dport=53) / DNS(rd=1, qd=DNSQR(qname=domain))
        response = sr1(packet, verbose=0, timeout=2)
        if response and response.haslayer(DNS):
            features["TTL.1"] = response[DNS].ancount  # Use answer count as TTL proxy
            
    except Exception as e:
        print(f"Error processing domain {domain}: {e}")

    # Change here
    for key, value in features.items():
        features[key] = str(value).replace(", ", " ")

    features['IP'] = str(features["IP"]).replace(".","")
    # print("IP after replace", features["IP"])

    # print("All features")

    features = {key: features[key] for key in cols if key in features}
    # print(features)

    return features

