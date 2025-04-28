import pandas as pd
import re


# Load the blacklist file into memory when the application starts
try:
    df = pd.read_csv("whitelist.csv", usecols=["Domain"])
    domain_set = set(df["Domain"])  # Convert to a set for O(1) lookups
except FileNotFoundError:
    raise RuntimeError(f"File 'list.csv' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading CSV file: {e}")





def in_whitelist(domain):
    # Remove the "www." prefix if it exists
    if domain.split(".")[0] == "www":
        domain = ".".join(domain.split(".")[1:])

    # Check if the domain exists in the set
    result = domain in domain_set

    return result