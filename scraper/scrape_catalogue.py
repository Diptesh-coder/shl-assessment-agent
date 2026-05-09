import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.shl.com"

CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

# -----------------------------------
# STEP 1 — Load Catalog Page
# -----------------------------------

response = requests.get(CATALOG_URL, headers=headers)

print("Status Code:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

print("Page Title:", soup.title)

# -----------------------------------
# STEP 2 — Extract Assessment URLs
# -----------------------------------

links = soup.find_all("a")

print(f"Total Links Found: {len(links)}")

assessment_links = []

for link in links:

    href = link.get("href")

    if href and "/products/product-catalog/view/" in href:

        # Create full URL
        if href.startswith("http"):
            full_url = href
        else:
            full_url = BASE_URL + href

        # Avoid duplicates
        if full_url not in assessment_links:
            assessment_links.append(full_url)

print(f"\nAssessment URLs Found: {len(assessment_links)}\n")

# Print first few URLs
for url in assessment_links[:5]:
    print(url)

# -----------------------------------
# STEP 3 — Scrape Assessment Details
# -----------------------------------

assessment_data = []

for url in assessment_links[:10]:

    print(f"\nScraping: {url}")

    try:

        page = requests.get(url, headers=headers)

        page_soup = BeautifulSoup(page.text, "html.parser")

        # -----------------------------------
        # Title
        # -----------------------------------

        title = page_soup.find("h1")

        title_text = (
            title.get_text(strip=True)
            if title else "N/A"
        )

        # -----------------------------------
        # Description
        # -----------------------------------

        description = page_soup.find(
            "meta",
            attrs={"name": "description"}
        )

        description_text = (
            description.get("content")
            if description
            else "N/A"
        )

        # -----------------------------------
        # Full Page Text
        # -----------------------------------

        page_text = page_soup.get_text(
            " ",
            strip=True
        )

        # -----------------------------------
        # Duration Extraction
        # -----------------------------------

        duration = "N/A"

        if "Approximate Completion Time" in page_text:

            try:

                duration = (
                    page_text.split(
                        "Approximate Completion Time"
                    )[1][:50]
                )

            except:
                pass

        # -----------------------------------
        # Remote Testing
        # -----------------------------------

        remote_testing = (
            "Yes"
            if "Remote Testing" in page_text
            else "No"
        )

        # -----------------------------------
        # Adaptive / IRT Support
        # -----------------------------------

        adaptive_support = (
            "Yes"
            if "Adaptive" in page_text
            else "No"
        )

        # -----------------------------------
        # Store Data
        # -----------------------------------

        assessment_data.append({

            "title": title_text,

            "url": url,

            "description": description_text,

            "duration": duration,

            "remote_testing": remote_testing,

            "adaptive_support": adaptive_support

        })

        print("Title:", title_text)

        print("Duration:", duration)

        print("Remote Testing:", remote_testing)

        print("Adaptive Support:", adaptive_support)

        # Delay to avoid rate limiting
        time.sleep(1)

    except Exception as e:

        print(f"Error scraping {url}")

        print(e)

# -----------------------------------
# STEP 4 — Create DataFrame
# -----------------------------------

df = pd.DataFrame(assessment_data)

print("\nData Preview:\n")

print(df.head())

# -----------------------------------
# STEP 5 — Save CSV
# -----------------------------------

df.to_csv("shl_assessments.csv", index=False)

print("\nCSV Saved Successfully")

print("File Name: shl_assessments.csv")