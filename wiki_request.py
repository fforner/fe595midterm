import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_wiki(link):
    r = requests.get(link)
    soup = BeautifulSoup(r.text)

    content = []

    if soup.find("div", {"class": "toclimit-3"}) is not None:
        tag = soup.find("div", {"class": "toclimit-3"})
    else:
        tag = soup.find("div", {"class": "toc"})

    for h2 in tag.findAllNext(['h2','h3']):
        if 'h2' in h2.name:
            h2text = h2.text
            pass
        elif 'h3' in h2.name:
            for h3 in h2.findAllNext(['p','h3']):
                if 'h3' in h3.name:
                    break
                elif 'p' in h3.name:
                    content.append((h2.text,h3.text))

    df = pd.DataFrame(content,columns=['Topic','text'])
    return df