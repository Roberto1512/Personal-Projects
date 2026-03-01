import os

from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

def geturl(url):
    r = requests.get(url)
    return r.text
html_page = geturl('https://www.dni.gov/nctc/ftos.html')
soup = BeautifulSoup(html_page, 'html.parser')
url = 'https://www.dni.gov/nctc/'
for item in soup.findAll('img'):
    abs_url = urljoin(url, item['src'])
    response = requests.get(abs_url)

    img_name = os.path.basename(item['src'])
    if response.status_code == 200:
        with open(f'image_{img_name}', 'wb') as f:
            f.write(response.content)

