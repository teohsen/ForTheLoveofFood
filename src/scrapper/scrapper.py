import requests
from bs4 import BeautifulSoup
import bs4
import urllib3

url = "https://coinmarketcap.com/currencies/ethereum/news/"


page = requests.get(url)


soup = BeautifulSoup(page.content, 'html.parser')

soup.prettify("utf-8")
print(soup.prettify())

list(soup.children)

[type(item) for item in list(soup.children)]

[bs4.element.Doctype, bs4.element.NavigableString, bs4.element.Tag]


