import time
import pandas
from bs4 import BeautifulSoup
from selenium import webdriver as wdd

wd = wdd.Edge()
topics = ["koran", 'kuran']
outset = []
# Making a GET request
for topic in topics:
    wd.get(f"https://www.quora.com/search?q={topic}&type=question")
    SCROLL_PAUSE_TIME = 2.5

    # Get scroll height
    last_height = wd.execute_script("return document.body.scrollHeight")
    sax = 0
    while sax < 100:
        sax += 1
        time.sleep(3)
        # Scroll down to bottom
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = wd.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

        x = wd.page_source
        soup = BeautifulSoup(x, 'html.parser')
        s = soup.find_all('span', class_='q-box qu-userSelect--text')
        for sa in s:
            content = sa.find_all('span')
            t = ""
            for line in content:
                t += line.text
            print(t)
            outset.append(str(t))

with open("BabaO.csv", 'a', encoding="utf-8", newline="") as o:
    p = pandas.DataFrame(outset)
    p.to_csv(o, encoding="utf-8", sep='`')
