import requests
from bs4 import BeautifulSoup

URL = "https://www.today.com/food/news"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
}


def news_parse():
    response = requests.get(URL, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")

    items = soup.find("div", class_="styles_itemsContainer__saJYW").findChildren(
        "div", recursive=False
    )

    comps = []
    for item in items:
        link = item.findChildren("div", recursive=False)[1].findChild(
            "a", recursive=False
        )
        comps.append(
            {
                "title": link.findChild("h2", recursive=False).get_text(strip=True),
                "date": item.findChildren("div", recursive=False)[0]
                .findChildren("div", recursive=False)[1]
                .get_text(strip=True),
                "link": link.get("href"),
            }
        )
    return comps
