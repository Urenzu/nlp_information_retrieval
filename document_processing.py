import os
import re

def find_total_relatives(path, query):
    """
    This function is used to get the number of relative article in data collection
    """
    total_count = 0
    query = query.lower()

    for j in range(22):
        rem = j % 10
        mult = j // 10
        file_path = f"{path}/reut2-0{mult}{rem}.sgm"

        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='iso-8859-1') as f:
            content = f.read()
            articles = re.findall(r"<REUTERS.*?>.*?</REUTERS>", content, re.DOTALL)

            for article in articles:
                if query in article.lower():
                    total_count += 1

    return total_count

def find_article(path, ID):
    """
    This function is used to return the article context depends on the ID
    """
    ID = str(ID)

    for j in range(22):
        rem = j % 10
        mult = j // 10
        file_path = f"{path}/reut2-0{mult}{rem}.sgm"

        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='iso-8859-1') as f:
            content = f.read()
            articles = re.findall(r"<REUTERS.*?>.*?</REUTERS>", content, re.DOTALL)

            for article in articles:
                match = re.search(r'NEWID="(\d+)"', article)
                if match and match.group(1) == ID:
                    body_match = re.search(r"<BODY>(.*?)</BODY>", article, re.DOTALL)
                    if body_match:
                        return body_match.group(1).strip()
                    return None
    return None