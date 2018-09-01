import pandas as pd
import sqlite3
import time

connect = sqlite3.connect("/home/renhao/github/danmu.db")
cursor = connect.cursor()
query_danmu = "select aid, content from filtered_danmu_without_sc"
results = cursor.execute(query_danmu)

for result in results:
    
