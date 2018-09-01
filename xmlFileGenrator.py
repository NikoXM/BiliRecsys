import sqlite3

def get_videos_list():
    return ['']

if __name__=='__main__':
    connect = sqlite3.connect('danmu.db')
    cursor = connect.cursor()
    videos_list = get_videos_list()
    query_for_danmu_sql = "select "
    for video_aid in videos_list:
        
