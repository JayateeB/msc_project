from twitter.crawl import TwitterCrawlManager
from twitter.crawl_friends import FriendsCrawler
from db.database import Database
from db.data_access import DataAccess


def main():
    _db = Database("app.db")
    data_access = DataAccess(db=_db)
    crawl = TwitterCrawlManager()
    friends_crawler = FriendsCrawler(crawl=crawl, data_access=data_access)
    friends_crawler.extract_friends("less_than_5000_infctd.csv")
    data_access.generate_dataset("less_than_5000_infctd_out.csv")


if __name__ == '__main__':
    main()
