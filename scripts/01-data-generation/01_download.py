import base64
import hashlib

from icrawler import ImageDownloader
from icrawler.builtin import GoogleImageCrawler
from six.moves.urllib.parse import urlparse


EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tiff", "gif", "ppm", "pgm"]


def short_url_hash(url_path):
    """See https://stackoverflow.com/a/2510733"""
    hasher = hashlib.sha1(url_path.encode())
    return base64.urlsafe_b64encode(hasher.digest()[:10]).decode()


class MyImageDownloader(ImageDownloader):
    """See https://github.com/hellock/icrawler/issues/34#issuecomment-328320534"""

    def get_filename(self, task, default_ext):
        url_path = urlparse(task["file_url"])[2]
        if "." in url_path:
            extension = url_path.split(".")[-1]
            if extension.lower() not in EXTENSIONS:
                extension = default_ext
        else:
            extension = default_ext
        filename = short_url_hash(url_path)
        return "{}.{}".format(filename, extension)


google_crawler = GoogleImageCrawler(
    downloader_cls=MyImageDownloader,
    downloader_threads=4,
    storage={"root_dir": "images"},
)
# google_crawler.crawl(keyword="window", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="window picture", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="window photo", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="window shutters", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="window curtains", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="awning window", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="bay window", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="bow window", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="double hung window", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="egress window", max_num=50, min_size=(1000, 1000))
# google_crawler.crawl(keyword="fixed window", max_num=50, min_size=(1000, 1000))
google_crawler.crawl(keyword="blank paper", max_num=50)
