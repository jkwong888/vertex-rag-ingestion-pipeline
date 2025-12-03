import os
import scrapy
import base64
import google_crc32c
from google.cloud import storage
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess

from markdownify import markdownify

START_URL = "https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number"
DOWNLOAD_REGEX = r'wiki/.*_\(Pok%C3%A9mon\)'
ALLOWED_DOMAINS = 'bulbapedia.bulbagarden.net'
GCS_BUCKET_NAME = 'jkwng-vertex-experiments'
GCS_BUCKET_PATH = 'pokemon/bulbapedia/html'

# Initialize GCS client and bucket globally
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

class WebSpider(CrawlSpider):
    name = "bulbapedia"
    start_urls = [
        START_URL,
    ]

    rules = (
        Rule(
            LinkExtractor(
                allow=DOWNLOAD_REGEX,
                allow_domains=ALLOWED_DOMAINS,
                deny=r'wiki/.*:.*',
            ),            
            callback="parse_page", 
            follow=True
        ),
    )

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'DEPTH_LIMIT': 2,
    }

    def parse_page(self, response):
        # drop the first part of the path (e.g. /wiki)
        page = response.url.split("/")[-1].split("?")[0]
        response_bytes = response.body

        try:
            blob_name = f"{GCS_BUCKET_PATH}/{page}.html"

            # Calculate local CRC32C
            local_crc32c = base64.b64encode(google_crc32c.Checksum(response_bytes).digest()).decode("utf-8")

            # Check existing blob
            existing_blob = bucket.get_blob(blob_name)

            if existing_blob and existing_blob.crc32c == local_crc32c:
                self.log(f"Skipping upload for {blob_name} (CRC32C match: {local_crc32c})")
            else:
                blob = bucket.blob(blob_name)
                blob.metadata = {'original_url': response.url}
                blob.upload_from_string(response_bytes, content_type='text/html')
                self.log(f"Uploaded raw HTML {blob_name} to GCS path gs://{GCS_BUCKET_NAME}/{blob_name}")
        except Exception as e:
            self.log(f"Failed to upload to GCS: {e}")

if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(WebSpider)
    process.start()