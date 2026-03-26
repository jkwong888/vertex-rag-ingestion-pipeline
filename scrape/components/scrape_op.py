from kfp import dsl

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "scrapy",
        "google-cloud-storage",
        "google-crc32c",
    ]
)
def scrape_bulbapedia_op(gcs_bucket_name: str, gcs_bucket_path: str):
    import os
    import base64
    import datetime
    from google.cloud import storage
    from scrapy.spiders import CrawlSpider, Rule
    from scrapy.linkextractors import LinkExtractor
    from scrapy.crawler import CrawlerProcess
    import google_crc32c
    
    START_URL = "https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number"
    DOWNLOAD_REGEX = r'wiki/.*_\(Pok%C3%A9mon\)'
    ALLOWED_DOMAINS = 'bulbapedia.bulbagarden.net'

    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)

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
            page = response.url.split("/")[-1].split("?")[0]
            response_bytes = response.body
            scrape_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%Z")

            try:
                blob_name = f"{gcs_bucket_path}/{page}.html"
                local_crc32c = base64.b64encode(google_crc32c.Checksum(response_bytes).digest()).decode("utf-8")
                existing_blob = bucket.get_blob(blob_name)

                if existing_blob and existing_blob.crc32c == local_crc32c:
                    self.log(f"Skipping upload for {blob_name} (CRC32C match: {local_crc32c})")
                else:
                    blob = bucket.blob(blob_name)
                    blob.metadata = {'original_url': response.url}
                    blob.metadata['scrape_time'] = scrape_time

                    blob.upload_from_string(response_bytes, content_type='text/html')
                    self.log(f"Uploaded raw HTML {blob_name} to gs://{gcs_bucket_name}/{blob_name}")
            except Exception as e:
                self.log(f"Failed to upload to GCS: {e}")

    process = CrawlerProcess()
    process.crawl(WebSpider)
    process.start()
