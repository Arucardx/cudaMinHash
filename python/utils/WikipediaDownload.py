import requests
import wget
from bs4 import BeautifulSoup

class WikipediaDownload:

    def __init__(self, year, month, language='de', output_path='./downloads'):
        self.output_path = output_path
        self.url = self._generate_url(year, month, language)

    def _generate_url(self, year, month, language):
        url = 'https://dumps.wikimedia.org/other/mediawiki_content_current/{WIKI}/{DATE}/xml/bzip2/'
        year = str(year) if year >= 2000 else f'20{year}'
        month = str(month) if month >= 10 else f'0{month}'
        return url.replace('{DATE}', f'{year}-{month}-01').replace('{WIKI}', f'{language}wiki')

    def _build_download_links(self):
        response = requests.get(self.url, timeout=1000)
        content = BeautifulSoup(response.content.decode('utf-8'))
        links = []
        for file in content.find_all('a'):
            if file.text == 'SHA256SUMS':
                links.append(f'{self.url}{file.text}')
            elif '.xml.bz2' in file.text:
                links.append(f'{self.url}{file.text}')
        return links
    
    
    def download_files(self, include_checksum = False):
        links = self._build_download_links()
        for link in links:
            print(f'start downloading {link}')
            wget.download(link, self.output_path)
            print('finished')