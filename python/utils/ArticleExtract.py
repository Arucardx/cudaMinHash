import os
import codecs
import bz2
import wikitextparser as wtp
import regex as rx
from xml.etree import ElementTree

class ExtractorConfig:
    def __init__(self, ignore_patterns = [], ignore_section_names = [], minimal_section_length = 1):
        self.ignore_patterns = ignore_patterns
        self.ignore_section_names = ignore_section_names
        self.minimal_section_length = minimal_section_length
    
    def __str__(self):
        return f"""
            ignore patterns: {[str(pattern) for pattern in self.ignore_patterns]}
            ignore section naems: {[str(section_name) for section_name in self.ignore_section_names]}
            minimal section-length: {self.minimal_section_length}
        """
    
    def get_default_config():
        return ExtractorConfig(
            ignore_patterns=[
                rx.compile(r'{\|(.|\n)*?\|}'),
                rx.compile(r'<gallery>?(.|\n)*?(</gallery>|/>)'),
                rx.compile(r'<ref>?(.|\n)*?(</ref>|/>)'),
                #rx.compile(r'\n\*\s*(.*?)$')
            ],
            ignore_section_names=['Siehe auch', 'Literatur', 'Weblinks', 'Einzelnachweise', 'Quellen'],
            minimal_section_length=100
        )
        


class Extract:

    def __init__(self, input_path: str, extractor_config: ExtractorConfig = None):
        self.input_path = input_path

        if extractor_config is None:
            extractor_config = ExtractorConfig()

        self.section_title_rx = rx.compile(r'=+\s?(.*?)\s?=+')

        self.ignore_patterns = extractor_config.ignore_patterns
        self.minimal_section_length = extractor_config.minimal_section_length
        self.set_section_filter(extractor_config.ignore_section_names)

    
    def set_section_filter(self, ignored_section_names):
        self.ignore_section_names = ignored_section_names
        rgx = r'\s*=+\s*(' + r'|'.join(f'({topic})' for topic in self.ignore_section_names) + r')\s*=+'
        self.ignore_sections_rx = rx.compile(rgx)


    def substitute_rx(self, page_content):
        for rx_s in self.ignore_patterns:
            page_content = rx_s.sub('', page_content)
        return page_content

    
    def extract_pages(self, store_func, text_transform_func, limit = None):
        article_count, section_count = 0, 0
        for file in os.listdir(self.input_path):
            file = bz2.BZ2File(os.path.join(self.input_path, file))
            reader = codecs.getreader('utf-8')(file)
            context = ElementTree.iterparse(reader)
            for _, elm in context:
                # only extract pages
                if elm.tag.endswith('page'):
                    # extract namespace
                    ns = elm.tag[:-4]

                    # articles have namespace 0
                    if elm.find(f'./{ns}ns').text != '0':
                        #elm.clear()
                        continue

                    redirect = elm.find(f'./{ns}redirect')
                    rev = elm.find(f'./{ns}revision')

                    # ignore redirects                        
                    if redirect is not None or rev is None:
                        #elm.clear()
                        continue

                    # some elements have no content (?)
                    txt = rev.find(f'./{ns}text')
                    if txt is None or txt.text is None:
                        #elm.clear()
                        continue

                    article_count += 1

                    id = elm.find(f'./{ns}id').text
                    title = elm.find(f'./{ns}title').text

                    # preprocess whole article
                    text = self.substitute_rx(txt.text)

                    parsed = wtp.parse(text)

                    # parse sections
                    for i, section in enumerate(parsed.get_sections()):
                        section_text = section.plain_text()
                        
                        # ignore specific sections based on the title 
                        if bool(self.ignore_sections_rx.match(section_text)):
                            continue
                        
                        # replace whitespaces
                        section_text = rx.sub(r'\s+', ' ', section_text)

                        section_title = self.section_title_rx.match(section_text)
                        if section_title is not None:
                            section_title = section_title.groups()[0]
                            section_text = self.section_title_rx.sub('', section_text)
                        else:
                            section_title = 'Header'
                            
                        if len(section_text) < self.minimal_section_length:
                            continue

                        #!!! match further subsections (;abc)

                        meta = [article_count << 9 | i, title, section_title]
                        section_text = text_transform_func(section_text)

                        section_count += 1
                        store_func(meta, section_text)

                        if limit is not None and section_count >= limit:
                            return

                #elm.clear()
    

    def decompress(self, out_path):
        for file in os.listdir(self.input_path):
            if not '.bz2' in file:
                continue
        out_file = os.path.join(out_path, file.split('.bz2')[0])
        in_file = os.path.join(self.input_path, file)
        with open(out_file, 'wb') as out_fd, open(in_file, 'rb') as in_fd:
            decompressor = bz2.BZ2Decompressor()
            for data in iter(lambda: in_fd.read(1024 * 1024), b''):
                out_fd.write(decompressor.decompress(data))