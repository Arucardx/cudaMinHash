import regex as rx

class Stopword:

    def __init__(self, remove_words: list[str] = None, replace_chars: list[tuple[str]] = None):

        if replace_chars is not None:
            self.replacement_chars = {tup[0]: tup[1] for tup in replace_chars}
        # regex should be way faster than iterating, especially if we use it for many texts
        if remove_words is not None:
            self.remove_rx = rx.compile('(' + ')|('.join(remove_words) + ')', rx.IGNORECASE)
        else:
            self.remove_rx = None
    
    def text_to_bytes(self, text: str) -> bytes:
        text = text.lower()
        text = text.replace('\'', '')
        text_bytes = []

        for c in text:
            if 0 <= ord(c) <= 127:
                #ascii
                text_bytes.append(ord(c))
                pass
            else:
                #non ascii
                if c == 'ä':
                    text_bytes.append(0b10000000)
                elif c == 'ö':
                    text_bytes.append(0b10000001)
                elif c == 'ü':
                    text_bytes.append(0b10000010)
                elif c == 'ß':
                    text_bytes.append(0b10000011)
                elif c == 'ó':
                    text_bytes.append(0b10000100)
                elif c == 'á':
                    text_bytes.append(0b10000101)
                else:
                    text_bytes.append((1 << 7) | (ord(c) % 127))

        return bytes(text_bytes)