import re
import enchant
import wordninja
# import splitter


d = enchant.Dict('en_UK')
dus = enchant.Dict('en_US')
space_pattern = '\s+'
giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_regex = '@[\w\-]+'
emoji_regex = '&#[0-9]{4,6};'

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

#This is the original preprocess method from Davidson
def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub('RT','', parsed_text) #Some RTs have !!!!! in front of them
    parsed_text = re.sub(emoji_regex,'',parsed_text) #remove emojis from the text
    parsed_text = re.sub('…','',parsed_text) #Remove the special ending character is truncated
    parsed_text = re.sub('#[\w\-]+', '',parsed_text)
    # parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text


def preprocess_clean(text_string, remove_hashtags=True, remove_special_chars=True):
    # Clean a string down to just text
    # text_string=preprocess(text_string)

    parsed_text = preprocess(text_string)
    # parsed_text = parsed_text.lower()
    parsed_text = re.sub('\'', '', parsed_text)
    parsed_text = re.sub('|', '', parsed_text)
    parsed_text = re.sub(':', '', parsed_text)
    parsed_text = re.sub(',', '', parsed_text)
    parsed_text = re.sub('/', ' ', parsed_text)
    parsed_text = re.sub("\*", '', parsed_text)
    parsed_text = re.sub(';', '', parsed_text)
    parsed_text = re.sub('\.', '', parsed_text)
    parsed_text = re.sub('&amp', '', parsed_text)
    parsed_text = re.sub('ð', '', parsed_text)

    if remove_hashtags:
        parsed_text = re.sub('#[\w\-]+', '', parsed_text)
    if remove_special_chars:
        # parsed_text = re.sub('(\!|\?)+','.',parsed_text) #find one or more of special char in a row, replace with one '.'
        parsed_text = re.sub('(\!|\?)+','',parsed_text)
    return parsed_text

def strip_hashtags(text):
    text = preprocess_clean(text,False,False)
    # hashtags = re.findall('#[\w\-]+', text)
    # for tag in hashtags:
    #     cleantag = tag[1:]
    #     if d.check(cleantag) or dus.check(cleantag):
    #         text = re.sub(tag, cleantag, text)
    #         pass
    #     else:
    #         hashtagSplit = ""
    #         for word in wordninja.split(cleantag):
    #             hashtagSplit = hashtagSplit + word + " "
    #         text = re.sub(tag, hashtagSplit, text)
    # print(text)
    return text

