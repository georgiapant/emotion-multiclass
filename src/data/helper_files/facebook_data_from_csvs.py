import os
import json
import codecs
from datetime import datetime
import matplotlib.pyplot as plt

import pandas as pd
import sys
import unicodedata
import six
import re
import emoji
import contractions
# from googletrans import Translator
from pathlib import Path
from pymongo import MongoClient

# import matplotlib.pyplot as plt
from collections import Counter

client = MongoClient('localhost', 27017)
db = client.emotion_index_database

activities_all_collection = db.emotions
queries_comments_reactions_collection = db.emotions

def get_reactions(usr_id):
    type_of_data = "comments_and_reactions"
    filename = "posts_and_comments"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    reactions = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'activity_type','on', 'session_id'])

    if Path(file).exists():
        with codecs.open(file, 'r', 'utf-8') as infile:
            data = json.load(infile)

        for i in range(len(data['reactions_v2'])):
            try:
                temp = {'user_id': usr_id,
                        'timestamp': data['reactions_v2'][i]['timestamp'],
                        'date_time': datetime.fromtimestamp(data['reactions_v2'][i]['timestamp']),
                        'content': data['reactions_v2'][i]['data'][0]['reaction']['reaction'],
                        'activity_type': 'fb_reaction',
                        'on': '',
                        'session_id': ''}
                reactions = reactions.append(temp, ignore_index=True)
            except:
                continue
    return reactions


def get_comments(usr_id):
    type_of_data = "comments_and_reactions"
    filename = "comments"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    comments = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'activity_type', 'on', 'session_id'])

    if Path(file).exists():
        with codecs.open(file, 'r', 'utf-8') as infile:
            data = json.load(infile)

        for i in range(len(data['comments_v2'])):
            try:

                temp = {'user_id': usr_id,
                        'timestamp': data['comments_v2'][i]['timestamp'],
                        'date_time': datetime.fromtimestamp(data['comments_v2'][i]['timestamp']),
                        'content': data['comments_v2'][i]['data'][0]['comment']['comment'],
                        'activity_type': 'fb_comment',
                        'on':'',
                        'session_id': ''}
                comments = comments.append(temp, ignore_index=True)
            except:
                continue
    return comments


def get_posts(usr_id):
    """
    There are several data that are not utilised yet. Such as attachment data
    """
    type_of_data = "posts"
    filename = "your_posts_1"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    posts = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'activity_type','on','session_id'])
    # print(len(data))

    if Path(file).exists():
        with codecs.open(file, 'r', 'utf-8') as infile:
            data = json.load(infile)

        for i in range(len(data)):

            try:

                temp = {'user_id': usr_id,
                        'timestamp': data[i]['timestamp'],
                        'date_time': datetime.fromtimestamp(data[i]['timestamp']),
                        'content': data[i]['data'][0]['post'],
                        'activity_type': 'fb_posts',
                        'on': '',
                        'session_id': ''}
                posts = posts.append(temp, ignore_index=True)
            except:
                continue
    return posts


def get_posts_group(usr_id):
    """
    There are several data that are not utilised yet. Such as attachment data
    """
    type_of_data = "groups"
    filename = "your_posts_in_groups"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")


    posts_group = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'activity_type','on', 'session_id'])
    # print(len(data['group_posts_v2']))

    if Path(file).exists():
        with codecs.open(file, 'r', 'utf-8') as infile:
            data = json.load(infile)
        # print(len(data['group_posts_v2']))

        for i in range(len(data['group_posts_v2'])):
            try:
                temp = {'user_id': usr_id,
                        'timestamp': data['group_posts_v2'][i]['timestamp'],
                        'date_time': datetime.fromtimestamp(data['group_posts_v2'][i]['timestamp']),
                        'content': data['group_posts_v2'][i]['data'][0]['post'],
                        'activity_type': 'fb_post_groups',
                        'on': '',
                        'session_id': ''}
                posts_group = posts_group.append(temp, ignore_index=True)
            except:
                continue

    return posts_group


def get_comments_group(usr_id):
    """
    There are several data that are not utilised yet. Such as attachment data
    """
    type_of_data = "groups"
    filename = "your_comments_in_groups"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")

    comments_group = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'activity_type','on', 'session_id'])

    if Path(file).exists():
        with codecs.open(file, 'r', 'utf-8') as infile:
            data = json.load(infile)

        for i in range(len(data['group_comments_v2'])):
            try:
                temp = {'user_id': usr_id,
                        'timestamp': data['group_comments_v2'][i]['timestamp'],
                        'date_time': datetime.fromtimestamp(data['group_comments_v2'][i]['timestamp']),
                        'content': data['group_comments_v2'][i]['data'][0]['comment']['comment'],
                        'activity_type': 'fb_comment_groups',
                        'on': '',
                        'session_id': ''}
                comments_group = comments_group.append(temp, ignore_index=True)
            except:
                # print("one did not have data")
                # print(data['group_comments_v2'][i])
                continue

    return comments_group


def get_searches(usr_id):
    type_of_data = "search"
    filename = "your_search_history"

    path = r"D:\CERTH\REBECCA\WP3\Data\FACEBOOK\ADC_PRT_" + usr_id + "_Facebook"
    file = os.path.join(path, type_of_data, filename + ".json")
    searches = pd.DataFrame(columns=['user_id', 'timestamp', 'date_time', 'content', 'activity_type', 'on', 'session_id'])

    if Path(file).exists():
        with codecs.open(file, 'r', 'utf-8') as infile:
            data = json.load(infile)

        for i in range(len(data['searches_v2'])):
            try:
                temp = {'user_id': usr_id,
                        'timestamp': data['searches_v2'][i]['timestamp'],
                        'date_time': datetime.fromtimestamp(data['searches_v2'][i]['timestamp']),
                        'content': data['searches_v2'][i]['data'][0]['text'],
                        'activity_type': 'fb_query',
                        'on': '',
                        'session_id': ''}
                searches = searches.append(temp, ignore_index=True)
            except:
                # print("one did not have data")
                # print(data['searches_v2'][i])
                continue

    return searches


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding='utf-8', errors='strict'):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")


def run_strip_accents(text):
    """
    Strips accents from a piece of text.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def text_preprocess(text):
    text = re.sub("[\xc2-\xf4][\x80-\xbf]+", lambda m: m.group(0).encode('latin1').decode('utf8'), text)
    text = convert_to_unicode(text.rstrip().lower())
    text = run_strip_accents(text)

    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(r" +", ' ', text)

    text = text.lower()
    # Demojize
    text = emoji.demojize(text)

    # Expand contraction
    text = contractions.fix(text)

    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = str(re.sub("\S*\d\S*", "", text).strip())

    return text


def main():
    # usr = "02"
    usr = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18",
           "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]

    activities_all = pd.DataFrame()
    for usr_id in usr:

        reactions = get_reactions(usr_id)
        comments = get_comments(usr_id)
        posts = get_posts(usr_id)
        posts_groups = get_posts_group(usr_id)
        comments_groups = get_comments_group(usr_id)
        searches = get_searches(usr_id)

        activities_all = pd.concat([activities_all, comments, comments_groups, posts, posts_groups, searches, reactions])

    # activities_all.drop(columns=['date_time'], inplace=True)
    # activities_all_limited_info = activities_all[['session_id', 'timestamp', 'user_id','activity_type']]
    # print(activities_all)
    # print(activities_all_limited_info)

    # Load to mongoDB
    # activities_all_limited_info.reset_index(inplace=True)
    # activities_all_limited_info.drop(columns=['index'], inplace=True)
    # activities_all_for_collection = json.loads(activities_all_limited_info.T.to_json()).values()
    # db.activities_all_collection.insert_many(activities_all_for_collection)
    #
    # activities_all.reset_index(inplace=True)
    # activities_all.drop(columns=['index'], inplace=True)
    # queries_comments_reactions_for_collection = json.loads(activities_all.T.to_json()).values()
    # db.queries_comments_reactions_collection.insert_many(queries_comments_reactions_for_collection)


    """
    # Process and translate search, posts and comments text - since they are in Swedish ################################
    
    posts_comments = pd.concat([comments, comments_groups, posts,posts_groups])
    posts_comments['processed'] = posts_comments.apply(lambda row: text_preprocess(row['content']), axis=1)
    
    translator = Translator()
    posts_comments['translated'] = posts_comments.apply(lambda row: translator.translate(row['processed']).text, axis=1)

    # path = r"D:\CERTH\REBECCA\WP3\Data\processed_data\FACEBOOK\posts_comments"
    # posts_comments.to_csv(os.path.join(path, usr+'_posts_comments_translated.csv'))
    # translator = Translator()
    searches['preprocessed'] = searches.apply(lambda row: text_preprocess(row['content']), axis=1)
    searches['translated'] = searches.apply(lambda row: translator.translate(row['preprocessed']).text, axis=1)
    ####################################################################################################################
    """


if __name__ == '__main__':
    main()
