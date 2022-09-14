import os
import urllib.request
from tqdm import tqdm

actors_dir = './data/laion_german_non_members/actors/'
actresses_dir = './data/laion_german_non_members/actresses/'

def read_actor_files(folder_path):
    urls = {}
    for csv_file in os.listdir(folder_path):
        if not csv_file.endswith('.txt'):
            continue

        file_name_without_ext = os.path.splitext(csv_file)[0]
        with open(os.path.join(folder_path, csv_file)) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        
        urls[file_name_without_ext] = lines

    return urls

def save_images_to_folder(folder_path, url_dict):
    url_opener = urllib.request.URLopener()
    url_opener.addheader('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36')

    for name, url_list in tqdm(url_dict.items()):
        base_folder = os.path.join(folder_path, name)
        if os.path.exists(base_folder):
            print(f'The image folder {base_folder} already exists. Skipping folder.')
        os.makedirs(base_folder)
        for i, url in tqdm(enumerate(url_list), desc=name, leave=False):
            url = urllib.parse.quote(url, safe='://?=&(),%+')
            url_opener.retrieve(url, os.path.join(base_folder, f'{name}_{i}.jpg'))

actor_urls = read_actor_files(actors_dir)
save_images_to_folder(os.path.join(actors_dir, 'images'), actor_urls)
actresses_urls = read_actor_files(actresses_dir)
save_images_to_folder(os.path.join(actresses_dir, 'images'), actresses_urls)

