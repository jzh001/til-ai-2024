import pandas as pd
from PIL import Image
import re
from tqdm import tqdm

file_path = "../../../../advanced/vlm.jsonl"

image_dir = '../../../../advanced/images/'

def createJSONL():
    df = pd.read_json(path_or_buf=file_path, lines=True)
    new_df = pd.DataFrame([sample for sample in tqdm(dataset_generator(df))])
    new_df.to_json('vlm_od.jsonl',orient='records', lines=True)
    
def createCroppedFolder():
    df = pd.read_json(path_or_buf=file_path, lines=True)
    new_df = pd.DataFrame([sample for sample in tqdm(cropped_dataset_gen(df))])
    new_df.to_json('vlm_caption_expanded-10.jsonl',orient='records', lines=True)


def imageFolderDatasetGenerator():
    df = pd.read_json(path_or_buf=file_path, lines=True)
    label2id, _ = get_mappings(df)
    metadata = []
    for i,row in tqdm(df.iterrows()):
        image_path = image_dir + row['image']
        im = Image.open(image_path)
        image_id = int(row['image'].split('_')[-1].split('.')[0])
        new_path = f'./imagefolder/train/{image_id}.jpg'
        im.save(new_path)
        metadata.append({
            'file_name':new_path,
            'image_id': image_id,
            'category': [label2id[word_tokenize(a['caption'])[-1]] for a in row['annotations']],
            'caption': [a['caption'] for a in row['annotations']],
            'area': [calculate_area(a['bbox']) for a in row['annotations']],
            'id': [i1 for i1,_ in enumerate(row['annotations'])],
            'bbox': [trim_bounding_box(im.size[0],im.size[1],tuple(a['bbox'])) for a in row['annotations']],
        })
    metadata = pd.DataFrame(metadata)
    metadata.to_json('./imagefolder/metadata.jsonl',orient='records', lines=True)
    df = pd.read_json('imagefolder/metadata.jsonl', lines = True)
    df['objects'] = [{
        'text': row['caption'],
        'area': row['area'],
        'category': row['category'],
        'id': row['id'],
        'bbox': row['bbox'],
    } for i,row in df.iterrows()]
    df = df[['file_name','image_id','objects']]

    df.to_json('imagefolder/train/metadata.jsonl',orient='records', lines=True)
    
    df = pd.read_json('imagefolder/train/metadata.jsonl', lines = True)
    df['file_name'] = [f'{image_id}.jpg' for image_id in df['image_id']]
    df.to_json('imagefolder/train/metadata.jsonl',orient='records', lines=True)
    df


def cropped_dataset_gen(dataframe):
    for i,row in dataframe.iterrows():
        image_path = image_dir + row['image']
        image = Image.open(image_path)
        
        for j,a in enumerate(row['annotations']):
            example = {}
            bbox = trim_bounding_box(image.size[0],image.size[1],tuple(a['bbox']))
            x,y,w,h = expand_bbox(image.size[0],image.size[1],bbox)
            im = image.crop((x,y,x+w,y+h))
            annotation_id = 20*i+j
            im_path = f"./cropped_expanded-10/{annotation_id}.jpg"
            im.save(im_path)
            example['image_path'] = im_path
            example['image_id'] = annotation_id
            example['original_id'] = int(row['image'].split('_')[-1].split('.')[0])
            example['caption'] = a['caption']
            yield example

def expand_bbox(w_max, h_max, bbox, units=10):
    x,y,w,h = bbox
    x2 = x + w
    y2 = y + h
    x = max(0, x-units)
    y = max(0, y-units)
    x2 = min(w_max-1, x2+units)
    y2 = min(h_max-1, y2+units)
    return (x,y,x2-x,y2-y)
    
    
def dataset_generator(dataframe, image_column = 'image'):
    label2id, _ = get_mappings(dataframe)
    for i,row in dataframe.iterrows():
        example = {}
        image_path = image_dir + row['image']
        example[image_column] = image_path
        image = Image.open(image_path)
        example['image_id']= int(row['image'].split('_')[-1].split('.')[0])
        example['objects'] = { # annotations
            'id': [i1 for i1,_ in enumerate(row['annotations'])],
            'bbox': [trim_bounding_box(image.size[0],image.size[1],tuple(a['bbox'])) for a in row['annotations']],
            #'bbox': [a['bbox'] for a in row['annotations']],
            'area': [calculate_area(a['bbox']) for a in row['annotations']],
            'category': [label2id[word_tokenize(a['caption'])[-1]] for a in row['annotations']]
        }
        
        yield example


def trim_bounding_box(image_width, image_height, bbox):
    x_min, y_min, w, h = bbox
    
    # Ensure x_min and y_min are not less than 0
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    
    # Ensure x_max and y_max are not greater than image width and height respectively
    w = min(w,image_width-x_min)
    h = min(h, image_height-y_min)
    
    # Ensure x_min is not greater than x_max and y_min is not greater than y_max
    x_min = min(x_min, x_min + w)
    y_min = min(y_min, y_min + h)

    return x_min, y_min, w, h
        
def calculate_area(bbox):
    x,y,w,h = bbox
    return w*h

def get_mappings(df= pd.read_json(path_or_buf=file_path, lines=True)):
    categories = []
    for i,row in df.iterrows():
        for a in row['annotations']:
            categories.append(word_tokenize(a['caption'])[-1])

    categories = set(categories)
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}
    return label2id, id2label

def word_tokenize(sent):
    return [x for x in re.split('[., ]', sent) if x != '']


