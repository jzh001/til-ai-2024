'''REQUIREMENTS:
torch==2.3.0
torchvision==0.18.0

'''

# with open('models.txt','r') as f:
#     lines = f.read().split('\n')
#     od_model_path = f'auto_od/{lines[0]}'
#     caption_model_path = f'auto_caption/{lines[1]}'

from torch import tensor, no_grad, cuda
from torch import minimum as torch_min
from torch import maximum as torch_max
import torch
import torchvision.ops as ops
from transformers import VisionTextDualEncoderConfig, VisionTextDualEncoderModel, AutoTokenizer, AutoImageProcessor, VisionTextDualEncoderProcessor, AutoModelForObjectDetection
from PIL import Image

MAX_BATCH = 100
SIZES_TO_TRY = [1]

import json
with open('params.json','r') as f:
    params = json.load(f)
CAPT_ENSEM_STRAT = params.get("CAPT_ENSEM_STRAT", 'sum')
BUFFER_THRESH = params.get("BUFFER_THRESH", 7)

################### Predict bounding box based on text ###################
def predict(images, texts):
    '''
    params:
    - PIL Image
    - string
    returns:
    - tuple containing best bounding box
    '''
    if len(images) > MAX_BATCH: # prevent OOM
        preds = []
        for i in range(0, len(images), MAX_BATCH):
            preds += predict(images[i:(i+MAX_BATCH)], texts[i:(i+MAX_BATCH)])
        return preds

    all_bboxes = [
        [tuple(box) for bbox in bboxes for box in try_various_size(bbox.tolist())]
        for bboxes in object_detector(images)
    ]
    all_cropped_imgs, copies = image_cropper(images, all_bboxes, texts)
    scores = [image_caption_score(cropped_imgs, text) for cropped_imgs, text in zip(all_cropped_imgs,texts)]
    preds = []
    for i in range(len(all_bboxes)):
        x1,y1,x2,y2 = all_bboxes[i][scores[i].argmax()//copies]
        preds.append((x1,y1,x2-x1,y2-y1))
    return preds

def scale(x1, y1, x2, y2, k):
    wid, height = (x2-x1)*k, (y2-y1)*k
    mid = (x1+x2)/2, (y1+y2)/2
    bbox = [mid[0]-wid/2, mid[1]-height/2, mid[0]+wid/2, mid[1]+height/2]
    return [t for t in bbox]
def try_various_size(bbox):
    x1,y1,x2,y2 = bbox
    ans = []
    for delt in SIZES_TO_TRY:
        ans.append(scale(x1,y1,x2,y2, delt))
    return ans


################### Vision-Text Model to calculate image-caption score ###################
# caption_model_path="caption_models/roberta-vit-v1-1"
# caption_vision_text_config = VisionTextDualEncoderConfig.from_pretrained(caption_model_path)
# caption_model = VisionTextDualEncoderModel.from_pretrained(caption_model_path, config=caption_vision_text_config)
# caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_path)
# caption_image_processor = AutoImageProcessor.from_pretrained(caption_model_path)
# caption_processor = VisionTextDualEncoderProcessor(caption_image_processor, caption_tokenizer)

caption_weights = None
caption_model_paths = ["caption_models/simcse-beit-v1", "caption_models/mpnet-planes-v1", "caption_models/roberta-vit-v1-1", "caption_models/roberta-beit-v1"]

caption_models = []
caption_processors = []
for caption_model_path in caption_model_paths:
    caption_vision_text_config = VisionTextDualEncoderConfig.from_pretrained(caption_model_path)
    caption_model = VisionTextDualEncoderModel.from_pretrained(caption_model_path, config=caption_vision_text_config)
    caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_path)
    caption_image_processor = AutoImageProcessor.from_pretrained(caption_model_path)
    caption_processor = VisionTextDualEncoderProcessor(caption_image_processor, caption_tokenizer)
    
    caption_models.append(caption_model)
    caption_processors.append(caption_processor)


def image_caption_score(images, text):
    
    global caption_models, caption_processors
    
    device = checkDevice()
    
    acc_logits = None #sum the image caption scores (same as average since taking max anw)
    i = 0
    
    for caption_model, caption_processor in zip(caption_models, caption_processors):
        with no_grad():
            inputs = caption_processor(text=text, images=images, return_tensors="pt", padding=True).to(device)
            caption_model.to(device)
            outputs_raw = caption_model(**inputs)
            logits_per_image = outputs_raw.logits_per_image
            outputs = logits_per_image.sigmoid()
        # print(logits_per_image)
        if acc_logits is None:
            acc_logits = logits_per_image
        else:
            if CAPT_ENSEM_STRAT=='min':
                acc_logits = torch_min(acc_logits, logits_per_image)
            elif CAPT_ENSEM_STRAT=='max':
                acc_logits = torch_max(acc_logits, logits_per_image)
            else:
                if caption_weights:
                    outputs *= caption_weights[i]
                acc_logits += logits_per_image
        #free up memory
        del inputs, outputs, outputs_raw, logits_per_image
        torch.cuda.empty_cache()
        i+=1
    # print(acc_logits)
    return acc_logits


# from ultralytics import YOLO
from ultralytics import RTDETR
# model = YOLO("yolov8m-v1.pt")
model = RTDETR("rtdetr-l-v1.pt")
#model.overrides['conf'] = 0.5  # NMS confidence threshold
# model.overrides['iou'] = IOU_THRESH  # NMS IoU threshold
def object_detector(images):
    results = model(source=images, device=checkDevice(), save=False, save_txt=False)

    return [(result.boxes.xyxy) for result in results]


################### Object Detection with DETR ###################'

# od_model_paths= ["od_models/detr_model_v2_0.47620"] # , "od_models/detr_model_1.25093"]
# confidence_thresh = [0.7 ]#, 0.1]
# od_image_processors = [AutoImageProcessor.from_pretrained(od_model_path) for od_model_path in od_model_paths]
# od_models = [AutoModelForObjectDetection.from_pretrained(od_model_path) for od_model_path in od_model_paths]

# def object_detector(images):
#     global od_image_processors, od_models

#     device = checkDevice()
    
#     bbox_lists=None
    
#     for od_model, od_image_processor, thres in zip(od_models, od_image_processors, confidence_thresh):
#         od_model.to(device)
    
#         results = []

#         with no_grad():
#             inputs = od_image_processor(images=images, return_tensors="pt").to(device)
#             outputs = od_model(**inputs)
#             target_sizes = tensor([image.size[::-1] for image in images])
#             results = od_image_processor.post_process_object_detection(outputs, threshold=thres, target_sizes=target_sizes)        

#         all_bboxes = []

#         for result in results:

#             boxes = result['boxes']
#             scores = result['scores']

#             # Apply non-maximum suppression
#             keep = ops.nms(boxes, scores, iou_threshold=0.5)

#             all_bboxes.append([bbox for i, bbox in enumerate(boxes) if i in keep])
#         if bbox_lists is None: bbox_lists = all_bboxes
#         else:
#             for i,bboxes in enumerate(all_bboxes):
#                 bbox_lists[i].extend(bboxes)
#     return bbox_lists

################### Crop images based on bounding box ###################

def upsample(image, scale_factor = 1.1):
    # Define the scaling factor or the desired size
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    # new_size = (224, 224)

    # Upsample the image
    return image.resize(new_size, Image.LANCZOS)

# from diffusion_upscale import mass_upscale

def image_cropper(images, bboxes, texts):
    # add small buffer for better captioning accuracy
    all_cropped_images = []
    for i,image in enumerate(images):
        cropped_imgs = []
        for bbox in bboxes[i]: #x1,y1,x2,y2
            x1,y1,x2,y2 = bbox
            w,h = image.size
            k = BUFFER_THRESH
            x1 = max(x1-k, 0)
            y1 = max(y1-k, 0)
            x2 = min(x2+k, w-1)
            y2 = min(y2+k, h-1)
            small_img = images[i].crop((x1,y1,x2,y2))
            cropped_imgs.append(small_img)
            # cropped_imgs.append(upsample(small_img))
        all_cropped_images.append(cropped_imgs)
    # return mass_upscale(all_cropped_images, texts), 1
    return all_cropped_images, 1

def checkDevice():
    return 'cuda' if cuda.is_available() else 'cpu'

'''REQUIREMENTS:
torch==2.3.0
torchvision==0.18.0

'''


# ################### Archives ###################

# from torch import tensor, no_grad, cuda
# import torchvision.ops as ops
# from transformers import VisionTextDualEncoderConfig, VisionTextDualEncoderModel, AutoTokenizer, AutoImageProcessor, VisionTextDualEncoderProcessor, AutoModelForObjectDetection

# ################### Predict bounding box based on text ###################
# def predict(image, text):
#     '''
#     params:
#     - PIL Image
#     - string
#     returns:
#     - tuple containing best bounding box
#     '''
#     bboxes = [tuple(bbox.tolist()) for bbox in object_detector(image)]
#     cropped_imgs = image_cropper(image, bboxes)
#     scores = image_caption_score(cropped_imgs, text)
#     x1,y1,x2,y2 = bboxes[scores.argmax()]
#     return (x1,y1,x2-x1,y2-y1)


# ################### CLIP Model to calculate image-caption score ###################

# def image_caption_score(images, text, model_path="clip_models/clip-finetune-v1"):
    
#     vision_text_config = VisionTextDualEncoderConfig.from_pretrained(model_path)
#     model = VisionTextDualEncoderModel.from_pretrained(model_path, config=vision_text_config)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     image_processor = AutoImageProcessor.from_pretrained(model_path)
#     processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
    
#     device = checkDevice()
    
#     inputs = processor(text=text, images=images, return_tensors="pt", padding=True).to(device)
#     model = model.to(device)
#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image
#     return logits_per_image

# ################### Object Detection with DETR ###################

# def object_detector(image, model_path="od_models/detr_model_1.25093"):

#     image_processor = AutoImageProcessor.from_pretrained(model_path)
#     model = AutoModelForObjectDetection.from_pretrained(model_path)

#     device = checkDevice()

#     model.to(device)

#     with no_grad():
#         inputs = image_processor(images=image, return_tensors="pt").to(device)
#         outputs = model(**inputs)
#         target_sizes = tensor([image.size[::-1]])
#         results = image_processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

#     boxes = results['boxes']
#     scores = results['scores']

#     # Apply non-maximum suppression
#     keep = ops.nms(boxes, scores, iou_threshold=0.5)
    
#     return [bbox for i, bbox in enumerate(boxes) if i in keep]

# ################### Crop images based on bounding box ###################

# def image_cropper(image, bboxes):
#     cropped_imgs = []
#     for bbox in bboxes: #x1,y1,x2,y2
#         cropped_imgs.append(image.crop(bbox))
#     return cropped_imgs

# def checkDevice():
#     return 'cuda' if cuda.is_available() else 'cpu'

