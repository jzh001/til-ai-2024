import re
from transformers import AutoModelForQuestionAnswering, AutoTokenizer #, BertForQuestionAnswering
import torch

'''
TAG='v1-fixed'
docker build -t anti-heroes-$TAG-nlp .
docker tag anti-heroes-$TAG-nlp asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-anti-heroes/anti-heroes-$TAG-nlp:latest
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-anti-heroes/anti-heroes-$TAG-nlp:latest
gcloud ai models upload --region asia-southeast1 --display-name anti-heroes-$TAG-nlp --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-anti-heroes/anti-heroes-$TAG-nlp:latest --container-health-route /health --container-predict-route /extract --container-ports 5002 --version-aliases default
'''

############################### Predict ###############################

model_dir = 'models/roberta-base-squad2-v2'

def checkDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initModelAndTokenizer(model_dir):
    device = checkDevice()
    model_path = model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = None
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model = model.to(device)
    return model, tokenizer

model, tokenizer = initModelAndTokenizer(model_dir)  # init on first run
# model, tokenizer = None, None

def predict(transcript):
    global model, tokenizer, model_dir
    # Try classical methods
    # Note: None means error
    
    device = checkDevice()
    
    headings = findHeading(transcript)
    
    # heading = None # Uncomment only during testing
    if len(headings)==1:
        heading = headings[0]
    elif len([h for h in headings if len(h)==3])==1:
        heading = [h for h in headings if len(h)==3][0]
    else:
        if model == None:
            model, tokenizer = initModelAndTokenizer(model_dir)
        heading = askModel(model, tokenizer, "Where is the heading?", transcript)
        heading = words2digit(strict_word_tokenize(heading.strip())[0])
        if len(headings)>0 and heading not in headings:
            heading = headings[0]
    
    if heading == None:
        heading = "000"
    
    # tool = findTool(transcript)
    
    tool = None # Uncomment only during testing
    if tool == None:
        if model == None:
            model, tokenizer = initModelAndTokenizer(model_dir)
        tool = askModel(model, tokenizer, "What is the tool to deploy?", transcript).replace(' - ','-')

        if tool == 'emp':
            tool = "EMP"
    
    # target = findTarget(transcript)
    
    target = None # Uncomment only during testing
    if target == None:
        if model == None:
            model, tokenizer = initModelAndTokenizer(model_dir)
        target = askModel(model, tokenizer, "What is the target to engage?", transcript)

    return {
        'target': target.strip(),
        'heading': heading,
        'tool': tool.strip(),
    }
    
############################### BERT QA ###############################


def askModel(model, tokenizer, question, context):
    device = checkDevice()
    # Tokenize the context to find the exact start and end position of the answer
    encoded = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = encoded["input_ids"].tolist()[0]
    
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded.to(device))

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Convert tokens to answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    return answer

############################### Heading ###############################



def findHeading(sent):
    words, _ = strict_word_tokenize(sent.lower())

    possible_headings = sliding_window_heading(words)
    return [heading for heading in possible_headings if int(heading)>=0 and int(heading)<=360]

def words2digit(words):
    '''
    Converts heading in words to digits.
    words:list(string)
    return:int
    '''
    digits = ['zero','one','two','three','four','five','six','seven','eight','nine']

    word2digit = {word:i for i,word in enumerate(digits)}
    word2digit['niner'] = 9
    if any([word not in word2digit for word in words]):
        return None
    
    return ''.join(str(word2digit[word]) for word in words)

def sliding_window_heading(words):
    possible_answers = []
    for i in range(len(words)-2):
        res = words2digit(words[i:i+3])
        if res != None:
            possible_answers.append(res) # if no error
    for i in range(len(words)-1):
        if i>0 and words2digit(words[i-1:i]) is not None:
            continue
        if i+2<len(words) and words2digit(words[i+2:i+3]) is not None:
            continue
        res = words2digit(words[i:i+2])
        if res != None:
            possible_answers.append(res)
    return possible_answers
    
############################### Tool ###############################

def findTool(sent):
    '''
    Use sliding window to find tool
    '''
    tools = ['electromagnetic pulse', 'surface-to-air missiles', 'EMP',
       'machine gun', 'anti-air artillery', 'interceptor jets',
       'drone catcher']
    for idx in range(len(sent)):
        for tool in tools:
            if idx + len(tool) <= len(sent) and tool.lower() == sent[idx:idx+len(tool)].lower():
                return tool
    return None

############################### Target ###############################

def findTarget(sent):
    targets = ['aircraft', 'drone', 'helicopter', 'jet', 'missile', 'plane']
    attributes = ['black', 'blue', 'brown','camouflage','cargo','commercial','fighter','green','grey','light','orange','purple','red','silver','white','yellow']
    sent = sent.lower()
    tokens, indexes = strict_word_tokenize(sent)
    st, en = -1,-1
    for token, idx in zip(tokens, indexes):
        if token in attributes and st == -1:
            st = idx[0]
        elif (token in targets or token[:-1] in targets) and st != -1:
            en = idx[1]
            if sent[en-1] == 's':
                en -= 1
            return sent[st:en]
    
    return None
    
############################### Tokenization ###############################

def sent_tokenize(transcript):
    return re.split('[.,]', transcript)

def word_tokenize(sent):
    return sent.split()

def strict_word_tokenize(sent):
    #return [x for x in re.split('[., ]', sent) if x != '']
    words = [m.group(0) for m in re.finditer(r'[\w]+', sent)]
    indexes = [(m.start(), m.end()) for m in re.finditer(r'[\w]+', sent)]
    return words, indexes