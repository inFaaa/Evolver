
import argparse
import os
from tqdm import tqdm
from copy import copy 
import torch
from utils import (
    load_image, 
    set_seed, 
    save_file,
    load_inference_model,
    load_extract_model
)
from PIL import Image 
import json
from sklearn.metrics import accuracy_score, roc_auc_score

IMAGE_PROMPT =  """Image-1: <image>\nImage-2: <image>\nImage-3: <image>\nImage-4: <image>\nImage-5: <image>"""

def evaluate(output_path):

    with open('./result3.json', 'r') as f:
        output = json.load(f)

    labels = []
    preds = []

    idx = 0
    for item in output:
        label = item['label']
        labels.append(label)
        try:
            result = item['result'].split("classification")[1].split(',')[0].replace(":", "").replace('"', '').strip()
            if 'not' in result.lower():
                preds.append(0)
            else:
                preds.append(1)
        except:
            print(item['result'])

    acc = accuracy_score(preds, labels)
    auc = roc_auc_score(labels, preds)
    print(f'Acc: {acc:.3f}')
    print(f'AUC: {auc:.3f}')
     

def inference_multi(model, tokenizer, instruction, data, image_folder, save_path):

    generation_config = dict(max_new_tokens=512, temperature=0.2, do_sample=True)
    output = copy(data)
    pbar = tqdm(total=len(data))
    prompt = IMAGE_PROMPT + '\n' + instruction

    for idx, item in enumerate(data):
        evolution = item['evolution']
        imgs = [load_image(os.path.join(image_folder, e['img']), max_num=5).to(torch.bfloat16).cuda() for e in evolution]
        pixel_values = torch.cat(imgs, dim=0)
        num_patches_list = [item.size(0) for item in imgs]

        response, history = model.chat(tokenizer, pixel_values, prompt, generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        output[idx]['summary'] = copy(response)
        pbar.update(1)
        save_file(save_path=save_path, output=output)

    pbar.close()
    return output

def inference_evolution(model, processor, data, image_folder, instruction, definition, save_path):

  output = copy(data)
  pbar = tqdm(total=len(data))

  for idx, item in enumerate(data):
    img = Image.open(os.path.join(image_folder, item['img'])).convert('RGB')
    ocr_text = item['text']
    evolution = item['summary']

    prompt = f"""Instruction:

      You are provided with an image and its caption:

      Caption: {ocr_text}

      Context:

      The image-text pair is derived from the following evolutionary pairs summarized below:

      Summarization:
      {evolution}

      Task:

      {instruction}

      {definition}

      Response Format:

      Provide your answer in the following JSON format:
      {{
        "classification": "<YOUR PREDICTION HERE>",
        "reason": "<YOUR REASON HERE>"
      }}

      """

    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": "You are an expert on hateful speech detection. "},
              {"type": "image"},
              {"type": "text", "text": f"{prompt}"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=img, text=prompt, return_tensors='pt').to("cuda:0", torch.float16)
    start = inputs.input_ids.shape[1]
    out = model.generate(**inputs, max_new_tokens=50, temperature=0.0002, do_sample=True)
    out = processor.decode(out[0][start:], skip_special_tokens=True).strip()
    
    output[idx]['result'] = copy(out)
    pbar.update(1)
    save_file(save_path=save_path, output=output)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=['eie', 'cra'],
        help='mode to summarize similar meme or inference with summarization',
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        help='model path',
    )

    parser.add_argument(
        "--test_path",
        type=str,
        help='test data path',
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        help='path to save final result',
    )

    parser.add_argument(
        "--extract_path",
        type=str,
        help='path to save summarization',
    )

    parser.add_argument(
        "--pool_path",
        type=str,
        help=''
    )

    parser.add_argument(
        "--image_folder",
        type=str,
        help='image folder'
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0", 
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    return parser.parse_known_args()

if __name__ == '__main__':
    args, _ = parse_args()
    set_seed(args.seed)

    if args.mode.lower() == 'eie':
        model, tokenizer = load_extract_model(args.model_path, args.device)
        data = json.load(open(args.pool_path))
        instruction = "<YOUR INSTRUCTION HERE>"

        inference_multi(
            model=model, 
            tokenizer=tokenizer, 
            instruction=instruction, 
            data=data, 
            image_folder=args.image_folder,
            save_path=args.extract_path
        )

    elif args.mode.lower() == 'cra':
        model, processor = load_inference_model(args.model_path, args.device)
        data = json.load(open(args.test_path))
        instruction = "<YOUR INSTRUCTION HERE>"
        definition = "<YOUR HATE DEFINITION HERE>"

        inference_evolution(
            model=model, 
            processor=processor, 
            data=data, 
            image_folder=args.image_folder, 
            instruction=instruction, 
            definition=definition, 
            save_path=args.save_path)
        evaluate(args.save_path)
    print('Finished')

