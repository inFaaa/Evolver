
import argparse
import clip
import torch
from PIL import Image
import os
import json 
import numpy as np



def extract(root, data_path, model, preprocess, save_path):

    embed = []
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    with torch.no_grad():
      for item in data:
          text = item['text']
          img_path = os.path.join(root, item['img'])
          text = clip.tokenize([text], context_length=77, truncate=True).to('cuda')
          image = preprocess(Image.open(img_path)).unsqueeze(0).to('cuda')
          image_features = model.encode_image(image)
          text_features = model.encode_text(text)

          feature = image_features * .5 + text_features * .5
          embed.append(feature)
    embed = torch.cat(embed, dim=0)
    embed = embed.cpu().numpy()

    print('Saving Embedding')
    with open(save_path, 'wb') as f:
        np.save(f, embed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
    )
    parser.add_argument(
        "--root",
        type=str,
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
    )
    return parser.parse_known_args()


if __name__ == '__main__':

    args, _ = parse_args()
    print('Load Model')
    model, preprocess = clip.load("ViT-L/14@336px", device="cuda")
    model = model.to('cuda')
    model.eval()

    print('Mining')
    extract(args.root, args.data_path, model, preprocess, args.save_path)


