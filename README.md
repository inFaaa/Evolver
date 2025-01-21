<div align=center>
<img src="assets/evoler_logo.png" width="150px">
</div>
<h2 align="center"> <a href="https://arxiv.org/abs/2407.21004">Evolver: Chain-of-Evolution Prompting to Boost Large Multimodal Models for Hateful Meme Detection</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2407.21004-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2407.21004) 
   [![YouTube](https://img.shields.io/badge/-YouTube-000000?logo=youtube&logoColor=FF0000)](https://www.youtube.com/watch?v=SPloR9BF2-c)
[![github](https://img.shields.io/github/stars/inFaaa/Evolver.svg?style=social)](https://github.com/inFaaa/Evolver)

</h5>

<div align="center">
This repository is the official implementation of Evolver,  which incorporates LMMs via Chain-of-Evolution (CoE) Prompting, by integrating the evolution attribute and in-context information of memes.
</div>


# Poster
<div align=center>
<img src="assets/evolver_poster_COLING2025.png" width="500px">
</div>

Welcome to the COLING 2025 visual presentation at Jan 27, 10:00-11:30 AM EST, our paper id is 1767, if you have any question, we can discuss about it.

## Environment Installation
    pip install -r requirements.txt


## Test

#### Step 1: Extract embedding
need to run twice, to extract embedding for both the test set and the training set

      python src/extract_embed.py --save_path <Your_SAVE_PATH> --root <PATH_TO_DIR_OF_IMAGE_FOLDER> \
      --data_path <PATH_TO_TEST_SET_OR_TRAIN_SET>

#### Step 2: Pair mining

      python src/rank.py --db_embed_path <PATH_TO_TRAIN_EMBEDDING> \
      --test_embed_path <PATH_TO_TEST_EMBEDDING> \
      --test_data_path <PATH_TO_TEST_SET> \
      --train_data_path <PATH_TO_TRAIN_SET> \
      --save_path <SAVE_PATH> \
      --top_k <K_MOST_SIMILAR_DATA>

Then specify your hatefulness definition and instruction in <b> inference.py </b>

#### Step 3: EIE

    python inference.py --mode eie \
    --model_path <MODEL_PATH> \
    --extract_path <PATH_TO_SAVE_RESULT> \
    --image_folder <PATH_TO_DIR_OF_IMAGE_FOLDER> \
    --pool_path <PAIR_MINING_SAVE_PATH> \

#### Step 4: CRA

    python inference.py --mode cra \
    --model_path <MODEL_PATH> \
    --test_path <PATH_OF_EXTRACT_RESULT> \
    --save_path <SAVE_PATH> \
    --image_folder <PATH_TO_DIR_OF_IMAGE_FOLDER> \


## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{huang2024evolver,
  title={Evolver: Chain-of-Evolution Prompting to Boost Large Multimodal Models for Hateful Meme Detection},
  author={Huang, Jinfa and Pan, Jinsheng and Wan, Zhongwei and Lyu, Hanjia and Luo, Jiebo},
  journal={arXiv preprint arXiv:2407.21004},
  year={2024}
}
```

</a>
