
python src/extract_embed.py --save_path <Your_SAVE_PATH> \
--root <PATH_TO_DIR_OF_IMAGE_FOLDER> \
--data_path <PATH_TO_TEST_SET_OR_TRAIN_SET>

python src/extract_embed.py --save_path <Your_SAVE_PATH> --root <PATH_TO_DIR_OF_IMAGE_FOLDER> \
--data_path <PATH_TO_TEST_SET_OR_TRAIN_SET>

python src/rank.py --db_embed_path <PATH_TO_TRAIN_EMBEDDING> \
--test_embed_path <PATH_TO_TEST_EMBEDDING> \
--test_data_path <PATH_TO_TEST_SET> \
--train_data_path <PATH_TO_TRAIN_SET> \
--save_path <SAVE_PATH> \
--top_k <K_MOST_SIMILAR_DATA>

python inference.py --mode eie \
--model_path OpenGVLab/InternVL2-8B \
--extract_path <PATH_TO_SAVE_RESULT> \
--image_folder <PATH_TO_DIR_OF_IMAGE_FOLDER> \
--pool_path <PAIR_MINING_SAVE_PATH> \

python inference.py --mode cra \
--model_path llava-hf/llava-v1.6-vicuna-13b-hf \
--test_path <PATH_OF_EXTRACT_RESULT> \
--save_path <SAVE_PATH> \
--image_folder <PATH_TO_DIR_OF_IMAGE_FOLDER> \