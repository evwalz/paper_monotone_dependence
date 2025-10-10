# Set your Rank-Calibration directory path here
RANK_CALIBRATION_PATH="path/to/Rank-Calibration/"

# Validate the path exists
if [ ! -d "$RANK_CALIBRATION_PATH" ]; then
    echo "Error: RANK_CALIBRATION_PATH directory does not exist: $RANK_CALIBRATION_PATH"
    echo "Please set the correct path in the USER CONFIGURATION section above"
    exit 1
fi

file_path="$RANK_CALIBRATION_PATH"

python compute_table.py --rank_calibration_path "$file_path" --correctness bert_similarity
python compute_table.py --rank_calibration_path "$file_path" --correctness rouge
python compute_table.py --rank_calibration_path "$file_path" --correctness rouge1
python compute_table.py --rank_calibration_path "$file_path" --correctness meteor

python compute_table.py --rank_calibration_path "$file_path" --dataset nq-open --correctness bert_similarity
python compute_table.py --rank_calibration_path "$file_path" --dataset nq-open --correctness rouge
python compute_table.py --rank_calibration_path "$file_path" --dataset nq-open --correctness rouge1
python compute_table.py --rank_calibration_path "$file_path" --dataset nq-open --correctness meteor

python compute_table.py --rank_calibration_path "$file_path" --dataset squad --correctness bert_similarity
python compute_table.py --rank_calibration_path "$file_path" --dataset squad --correctness rouge
python compute_table.py --rank_calibration_path "$file_path" --dataset squad --correctness rouge1
python compute_table.py --rank_calibration_path "$file_path" --dataset squad --correctness meteor


python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness bert_similarity
python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness rouge
python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness rouge1
python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness meteor

python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness bert_similarity
python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge
python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge1
python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness meteor

python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness bert_similarity
python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge
python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge1
python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness meteor


python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness bert_similarity
python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness meteor
python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness rouge
python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness rouge1

python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness bert_similarity
python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness meteor
python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness rouge
python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness rouge1

python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness bert_similarity
python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness meteor
python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness rouge
python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness rouge1