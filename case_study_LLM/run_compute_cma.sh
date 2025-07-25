file_path='/Users/eva/Documents/Work/after_promotion/CPA/example/github_repos/Rank-Calibration/'

python compute_cma.py --rank_calibration_path "$file_path" --correctness bert_similarity
python compute_cma.py --rank_calibration_path "$file_path" --correctness rouge
python compute_cma.py --rank_calibration_path "$file_path" --correctness rouge1
python compute_cma.py --rank_calibration_path "$file_path" --correctness meteor

python compute_cma.py --rank_calibration_path "$file_path" --dataset nq-open --correctness bert_similarity
python compute_cma.py --rank_calibration_path "$file_path" --dataset nq-open --correctness rouge
python compute_cma.py --rank_calibration_path "$file_path" --dataset nq-open --correctness rouge1
python compute_cma.py --rank_calibration_path "$file_path" --dataset nq-open --correctness meteor

python compute_cma.py --rank_calibration_path "$file_path" --dataset squad --correctness bert_similarity
python compute_cma.py --rank_calibration_path "$file_path" --dataset squad --correctness rouge
python compute_cma.py --rank_calibration_path "$file_path" --dataset squad --correctness rouge1
python compute_cma.py --rank_calibration_path "$file_path" --dataset squad --correctness meteor


python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness bert_similarity
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness rouge
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness rouge1
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness meteor

python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness bert_similarity
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge1
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness meteor

python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness bert_similarity
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge1
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness meteor


python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --correctness bert_similarity
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --correctness rouge
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --correctness rouge1
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --correctness meteor

python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness bert_similarity
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness rouge
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness rouge1
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness meteor

python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness bert_similarity
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness rouge
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness rouge1
python compute_cma.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness meteor