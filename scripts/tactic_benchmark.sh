#!/usr/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

source /home/thomaszh/.bashrc
cd /home/thomaszh/ntp-toolkit
lake build
cd /home/thomaszh/premise-retrieval
conda activate lm

mkdir -p /scratch/thomaszh

MAX_WORKERS=16
SPLIT=test

    # all-mpnet-base-v2-lr7e-5-bs256-nneg3-ml \
    # all-roberta-large-v1-lr5e-5-bs256-nneg3-ml \
    # all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-naive \
    # gt-naive-blacklist \

    # all-MiniLM-L6-v2-lr2e-4-bs256-nneg3-ml-ne5 \
    # gt \
    # gt-naive \
    # all-MiniLM-L12-v2-lr2e-4-bs256-nneg3-ne5 \
    # all-MiniLM-L12-v2-lr2e-4-bs256-nneg0-ml-ne5 \
    # all-MiniLM-L12-v2-lr2e-4-bs256-nneg0-ne5 \
    # all-MiniLM-L12-v2-lr2e-4-bs256-nneg3-ml-ne5-naive \
    # leandojo-lean4-retriever-byt5-small-hammer \
    # rf \
    # knn \
    # mepo \
    # all-mpnet-base-v2-lr1e-4-bs256-nneg3-ml-ne5 \
    # all-MiniLM-L12-v2-lr2e-4-bs256-nneg3-ml-ne5 \
for MODEL_NAME in \
    all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-ne5 \
; do
    PREMISES_RETRIEVED=retrieved_premises/dot_$SPLIT-$MODEL_NAME.json

    # run export_decls.py
    DECL_NAMES=retrieved_premises/${SPLIT}_decls_apr25.json
    if [[ "$MODEL_NAME" == "gt-naive" ]]; then
        DECL_NAMES=retrieved_premises/${SPLIT}_decls_naive.json
    elif [[ "$MODEL_NAME" == "gt-naive-blacklist" ]]; then
        DECL_NAMES=retrieved_premises/${SPLIT}_decls_naive-blacklist.json
    fi

    # PREMISES_RETRIEVED_WITH_HINTS=retrieved_premises/simp_all_hints_$SPLIT-ce-all-distilroberta-v1-lr2e-4-bs1024-nneg3-mlbs-lr1e-5-bs64-nneg8-ml.json

    # dec13:
    # jan13: v4.14, add simp all hints
    # feb04: notation change for training data
    # feb10: notation change for declarations
    # feb22: v4.15
    # mar03: v4.16
    # mar11: fix contamination
    # mar30: server refactor
    # apr25: v4.18
    RESULTS_DIR=results_apr25-$SPLIT/$MODEL_NAME

    K=32
    KHC=16
    AHP=10
    APP=20

    common_args="--max_workers $MAX_WORKERS --decl_names_file $DECL_NAMES --out_dir $RESULTS_DIR"

    ## GT
    if [[ $MODEL_NAME =~ ^gt ]]; then # starts with "gt"
        # 1
        python tactic_benchmark.py $common_args --benchmark_type aesop_with_premises
        # 9
        python tactic_benchmark.py $common_args --benchmark_type hammerCore_nosimp
        # 13
        python tactic_benchmark.py $common_args --benchmark_type aesop_hammerCore_nosimp
        # 17: gt, aesop+premises+hammer
        python tactic_benchmark.py $common_args --benchmark_type aesop_hammerCore_nosimp_with_premises

        # Baselines without retrieval
        # python tactic_benchmark.py $common_args --benchmark_type simp_all
        # python tactic_benchmark.py $common_args --benchmark_type rfl
        # python tactic_benchmark.py $common_args --benchmark_type exact
        # python tactic_benchmark.py $common_args --benchmark_type aesop
        # python tactic_benchmark.py $common_args --benchmark_type omega

    else
        # 7 (TODO: think about simp hints)
        # python tactic_benchmark.py $common_args --benchmark_type hammerCore --pred_simp_all_hint
        # 11 (TODO: think about simp hints)
        # python tactic_benchmark.py $common_args --benchmark_type aesop_hammerCore --pred_simp_all_hint

        ## On retrieved
        # 2
        python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED --benchmark_type aesop_with_premises --k $K
        # # 4
        # # Don't python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED --benchmark_type simp_all_with_premises --k $K
        # 10
        # for KHC in 8 12 16 20 24; do
        python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED --benchmark_type hammerCore_nosimp --k $KHC
        python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED --benchmark_type duper --k $KHC
        # done
        # 14
        python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED --benchmark_type aesop_hammerCore_nosimp --k $KHC
        # 18
        # for KHC in 8 12 16 20 24; do
        #     for K in 48 40 32 24 16; do
        # for APP in 5 10 20 40 80; do
        python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED --benchmark_type aesop_hammerCore_nosimp_with_premises --k $K --k_hammerCore $KHC --aesopHammerPriority $AHP --aesopPremisePriority $APP
        #     done
        # done

        # 8 (TODO: think about simp hints)
        # python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED_WITH_HINTS --benchmark_type hammerCore --k $K --pred_simp_all_hint --rerank
        # python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED_WITH_HINTS --benchmark_type hammerCore --k $K --rerank
        # python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED_WITH_HINTS --benchmark_type hammerCore --k $K --pred_simp_all_hint
        # 12 (TODO: think about simp hints)
        # python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED_WITH_HINTS --benchmark_type aesop_hammerCore --k $K --pred_simp_all_hint --rerank
        # python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED_WITH_HINTS --benchmark_type aesop_hammerCore --k $K --rerank
        # python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED_WITH_HINTS --benchmark_type aesop_hammerCore --k $K --pred_simp_all_hint
        # 15 (TODO: think about simp hints)
        # python tactic_benchmark.py $common_args --premises_file $PREMISES_RETRIEVED --benchmark_type aesop_hammer --k $K --pred_simp_all_hint

        # ## On sever
        # # Expect same as 2
        # python tactic_benchmark.py $common_args --max_workers 1 --premises_file $PREMISES_RETRIEVED --benchmark_type aesop_with_selector --k $K
        # # Expect to be the same as 10
        # python tactic_benchmark.py $common_args --max_workers 1 --premises_file $PREMISES_RETRIEVED --benchmark_type hammer_nosimp --k $KHC
        # # 16 (lower the worker count so that server is ok)
        # python tactic_benchmark.py $common_args --max_workers 1 --premises_file $PREMISES_RETRIEVED --benchmark_type aesop_hammer_nosimp --k $KHC
        # # Expect same as 18
        # python tactic_benchmark.py $common_args --max_workers 1 --premises_file $PREMISES_RETRIEVED --benchmark_type aesop_hammer_nosimp_with_selector --k $K --k_hammerCore $KHC --aesopHammerPriority $AHP --aesopPremisePriority $APP
    fi
done
