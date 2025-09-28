declare -a values=("nonhate" "fairness" "objectivity" "honesty" "noncondescension" "clarity")

for v1 in "${values[@]}"
    do
        for v2 in "${values[@]}"
            do
                if [[ "$v1" < "$v2" ]]; then
                    python src/generate_scenarios.py -o data/modelspec/scenarios -m claude-3-5-sonnet-latest -v modelspec -n 240 -d -dt 0.8 -b 40 --max-tokens 4096 -v1 $v1 -v2 $v2
                    python src/filter_scenarios.py -i data/modelspec/scenarios/claude-3-5-sonnet-latest_${v1}_${v2}.json -o data/modelspec/outputs -m gpt-4.1 -v modelspec
                fi
            done
    done