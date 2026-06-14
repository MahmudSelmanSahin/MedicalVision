from ablation_csv_factory import run_scenarios, scenarios_for_models


if __name__ == "__main__":
    run_scenarios(scenarios_for_models(["knn"]))
