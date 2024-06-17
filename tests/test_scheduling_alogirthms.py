import pytest
import os
import json
import sys

# Adjust path to include the 'src' directory for importing algorithms
script_dir = os.path.dirname(__file__)
input_models_dir = os.path.join(script_dir, "input_models")
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "src")))
from algorithms import edf_multinode, ll_multinode, ldf_multinode


# Utility function to load models and run scheduling algorithm
def load_and_schedule(filename):
    model_path = os.path.join(input_models_dir, filename)
    with open(model_path) as f:
        model_data = json.load(f)

    application_model = model_data["application"]
    platform_model = model_data["platform"]
    results = []
    for algo in [edf_multinode, ll_multinode, ldf_multinode]:
        result = algo(application_model, platform_model)
        results.append((result, application_model))
    return results


@pytest.mark.parametrize("filename", os.listdir(input_models_dir))
def test_task_duration(filename):
    """Test that each task completes within its estimated duration."""
    for result, app_model in load_and_schedule(filename):
        for task in result["schedule"]:
            start_time = task["start_time"]
            end_time = task["end_time"]
            task_id = task["task_id"]
            wcet = next(
                (t["wcet"] for t in app_model["tasks"] if t["id"] == task_id), None
            )
            assert end_time == start_time + wcet, "Incorrect task duration calculation"


@pytest.mark.parametrize("filename", os.listdir(input_models_dir))
def test_task_deadline(filename):
    """Test that each task respects its deadline."""
    for result, app_model in load_and_schedule(filename):
        for task in result["schedule"]:
            end_time = task["end_time"]
            task_id = task["task_id"]
            deadline = next(
                (t["deadline"] for t in app_model["tasks"] if t["id"] == task_id), None
            )
            assert end_time <= deadline, "Task exceeds deadline"


@pytest.mark.parametrize("filename", os.listdir(input_models_dir))
def test_task_dependencies(filename):
    """Test that each task respects the completion times of its predecessors."""
    for result, app_model in load_and_schedule(filename):
        for task in result["schedule"]:
            start_time = task["start_time"]
            task_id = task["task_id"]
            predecessors = [
                msg["sender"]
                for msg in app_model["messages"]
                if msg["receiver"] == task_id
            ]
            predecessors_end_times = [
                t["end_time"]
                for t in result["schedule"]
                if t["task_id"] in predecessors
            ]
            assert start_time >= max(
                predecessors_end_times, default=0
            ), "Task starts before predecessor ends"
